#!/usr/bin/env python3
"""Caption purchase JSONL shards from tar ranges without materializing images."""
from __future__ import annotations

import argparse
import base64
import ctypes
import gc
import hashlib
import json
import os
import re
import resource
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from . import label_caption_jsonl as base
    from .final_caption_prompt_tar import render_final_prompt
    from .vlm_client import VLMConfig
except ImportError:
    import label_caption_jsonl as base
    from final_caption_prompt_tar import render_final_prompt
    from vlm_client import VLMConfig


DEFAULT_INPUT = Path("/home/ma-user/work/others/w00911759/dataset_0827/purchase_panguml_with_meta_info/purchase_caption/0814_data_scv1/annotation")
DEFAULT_IMAGE_ROOT = Path("/mnt-mm-pro/s3-g-szc-asset-aifm-mm-exp/j00939224/data/0814_data_scv1/images")
PROJECT = Path(__file__).resolve().parents[1]
QWEN_38_API_BASE = "http://127.0.0.1:8000/v1"
QWEN_38_MODEL = "/home/ma-user/work/others/w00911759/models/qwen-3.8-27B"


def load_jsonl_rows(
    path: Path,
    *,
    attempts: int = 5,
    retry_delay_seconds: float = 1.0,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Load object records, logging and skipping lines that remain malformed."""
    if attempts < 1:
        raise ValueError("attempts must be positive")

    for attempt in range(1, attempts + 1):
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if attempt < attempts:
                            if logger is not None:
                                logger(
                                    f"JSONL parse failed at {path}:{line_number}:{exc.colno} "
                                    f"({exc.msg}); retrying {attempt}/{attempts}"
                                )
                            time.sleep(retry_delay_seconds)
                            break
                        if logger is not None:
                            excerpt = line[
                                max(0, exc.pos - 80):exc.pos + 80
                            ].rstrip("\r\n")
                            logger(
                                f"FAILURE line={line_number} stage=jsonl_parse "
                                f"error={exc.msg} column={exc.colno} near={excerpt!r}"
                            )
                        continue
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
                    rows.append(value)
                else:
                    return rows
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"input JSONL is not valid UTF-8 at {path}, byte {exc.start}"
            ) from exc

    raise AssertionError("unreachable")


def read_image(row: dict[str, Any], image_root: Path, tarinfo_cache: dict[Path, dict[str, Any]]) -> bytes:
    relative_path: str | None = None

    def find(value: Any) -> None:
        nonlocal relative_path
        if relative_path is not None:
            return
        if isinstance(value, dict):
            if value.get("type") == "relative_path" and isinstance(value.get("relative_path"), str):
                relative_path = value["relative_path"]
                return
            for child in value.values():
                find(child)
        elif isinstance(value, list):
            for child in value:
                find(child)

    find(row.get("data"))
    if not relative_path:
        raise ValueError("record has no data image relative_path")
    # The shard name is authoritative; img_path is retained as a fallback for
    # older records that do not carry the internal shard marker.
    tar_name = str(row.get("_tar_name") or Path(str(row.get("img_path", "")).split("|", 1)[0]).name)
    tar_path = image_root / tar_name
    tarinfo_path = image_root / f"{Path(tar_name).stem}_tarinfo.json"
    if tarinfo_path not in tarinfo_cache:
        tarinfo_cache[tarinfo_path] = json.loads(tarinfo_path.read_text(encoding="utf-8"))
    entry = tarinfo_cache[tarinfo_path].get(relative_path)
    if entry is None:
        entry = tarinfo_cache[tarinfo_path].get(relative_path.lstrip("/"))
    if entry is None:
        raise KeyError(f"{relative_path!r} not found in {tarinfo_path}")
    offset, size = int(entry["offset_data"]), int(entry["size"])
    with tar_path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated tar range: {tar_path}|{offset}|{size}")
    return data


def image_md5(data: bytes) -> str:
    digest = hashlib.md5()
    for start in range(0, len(data), 1024 * 1024):
        digest.update(data[start:start + 1024 * 1024])
    return digest.hexdigest()


def labels(row: dict[str, Any], field: str) -> list[str]:
    value = row.get("meta_info", {}).get("label", {}).get(field)
    if not value:
        raise ValueError(f"missing meta_info.label.{field}")
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        text = item.get("label", item) if isinstance(item, dict) else item
        result.extend(part.strip() for part in str(text).split("-") if part.strip())
    return result



def taxonomy_label_names(forms: list[dict[str, Any]], *, domain: bool) -> dict[str, str]:
    names: dict[str, str] = {}
    for form in forms:
        identifier = form.get("id")
        label = form.get("taxonomy_label") or form.get("label")
        if isinstance(identifier, str) and isinstance(label, str) and label.strip():
            names[identifier] = label.strip()
        for child in form.get("children", []):
            child_id = child.get("leaf_id") if domain else child.get("id")
            child_label = child.get("leaf_label") or child.get("label")
            if isinstance(child_id, str) and isinstance(child_label, str) and child_label.strip():
                names[child_id] = child_label.strip()
    return names


def render_label_prior(
    refs: tuple[base.LabelRef, ...],
    english_names: dict[str, str],
    taxonomy_names: dict[str, str],
    name: str,
) -> str:
    rendered: list[str] = []
    for ref in refs:
        identifier = ref.child_id or ref.parent_id
        label = english_names.get(identifier) or taxonomy_names.get(identifier)
        if not label:
            raise ValueError(f"no English or Chinese {name} label for {identifier}")
        rendered.append(label)
    return "; ".join(rendered)


def make_job(
    row: dict[str, Any],
    key: str,
    domain_index: dict[str, set[base.LabelRef]],
    genre_index: dict[str, set[base.LabelRef]],
    domain_names: dict[str, str],
    genre_names: dict[str, str],
) -> dict[str, Any]:
    domain = base.resolve_prior(labels(row, "domain_label"), domain_index, "domain")
    genre = base.resolve_prior(labels(row, "genre_label"), genre_index, "genre")
    domain_prior = render_label_prior(domain.refs, base.DOMAIN_ENGLISH_LABELS, domain_names, "domain")
    genre_prior = render_label_prior(genre.refs, base.GENRE_ENGLISH_LABELS, genre_names, "genre")
    stem = row.get("meta_info", {}).get("stem_ext", {})
    prompt = render_final_prompt(
        domain_prior=domain_prior,
        genre_prior=genre_prior,
        subject_prior=stem.get("subject_labels", {}).get("primary_label"),
        course_prior="; ".join(stem.get("course", [])),
        knowledge_points_prior="; ".join(stem.get("knowledge_points", [])),
    )
    return {
        "key": key, "row": row, "md5": row["_image_md5"], "prompt": prompt,
        "prompt_file": None,
        "genre_prompt_file": None,
        "domain_prior": domain_prior, "genre_prior": genre_prior,
        "route_candidates": [],
        "genre_route_candidates": [],
    }


def call_api(job: dict[str, Any], args: argparse.Namespace, config: Any, client: Any) -> tuple[str, Any, Any]:
    data: bytes | None = None
    data_url: str | None = None
    content: list[dict[str, Any]] | None = None
    parts: list[str] | None = None
    response: Any = None
    try:
        data = read_image(job["row"], args.image_root, job.get("_tarinfo_cache", {}))
        data_url = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        # Raw bytes are no longer needed after encoding.  Release them before
        # the blocking request so 200 workers do not retain two image copies.
        data = None
        content = []
        parts = job["prompt"].split("<image 1>")
        for index, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if index < len(parts) - 1:
                content.append({"type": "image_url", "image_url": {"url": data_url, "detail": args.image_detail}})
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": content}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body={
                    "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
            #         reasoning_effort=os.environ.get(
            #     "VLM_REASONING_EFFORT", "medium" if args.use_qwen_38 else "high"
            # ),
        choices = response.get("choices") if isinstance(response, dict) else response.choices
        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else choice.message
        prediction = message.get("content") if isinstance(message, dict) else message.content
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else choice.finish_reason
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        return prediction, finish_reason, usage
    finally:
        # Always execute for image errors, API failures, timeouts and malformed
        # responses.  A failed request must not retain image payloads locally.
        data = None
        data_url = None
        content = None
        parts = None
        response = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-jsonl", type=Path, help="只处理指定的单个 JSONL 文件，处理完成后退出")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-file", type=Path, help="同时写入终端和该文件的运行日志")
    parser.add_argument("--log-dir", type=Path, help="每个输入 JSONL 单独保存一个日志文件的目录")
    parser.add_argument("--resume", action="store_true", help="复用已有输出，跳过已经处理的记录")
    parser.add_argument("--start-annotation", type=int, help="从指定 annotation 序号开始处理，例如 34 对应 part_00034.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录中的同名 JSONL")
    parser.add_argument("--dry-run", action="store_true", help="只读图、计算 MD5、路由并生成 prompt，不调用 VLM API")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--tar-name", help="固定使用的 tar 分片名（不含 .tar），例如 part_000")
    parser.add_argument("--data-workers", type=int, default=16)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--md5-log-interval", type=int, default=100, help="MD5 进度日志间隔")
    parser.add_argument("--memory-log-interval", type=int, default=100, help="API 完成进度和内存日志间隔")
    parser.add_argument("--prompt-dir", type=Path, default=PROJECT / "caption_promptV3/caption_prompt")
    parser.add_argument("--assignments", type=Path, default=PROJECT / "domain_and_genre_labels/caption_prompt_label_assignments_v1.2.json")
    parser.add_argument("--domain-mapping", type=Path, default=PROJECT / "domain_and_genre_labels/domain_label_mapping_v1.1.json")
    parser.add_argument("--genre-mapping", type=Path, default=PROJECT / "domain_and_genre_labels/genre_label_mapping_v1.1.json")
    parser.add_argument("--config", type=Path, default=PROJECT / "vlm_api.yaml")
    parser.add_argument(
        "--use-qwen-38",
        action="store_true",
        help="使用当前部署在 127.0.0.1:8000 的 qwen-3.8-27B；不传时保持原配置文件逻辑",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="auto")
    return parser.parse_args()


def memory_usage_mb() -> tuple[float, float]:
    """Return current and peak RSS in MiB without allocating large objects."""
    current_kib = 0
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    current_kib = int(line.split()[1])
                    break
    except (FileNotFoundError, OSError, ValueError):
        pass
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return current_kib / 1024, peak_kib / 1024

def release_memory() -> None:
    """Collect Python cycles and return free glibc heap pages when available."""
    gc.collect()
    try:
        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim")
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def main() -> int:
    args = parse_args()
    if args.data_workers < 1 or args.api_workers < 1:
        raise SystemExit("worker counts must be positive")
    if args.md5_log_interval < 1:
        raise SystemExit("--md5-log-interval must be positive")
    if args.memory_log_interval < 1:
        raise SystemExit("--memory-log-interval must be positive")
    args.image_root = args.image_root.resolve()
    output_dir = args.output_dir or args.input_dir.with_name(args.input_dir.name + "_captioned")
    log_dir = args.log_dir or (args.log_file.parent if args.log_file else output_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    current_log_handle: Any = None
    log_lock = threading.Lock()
    # With --tar-name all split JSONLs use the same (potentially very large)
    # tarinfo.  Keep exactly one parsed copy for the complete process.
    fixed_tarinfo_cache: dict[Path, dict[str, Any]] = {}

    if args.input_jsonl is not None:
        input_path = args.input_jsonl
        if not input_path.is_absolute() and not input_path.is_file():
            input_path = args.input_dir / input_path
        input_path = input_path.resolve()
        if input_path.suffix != ".jsonl":
            raise SystemExit(f"--input-jsonl must point to a .jsonl file: {input_path}")
        if not input_path.is_file():
            raise SystemExit(f"--input-jsonl does not exist or is not a file: {input_path}")
        input_paths = [input_path]
    else:
        input_paths = sorted(args.input_dir.glob("*.jsonl"))

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, file=sys.stderr, flush=True)
        if current_log_handle is not None:
            with log_lock:
                current_log_handle.write(line + "\n")
                current_log_handle.flush()

    def log_memory(prefix: str) -> None:
        rss_mb, peak_mb = memory_usage_mb()
        log(f"MEMORY {prefix}: rss={rss_mb:.1f}MiB peak_rss={peak_mb:.1f}MiB")

    def close_current_log() -> None:
        nonlocal current_log_handle
        if current_log_handle is not None:
            current_log_handle.flush()
            current_log_handle.close()
            current_log_handle = None

    domains = base.load_json(args.domain_mapping)["forms"]
    genres = base.load_json(args.genre_mapping)["forms"]
    domain_index = base.build_label_index(domains, domain=True)
    genre_index = base.build_label_index(genres, domain=False)
    domain_names = taxonomy_label_names(domains, domain=True)
    genre_names = taxonomy_label_names(genres, domain=False)
    if args.dry_run:
        config = None
    elif args.use_qwen_38:
        config = VLMConfig(
            api_base=os.environ.get("QWEN_38_API_BASE", QWEN_38_API_BASE),
            api_key=os.environ.get("QWEN_38_API_KEY", "EMPTY"),
            model=os.environ.get("QWEN_38_MODEL", QWEN_38_MODEL),
        )
        log(f"API backend=qwen_38 base={config.api_base} model={config.model}")
    else:
        config = base.load_config(args.config)
    try:
      for input_path in input_paths:
        annotation_match = re.search(r"(\d+)$", input_path.stem)
        annotation_number = int(annotation_match.group(1)) if annotation_match else None
        if args.input_jsonl is None and args.start_annotation is not None and annotation_number is not None and annotation_number < args.start_annotation:
            continue
        close_current_log()
        current_log_handle = (log_dir / f"{input_path.stem}.log").open("w", encoding="utf-8")
        log(f"start processing: {input_path}")
        log_memory("jsonl_start")
        output_path = output_dir / input_path.name
        if output_path.exists() and not args.resume and not args.overwrite:
            raise FileExistsError(f"output exists: {output_path}; use --resume or --overwrite")
        existing_rows: list[dict[str, Any]] = []
        completed_ids: set[tuple[Any, Any]] = set()
        if args.resume:
            # Results are now stored in one stable directory, independent of
            # the worker/node that owns this shard.  Also discover outputs
            # created by the previous machine_XXX layout so scaling from N to
            # M nodes can resume rather than processing those shards again.
            resume_paths = [output_path]
            resume_paths.extend(sorted(output_dir.glob(f"machine_*/{input_path.name}")))
            resume_paths = [path for path in resume_paths if path.is_file()]

            identified_rows: dict[tuple[Any, Any], dict[str, Any]] = {}
            identity_order: list[tuple[Any, Any]] = []
            unidentified_rows: dict[str, dict[str, Any]] = {}
            malformed_lines = 0
            for resume_path in resume_paths:
                with resume_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            old = json.loads(line)
                        except json.JSONDecodeError:
                            malformed_lines += 1
                            log(f"resume warning: ignore malformed line {resume_path}:{line_number}")
                            continue
                        identity = (old.get("uuid"), old.get("data_uuid"))
                        is_error = isinstance(old.get("caption"), str) and old["caption"].startswith(("API_ERROR:", "ERROR:"))
                        if identity == (None, None):
                            key = json.dumps(old, ensure_ascii=False, sort_keys=True)
                            unidentified_rows.setdefault(key, old)
                            continue
                        previous = identified_rows.get(identity)
                        if previous is None:
                            identified_rows[identity] = old
                            identity_order.append(identity)
                        else:
                            previous_is_error = isinstance(previous.get("caption"), str) and previous["caption"].startswith(("API_ERROR:", "ERROR:"))
                            if previous_is_error and not is_error:
                                identified_rows[identity] = old

            merged_rows = [identified_rows[identity] for identity in identity_order]
            merged_rows.extend(unidentified_rows.values())
            # Failed rows must be retried, but retaining them here and then
            # appending the retry result writes the same identity twice when
            # the failure persists. Keep only successful rows in the resumed
            # output; this run will append exactly one fresh success/error for
            # every retried input record.
            def is_error_row(old: dict[str, Any]) -> bool:
                return isinstance(old.get("caption"), str) and old["caption"].startswith(("API_ERROR:", "ERROR:"))

            retryable_error_rows = [old for old in merged_rows if is_error_row(old)]
            existing_rows = [old for old in merged_rows if not is_error_row(old)]
            for old in existing_rows:
                is_error = isinstance(old.get("caption"), str) and old["caption"].startswith(("API_ERROR:", "ERROR:"))
                identity = (old.get("uuid"), old.get("data_uuid"))
                if not is_error and identity != (None, None):
                    completed_ids.add(identity)

            if resume_paths:
                # Consolidate legacy and current results atomically before new
                # rows are appended. Current modulo assignment guarantees one
                # active owner for each input JSONL.
                output_dir.mkdir(parents=True, exist_ok=True)
                resume_tmp = output_path.with_name(f".{output_path.name}.resume.{os.getpid()}.tmp")
                with resume_tmp.open("w", encoding="utf-8") as handle:
                    for old in existing_rows:
                        print(json.dumps(old, ensure_ascii=False), file=handle)
                os.replace(resume_tmp, output_path)
                sources = ",".join(str(path) for path in resume_paths)
                log(
                    f"resume: {input_path.name}, existing_output_rows={len(existing_rows)} "
                    f"completed={len(completed_ids)} retrying_errors={len(retryable_error_rows)} "
                    f"malformed={malformed_lines} sources={sources}"
                )

        rows = load_jsonl_rows(input_path, logger=log)
        for row in rows:
            row["_tar_name"] = f"{args.tar_name}.tar" if args.tar_name else input_path.stem + ".tar"
        tarinfo_cache = fixed_tarinfo_cache if args.tar_name else {}
        tarinfo_path = args.image_root / (Path(rows[0]["_tar_name"]).stem + "_tarinfo.json") if rows else None
        if tarinfo_path is not None and tarinfo_path not in tarinfo_cache:
            log(f"tarinfo load start: {tarinfo_path}")
            with tarinfo_path.open("r", encoding="utf-8") as handle:
                tarinfo_cache[tarinfo_path] = json.load(handle)
            log(f"tarinfo load complete: {tarinfo_path} entries={len(tarinfo_cache[tarinfo_path])}")
            release_memory()
            log_memory("tarinfo_loaded")

        md5_progress = {"completed": 0}
        md5_progress_lock = threading.Lock()
        log(f"MD5 start: total={len(rows)} workers={min(args.data_workers, max(1, len(rows)))}")

        def inspect(row: dict[str, Any]) -> dict[str, Any]:
            data: bytes | None = None
            try:
                data = read_image(row, args.image_root, tarinfo_cache)
                row["_image_md5"] = image_md5(data)
            except Exception as exc:
                row["_image_read_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                data = None
                with md5_progress_lock:
                    md5_progress["completed"] += 1
                    completed = md5_progress["completed"]
                if completed % args.md5_log_interval == 0 or completed == len(rows):
                    log(f"MD5 progress: {completed}/{len(rows)} ({completed / max(1, len(rows)):.1%})")
            return row

        with ThreadPoolExecutor(max_workers=min(args.data_workers, max(1, len(rows)))) as executor:
            rows = list(executor.map(inspect, rows))

        jobs: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            if "_image_read_error" in row:
                error = row["_image_read_error"]
                skipped.append({"uuid": row.get("uuid"), "data_uuid": row.get("data_uuid"), "md5": None, "prompt": None, "caption": f"API_ERROR: image read failed: {error}"})
                log(f"FAILURE line={index} uuid={row.get('uuid')} data_uuid={row.get('data_uuid')} stage=read error={error}")
                continue
            if (row.get("uuid"), row.get("data_uuid")) in completed_ids:
                continue
            try:
                jobs.append(make_job(
                    row,
                    f"{input_path.name}:{index}",
                    domain_index,
                    genre_index,
                    domain_names,
                    genre_names,
                ))
                jobs[-1]["_tarinfo_cache"] = tarinfo_cache
            except Exception as exc:
                skipped.append({
                    "uuid": row.get("uuid"),
                    "data_uuid": row.get("data_uuid"),
                    "md5": row.get("_image_md5"),
                    "prompt": None,
                    "caption": f"API_ERROR: {type(exc).__name__}: {exc}",
                })
                log(f"FAILURE line={index} uuid={row.get('uuid')} data_uuid={row.get('data_uuid')} stage=routing error={type(exc).__name__}: {exc}")

        if args.dry_run:
            results = [{
                "uuid": job["row"].get("uuid"),
                "data_uuid": job["row"].get("data_uuid"),
                "md5": job["md5"],
                "prompt": job["prompt"],
                "caption": None,
            } for job in jobs] + skipped
            for job in jobs:
                log(f"SUCCESS line={job['key'].rsplit(':', 1)[-1]} uuid={job['row'].get('uuid')} data_uuid={job['row'].get('data_uuid')} stage=dry-run")
            if args.resume:
                results = existing_rows + results
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
            log(f"{input_path.name}: dry-run input={len(rows)} routed={len(jobs)} skipped={len(skipped)} output={output_path}")
            if not args.tar_name:
                tarinfo_cache.clear()
            del rows, jobs, skipped, results, existing_rows, completed_ids
            release_memory()
            log_memory("jsonl_complete")
            close_current_log()
            continue

        state = threading.local()
        clients: list[Any] = []
        clients_lock = threading.Lock()

        # Persist every completed record immediately. Resume appends to the
        # existing JSONL; a fresh/overwrite run starts a new file.
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mode = "a" if args.resume and output_path.is_file() else "w"
        output_handle = output_path.open(output_mode, encoding="utf-8", buffering=1)

        def write_output_row(result_row: dict[str, Any]) -> None:
            result_row.pop("_tar_name", None)
            result_row.pop("_image_md5", None)
            print(json.dumps(result_row, ensure_ascii=False), file=output_handle, flush=True)

        for result_row in skipped:
            write_output_row(result_row)

        def thread_client() -> Any:
            if not hasattr(state, "client"):
                client = base.create_client(config, timeout=args.timeout, max_retries=args.max_retries)
                state.client = client
                with clients_lock:
                    clients.append(client)
            return state.client

        def run(job: dict[str, Any]) -> dict[str, Any]:
            # Catch inside the worker.  The Future therefore stores a small
            # normal result instead of an exception traceback retaining image
            # bytes, base64 strings and HTTP request objects.
            try:
                prediction, finish_reason, usage = call_api(job, args, config, thread_client())
                parsed = base.parse_model_output(prediction)
                return {
                    "ok": True,
                    "result": {
                        "uuid": job["row"].get("uuid"),
                        "data_uuid": job["row"].get("data_uuid"),
                        "md5": job["md5"],
                        "prompt": job["prompt"],
                        "caption": parsed.caption if not parsed.rejected else parsed.refusal,
                    },
                }
            except Exception as exc:
                return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

        completed = 0
        api_successes = 0
        api_failures = 0
        futures: dict[Any, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(args.api_workers, max(1, len(jobs)))) as executor:
            futures = {executor.submit(run, job): job for job in jobs}
            for future in as_completed(tuple(futures)):
                job = futures.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    # Defensive fallback for failures outside run(); ordinary
                    # API/image failures are already converted inside run().
                    outcome = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
                if outcome["ok"]:
                    result = outcome["result"]
                    api_successes += 1
                    status = "SUCCESS_REFUSAL" if result["caption"].startswith(("Yes; Reason:", "REFUSAL:")) else "SUCCESS"
                    log(f"{status} line={job['key'].rsplit(':', 1)[-1]} uuid={job['row'].get('uuid')} data_uuid={job['row'].get('data_uuid')}")
                else:
                    api_failures += 1
                    error_text = f"{outcome['error_type']}: {outcome['error']}"
                    result = {
                        "uuid": job["row"].get("uuid"),
                        "data_uuid": job["row"].get("data_uuid"),
                        "md5": job["row"].get("_image_md5"),
                        "prompt": job["prompt"],
                        "caption": f"API_ERROR: {error_text}",
                    }
                    log(f"FAILURE line={job['key'].rsplit(':', 1)[-1]} uuid={job['row'].get('uuid')} data_uuid={job['row'].get('data_uuid')} stage=api error={error_text}")
                write_output_row(result)
                completed += 1
                if completed % args.memory_log_interval == 0 or completed == len(jobs):
                    rss_mb, peak_mb = memory_usage_mb()
                    log(f"API progress: {completed}/{len(jobs)} success={api_successes} failure={api_failures} pending={len(futures)} rss={rss_mb:.1f}MiB peak_rss={peak_mb:.1f}MiB")
                del future, job, outcome

        output_handle.close()

        close_failures = 0
        for client in clients:
            try:
                client.close()
            except Exception as exc:
                close_failures += 1
                log(f"WARNING stage=client_close error={type(exc).__name__}: {exc}")
        clients.clear()
        if close_failures:
            log(f"API clients close complete: failures={close_failures}")

        log(f"{input_path.name}: input={len(rows)} processed={len(jobs)} api_success={api_successes} api_failure={api_failures} skipped={len(skipped)} output={output_path}")
        if not args.tar_name:
            tarinfo_cache.clear()
        del rows, jobs, skipped, existing_rows, completed_ids
        del futures, state, clients
        release_memory()
        log_memory("jsonl_complete")
        close_current_log()
    finally:
        close_current_log()
        fixed_tarinfo_cache.clear()
        release_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
