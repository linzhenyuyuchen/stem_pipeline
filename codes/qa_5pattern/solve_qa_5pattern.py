#!/usr/bin/env python3
"""Use a vision chat API to rewrite QA answers with the five-pattern prompt.

The input is a Pangu-ML data config.  For every JSONL below ``annotation_dir``
the matching uncompressed TAR and ``*_tarinfo.json`` are read from
``tar_image_root``.  Successful API responses replace only
``data[1].content[0].text.string``; failed records are omitted.

By default, an annotation directory named ``foo`` is written to its sibling
``foo_modified`` and the updated config is written as ``<config>_modified.json``.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
import json
import mimetypes
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import requests


HERE = Path(__file__).resolve().parent
DEFAULT_META_PROMPT = (HERE / "meta_prompt.txt"
                       if (HERE / "meta_prompt.txt").is_file()
                       else HERE / "sys_prompt.txt")
DEFAULT_USER_PROMPT = HERE / "user_prompt.txt"
REFERENCE_KEY_FILE = Path(
    "/mnt-szc-mm-exp/others/j00939224/captioner_production/qa/qa_api_key.txt"
)
REMOVE_OPTION_LETTER_INSTRUCTIONS = (
    "Answer with the option letter from the given choices directly.",
    "Answer with the option's letter from the given choices directly.",
)


def load_api_key() -> str:
    for env_name in ("QA_API_KEY", "OPENAI_API_KEY", "GPT55_API_KEY"):
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()
    path = Path(os.getenv("QA_API_KEY_FILE", str(REFERENCE_KEY_FILE)))
    if not path.is_file():
        raise SystemExit(f"未找到 API Key：设置 QA_API_KEY，或创建 {path}")
    value = path.read_text(encoding="utf-8-sig").strip()
    if not value:
        raise SystemExit(f"API Key 文件为空：{path}")
    return value


@dataclass(frozen=True)
class Source:
    config_name: str
    dataset: str
    annotation_root: Path
    jsonl: Path
    tar_path: Path
    tarinfo_path: Path
    output: Path
    index: int
    count: int


@dataclass(frozen=True)
class Task:
    source: Source
    offset: int
    line_no: int


class TarShard:
    """Thread-safe random reads from an uncompressed TAR using its tarinfo."""

    def __init__(self, tar_path: Path, tarinfo_path: Path):
        with tarinfo_path.open("r", encoding="utf-8") as fin:
            self.index = json.load(fin)
        self.tar_path = tar_path
        self.fd = os.open(tar_path, os.O_RDONLY)

    def read(self, member: str) -> bytes:
        key = member.replace("\\", "/").lstrip("./")
        info = self.index.get(key) or self.index.get("./" + key)
        if info is None:
            raise KeyError(f"tarinfo 中不存在图片: {member}")
        offset = int(info["offset_data"])
        size = int(info["size"])
        chunks: list[bytes] = []
        read_size = 0
        while read_size < size:
            chunk = os.pread(self.fd, size - read_size, offset + read_size)
            if not chunk:
                raise EOFError(
                    f"TAR 提前结束: {self.tar_path}, member={member}, "
                    f"expected={size}, actual={read_size}"
                )
            chunks.append(chunk)
            read_size += len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        os.close(self.fd)


class CheckpointStore:
    """Durable per-record results used for real crash-safe resume."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(
            path, timeout=60, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_key TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                source_key TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                uid TEXT,
                record_json TEXT NOT NULL,
                PRIMARY KEY (source_key, line_no)
            )
        """)

    @staticmethod
    def source_key(source: Source) -> str:
        return str(source.jsonl)

    def register_source(self, source: Source) -> None:
        key = self.source_key(source)
        stat = source.jsonl.stat()
        row = self.conn.execute(
            "SELECT size, mtime_ns FROM sources WHERE source_key = ?",
            (key,),
        ).fetchone()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if row is not None and tuple(row) != fingerprint:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM results WHERE source_key = ?", (key,)
            ).fetchone()[0]
            if count:
                raise ValueError(
                    f"源 JSONL 已变化，不能安全 resume: {source.jsonl}; "
                    f"请删除 checkpoint 后重新运行: {self.path}")
        self.conn.execute(
            "INSERT OR REPLACE INTO sources(source_key, size, mtime_ns) "
            "VALUES (?, ?, ?)",
            (key, *fingerprint),
        )

    def line_uids(self, source: Source) -> dict[int, str | None]:
        rows = self.conn.execute(
            "SELECT line_no, uid FROM results WHERE source_key = ?",
            (self.source_key(source),),
        )
        return {int(line_no): uid for line_no, uid in rows}

    def save(self, source: Source, line_no: int,
             uid: str | None, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        # Autocommit + synchronous=FULL makes each successful response durable
        # before the progress log reports it as "ok".
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO results"
                "(source_key, line_no, uid, record_json) VALUES (?, ?, ?, ?)",
                (self.source_key(source), line_no, uid, payload),
            )

    def iter_results(self, source: Source):
        rows = self.conn.execute(
            "SELECT line_no, record_json FROM results "
            "WHERE source_key = ? ORDER BY line_no",
            (self.source_key(source),),
        )
        for line_no, payload in rows:
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError(
                    f"checkpoint record 不是 object: "
                    f"{source.jsonl}:{line_no}")
            yield int(line_no), record

    def clear_source(self, source: Source) -> None:
        with self.lock:
            key = self.source_key(source)
            self.conn.execute(
                "DELETE FROM results WHERE source_key = ?", (key,))
            self.conn.execute(
                "DELETE FROM sources WHERE source_key = ?", (key,))

    def clear_all(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM results")
            self.conn.execute("DELETE FROM sources")

    def close(self) -> None:
        self.conn.close()

    def remove_files(self) -> None:
        for path in (self.path, Path(str(self.path) + "-wal"),
                     Path(str(self.path) + "-shm")):
            path.unlink(missing_ok=True)


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as fin:
        return sum(bool(line.strip()) for line in fin)


def modified_annotation_dir(annotation_root: Path) -> Path:
    """Return a sibling whose final directory name has the _modified suffix."""
    normalized = Path(str(annotation_root).rstrip("/"))
    return normalized.with_name(normalized.name + "_modified")


def shard_paths(annotation_root: Path, jsonl: Path,
                tar_root: Path) -> tuple[Path, Path]:
    rel = jsonl.relative_to(annotation_root)
    tar_path = (tar_root / rel).with_suffix(".tar")
    tarinfo_path = tar_path.with_name(tar_path.stem + "_tarinfo.json")
    if not tar_path.is_file():
        raise FileNotFoundError(f"JSONL 对应的 TAR 不存在: {tar_path}")
    if not tarinfo_path.is_file():
        raise FileNotFoundError(f"JSONL 对应的 tarinfo 不存在: {tarinfo_path}")
    return tar_path, tarinfo_path


def load_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("config 顶层必须是 JSON object")
    return config


def discover_sources(config: dict) -> tuple[list[Source], dict[str, Path], int]:
    sources: list[Source] = []
    output_roots: dict[str, Path] = {}
    total = 0
    for config_name, item in config.items():
        if not isinstance(item, dict):
            raise ValueError(f"config[{config_name!r}] 必须是 object")
        dataset = str(item.get("dataset_name") or config_name)
        annotation_root = Path(item["annotation_dir"]).expanduser().resolve()
        tar_root = Path(item["tar_image_root"]).expanduser().resolve()
        if not annotation_root.is_dir():
            raise FileNotFoundError(f"annotation_dir 不存在: {annotation_root}")
        if not tar_root.is_dir():
            raise FileNotFoundError(f"tar_image_root 不存在: {tar_root}")
        output_root = modified_annotation_dir(annotation_root)
        output_roots[config_name] = output_root
        jsonls = sorted(path for path in annotation_root.rglob("*.jsonl")
                        if path.is_file())
        if not jsonls:
            print(f"警告: {dataset} 下没有 JSONL: {annotation_root}",
                  file=sys.stderr)
        for jsonl_path in jsonls:
            count = count_nonempty_lines(jsonl_path)
            tar_path, tarinfo_path = shard_paths(
                annotation_root, jsonl_path, tar_root)
            output = output_root / jsonl_path.relative_to(annotation_root)
            sources.append(Source(
                config_name=config_name,
                dataset=dataset,
                annotation_root=annotation_root,
                jsonl=jsonl_path,
                tar_path=tar_path,
                tarinfo_path=tarinfo_path,
                output=output,
                index=len(sources),
                count=count,
            ))
            total += count
    return sources, output_roots, total


def record_uid(record: dict) -> str | None:
    uid = record.get("data_uuid") or record.get("uuid")
    return str(uid) if uid is not None else None


def completed_uid_counts(path: Path) -> Counter[str]:
    """Count successfully saved records in an existing output JSONL."""
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"resume 输出包含无效 JSON: {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"resume 输出行不是 object: {path}:{line_no}")
            uid = record_uid(record)
            if uid is None:
                raise ValueError(
                    f"resume 输出记录缺少 data_uuid/uuid: {path}:{line_no}")
            counts[uid] += 1
    return counts


def build_tasks(sources: list[Source], resume: bool,
                checkpoints: CheckpointStore | None) -> tuple[
        list[Task], dict[int, set[int]], dict[str, int], int, int]:
    tasks: list[Task] = []
    reused_output_lines: dict[int, set[int]] = {}
    success_by_config: dict[str, int] = {}
    reused_output_count = 0
    reused_checkpoint_count = 0
    for source in sources:
        remaining = (completed_uid_counts(source.output)
                     if resume else Counter())
        checkpoint_uids = (checkpoints.line_uids(source)
                           if resume and checkpoints is not None else {})
        unmatched_checkpoint_lines = set(checkpoint_uids)
        reused_output: set[int] = set()
        reused_checkpoint: set[int] = set()
        with source.jsonl.open("rb") as fin:
            line_no = 0
            while True:
                offset = fin.tell()
                line = fin.readline()
                if not line:
                    break
                line_no += 1
                if not line.strip():
                    continue
                if remaining:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        record = {}
                    uid = record_uid(record) if isinstance(record, dict) else None
                    if uid is not None and remaining[uid] > 0:
                        remaining[uid] -= 1
                        if remaining[uid] == 0:
                            del remaining[uid]
                        reused_output.add(line_no)
                        unmatched_checkpoint_lines.discard(line_no)
                        reused_output_count += 1
                        continue
                if line_no in checkpoint_uids:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError(
                            f"checkpoint 对应的原始行不是有效 JSON: "
                            f"{source.jsonl}:{line_no}") from exc
                    uid = record_uid(record) if isinstance(record, dict) else None
                    if uid != checkpoint_uids[line_no]:
                        raise ValueError(
                            f"checkpoint UUID 与原始数据不一致: "
                            f"{source.jsonl}:{line_no}")
                    reused_checkpoint.add(line_no)
                    unmatched_checkpoint_lines.discard(line_no)
                    reused_checkpoint_count += 1
                    continue
                tasks.append(Task(source, offset, line_no))
        if remaining:
            examples = ", ".join(
                f"{uid}×{count}" for uid, count in remaining.most_common(3))
            raise ValueError(
                f"resume 输出含原始 JSONL 中无法匹配的 UUID: "
                f"{source.output} ({examples})")
        if unmatched_checkpoint_lines:
            examples = ", ".join(
                str(line_no)
                for line_no in sorted(unmatched_checkpoint_lines)[:3])
            raise ValueError(
                f"checkpoint 含原始 JSONL 中无法匹配的行: "
                f"{source.jsonl} (line {examples})")
        reused_output_lines[source.index] = reused_output
        success_by_config[source.config_name] = (
            success_by_config.get(source.config_name, 0) +
            len(reused_output) + len(reused_checkpoint))
    return (tasks, reused_output_lines, success_by_config,
            reused_output_count, reused_checkpoint_count)


def split_evenly(tasks: list[Task], workers: int) -> list[list[Task]]:
    workers = min(workers, len(tasks))
    quotient, remainder = divmod(len(tasks), workers)
    chunks: list[list[Task]] = []
    start = 0
    for index in range(workers):
        size = quotient + (index < remainder)
        chunks.append(tasks[start:start + size])
        start += size
    return chunks


def parse_base_urls(value: str) -> list[str]:
    urls = [item for item in re.split(r"[\s,;]+", value.strip()) if item]
    if not urls:
        raise ValueError("至少需要一个 base URL")
    return urls


def read_record(task: Task, handles: dict[Path, BinaryIO]) -> dict:
    fin = handles.get(task.source.jsonl)
    if fin is None:
        fin = task.source.jsonl.open("rb")
        handles[task.source.jsonl] = fin
    fin.seek(task.offset)
    record = json.loads(fin.readline())
    if not isinstance(record, dict):
        raise ValueError("JSONL 行不是 object")
    return record


def image_mime(data: bytes, relative_path: str, declared: object) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if isinstance(declared, str) and declared.startswith("image/"):
        return declared
    return mimetypes.guess_type(relative_path)[0] or "image/jpeg"


def get_original_content(record: dict) -> list[dict]:
    data = record.get("data")
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("记录缺少 data[0] 或 data[1]")
    content = data[0].get("content") if isinstance(data[0], dict) else None
    if not isinstance(content, list) or not content:
        raise ValueError("记录缺少 data[0].content")
    # Validate the exact output location before spending an API request.
    try:
        old_answer = data[1]["content"][0]["text"]["string"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "记录缺少 data[1].content[0].text.string") from exc
    if not isinstance(old_answer, str):
        raise ValueError("data[1].content[0].text.string 不是字符串")
    return content


def make_messages(record: dict, archive: TarShard, meta_prompt: str,
                  user_prompt: str) -> list[dict]:
    content = get_original_content(record)
    question_parts: list[str] = []
    images: list[dict] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("data[0].content 元素不是 object")
        kind = item.get("type")
        if kind == "image":
            image = item.get("image") or {}
            relative_path = image.get("relative_path")
            if not relative_path:
                raise ValueError("image 缺少 relative_path")
            image_bytes = archive.read(str(relative_path))
            mime = image_mime(
                image_bytes, str(relative_path), image.get("format"))
            encoded = base64.b64encode(image_bytes).decode("ascii")
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        elif kind == "text":
            text = item.get("text")
            value = text.get("string") if isinstance(text, dict) else text
            if not isinstance(value, str):
                raise ValueError("text 内容不是字符串")
            for instruction in REMOVE_OPTION_LETTER_INSTRUCTIONS:
                value = value.replace(instruction, "")
            if value.strip():
                question_parts.append(value.strip())
        else:
            raise ValueError(f"不支持的 content type: {kind!r}")
    if not question_parts:
        raise ValueError("问题文本为空")
    question = "\n\n".join(question_parts)
    rendered_prompt = user_prompt.replace("{{question}}", question)
    return [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": [
            *images, {"type": "text", "text": rendered_prompt}
        ]},
    ]


def replace_answer(record: dict, answer: str) -> dict:
    output = copy.deepcopy(record)
    output["data"][1]["content"][0]["text"]["string"] = answer
    return output


class Client:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float,
                 retries: int, max_tokens: int | None, extra: dict,
                 insecure: bool):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.extra = extra
        self.key = api_key
        self.session = requests.Session()
        bypass = os.getenv("QA_BYPASS_PROXY", "true").strip().lower()
        self.session.trust_env = bypass in {"0", "false", "no"}
        self.verify = not insecure
        if insecure:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def solve(self, messages: list[dict]) -> str:
        payload: dict = {
            "model": self.model,
            "stream": False,
            "temperature": 0.0,
            "messages": messages,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra)
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                    verify=self.verify,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("message.content 为空")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                status = (exc.response.status_code
                          if isinstance(exc, requests.HTTPError)
                          and exc.response is not None else None)
                if status in {400, 401, 403, 404} or attempt == self.retries:
                    break
                delay = min(2 ** (attempt - 1) + random.random(), 30)
                print(f"  [retry {attempt}] {type(exc).__name__}: "
                      f"{str(exc)[:120]} -> {delay:.1f}s",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
        raise RuntimeError(f"{type(last).__name__}: {last}")


def copy_non_jsonl_files(source_root: Path, output_root: Path) -> None:
    """Preserve auxiliary files and empty directories from annotation_dir."""
    for root, dirnames, filenames in os.walk(source_root):
        root_path = Path(root)
        relative = root_path.relative_to(source_root)
        (output_root / relative).mkdir(parents=True, exist_ok=True)
        dirnames.sort()
        for filename in sorted(filenames):
            source = root_path / filename
            if source.suffix.lower() != ".jsonl":
                destination = output_root / relative / filename
                shutil.copy2(source, destination)


def iter_jsonl_records(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"无效 JSON: {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 行不是 object: {path}:{line_no}")
            yield record


def merge_source_output(source: Source, checkpoints: CheckpointStore,
                        reused_lines: set[int], resume: bool) -> int:
    """Atomically merge formal output and durable checkpoints in input order."""
    existing_iter = (iter_jsonl_records(source.output)
                     if resume and source.output.is_file() else iter(()))
    checkpoint_iter = checkpoints.iter_results(source)
    existing_next = next(existing_iter, None)
    checkpoint_next = next(checkpoint_iter, None)
    written = 0

    source.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{source.output.name}.merge_",
        dir=source.output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout:
            with source.jsonl.open("rb") as fin:
                for line_no, line in enumerate(fin, 1):
                    if not line.strip():
                        continue
                    output_record = None
                    if line_no in reused_lines:
                        if existing_next is None:
                            raise ValueError(
                                f"resume 输出记录数不足: {source.output}")
                        output_record = existing_next
                        existing_next = next(existing_iter, None)
                        # A crash may happen after formal output replacement but
                        # before checkpoint cleanup. The formal output wins.
                        if (checkpoint_next is not None and
                                checkpoint_next[0] == line_no):
                            checkpoint_next = next(checkpoint_iter, None)
                    elif (checkpoint_next is not None and
                          checkpoint_next[0] == line_no):
                        output_record = checkpoint_next[1]
                        checkpoint_next = next(checkpoint_iter, None)
                    elif (checkpoint_next is not None and
                          checkpoint_next[0] < line_no):
                        raise ValueError(
                            f"checkpoint 行号顺序异常: {source.jsonl}, "
                            f"line={checkpoint_next[0]}")
                    # No reused/checkpoint result means this record failed or was
                    # excluded by --limit, so it is deliberately omitted.
                    if output_record is not None:
                        fout.write(json.dumps(
                            output_record, ensure_ascii=False) + "\n")
                        written += 1
        if existing_next is not None or next(existing_iter, None) is not None:
            raise ValueError(f"resume 输出存在多余记录: {source.output}")
        if (checkpoint_next is not None or
                next(checkpoint_iter, None) is not None):
            raise ValueError(f"checkpoint 存在多余记录: {source.jsonl}")
        os.replace(temporary, source.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return written


def write_updated_config(config: dict, output_roots: dict[str, Path],
                         success_by_config: dict[str, int],
                         output_path: Path) -> None:
    updated = copy.deepcopy(config)
    for config_name, item in updated.items():
        item["annotation_dir"] = str(output_roots[config_name])
        count = success_by_config.get(config_name, 0)
        item["sample_num"] = count
        item["sample_num_all"] = count
        # tar_image_root is deliberately not normalized or rewritten.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, default=None)
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="持久化断点数据库路径（默认位于输出 config 旁）")
    parser.add_argument("--meta-prompt", type=Path,
                        default=DEFAULT_META_PROMPT)
    parser.add_argument("--user-prompt", type=Path,
                        default=DEFAULT_USER_PROMPT)
    parser.add_argument("--base-url", default=os.getenv(
        "QA_BASE_URL", "http://127.0.0.1:8891/v1"),
        help="一个或多个 API 地址；多个地址用空格、逗号或分号分隔")
    parser.add_argument("--model", default=os.getenv(
        "QA_MODEL", "gemini-3.1-flash-lite"))
    parser.add_argument("--workers", "-M", type=int, default=32,
                        help="每个 base URL 的并发数（默认 32）")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.getenv("QA_MAX_OUTPUT_TOKENS", "0"))
                        or None)
    parser.add_argument("--extra", default="",
                        help='追加请求体 JSON，如 \'{"enable_thinking":false}\'')
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅处理全局前 N 条；0 表示不限")
    parser.add_argument("--resume", action="store_true",
                        help="跳过输出中已有的成功记录，仅重试未处理/失败记录")
    parser.add_argument("--dry-run", action="store_true",
                        help="校验数据、取图并组装请求，但不调用 API 或写输出")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers 必须大于 0")
    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.retries < 1:
        parser.error("--retries 必须大于 0（1 表示不重试）")
    try:
        base_urls = parse_base_urls(args.base_url)
        extra = json.loads(args.extra) if args.extra else {}
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not isinstance(extra, dict):
        parser.error("--extra 必须是 JSON object")

    config_path = args.config.expanduser().resolve()
    output_config = (args.output_config.expanduser().resolve()
                     if args.output_config else
                     config_path.with_name(config_path.stem + "_modified.json"))
    checkpoint_path = (args.checkpoint_db.expanduser().resolve()
                       if args.checkpoint_db else
                       output_config.with_name(
                           f".{output_config.stem}.qa_5pattern_resume.sqlite3"))
    meta_prompt = args.meta_prompt.read_text(encoding="utf-8-sig").strip()
    user_prompt = args.user_prompt.read_text(encoding="utf-8-sig")
    if not meta_prompt:
        parser.error("meta prompt 为空")
    if "{{question}}" not in user_prompt:
        parser.error("user prompt 中缺少 {{question}} 占位符")

    config = load_config(config_path)
    sources, output_roots, total = discover_sources(config)
    checkpoints = None if args.dry_run else CheckpointStore(checkpoint_path)
    try:
        if checkpoints is not None:
            if not args.resume:
                checkpoints.clear_all()
            for source in sources:
                checkpoints.register_source(source)
        (tasks, reused_lines, success_by_config, reused_output,
         reused_checkpoint) = build_tasks(
            sources, args.resume, checkpoints)
    except BaseException:
        if checkpoints is not None:
            checkpoints.close()
        raise
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"扫描完成: {len(sources)} 个 JSONL，共 {total} 条；"
          f"正式输出复用 {reused_output} 条，checkpoint 恢复 "
          f"{reused_checkpoint} 条；待处理 {len(tasks)} 条", flush=True)
    if checkpoints is not None:
        print(f"持久化 checkpoint: {checkpoint_path}", flush=True)
    if not tasks:
        if args.resume and not args.dry_run:
            assert checkpoints is not None
            for source in sources:
                merge_source_output(
                    source, checkpoints, reused_lines[source.index], True)
                checkpoints.clear_source(source)
            for config_name, item in config.items():
                source_root = Path(
                    item["annotation_dir"]).expanduser().resolve()
                copy_non_jsonl_files(source_root, output_roots[config_name])
            write_updated_config(
                config, output_roots, success_by_config, output_config)
            checkpoints.close()
            checkpoints.remove_files()
            print("所有记录均已处理，无需请求 API。", flush=True)
            print(f"输出 config: {output_config}", flush=True)
            return 0
        if checkpoints is not None:
            checkpoints.close()
        print("没有需要处理的数据。", file=sys.stderr)
        return 0 if args.resume else 1

    total_workers = len(base_urls) * args.workers
    chunks = split_evenly(tasks, total_workers)
    print(f"API 地址数={len(base_urls)}，每地址并发={args.workers}，"
          f"实际 worker={len(chunks)}", flush=True)

    api_key = "" if args.dry_run else load_api_key()
    archives: dict[Path, TarShard] = {}
    active_sources = {task.source.index: task.source for task in tasks}.values()
    for source in active_sources:
        if source.tar_path not in archives:
            archives[source.tar_path] = TarShard(
                source.tar_path, source.tarinfo_path)

    progress_lock = threading.Lock()
    stats = {"finished": 0, "ok": 0, "fail": 0}

    def run_chunk(worker_id: int, chunk: list[Task], base_url: str) -> None:
        handles: dict[Path, BinaryIO] = {}
        client = None if args.dry_run else Client(
            api_key, base_url, args.model, args.timeout, args.retries,
            args.max_tokens, extra, args.insecure)
        try:
            for task in chunk:
                started = time.time()
                uid: object = None
                error = ""
                try:
                    record = read_record(task, handles)
                    uid = record.get("data_uuid") or record.get("uuid")
                    messages = make_messages(
                        record, archives[task.source.tar_path],
                        meta_prompt, user_prompt)
                    if args.dry_run:
                        answer = ""
                    else:
                        assert client is not None and checkpoints is not None
                        answer = client.solve(messages)
                        output_record = replace_answer(record, answer)
                        checkpoints.save(
                            task.source, task.line_no,
                            record_uid(record), output_record)
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    ok = False

                elapsed = time.time() - started
                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds")
                with progress_lock:
                    stats["finished"] += 1
                    stats["ok" if ok else "fail"] += 1
                    if ok and not args.dry_run:
                        success_by_config[task.source.config_name] += 1
                    number = stats["finished"]
                    suffix = "" if ok else f" error={error[:240]}"
                    print(f"{timestamp} [{number}/{len(tasks)}] "
                          f"{'ok  ' if ok else 'FAIL'} worker={worker_id} "
                          f"file={task.source.jsonl.name}:"
                          f"{task.line_no} uuid={uid} "
                          f"elapsed={elapsed:.2f}s{suffix}", flush=True)
        finally:
            for fin in handles.values():
                fin.close()
            if client is not None:
                client.session.close()

    threads = [threading.Thread(
        target=run_chunk,
        args=(index, chunk, base_urls[index % len(base_urls)]),
        name=f"api-worker-{index}",
    ) for index, chunk in enumerate(chunks)]

    finalized = False
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if not args.dry_run:
            assert checkpoints is not None
            for source in sources:
                written = merge_source_output(
                    source, checkpoints, reused_lines[source.index],
                    args.resume)
                checkpoints.clear_source(source)
                print(f"合并完成: {source.output}，共 {written} 条", flush=True)
            for config_name, item in config.items():
                source_root = Path(item["annotation_dir"]).expanduser().resolve()
                copy_non_jsonl_files(source_root, output_roots[config_name])
            write_updated_config(
                config, output_roots, success_by_config, output_config)
            finalized = True
    finally:
        for archive in archives.values():
            archive.close()
        if checkpoints is not None:
            checkpoints.close()
            if finalized:
                checkpoints.remove_files()

    mode = "dry-run" if args.dry_run else "API"
    print(f"{mode} 完成: 正式输出复用 {reused_output}，"
          f"checkpoint 恢复 {reused_checkpoint}，"
          f"新增成功 {stats['ok']}，失败 {stats['fail']}", flush=True)
    if not args.dry_run:
        for name, root in output_roots.items():
            print(f"annotation_dir[{name}]: {root}", flush=True)
        print(f"输出 config: {output_config}", flush=True)
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
