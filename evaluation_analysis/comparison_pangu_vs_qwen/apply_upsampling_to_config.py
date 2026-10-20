#!/usr/bin/env python3
"""Expand a dataset config according to Markdown weights and a token budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


def parse_number(value: Any, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{field} 不能为空")
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(f"{field} 不是有效数字: {value!r}") from exc


def split_md_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def read_weights(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        cells = split_md_row(line)
        if "排名" in cells and "数据集" in cells and "上采样权重" in cells:
            header = cells
            continue
        if header is None or len(cells) != len(header) or not cells[0].isdigit():
            continue
        record = dict(zip(header, cells))
        raw_weight = record["上采样权重"].rstrip("×xX ")
        weight = parse_number(raw_weight, "上采样权重")
        if weight < 1:
            raise ValueError(f"排名 {record['排名']} 的上采样权重小于 1: {weight}")
        rows.append(
            {
                "rank": int(record["排名"]),
                "dataset": record["数据集"],
                "weight": weight,
            }
        )
    if not rows:
        raise ValueError("Markdown 中未找到包含 排名/数据集/上采样权重 的明细表")
    ranks = [row["rank"] for row in rows]
    if len(ranks) != len(set(ranks)):
        raise ValueError("Markdown 的排名列不唯一，不能作为稳定映射键")
    return rows


def read_token_plan(path: Path) -> tuple[float | None, list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return None, rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list):
        raise ValueError("token JSON 必须是包含 target_tokens 和 datasets 数组的对象")
    target = data.get("target_tokens")
    return (parse_number(target, "target_tokens") if target is not None else None), data["datasets"]


def resolve_weight(
    token_row: dict[str, Any],
    by_rank: dict[int, dict[str, Any]],
    by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    config_key = str(token_row.get("config_key", "")).strip()
    rank_value = token_row.get("weight_rank")
    requested_name = str(token_row.get("weight_dataset", "")).strip()
    if rank_value not in (None, ""):
        rank = int(parse_number(rank_value, f"{config_key}.weight_rank"))
        if rank not in by_rank:
            raise ValueError(f"{config_key}: weight_rank={rank} 在 Markdown 中不存在")
        result = by_rank[rank]
        if requested_name and requested_name != result["dataset"]:
            raise ValueError(
                f"{config_key}: weight_rank={rank} 对应 {result['dataset']!r}，"
                f"与 weight_dataset={requested_name!r} 不一致"
            )
        return result

    lookup_name = requested_name or config_key
    candidates = by_name.get(lookup_name, [])
    if not candidates:
        raise ValueError(
            f"{config_key}: 无法在 Markdown 中匹配 {lookup_name!r}；"
            "请在 token 文件中填写 weight_rank"
        )
    if len(candidates) > 1:
        ranks = ", ".join(str(row["rank"]) for row in candidates)
        raise ValueError(
            f"{config_key}: Markdown 中的数据集 {lookup_name!r} 有重名，"
            f"候选排名为 {ranks}；请显式填写 weight_rank"
        )
    return candidates[0]


def round_samples(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_int(value: float) -> str:
    return f"{value:,.0f}"


def build_audit(
    source_config: Path,
    weights_md: Path,
    tokens_file: Path,
    output_config: Path,
    target_tokens: float,
    base_tokens: float,
    full_recommendation_tokens: float,
    alpha: float,
    estimated_tokens: float,
    audit_rows: list[dict[str, Any]],
) -> str:
    error = estimated_tokens - target_tokens
    lines = [
        "# 上采样 config 生成审计",
        "",
        "## 结果",
        "",
        f"- 原始 token：{format_int(base_tokens)}",
        f"- 目标 token：{format_int(target_tokens)}",
        f"- 完整采用 Markdown 权重时的预计 token：{format_int(full_recommendation_tokens)}",
        f"- 实际权重强度系数 α：{alpha:.8f}",
        f"- 离散 sample_num 后预计 token：{format_int(estimated_tokens)}",
        f"- 预计误差：{error:+,.0f} token（{error / target_tokens * 100:+.6f}%）",
        f"- 输出 config 条目数：{sum(row['copies'] for row in audit_rows):,}",
        "",
        "## 数据集明细",
        "",
        "| config 数据集 | 权重排名 | 权重数据集 | 建议权重 | 实际倍数 | 原 sample_num | 完整副本 | 小数副本 sample_num | 输出条目数 | 预计 token |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            "| {config_key} | {weight_rank} | {weight_dataset} | {recommended_weight:.3f}× | "
            "{actual_multiplier:.6f}× | {sample_num:,} | {full_copies:,} | "
            "{partial_sample_num:,} | {copies:,} | {estimated_tokens:,.0f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 输入与输出",
            "",
            f"- 原始 config：`{source_config}`",
            f"- 权重 Markdown：`{weights_md}`",
            f"- token 预算：`{tokens_file}`",
            f"- 输出 config：`{output_config}`",
            "",
            "> token 为按原数据集平均 token/sample 线性估计；最终训练框架实读值可能因过滤、截断或动态拼接略有差异。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights-md", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--target-tokens", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="允许目标 token 超过完整采用建议权重时的 token，按相同方向外推权重",
    )
    parser.add_argument("--max-copies", type=int, default=100)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config:
        raise ValueError("config 顶层必须是非空 JSON 对象，键为数据集名称")
    weights = read_weights(args.weights_md)
    token_target_from_file, token_rows = read_token_plan(args.tokens)
    target_tokens = args.target_tokens if args.target_tokens is not None else token_target_from_file
    if target_tokens is None or target_tokens <= 0:
        raise ValueError("必须通过 token JSON 的 target_tokens 或 --target-tokens 提供正数目标")

    by_rank = {row["rank"]: row for row in weights}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in weights:
        by_name.setdefault(row["dataset"], []).append(row)

    token_by_key: dict[str, dict[str, Any]] = {}
    for row in token_rows:
        key = str(row.get("config_key", "")).strip()
        if not key:
            raise ValueError("token 文件每行都必须包含 config_key")
        if key in token_by_key:
            raise ValueError(f"token 文件中 config_key 重复: {key}")
        token_by_key[key] = row

    missing_tokens = sorted(set(config) - set(token_by_key))
    if missing_tokens:
        raise ValueError("以下 config 数据集缺少 token 信息: " + ", ".join(missing_tokens))

    prepared: list[dict[str, Any]] = []
    for config_key, spec in config.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{config_key}: config 值必须是对象")
        sample_num = spec.get("sample_num")
        if not isinstance(sample_num, int) or sample_num <= 0:
            raise ValueError(f"{config_key}: sample_num 必须是正整数")
        token_row = token_by_key[config_key]
        total_tokens = parse_number(token_row.get("total_tokens"), f"{config_key}.total_tokens")
        if total_tokens <= 0:
            raise ValueError(f"{config_key}: total_tokens 必须为正数")
        weight_row = resolve_weight(token_row, by_rank, by_name)
        prepared.append(
            {
                "config_key": config_key,
                "spec": spec,
                "sample_num": sample_num,
                "total_tokens": total_tokens,
                "weight_rank": weight_row["rank"],
                "weight_dataset": weight_row["dataset"],
                "recommended_weight": weight_row["weight"],
            }
        )

    base_tokens = sum(row["total_tokens"] for row in prepared)
    full_tokens = sum(row["total_tokens"] * row["recommended_weight"] for row in prepared)
    tolerance = max(1.0, base_tokens * 1e-12)
    if target_tokens < base_tokens - tolerance:
        raise ValueError(
            f"目标 token {target_tokens:,.0f} 小于原始 token {base_tokens:,.0f}；"
            "本工具只上采样，不执行下采样"
        )
    if target_tokens > full_tokens + tolerance and not args.allow_extrapolation:
        raise ValueError(
            f"目标 token {target_tokens:,.0f} 超过完整采用建议权重后的 {full_tokens:,.0f}；"
            "如确认需要继续放大，请添加 --allow-extrapolation"
        )
    if math.isclose(full_tokens, base_tokens, abs_tol=tolerance):
        if not math.isclose(target_tokens, base_tokens, abs_tol=tolerance):
            raise ValueError("所有建议权重均为 1，无法增加到目标 token")
        alpha = 0.0
    else:
        alpha = (target_tokens - base_tokens) / (full_tokens - base_tokens)

    output: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for row in prepared:
        multiplier = 1.0 + alpha * (row["recommended_weight"] - 1.0)
        full_copies = int(math.floor(multiplier + 1e-12))
        fraction = multiplier - full_copies
        if fraction < 1e-9:
            fraction = 0.0
        partial_sample_num = round_samples(row["sample_num"] * fraction) if fraction else 0
        if partial_sample_num >= row["sample_num"]:
            full_copies += 1
            partial_sample_num = 0
        copies = full_copies + (1 if partial_sample_num > 0 else 0)
        if copies > args.max_copies:
            raise ValueError(
                f"{row['config_key']}: 需要 {copies} 份，超过 --max-copies={args.max_copies}"
            )
        for copy_index in range(1, full_copies + 1):
            new_key = f"{row['config_key']}_{copy_index}"
            if new_key in output:
                raise ValueError(f"输出名称冲突: {new_key}")
            output[new_key] = deepcopy(row["spec"])
        if partial_sample_num > 0:
            new_key = f"{row['config_key']}_{full_copies + 1}"
            if new_key in output:
                raise ValueError(f"输出名称冲突: {new_key}")
            partial_spec = deepcopy(row["spec"])
            partial_spec["sample_num"] = partial_sample_num
            output[new_key] = partial_spec

        actual_multiplier = (
            full_copies + partial_sample_num / row["sample_num"]
        )
        estimated_tokens = row["total_tokens"] * actual_multiplier
        audit_rows.append(
            {
                **row,
                "actual_multiplier": actual_multiplier,
                "full_copies": full_copies,
                "partial_sample_num": partial_sample_num,
                "copies": copies,
                "estimated_tokens": estimated_tokens,
            }
        )

    estimated_total = sum(row["estimated_tokens"] for row in audit_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit_path = args.audit or args.output.with_name(args.output.stem + "_audit.md")
    audit_path.write_text(
        build_audit(
            args.config,
            args.weights_md,
            args.tokens,
            args.output,
            target_tokens,
            base_tokens,
            full_tokens,
            alpha,
            estimated_total,
            audit_rows,
        ),
        encoding="utf-8",
    )
    print(f"output={args.output}")
    print(f"audit={audit_path}")
    print(f"datasets={len(config)} entries={len(output)}")
    print(
        f"target_tokens={target_tokens:.0f} estimated_tokens={estimated_total:.0f} "
        f"error={estimated_total - target_tokens:+.0f}"
    )


if __name__ == "__main__":
    main()
