#!/usr/bin/env python3
"""Concurrent label-gated captioning with routed category prompts.

Input records are JSON or JSONL objects with:
  image_path: required
  domain_labels/domain_label/domain: required string, list or taxonomy object
  genre_labels/genre_label/genre: optional string, list or taxonomy object
  prompt_key: optional file name, stem or numeric prefix used to resolve route ties
  id: optional stable output key; image_path is used when absent

The final prompt receives canonical English label names from the selected route;
the original supplied labels are retained in the output for traceability.

TXT input is UTF-8, one tab-separated record per line:
  absolute_image_path<TAB>domain_labels[<TAB>genre_labels[<TAB>prompt_key]]
Use | inside a label field for multiple labels. Blank lines are ignored.

Example:
  {
    "id": "sample-1",
    "image_path": "images/1.png",
    "domain_labels": {"id": "D11", "label": "物理",
                      "children": [{"id": "D11.03", "label": "电磁学"}]},
    "genre_labels": {"id": "GNR.04", "label": "示意图",
                     "children": [{"id": "GNR.04.06", "label": "其他示意图"}]},
    "prompt_key": "12"
  }
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl

try:
    from .final_caption_prompt import render_final_prompt
    from .vlm_client import (
        DEFAULT_CONFIG,
        VLMResponse,
        create_client,
        generate_response,
        load_config,
    )
except ImportError:
    from final_caption_prompt import render_final_prompt
    from vlm_client import (
        DEFAULT_CONFIG,
        VLMResponse,
        create_client,
        generate_response,
        load_config,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "caption_promptV3" / "caption_prompt"
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "domain_and_genre_labels"
    / "caption_prompt_label_assignments_v1.2.json"
)
DEFAULT_DOMAIN_MAPPING = (
    PROJECT_ROOT / "domain_and_genre_labels" / "domain_label_mapping_v1.1.json"
)
DEFAULT_GENRE_MAPPING = (
    PROJECT_ROOT / "domain_and_genre_labels" / "genre_label_mapping_v1.1.json"
)
DEFAULT_WEAK_CATEGORY_MAP = (
    PROJECT_ROOT
    / "domain_and_genre_labels"
    / "mmmupro_weak_categories_with_ids_v1.1.json"
)
DEFAULT_MD5_FILE = PROJECT_ROOT / "outputs" / "processed_image_md5.jsonl"
PROMPT_BLOCK_RE = re.compile(r"```text\s*(.*?)\s*```", re.DOTALL)
ID_TOKEN_RE = re.compile(r"(?:D\d{2}(?:\.\d{2})?|GNR\.\d{2}(?:\.\d{2})?)")
LABEL_KEYS = ("id", "leaf_id", "slug", "label", "leaf_label", "taxonomy_label")
DOMAIN_FALLBACK_PROMPTS = {
    "01", "02", "03", "09", "10", "11", "12", "13",
    "14", "15", "16", "17", "18", "19", "20",
}
GENRE_FALLBACK_PROMPTS = {
    "04", "05", "06", "07", "08",
    "21", "22", "23", "24", "25", "26", "27",
}

DOMAIN_ENGLISH_LABELS = {
    "D00": "Out of Scope", "D01": "Humanities and Social Sciences", "D01.01": "History", "D01.02": "Linguistics", "D01.03": "Literature", "D01.05": "Religion", "D01.06": "Folklore and Culture", "D01.07": "Sociology", "D01.08": "Psychology", "D01.09": "Politics",
    "D02": "Business", "D02.01": "Economics", "D02.02": "Finance", "D02.03": "Accounting", "D02.04": "Management", "D02.05": "Marketing",
    "D03": "Arts", "D03.01": "Artworks", "D03.02": "Graphic Design",
    "D04": "Entertainment", "D04.01": "Music", "D04.03": "Anime", "D04.04": "Games",
    "D05": "Sports", "D05.01": "Land Ball Sports", "D06": "Food and Dining", "D09": "Home and Living", "D09.01": "Furniture and Furnishings",
    "D10": "Mathematics", "D10.01": "Arithmetic and Number Theory", "D10.02": "Algebra", "D10.03": "Geometry", "D10.04": "Mathematical Analysis", "D10.05": "Probability and Statistics", "D10.06": "Discrete Mathematics", "D10.07": "Operations Research",
    "D11": "Physics", "D11.01": "Mechanics", "D11.02": "Thermal Physics", "D11.03": "Electromagnetism", "D11.04": "Optics", "D11.06": "Astronomy",
    "D12": "Chemistry", "D12.01": "Inorganic Chemistry", "D12.02": "Organic Chemistry", "D12.03": "Physical Chemistry", "D12.04": "Analytical Chemistry", "D12.05": "Biochemistry",
    "D13": "Biology", "D13.01": "Biological Entities", "D13.02": "Biological Knowledge",
    "D14": "Geography and Environment", "D14.02": "Geomorphology and Physical Geography", "D14.03": "Human Geography", "D14.04": "Geology", "D14.05": "Meteorology", "D14.06": "Hydrology and Oceanography", "D14.07": "Environmental Issues", "D14.08": "Maps and GIS",
    "D15": "Medicine", "D15.01": "Basic Medicine", "D15.02": "Medical Examination and Testing", "D15.03": "Clinical Medicine", "D15.05": "Public Health",
    "D16": "Agriculture", "D16.01": "Crop Production", "D16.02": "Crop Diseases and Pests", "D16.06": "Forestry", "D16.07": "Fisheries and Aquaculture",
    "D17": "Engineering", "D17.01": "Civil and Architectural Engineering", "D17.02": "Mechanical Engineering", "D17.03": "Electrical and Electronic Engineering", "D17.04": "Materials Engineering", "D17.05": "Industrial Design and Manufacturing", "D17.08": "Energy and Power Engineering", "D17.09": "Control and Automation",
    "D18": "Information Technology", "D18.02": "Applications and Systems Software", "D18.03": "Programming and Algorithms", "D18.04": "Artificial Intelligence", "D18.05": "Networks and Security",
    "D19": "Transportation", "D19.01": "Road Vehicles", "D20": "Public Safety and Military", "D20.02": "Military",
}

GENRE_ENGLISH_LABELS = {
    "GNR.01": "Photography", "GNR.01.01": "People and Portraits", "GNR.01.02": "Landscape and Nature", "GNR.01.03": "Architecture and Places", "GNR.01.04": "Animals and Plants", "GNR.01.05": "Objects and Still Life", "GNR.01.06": "Other Photographs",
    "GNR.02": "Painting", "GNR.02.01": "Traditional Media Painting", "GNR.02.02": "Ink Wash and Gongbi", "GNR.02.04": "Digital Illustration", "GNR.02.05": "Sketches and Design Drafts",
    "GNR.03": "3-D Rendering", "GNR.03.01": "Product Rendering", "GNR.03.02": "Architecture and Interior Rendering", "GNR.03.03": "Other Rendering",
    "GNR.04": "Diagram", "GNR.04.01": "Annotated Structural Diagram", "GNR.04.02": "Circuit Diagram", "GNR.04.03": "Flow and Workflow Diagram", "GNR.04.04": "Relationship and Hierarchy Diagram", "GNR.04.05": "Timeline", "GNR.04.06": "Other Diagram",
    "GNR.05": "Charts and Plots", "GNR.05.01": "Bar and Column Chart", "GNR.05.02": "Line and Area Chart", "GNR.05.03": "Scatter and Bubble Chart", "GNR.05.06": "Cartesian Function Plot", "GNR.05.07": "Spectrum and Waveform Plot", "GNR.05.08": "Other Charts and Plots",
    "GNR.06": "Table", "GNR.07": "Map", "GNR.07.01": "Administrative and Topographic Map", "GNR.07.03": "Thematic Map", "GNR.08": "Document", "GNR.08.02": "Article, Report and Paper Page", "GNR.08.03": "Handwritten Manuscript and Notes", "GNR.08.05": "Test and Exercises", "GNR.08.06": "Slide Page", "GNR.08.09": "Other Document",
    "GNR.11": "Symbol", "GNR.11.01": "Mathematical Formula and Notation", "GNR.11.02": "Chemical Formula and Structural Formula", "GNR.11.04": "Logos and Icons", "GNR.11.06": "Other Symbols", "GNR.12": "Geometry",
    "GNR.13": "Micrograph", "GNR.13.01": "Optical Micrograph", "GNR.13.03": "Fluorescence Micrograph", "GNR.13.04": "Histopathology Slide", "GNR.13.05": "Other Micrograph", "GNR.14": "Medical Imaging", "GNR.14.01": "X-ray Image", "GNR.14.02": "CT and MRI", "GNR.14.03": "Ultrasound Image", "GNR.14.04": "Other Medical Image",
    "GNR.15": "Instrument Imaging", "GNR.15.03": "Satellite Remote Sensing Image", "GNR.15.05": "Other Instrument Image", "GNR.16": "Poster, Cover and Advertisement", "GNR.17": "Infographic", "GNR.18": "Meme", "GNR.19": "Comic", "GNR.20": "Sheet Music", "GNR.21": "Blueprint",
}


class RoutingError(ValueError):
    """The supplied labels do not select one auditable category prompt."""


class OutputFormatError(ValueError):
    """The model response does not obey the required wrapper."""


def image_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MD5Registry:
    """File-backed, multi-process-safe image MD5 reservations and outcomes."""

    def __init__(self, path: Path, reservation_ttl: float) -> None:
        self.path = path
        self.reservation_ttl = reservation_ttl

    def _read_latest_locked(self, handle: Any) -> dict[str, dict[str, Any]]:
        handle.seek(0)
        latest: dict[str, dict[str, Any]] = {}
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            md5 = event.get("md5")
            if isinstance(md5, str) and md5:
                latest[md5] = event
        return latest

    def _append_locked(self, handle: Any, event: dict[str, Any]) -> None:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def claim(self, md5: str, image_path: Path) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                latest = self._read_latest_locked(handle).get(md5)
                if latest:
                    status = latest.get("status")
                    timestamp = float(latest.get("timestamp", 0))
                    if status == "done":
                        return False
                    if status == "reserved" and now - timestamp < self.reservation_ttl:
                        return False
                self._append_locked(
                    handle,
                    {
                        "md5": md5,
                        "status": "reserved",
                        "image_path": str(image_path),
                        "timestamp": now,
                        "pid": os.getpid(),
                    },
                )
                return True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def finish(self, md5: str, image_path: Path, status: str, error: str | None = None) -> None:
        if status not in {"done", "failed"}:
            raise ValueError(f"invalid MD5 registry status: {status}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "md5": md5,
            "status": status,
            "image_path": str(image_path),
            "timestamp": time.time(),
            "pid": os.getpid(),
        }
        if error:
            event["error"] = error
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._append_locked(handle, event)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, order=True)
class LabelRef:
    parent_id: str
    child_id: str | None = None


@dataclass(frozen=True)
class ResolvedPrior:
    display: str
    refs: tuple[LabelRef, ...]


@dataclass(frozen=True)
class RouteCandidate:
    prompt_file: str
    score: int
    match_level: str


@dataclass(frozen=True)
class ParsedOutput:
    refusal: str
    rejected: bool
    caption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--domain-mapping", type=Path, default=DEFAULT_DOMAIN_MAPPING)
    parser.add_argument("--genre-mapping", type=Path, default=DEFAULT_GENRE_MAPPING)
    parser.add_argument(
        "--weak-category-map",
        type=Path,
        default=DEFAULT_WEAK_CATEGORY_MAP,
        help="ID mapping generated from mmmupro弱项分类.txt",
    )
    parser.add_argument(
        "--filter-weak-categories",
        action="store_true",
        help="require the input domain L1 + genre L1 pair to occur in the weak-category mapping",
    )
    parser.add_argument(
        "--md5-file",
        type=Path,
        default=DEFAULT_MD5_FILE,
        help="append-only image MD5 registry used for deduplication",
    )
    parser.add_argument(
        "--md5-reservation-ttl",
        type=float,
        default=86400.0,
        help="seconds before an abandoned MD5 reservation can be reclaimed",
    )
    parser.add_argument(
        "--disable-md5-dedupe",
        action="store_true",
        help="disable image MD5 deduplication for this run",
    )
    parser.add_argument(
        "--md5-progress-interval",
        type=int,
        default=1000,
        help="print MD5 progress every N completed images",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--concurrency", type=int, default=8, help="Gemini request concurrency.")
    parser.add_argument("--data-concurrency", type=int, default=16, help="Local data/image preparation concurrency; defaults to --concurrency.")
    parser.add_argument(
        "--md5-concurrency",
        type=int,
        help="number of workers for MD5 calculation and deduplication; defaults to --concurrency",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Include the rendered final prompt in normal output rows.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_txt_label_field(value: str, field_name: str) -> Any:
    value = value.strip()
    if not value:
        raise ValueError(f"TXT {field_name} cannot be empty")
    if value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"TXT {field_name} JSON is invalid: {exc.msg}"
            ) from exc
    labels = [item.strip() for item in value.split("|") if item.strip()]
    return labels if len(labels) > 1 else labels[0]


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) not in (2, 3, 4):
                    raise ValueError(
                        f"TXT line {line_number} must contain 2, 3 or 4 "
                        "tab-separated fields"
                    )
                image_path = fields[0].strip()
                if not Path(image_path).is_absolute():
                    raise ValueError(
                        f"TXT line {line_number}: image_path must be absolute"
                    )
                record: dict[str, Any] = {
                    "image_path": image_path,
                    "domain_labels": parse_txt_label_field(
                        fields[1], "domain_labels"
                    ),
                    **(
                        {"genre_labels": parse_txt_label_field(fields[2], "genre_labels")}
                        if len(fields) >= 3 and fields[2].strip()
                        else {}
                    ),
                }
                if len(fields) == 4 and fields[3].strip():
                    record["prompt_key"] = fields[3].strip()
                yield record
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                yield value
        return

    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("records", value.get("result_details"))
    if not isinstance(value, list):
        raise ValueError("JSON input must be an array or contain records/result_details")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"JSON record {index} is not an object")
        yield record


def first_present(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    raise ValueError(f"missing required field; expected one of: {', '.join(names)}")


def flatten_label_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, str):
        if value.strip():
            tokens.append(value.strip())
    elif isinstance(value, (int, float)):
        tokens.append(str(value))
    elif isinstance(value, list):
        for item in value:
            tokens.extend(flatten_label_tokens(item))
    elif isinstance(value, dict):
        found = False
        for key in LABEL_KEYS:
            if key in value:
                tokens.extend(flatten_label_tokens(value[key]))
                found = True
        if "children" in value:
            tokens.extend(flatten_label_tokens(value["children"]))
            found = True
        if not found:
            raise ValueError(f"label object has no recognised keys: {sorted(value)}")
    else:
        raise ValueError(f"unsupported label value: {type(value).__name__}")
    return tokens


def normalise_token(token: str) -> str:
    return " ".join(token.strip().split()).casefold()


def build_label_index(forms: list[dict[str, Any]], *, domain: bool) -> dict[str, set[LabelRef]]:
    index: dict[str, set[LabelRef]] = defaultdict(set)

    def add(token: Any, ref: LabelRef) -> None:
        if isinstance(token, str) and token.strip():
            index[normalise_token(token)].add(ref)

    for form in forms:
        parent_id = form["id"]
        parent_ref = LabelRef(parent_id)
        for key in ("id", "slug", "label", "taxonomy_label"):
            add(form.get(key), parent_ref)
        for child in form.get("children", []):
            child_id = child.get("leaf_id") if domain else child.get("id")
            if not child_id:
                continue
            # Domain mappings repeat the parent as a leaf (for example
            # D11 -> leaf_id D11). Treat that self leaf as the parent itself;
            # otherwise a parent label becomes an impossible child constraint.
            child_ref = (
                parent_ref
                if child_id == parent_id
                else LabelRef(parent_id, child_id)
            )
            for key in ("id", "leaf_id", "label", "leaf_label"):
                add(child.get(key), child_ref)
    return index


def resolve_prior(value: Any, index: dict[str, set[LabelRef]], name: str) -> ResolvedPrior:
    tokens = flatten_label_tokens(value)
    if not tokens:
        raise ValueError(f"{name} labels are empty")

    refs: set[LabelRef] = set()
    unknown: list[str] = []
    for token in tokens:
        matches = index.get(normalise_token(token))
        if not matches:
            extracted = ID_TOKEN_RE.findall(token)
            matches = set()
            for item in extracted:
                matches.update(index.get(normalise_token(item), set()))
        if matches:
            refs.update(matches)
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(f"unknown {name} label token(s): {unknown}")

    parents_with_children = {ref.parent_id for ref in refs if ref.child_id is not None}
    refs = {
        ref
        for ref in refs
        if not (ref.child_id is None and ref.parent_id in parents_with_children)
    }
    display = value.strip() if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    )
    return ResolvedPrior(display=display, refs=tuple(sorted(refs)))


def _compact_assignment_forms(
    ids: list[str],
    taxonomy_forms: list[dict[str, Any]],
    *,
    domain: bool,
) -> list[dict[str, Any]]:
    """Expand v1.2 prompt->ID lists into the existing router form shape."""
    selected = set(ids)
    if not selected:
        return []
    labels: dict[str, str] = {}
    english_labels = DOMAIN_ENGLISH_LABELS if domain else GENRE_ENGLISH_LABELS
    for form in taxonomy_forms:
        labels[form["id"]] = english_labels.get(form["id"], form["id"])
        for child in form.get("children", []):
            child_id = child.get("id", child.get("leaf_id"))
            if child_id:
                labels[child_id] = english_labels.get(child_id, child_id)

    grouped: dict[str, set[str]] = defaultdict(set)
    for label_id in selected:
        is_level2 = (
            label_id.count(".") >= 2
            if not domain
            else label_id.count(".") >= 1
        )
        parent_id = label_id.rsplit(".", 1)[0] if is_level2 else label_id
        if label_id != parent_id:
            grouped[parent_id].add(label_id)
        else:
            grouped.setdefault(parent_id, set())

    result: list[dict[str, Any]] = []
    for parent_id in sorted(grouped):
        children = [
            {"id": child_id, "label_en": labels.get(child_id, child_id)}
            for child_id in sorted(grouped[parent_id])
        ]
        result.append(
            {
                "id": parent_id,
                "label_en": labels.get(parent_id, parent_id),
                "children": children,
            }
        )
    return result


def load_assignments(
    path: Path,
    domain_forms: list[dict[str, Any]] | None = None,
    genre_forms: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    value = load_json(path)
    assignments = value.get("assignments") if isinstance(value, dict) else None
    if assignments is None and isinstance(value, dict):
        domain_prompts = value.get("domain_prompts")
        genre_prompts = value.get("genre_prompts")
        if not isinstance(domain_prompts, dict) or not isinstance(genre_prompts, dict):
            raise ValueError(f"invalid assignments file: {path}")
        if domain_forms is None or genre_forms is None:
            raise ValueError("compact assignments require domain and genre mappings")
        prompt_files = sorted(set(domain_prompts) | set(genre_prompts))
        assignments = [
            {
                "prompt_file": name,
                "domains": _compact_assignment_forms(
                    domain_prompts.get(name, []), domain_forms, domain=True
                ),
                "genres": _compact_assignment_forms(
                    genre_prompts.get(name, []), genre_forms, domain=False
                ),
            }
            for name in prompt_files
        ]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(f"invalid assignments file: {path}")
    seen: set[str] = set()
    for item in assignments:
        name = item.get("prompt_file")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"invalid or duplicate prompt_file: {name!r}")
        seen.add(name)
        for dimension in ("domains", "genres"):
            forms = item.get(dimension)
            if not isinstance(forms, list):
                raise ValueError(f"{name}: {dimension} must be a list")
            for form in forms:
                children = form.get("children", [])
                if not isinstance(children, list):
                    raise ValueError(f"{name}: {dimension} children must be a list")
                for node in (form, *children):
                    if not isinstance(node.get("id"), str) or not node["id"]:
                        raise ValueError(f"{name}: invalid {dimension} label id")
                    if not isinstance(node.get("label_en"), str) or not node["label_en"].strip():
                        node_id = node["id"]
                        raise ValueError(
                            f"{name}: {dimension} label {node_id!r} has no label_en"
                        )
    return assignments


def load_weak_l1_pairs(path: Path) -> set[tuple[str, str]]:
    value = load_json(path)
    categories = value.get("categories") if isinstance(value, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"invalid weak-category mapping: {path}")
    pairs: set[tuple[str, str]] = set()
    for index, category in enumerate(categories, 1):
        if isinstance(category, list) and len(category) >= 3:
            domain_l1_id, genre_l1_id = category[0], category[2]
        elif isinstance(category, dict):
            domain_l1_id = category.get("domain_l1_id")
            genre_l1_id = category.get("genre_l1_id")
        else:
            raise ValueError(f"invalid weak-category row {index}: {category!r}")
        if not isinstance(domain_l1_id, str) or not isinstance(genre_l1_id, str):
            raise ValueError(f"invalid weak-category IDs at row {index}")
        pairs.add((domain_l1_id, genre_l1_id))
    return pairs


def matches_weak_l1_pair(
    domain: ResolvedPrior,
    genre: ResolvedPrior,
    weak_l1_pairs: set[tuple[str, str]],
) -> bool:
    domain_l1_ids = {ref.parent_id for ref in domain.refs}
    genre_l1_ids = {ref.parent_id for ref in genre.refs}
    return any(
        (domain_l1_id, genre_l1_id) in weak_l1_pairs
        for domain_l1_id in domain_l1_ids
        for genre_l1_id in genre_l1_ids
    )


def _dimension_matches(
    refs: tuple[LabelRef, ...], forms: list[dict[str, Any]], level: str
) -> bool:
    if not refs:
        return False
    if level == "level2" and any(ref.child_id is None for ref in refs):
        return False
    by_parent = {form["id"]: form for form in forms}
    for ref in refs:
        form = by_parent.get(ref.parent_id)
        if form is None:
            return False
        if level == "level1":
            continue
        child_ids = {
            child.get("id", child.get("leaf_id"))
            for child in form.get("children", [])
        }
        if ref.child_id not in child_ids:
            return False
    return True


def _axis_route_candidates(
    assignments: list[dict[str, Any]],
    refs: tuple[LabelRef, ...],
    *,
    dimension: str,
    fallback_prompts: set[str],
    level: str,
) -> list[tuple[dict[str, Any], RouteCandidate]]:
    """Return candidates for one taxonomy axis at one specificity level."""
    candidates: list[tuple[dict[str, Any], RouteCandidate]] = []
    for assignment in assignments:
        prompt_number = Path(assignment["prompt_file"]).stem.split("_", 1)[0]
        forms = assignment[dimension]
        if prompt_number not in fallback_prompts or not forms:
            continue
        if not _dimension_matches(refs, forms, level):
            continue
        candidates.append(
            (
                assignment,
                RouteCandidate(
                    prompt_file=assignment["prompt_file"],
                    score=len(forms),
                    match_level=f"{dimension}_{level}",
                ),
            )
        )
    return candidates


def prompt_aliases(prompt_file: str) -> set[str]:
    stem = Path(prompt_file).stem
    prefix = stem.split("_", 1)[0]
    return {prompt_file.casefold(), stem.casefold(), prefix.casefold()}


def choose_route(
    assignments: list[dict[str, Any]],
    domain: ResolvedPrior,
    genre: ResolvedPrior | None,
    prompt_key: str | None = None,
) -> tuple[dict[str, Any], list[RouteCandidate], dict[str, Any] | None, list[RouteCandidate], str, str | None]:
    """Select a required domain prompt, then an optional genre prompt.

    Domain and genre are intentionally routed independently. Domain specificity
    is resolved first; genre specificity is resolved only after a domain route
    exists and never prevents a valid domain-only job from being created.
    """
    del prompt_key  # prompt_key is intentionally ignored for auditable routing.

    def select_axis(
        refs: tuple[LabelRef, ...],
        dimension: str,
        fallback_prompts: set[str],
        required: bool,
    ) -> tuple[dict[str, Any] | None, list[RouteCandidate], str | None]:
        if not refs:
            if required:
                raise RoutingError(f"{dimension} labels are required for routing")
            return None, [], None
        candidates: list[tuple[dict[str, Any], RouteCandidate]] = []
        for level in ("level2", "level1"):
            candidates = _axis_route_candidates(
                assignments,
                refs,
                dimension=dimension,
                fallback_prompts=fallback_prompts,
                level=level,
            )
            if candidates:
                break
        if not candidates:
            if required:
                raise RoutingError(
                    f"no {dimension} prompt matches level2 or level1 labels"
                )
            return None, [], None
        route_candidates = [item[1] for item in candidates]
        best_count = min(candidate.score for candidate in route_candidates)
        tied = [item for item in candidates if item[1].score == best_count]
        selected, selected_candidate = random.choice(tied)
        return selected, route_candidates, selected_candidate.match_level

    domain_route, domain_candidates, domain_level = select_axis(
        domain.refs, "domains", DOMAIN_FALLBACK_PROMPTS, required=True
    )
    assert domain_route is not None and domain_level is not None
    genre_route, genre_candidates, genre_level = select_axis(
        genre.refs if genre is not None else (),
        "genres",
        GENRE_FALLBACK_PROMPTS,
        required=False,
    )
    return (
        domain_route,
        domain_candidates,
        genre_route,
        genre_candidates,
        domain_level,
        genre_level,
    )


def render_english_prior(
    refs: tuple[LabelRef, ...],
    forms: list[dict[str, Any]],
    name: str,
    level: str,
) -> str:
    """Render resolved labels as English names for prompt injection."""
    labels: dict[LabelRef, str] = {}
    for form in forms:
        parent_ref = LabelRef(form["id"])
        labels[parent_ref] = form["label_en"].strip()
        for child in form.get("children", []):
            child_id = child.get("id", child.get("leaf_id"))
            if child_id:
                labels[LabelRef(form["id"], child_id)] = child["label_en"].strip()

    rendered: list[str] = []
    for ref in refs:
        lookup_ref = ref if level == "level2" else LabelRef(ref.parent_id)
        label_en = labels.get(lookup_ref)
        if label_en is None:
            identifier = ref.child_id or ref.parent_id
            raise RoutingError(
                f"selected route has no English {name} label for {identifier}"
            )
        rendered.append(label_en)
    return "; ".join(rendered)


def load_category_prompt(prompt_dir: Path, prompt_file: str) -> str:
    path = prompt_dir / prompt_file
    if not path.is_file():
        raise FileNotFoundError(f"missing category prompt: {path}")
    text = path.read_text(encoding="utf-8")
    blocks = PROMPT_BLOCK_RE.findall(text)
    if len(blocks) != 1 or not blocks[0].strip():
        raise ValueError(f"{path} must contain exactly one non-empty text block")
    prompt = blocks[0].strip()
    if "<image 1>" in prompt:
        raise ValueError(f"category subprompt must not contain <image 1>: {path}")
    return prompt


def resolve_image(image_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_path must be a non-empty string")
    path = Path(value)
    path = path if path.is_absolute() else image_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return path


def parse_model_output(prediction: str) -> ParsedOutput:
    text = prediction.strip()
    if not text:
        raise OutputFormatError("model output is empty")

    # Preferred protocol: refusals carry the only reserved marker; accepted
    # responses are plain caption text and need no wrapper to be parsed.
    refusal_marker = "REFUSAL:"
    if text.casefold().startswith(refusal_marker.casefold()):
        reason = text[len(refusal_marker) :].strip()
        if not reason:
            raise OutputFormatError("REFUSAL output must include a reason")
        return ParsedOutput(
            refusal=f"Yes; Reason: {reason}",
            rejected=True,
            caption="",
        )

    # Backward compatibility for responses produced with the former wrapper.
    if text.startswith("{Refusal:") and text.endswith("}"):
        inner = text[len("{Refusal:") : -1].strip()
        if inner.startswith("No; Caption:"):
            caption = inner[len("No; Caption:") :].strip()
            if not caption:
                raise OutputFormatError("accepted output has an empty Caption")
            return ParsedOutput(refusal="No", rejected=False, caption=caption)
        if inner.startswith("Yes; Reason:"):
            reason = inner[len("Yes; Reason:") :].strip()
            if not reason:
                raise OutputFormatError("rejected output must include a Reason")
            if "Caption:" in reason:
                raise OutputFormatError("rejected output must not contain Caption")
            return ParsedOutput(
                refusal=f"Yes; Reason: {reason}",
                rejected=True,
                caption="",
            )

    # Any other non-empty response is the accepted caption itself.
    return ParsedOutput(refusal="No", rejected=False, caption=text)

def make_record_key(record: dict[str, Any], index: int) -> str:
    value = record.get("id", record.get("image_path"))
    if value is None:
        value = index
    return str(value)


def prepare_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domain_forms = load_json(args.domain_mapping)["forms"]
    genre_forms = load_json(args.genre_mapping)["forms"]
    assignments = load_assignments(args.assignments, domain_forms, genre_forms)
    weak_l1_pairs = (
        load_weak_l1_pairs(args.weak_category_map)
        if args.filter_weak_categories
        else set()
    )
    domain_index = build_label_index(domain_forms, domain=True)
    genre_index = build_label_index(genre_forms, domain=False)
    prompt_cache: dict[str, str] = {}
    assignment_genre_forms = [
        form
        for assignment in assignments
        for form in assignment["genres"]
    ]
    assignment_domain_forms = [
        form
        for assignment in assignments
        for form in assignment["domains"]
    ]
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    keys: set[str] = set()

    processed = 0
    source = iter_records(args.input)

    for index, record in enumerate(source):
        if args.limit is not None and index >= args.limit:
            break
        key = make_record_key(record, index)
        if key in keys:
            raise ValueError(f"duplicate record key: {key!r}")
        keys.add(key)
        image_path = resolve_image(args.image_root, record.get("image_path"))
        try:
            domain_value = first_present(record, ("domain_labels", "domain_label", "domain"))
            domain = resolve_prior(domain_value, domain_index, "domain")
            genre_value = next(
                (
                    record[name]
                    for name in ("genre_labels", "genre_label", "genre")
                    if name in record and record[name] is not None
                ),
                None,
            )
            # Genre is an optional second-stage route. An unknown or malformed
            # genre must not discard an otherwise usable domain route.
            if genre_value is not None:
                try:
                    genre = resolve_prior(genre_value, genre_index, "genre")
                except ValueError:
                    genre = None
            else:
                genre = None
            if args.filter_weak_categories and (
                genre is None or not matches_weak_l1_pair(domain, genre, weak_l1_pairs)
            ):
                raise RoutingError(
                    "domain L1 + genre L1 pair is not present in "
                    f"{args.weak_category_map.name}"
                )
            (
                domain_route,
                domain_candidates,
                genre_route,
                genre_candidates,
                domain_level,
                genre_level,
            ) = choose_route(assignments, domain, genre)
        except (RoutingError, ValueError) as exc:
            skipped.append(
                {
                    "_key": key,
                    "id": record.get("id"),
                    "image_path": str(image_path),
                    "domain_prior": record.get("domain_labels", record.get("domain_label", record.get("domain"))),
                    "genre_prior": record.get("genre_labels", record.get("genre_label", record.get("genre"))),
                    "domain_prior_en": None,
                    "genre_prior_en": None,
                    "prompt_file": None,
                    "domain_prompt_file": None,
                    "genre_prompt_file": None,
                    "route_score": None,
                    "route_candidates": [],
                    "genre_route_candidates": [],
                    "skipped": True,
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                    "rejected": None,
                    "refusal": None,
                    "caption": None,
                    "raw_prediction": None,
                    "format_valid": None,
                    "finish_reason": None,
                    "usage": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            processed += 1
            if processed % 1000 == 0:
                print(
                    f"prepare progress: processed={processed} jobs={len(jobs)} skipped={len(skipped)}",
                    file=sys.stderr,
                )
            continue

        domain_prior_en = render_english_prior(
            domain.refs, domain_route["domains"], "domain", domain_level.removeprefix("domains_")
        )
        if genre is None:
            genre_prior_en = "Unspecified genre"
        elif genre_route is None:
            try:
                genre_level_for_prior = (
                    "level2" if all(ref.child_id is not None for ref in genre.refs) else "level1"
                )
                genre_prior_en = render_english_prior(
                    genre.refs, assignment_genre_forms, "genre", genre_level_for_prior
                )
            except RoutingError:
                genre_prior_en = "Unspecified genre; no compatible genre prompt"
        else:
            genre_prior_en = render_english_prior(
                genre.refs, genre_route["genres"], "genre", genre_level.removeprefix("genres_")
            )

        domain_prompt_file = domain_route["prompt_file"]
        genre_prompt_file = genre_route["prompt_file"] if genre_route else None
        if domain_prompt_file not in prompt_cache:
            prompt_cache[domain_prompt_file] = load_category_prompt(args.prompt_dir, domain_prompt_file)
        if genre_prompt_file and genre_prompt_file not in prompt_cache:
            prompt_cache[genre_prompt_file] = load_category_prompt(args.prompt_dir, genre_prompt_file)
        final_prompt = render_final_prompt(
            domain_prior=domain_prior_en,
            genre_prior=genre_prior_en,
            domain_prompt=prompt_cache[domain_prompt_file],
            genre_prompt=prompt_cache[genre_prompt_file] if genre_prompt_file else None,
        )
        jobs.append(
            {
                "_key": key,
                "id": record.get("id"),
                "image_path": image_path,
                "domain_prior": domain.display,
                "genre_prior": genre.display if genre is not None else None,
                "domain_prior_en": domain_prior_en,
                "genre_prior_en": genre_prior_en,
                # Keep prompt_file as the required domain prompt for compatibility;
                # expose both axes explicitly for the new two-stage route.
                "prompt_file": domain_prompt_file,
                "domain_prompt_file": domain_prompt_file,
                "genre_prompt_file": genre_prompt_file,
                "route_level": domain_level,
                "genre_route_level": genre_level,
                "route_score": next(candidate.score for candidate in domain_candidates if candidate.prompt_file == domain_prompt_file),
                "route_candidates": [
                    {"prompt_file": candidate.prompt_file, "score": candidate.score, "match_level": candidate.match_level}
                    for candidate in sorted(
                        domain_candidates, key=lambda item: (item.score, item.prompt_file)
                    )
                ],
                "genre_route_candidates": [
                    {"prompt_file": candidate.prompt_file, "score": candidate.score, "match_level": candidate.match_level}
                    for candidate in sorted(
                        genre_candidates, key=lambda item: (item.score, item.prompt_file)
                    )
                ],
                "final_prompt": final_prompt,
            }
        )
        processed += 1
        if processed % 1000 == 0:
            print(
                f"prepare progress: processed={processed} jobs={len(jobs)} skipped={len(skipped)}",
                file=sys.stderr,
            )
    return jobs, skipped


def prepare_jobs_parallel(
    args: argparse.Namespace, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the existing per-record preparation logic concurrently by input chunks."""
    if workers <= 1:
        return prepare_jobs(args)
    if args.input.suffix.lower() not in {".txt", ".jsonl"}:
        raise ValueError("parallel prepare requires a line-oriented TXT or JSONL input")

    with tempfile.TemporaryDirectory(prefix="caption_prepare_") as temp_dir:
        chunk_paths = [Path(temp_dir) / f"chunk_{index:04d}.txt" for index in range(workers)]
        handles = [path.open("w", encoding="utf-8") for path in chunk_paths]
        try:
            with args.input.open("r", encoding="utf-8") as source:
                for index, line in enumerate(source):
                    handles[index % workers].write(line)
        finally:
            for handle in handles:
                handle.close()

        def run_chunk(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            chunk_args = copy.copy(args)
            chunk_args.input = path
            return prepare_jobs(chunk_args)

        all_jobs: list[dict[str, Any]] = []
        all_skipped: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_chunk, path) for path in chunk_paths]
            for future in futures:
                jobs, skipped = future.result()
                all_jobs.extend(jobs)
                all_skipped.extend(skipped)
        return all_jobs, all_skipped


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(iter_records(path)):
        key = record.get("_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"existing output row {index} has no _key")
        if key in existing:
            raise ValueError(f"existing output has duplicate _key: {key!r}")
        existing[key] = record
    return existing


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temp_path: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prompt_code(prompt_file: str | None) -> str | None:
    if not prompt_file:
        return None
    return Path(prompt_file).stem.split("_", 1)[0]


def base_result(job: dict[str, Any], save_prompt: bool) -> dict[str, Any]:
    domain_prompt_file = job.get("domain_prompt_file") or job.get("prompt_file")
    genre_prompt_file = job.get("genre_prompt_file")
    result = {
        "_key": job["_key"],
        "id": job["id"],
        "image_path": str(job["image_path"]),
        "image_md5": job.get("_image_md5"),
        "domain_prior": job["domain_prior"],
        "genre_prior": job["genre_prior"],
        "domain_prior_en": job["domain_prior_en"],
        "genre_prior_en": job["genre_prior_en"],
        # Keep prompt_file for backward compatibility, and persist both
        # matched axes so downstream viewers can reconstruct the final prompt.
        "prompt_file": job["prompt_file"],
        "domain_prompt_file": domain_prompt_file,
        "genre_prompt_file": genre_prompt_file,
        "prompt_codes": {
            "domain": prompt_code(domain_prompt_file),
            "genre": prompt_code(genre_prompt_file),
        },
        "prompt_files": [
            item for item in (domain_prompt_file, genre_prompt_file) if item
        ],
        "route_level": job.get("route_level"),
        "route_score": job["route_score"],
        "route_candidates": job["route_candidates"],
    }
    if save_prompt:
        result["final_prompt"] = job["final_prompt"]
    return result


def build_success(
    job: dict[str, Any],
    response: VLMResponse,
    parsed: ParsedOutput,
    save_prompt: bool,
) -> dict[str, Any]:
    return {
        **base_result(job, save_prompt),
        "rejected": parsed.rejected,
        "refusal": parsed.refusal,
        "caption": parsed.caption,
        "raw_prediction": response.prediction,
        "format_valid": True,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "error": None,
    }


def build_failure(
    job: dict[str, Any],
    exc: Exception,
    save_prompt: bool,
    response: VLMResponse | None = None,
) -> dict[str, Any]:
    return {
        **base_result(job, save_prompt),
        "rejected": None,
        "refusal": None,
        "caption": None,
        "raw_prediction": response.prediction if response else None,
        "format_valid": False if isinstance(exc, OutputFormatError) else None,
        "finish_reason": response.finish_reason if response else None,
        "usage": response.usage if response else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        print("error: --concurrency must be positive", file=sys.stderr)
        return 1
    if args.md5_concurrency is not None and args.md5_concurrency <= 0:
        print("error: --md5-concurrency must be positive", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit <= 0:
        print("error: --limit must be positive", file=sys.stderr)
        return 1
    if args.resume and args.overwrite:
        print("error: --resume and --overwrite are mutually exclusive", file=sys.stderr)
        return 1
    if args.retry_errors and not args.resume:
        print("error: --retry-errors requires --resume", file=sys.stderr)
        return 1
    if args.md5_reservation_ttl <= 0:
        print("error: --md5-reservation-ttl must be positive", file=sys.stderr)
        return 1
    if args.md5_progress_interval <= 0:
        print("error: --md5-progress-interval must be positive", file=sys.stderr)
        return 1
    md5_concurrency = args.data_concurrency

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_captioned.jsonl"
    )
    skipped_suffix = output_path.suffix or ".jsonl"
    skipped_output_path = output_path.with_name(
        f"{output_path.stem}_skipped{skipped_suffix}"
    )
    if output_path.exists() and not (args.resume or args.overwrite):
        print(
            f"error: output exists: {output_path}; use --resume or --overwrite",
            file=sys.stderr,
        )
        return 1

    try:
        prepare_concurrency = args.data_concurrency
        print(
            f"prepare start: workers={prepare_concurrency}",
            file=sys.stderr,
            flush=True,
        )
        jobs, skipped = prepare_jobs_parallel(args, prepare_concurrency)
        existing = load_existing(output_path) if args.resume else {}
        if args.retry_errors:
            done = {
                key for key, row in existing.items() if not row.get("error")
            }
        else:
            done = set(existing)
        jobs = [job for job in jobs if job["_key"] not in done]
        skipped = [row for row in skipped if row["_key"] not in done]
        # Inputs are deduplicated offline; skip online MD5 hashing, registry claims, and progress.
        md5_registry = None
        md5_jobs: list[dict[str, Any]] = jobs
        jobs = md5_jobs
        print(
            f"prepared={len(jobs)} skipped={len(skipped)} already_done={len(done)} "
            f"prepare_concurrency={prepare_concurrency} "
            f"model_concurrency={args.concurrency} data_concurrency={prepare_concurrency}",
            file=sys.stderr,
        )
        write_jsonl_atomic(
            skipped_output_path,
            ({**row, "dry_run": True} if args.dry_run else row for row in skipped),
        )
        print(f"skipped output: {skipped_output_path}", file=sys.stderr)

        if args.dry_run:
            rows = dict(existing)
            for job in jobs:
                rows[job["_key"]] = {
                    **base_result(job, save_prompt=True),
                    "dry_run": True,
                }
            write_jsonl_atomic(
                output_path, (rows[key] for key in sorted(rows))
            )
            print(f"dry-run output: {output_path}")
            return 0

        config = load_config(args.config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and output_path.exists() else "w"
        output = output_path.open(mode, encoding="utf-8", newline="\n")
        state = threading.local()

        def infer_one(job: dict[str, Any]) -> VLMResponse:
            if not hasattr(state, "client"):
                state.client = create_client(
                    config,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            response = generate_response(
                prompt=job["final_prompt"],
                image_path=job["image_path"],
                config=config,
                client=state.client,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                image_detail=args.image_detail,
            )
            return response

        succeeded = rejected = failed = 0
        replacements: dict[str, dict[str, Any]] = {}
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures: dict[Future[VLMResponse], dict[str, Any]] = {
                    executor.submit(infer_one, job): job for job in jobs
                }
                for position, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    response: VLMResponse | None = None
                    try:
                        response = future.result()
                        parsed = parse_model_output(response.prediction)
                        result = build_success(
                            job, response, parsed, args.save_prompt
                        )
                        succeeded += 1
                        if parsed.rejected:
                            rejected += 1
                        status = "REJECTED" if parsed.rejected else "OK"
                        if md5_registry is not None and not args.dry_run:
                            md5_registry.finish(
                                job["_image_md5"], job["image_path"], "done"
                            )
                    except Exception as exc:
                        result = build_failure(
                            job, exc, args.save_prompt, response
                        )
                        failed += 1
                        status = "FAILED"
                        if md5_registry is not None and not args.dry_run:
                            md5_registry.finish(
                                job["_image_md5"],
                                job["image_path"],
                                "failed",
                                f"{type(exc).__name__}: {exc}",
                            )
                    replacements[job["_key"]] = result
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    print(
                        f"[{timestamp}] [{position}/{len(jobs)}] {status} "
                        f"{job['_key']} -> {job['prompt_file']}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            output.close()

        if args.resume:
            merged = dict(existing)
            merged.update(replacements)
            write_jsonl_atomic(
                output_path, (merged[key] for key in sorted(merged))
            )

        print(
            f"finished: called={len(jobs)} accepted={succeeded - rejected} "
            f"rejected={rejected} failed={failed} skipped={len(skipped)} output={output_path} skipped_output={skipped_output_path}"
        )
        return 0 if failed == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
