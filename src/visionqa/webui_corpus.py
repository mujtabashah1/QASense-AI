"""Shared, deterministic rules for the WebUI-derived VisionQA v2 corpus."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable


SEED = 42
VIEWPORTS = ("desktop", "tablet", "mobile")
V2_CLASSES = (
    "button", "link", "input", "image", "select",
    "block-nav", "block-sidebar", "block-footer", "block-form", "block-table",
)
V2_CLASS_TO_ID = {name: index for index, name in enumerate(V2_CLASSES)}
ATOMBLOCK_CLASS_IDS = {
    "button": 0, "link": 1, "input": 2, "image": 3, "select": 7,
    "block-nav": 8, "block-sidebar": 9, "block-footer": 10,
    "block-form": 11, "block-table": 12,
}
WEBUI_SOURCE_LABELS = {
    "button": ["button"], "link": ["a"], "input": ["input", "textarea"],
    "image": ["img"], "select": ["select"], "block-nav": ["nav"],
    "block-sidebar": ["aside"], "block-footer": ["footer"],
    "block-form": ["form"], "block-table": ["table"],
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def group_id(original_split: str, sample_id: str) -> str:
    return f"webui-{original_split}-{sample_id}"


def screen_id(original_split: str, sample_id: str, viewport: str) -> str:
    return f"{group_id(original_split, sample_id)}-{viewport}"


def stable_rank(*parts: object, seed: int = SEED) -> str:
    return canonical_hash([seed, *parts])


def xyxy_to_yolo(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid clipped box {box} for {width}x{height}")
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def assign_source_grouped_split(groups: list[dict[str, Any]], validation_fraction: float = 0.2) -> dict[str, str]:
    """Assign complete groups while retaining both splits for sources with >=5 groups."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_source[group["source"]].append(group)
    assignments: dict[str, str] = {}
    for source, rows in sorted(by_source.items()):
        rows.sort(key=lambda row: stable_rank(source, row["group_id"]))
        if len(rows) >= 5:
            validation_count = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
        else:
            validation_count = 1 if len(rows) >= 2 else 0
        validation_ids = {row["group_id"] for row in rows[:validation_count]}
        for row in rows:
            assignments[row["group_id"]] = "val" if row["group_id"] in validation_ids else "train"
    return assignments


def class_schema() -> list[dict[str, Any]]:
    entries = []
    for name in V2_CLASSES:
        confidence = "medium" if name == "block-sidebar" else "high"
        mapping_type = "exact" if name in {"button", "link", "image", "select"} else "compatible"
        entries.append({
            "v2_class_id": V2_CLASS_TO_ID[name],
            "class_name": name,
            "original_atomblock_class_id": ATOMBLOCK_CLASS_IDS[name],
            "webui_source_labels": WEBUI_SOURCE_LABELS[name],
            "mapping_type": mapping_type,
            "mapping_confidence": confidence,
        })
    return entries
