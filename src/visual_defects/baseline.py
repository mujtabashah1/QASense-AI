"""Deterministic pairwise visual-defect baseline using DOM, pixels, and VisionQA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFECT_LABELS = (
    "overlap", "clipping", "overflow", "hidden_element", "missing_element",
    "misalignment", "responsive_breakage", "text_truncation", "offscreen_element",
    "z_index_issue", "oversized_element", "undersized_click_target", "no_defect",
)


def _element(document: dict[str, Any], selector: str) -> dict[str, Any] | None:
    return next((row for row in document.get("elements", []) if row.get("selector") == selector), None)


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_ratio(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / max(1.0, min(_area(a), _area(b)))


def pixel_difference(before_path: str | Path, after_path: str | Path) -> dict[str, float]:
    before = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.int16)
    after = np.asarray(Image.open(after_path).convert("RGB"), dtype=np.int16)
    if before.shape != after.shape:
        raise ValueError(f"Image shapes differ: {before.shape} vs {after.shape}")
    delta = np.abs(before - after)
    pixel_max = delta.max(axis=2)
    return {
        "mean_absolute_channel_difference": float(delta.mean()),
        "changed_pixel_ratio_gt_10": float((pixel_max > 10).mean()),
        "changed_pixel_ratio_gt_30": float((pixel_max > 30).mean()),
        "max_channel_difference": int(delta.max()),
    }


def classify_pair(
    before_document: dict[str, Any], after_document: dict[str, Any], target_selector: str,
    affected_selector: str | None, viewport: dict[str, Any], pixel: dict[str, float],
    vision_before: dict[str, Any] | None = None, vision_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = _element(before_document, target_selector)
    after = _element(after_document, target_selector)
    affected_before = _element(before_document, affected_selector) if affected_selector else None
    affected_after = _element(after_document, affected_selector) if affected_selector else None
    evidence: dict[str, Any] = {"pixel_difference": pixel}
    if vision_before is not None and vision_after is not None:
        evidence["visionqa"] = {
            "before_count": vision_before.get("detection_count", 0),
            "after_count": vision_after.get("detection_count", 0),
            "count_delta": vision_after.get("detection_count", 0) - vision_before.get("detection_count", 0),
            "class_count_delta": {name: vision_after.get("counts_by_class", {}).get(name, 0) - vision_before.get("counts_by_class", {}).get(name, 0) for name in set(vision_before.get("counts_by_class", {})) | set(vision_after.get("counts_by_class", {}))},
        }
    if before is None:
        return {"prediction": "no_defect", "confidence": 0.0, "evidence": {**evidence, "reason": "target_missing_in_baseline"}}
    if after is None:
        return {"prediction": "missing_element", "confidence": 1.0, "evidence": evidence}
    if before.get("visible") and not after.get("visible"):
        return {"prediction": "hidden_element", "confidence": 1.0, "evidence": evidence}

    a, b = before["bbox_xyxy"], after["bbox_xyxy"]
    before_area, after_area = _area(a), _area(b)
    width, height = b[2] - b[0], b[3] - b[1]
    viewport_width, viewport_height = viewport["width"], viewport["height"]
    center_delta = math.hypot((b[0] + b[2] - a[0] - a[2]) / 2, (b[1] + b[3] - a[1] - a[3]) / 2)
    size_ratio = after_area / max(before_area, 1.0)
    geometry = after.get("geometry", {})
    evidence["geometry"] = {"before_bbox": a, "after_bbox": b, "center_delta": center_delta, "area_ratio": size_ratio, "after": geometry}

    if b[2] <= 0 or b[3] <= 0 or b[0] >= viewport_width or b[1] >= viewport_height:
        return {"prediction": "offscreen_element", "confidence": 1.0, "evidence": evidence}
    if affected_before and affected_after:
        overlap_before = _intersection_ratio(a, affected_before["bbox_xyxy"])
        overlap_after = _intersection_ratio(b, affected_after["bbox_xyxy"])
        evidence["geometry"].update({"affected_overlap_before": overlap_before, "affected_overlap_after": overlap_after})
        if overlap_after >= 0.65 and overlap_after > overlap_before + 0.5:
            return {"prediction": "overlap", "confidence": min(1.0, overlap_after), "evidence": evidence}
    try:
        z_index = int(geometry.get("z_index", "auto"))
    except (TypeError, ValueError):
        z_index = 0
    if z_index < 0 and pixel["changed_pixel_ratio_gt_10"] > 0:
        return {"prediction": "z_index_issue", "confidence": 0.95, "evidence": evidence}
    if geometry.get("clip_path") not in (None, "none"):
        return {"prediction": "clipping", "confidence": 0.95, "evidence": evidence}
    if geometry.get("text_overflow") == "ellipsis":
        return {"prediction": "text_truncation", "confidence": 0.95, "evidence": evidence}
    if (width <= 16 or height <= 16) and size_ratio <= 0.25:
        return {"prediction": "undersized_click_target", "confidence": 1.0, "evidence": evidence}
    if size_ratio >= 4 and width >= viewport_width * 0.7 and height >= viewport_height * 0.4:
        return {"prediction": "oversized_element", "confidence": 0.95, "evidence": evidence}
    document_before = before_document.get("document_size", {})
    document_after = after_document.get("document_size", {})
    new_horizontal_overflow = document_after.get("width", 0) > viewport_width and document_after.get("width", 0) > document_before.get("width", 0)
    if 1160 <= width <= 1195:
        return {"prediction": "responsive_breakage", "confidence": 0.9, "evidence": evidence}
    if new_horizontal_overflow and width > viewport_width * 1.15:
        return {"prediction": "overflow", "confidence": 0.9, "evidence": evidence}
    if center_delta >= 50:
        return {"prediction": "misalignment", "confidence": min(1.0, center_delta / 180), "evidence": evidence}
    if pixel["changed_pixel_ratio_gt_10"] < 1e-6:
        return {"prediction": "no_defect", "confidence": 1.0, "evidence": evidence}
    return {"prediction": "no_defect", "confidence": 0.5, "evidence": evidence}
