"""Match viewport DOM geometry to VisionQA detections with explicit evidence."""

from __future__ import annotations

import math
from typing import Any


CLASS_COMPATIBILITY = {
    "button": {"button"}, "a": {"link"}, "link": {"link"},
    "img": {"image"}, "image": {"image"},
    "checkbox": {"checkbox", "input"}, "radio": {"radio", "input"},
    "select": {"select", "input"}, "combobox": {"select", "input"},
    "textbox": {"input"}, "input": {"input"}, "textarea": {"input"},
    "nav": {"block-nav", "block"}, "navigation": {"block-nav", "block"},
    "form": {"block-form", "block"}, "table": {"block-table", "block"},
    "contentinfo": {"block-footer", "block"}, "footer": {"block-footer", "block"},
    "aside": {"block-sidebar", "block"},
}


def _xyxy(box: Any) -> list[float]:
    if isinstance(box, dict):
        if "bbox_xyxy" in box:
            return [float(value) for value in box["bbox_xyxy"]]
        if "bbox" in box and isinstance(box["bbox"], dict):
            item = box["bbox"]
            return [item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"]]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(value) for value in box]
    raise ValueError(f"Unsupported bounding box: {box!r}")


def _geometry(a: list[float], b: list[float]) -> tuple[float, float, float]:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diagonal = max(math.hypot(a[2] - a[0], a[3] - a[1]), math.hypot(b[2] - b[0], b[3] - b[1]), 1.0)
    center_score = max(0.0, 1.0 - math.hypot(acx - bcx, acy - bcy) / diagonal)
    return iou, containment, center_score


def _compatible(element: dict[str, Any], class_name: str) -> tuple[bool, list[str]]:
    signals = {
        str(element.get("tag", "")).lower(), str(element.get("role", "")).lower(),
        str(element.get("input_type", "")).lower(),
    }
    expected: set[str] = set()
    for signal in signals:
        expected.update(CLASS_COMPATIBILITY.get(signal, set()))
    return class_name in expected, sorted(expected)


def _exact_control_class(element: dict[str, Any]) -> str | None:
    """Return a DOM-authoritative class only for native binary controls."""
    role = str(element.get("role", "")).lower()
    input_type = str(element.get("input_type", "")).lower()
    for value in (role, input_type):
        if value in {"checkbox", "radio"}:
            return value
    return None


def match_dom_to_detections(
    dom_elements: list[dict[str, Any]], detections: list[dict[str, Any]], minimum_score: float = 0.30,
) -> dict[str, Any]:
    """Create semantic one-to-one matches and retain geometry-only ambiguities."""
    candidates: list[dict[str, Any]] = []
    ambiguous_candidates: list[dict[str, Any]] = []
    visible_indices = [index for index, element in enumerate(dom_elements) if element.get("visible", False)]
    for dom_index in visible_indices:
        element = dom_elements[dom_index]
        dom_box = _xyxy(element)
        for detection_index, detection in enumerate(detections):
            detection_box = _xyxy(detection)
            iou, containment, center = _geometry(dom_box, detection_box)
            compatible, expected = _compatible(element, detection["class_name"])
            if not compatible:
                # Containment alone is weak for nested UI (a link is contained by nav).
                # Native checkbox/radio semantics are authoritative when an icon-like
                # detection has strong geometric agreement. Keep the raw detector
                # class in the evidence; this resolves integrated classification only.
                strong_geometry = containment >= 0.65 and iou >= 0.25
                resolved_class = _exact_control_class(element)
                if strong_geometry and detection["class_name"] == "icon" and resolved_class:
                    score = (0.45 * iou + 0.20 * containment + 0.15 * center + 0.20)
                    score *= float(detection.get("confidence", 1.0)) ** 0.25
                    if score >= minimum_score:
                        candidates.append({
                            "dom_index": dom_index,
                            "detection_index": detection_index,
                            "score": score,
                            "raw_vision_class": detection["class_name"],
                            "resolved_class": resolved_class,
                            "semantic_resolution_applied": True,
                            "evidence": {
                                "iou": iou,
                                "containment": containment,
                                "center_score": center,
                                "class_compatible": False,
                                "compatible_vision_classes": expected,
                                "detection_confidence": float(detection.get("confidence", 1.0)),
                                "resolution_source": "dom_native_control_semantics",
                            },
                        })
                        continue
                # Retain all other strong geometric conflicts as ambiguities.
                if strong_geometry:
                    ambiguous_candidates.append({
                        "dom_index": dom_index,
                        "detection_index": detection_index,
                        "selector": element.get("selector"),
                        "tag": element.get("tag"),
                        "role": element.get("role"),
                        "vision_class": detection["class_name"],
                        "reason": "strong_geometry_but_incompatible_class",
                        "evidence": {
                            "iou": iou,
                            "containment": containment,
                            "center_score": center,
                            "class_compatible": False,
                            "compatible_vision_classes": expected,
                            "detection_confidence": float(detection.get("confidence", 1.0)),
                        },
                    })
                continue
            score = (0.45 * iou + 0.20 * containment + 0.15 * center + 0.20)
            score *= float(detection.get("confidence", 1.0)) ** 0.25
            if score >= minimum_score:
                candidates.append({
                    "dom_index": dom_index, "detection_index": detection_index, "score": score,
                    "evidence": {"iou": iou, "containment": containment, "center_score": center,
                    "class_compatible": compatible, "compatible_vision_classes": expected,
                    "detection_confidence": float(detection.get("confidence", 1.0))},
                })
    matched_dom: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        dom_index, detection_index = candidate["dom_index"], candidate["detection_index"]
        if dom_index in matched_dom or detection_index in matched_detections:
            continue
        matched_dom.add(dom_index)
        matched_detections.add(detection_index)
        raw_class = detections[detection_index]["class_name"]
        matches.append({**candidate, "selector": dom_elements[dom_index].get("selector"),
            "tag": dom_elements[dom_index].get("tag"), "role": dom_elements[dom_index].get("role"),
            "vision_class": raw_class, "raw_vision_class": raw_class,
            "resolved_class": candidate.get("resolved_class", raw_class),
            "semantic_resolution_applied": candidate.get("semantic_resolution_applied", False),
            "match_confidence": candidate["score"]})
    return {
        "algorithm": "semantic_greedy_one_to_one_v3", "minimum_score": minimum_score,
        "dom_element_count": len(dom_elements), "visible_dom_element_count": len(visible_indices),
        "detection_count": len(detections), "match_count": len(matches), "matches": matches,
        "ambiguous_candidate_count": len(ambiguous_candidates),
        "ambiguous_candidates": sorted(
            ambiguous_candidates,
            key=lambda item: max(item["evidence"]["iou"], item["evidence"]["containment"]),
            reverse=True,
        ),
        "unmatched_dom_indices": [index for index in visible_indices if index not in matched_dom],
        "unmatched_detection_indices": [index for index in range(len(detections)) if index not in matched_detections],
        "limitation": "Geometry and semantic compatibility provide evidence, not guaranteed identity.",
    }
