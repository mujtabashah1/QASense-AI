"""Fixed mapping and box preparation rules for WebUI transfer evaluation."""

from __future__ import annotations

import re
from typing import Any


SCORABLE_MAPPINGS = {
    "button": ("button", "exact", "high", "Native button element."),
    "a": ("link", "exact", "high", "Anchor element; href availability is not encoded in bbox records."),
    "input": ("input", "compatible", "medium", "Input type is absent from bbox schema; checkbox/radio-like identifiers are excluded."),
    "textarea": ("input", "compatible", "high", "AtomBlock input covers text-entry controls."),
    "img": ("image", "exact", "high", "Native image element."),
    "select": ("select", "exact", "high", "Native select element."),
    "nav": ("block-nav", "compatible", "high", "Native navigation container."),
    "aside": ("block-sidebar", "compatible", "medium", "Aside usually represents sidebar content but can contain other tangential content."),
    "footer": ("block-footer", "compatible", "high", "Native footer container."),
    "form": ("block-form", "compatible", "high", "Native form container."),
    "table": ("block-table", "compatible", "high", "Native table container."),
}
SCORABLE_TARGET_CLASSES = {value[0] for value in SCORABLE_MAPPINGS.values()}

AMBIGUOUS = {
    "svg": ("icon", "Generic SVG may be an icon, logo, illustration, or chart."),
    "i": ("icon", "The i tag is frequently an icon hook but also represents alternate voice text."),
    "header": ("block-nav", "Header and navigation are not equivalent."),
    "div": ("block", "Generic containers do not identify a curated AtomBlock block."),
    "section": ("block", "Sections may or may not correspond to a visual block."),
    "article": ("block", "Articles may contain several visual blocks."),
}


def mapping_for_tag(tag: str) -> dict[str, Any]:
    tag = (tag or "").lower()
    if tag in SCORABLE_MAPPINGS:
        target, mapping_type, confidence, reason = SCORABLE_MAPPINGS[tag]
        return {"source_label": tag, "qasense_target_class": target, "mapping_type": mapping_type, "mapping_confidence": confidence, "include_in_quantitative_scoring": True, "reason": reason}
    if tag in AMBIGUOUS:
        target, reason = AMBIGUOUS[tag]
        return {"source_label": tag, "qasense_target_class": target, "mapping_type": "ambiguous", "mapping_confidence": "low", "include_in_quantitative_scoring": False, "reason": reason}
    return {"source_label": tag, "qasense_target_class": None, "mapping_type": "incompatible", "mapping_confidence": "high", "include_in_quantitative_scoring": False, "reason": "No defensible mapping to the 14 AtomBlock detector classes."}


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def prepare_boxes(boxes: dict[str, list], image_width: int, image_height: int) -> dict[str, Any]:
    fields = ("tag", "x", "y", "width", "height", "text", "id", "class", "parent_index")
    values = {field: boxes.get(field) or [] for field in fields}
    count = min((len(value) for value in values.values()), default=0)
    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    for index in range(count):
        tag = str(values["tag"][index] or "").lower()
        x, y, width, height = (int(values[field][index]) for field in ("x", "y", "width", "height"))
        raw = [x, y, x + width, y + height]
        record = {"source_index": index, "source_label": tag, "bbox_xyxy_original": raw, "text": str(values["text"][index] or "")[:80], "dom_id": str(values["id"][index] or ""), "css_class": str(values["class"][index] or "")[:300], "parent_index": values["parent_index"][index]}
        if width <= 0 or height <= 0:
            excluded.append({**record, "reason": "nonpositive_box"}); continue
        clipped = [max(0, x), max(0, y), min(image_width, x + width), min(image_height, y + height)]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            excluded.append({**record, "reason": "outside_viewport"}); continue
        mapping = mapping_for_tag(tag)
        semantic_hint = " ".join((record["dom_id"], record["css_class"], record["text"])).lower()
        if tag == "input" and re.search(r"(?:checkbox|radio|switch|toggle)", semantic_hint):
            excluded.append({**record, "bbox_xyxy": clipped, "reason": "input_subtype_ambiguous", "mapping": mapping}); continue
        if not mapping["include_in_quantitative_scoring"]:
            unmapped[tag] = unmapped.get(tag, 0) + 1
            continue
        candidate = {**record, "bbox_xyxy": clipped, "was_clipped_to_viewport": clipped != raw, "target_class": mapping["qasense_target_class"], "mapping_type": mapping["mapping_type"]}
        duplicate = next((item for item in mapped if item["target_class"] == candidate["target_class"] and box_iou(item["bbox_xyxy"], clipped) >= 0.95), None)
        if duplicate:
            excluded.append({**record, "bbox_xyxy": clipped, "reason": "duplicate_mapped_box", "duplicate_of_source_index": duplicate["source_index"]})
        else:
            mapped.append(candidate)
    return {"valid_ground_truth_boxes": mapped, "excluded_invalid_or_ambiguous_boxes": excluded, "unmapped_classes": unmapped}
