"""Conservative single-state DOM visual checks for runtime evidence."""

from __future__ import annotations

from typing import Any

INTERACTIVE = {"button", "link", "checkbox", "radio", "combobox", "textbox"}


def inspect_runtime_visuals(dom: dict[str, Any], viewport: dict[str, int], matching: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    unmatched = set(matching.get("unmatched_dom_indices", []))
    for index, element in enumerate(dom.get("elements", [])):
        if not element.get("visible"):
            continue
        selector = element.get("selector", f"dom[{index}]")
        box = element.get("bbox_xyxy", [0, 0, 0, 0])
        geometry = element.get("geometry", {})
        if box[2] <= 0 or box[3] <= 0 or box[0] >= viewport["width"] or box[1] >= viewport["height"]:
            issues.append({"defect_type": "offscreen_element", "selector": selector, "severity": "medium", "confidence": 1.0, "source": "dom.json", "evidence": {"bbox_xyxy": box}})
        overflow_x = geometry.get("scroll_width", 0) > geometry.get("client_width", 0) + 2
        overflow_y = geometry.get("scroll_height", 0) > geometry.get("client_height", 0) + 2
        if (overflow_x or overflow_y) and (geometry.get("overflow_x") in {"hidden", "clip"} or geometry.get("overflow_y") in {"hidden", "clip"}):
            issues.append({"defect_type": "clipping_or_hidden_overflow", "selector": selector, "severity": "medium", "confidence": .85, "source": "dom.json", "evidence": {"geometry": geometry}})
        role = element.get("role")
        width, height = max(0, box[2] - box[0]), max(0, box[3] - box[1])
        if role in INTERACTIVE and (width < 24 or height < 24):
            issues.append({"defect_type": "undersized_click_target", "selector": selector, "severity": "low", "confidence": .8, "source": "dom.json", "evidence": {"bbox_xyxy": box, "role": role}})
        if index in unmatched and role in INTERACTIVE:
            issues.append({"defect_type": "unmatched_interactive_control", "selector": selector, "severity": "info", "confidence": .4, "source": "dom_vision_matches.json", "evidence": {"role": role}, "limitation": "Missing VisionQA match does not prove a visual defect."})
    return issues
