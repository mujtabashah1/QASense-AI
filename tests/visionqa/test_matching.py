import pytest

from src.visionqa.matching import match_dom_to_detections


def test_matches_compatible_overlapping_elements_one_to_one():
    dom = [
        {"selector": "#submit", "tag": "button", "role": "button", "visible": True, "bbox_xyxy": [10, 10, 110, 50]},
        {"selector": "#hero", "tag": "img", "role": "img", "visible": True, "bbox_xyxy": [200, 20, 400, 120]},
    ]
    detections = [
        {"class_name": "button", "confidence": 0.9, "bbox_xyxy": [12, 11, 108, 51]},
        {"class_name": "image", "confidence": 0.8, "bbox_xyxy": [198, 18, 402, 122]},
    ]
    result = match_dom_to_detections(dom, detections)
    assert result["match_count"] == 2
    assert {row["selector"] for row in result["matches"]} == {"#submit", "#hero"}
    assert all(row["evidence"]["class_compatible"] for row in result["matches"])


def test_does_not_force_incompatible_distant_match():
    dom = [{"selector": "#submit", "tag": "button", "role": "button", "visible": True, "bbox_xyxy": [0, 0, 50, 30]}]
    detections = [{"class_name": "image", "confidence": 0.99, "bbox_xyxy": [500, 500, 700, 700]}]
    result = match_dom_to_detections(dom, detections)
    assert result["match_count"] == 0
    assert result["unmatched_dom_indices"] == [0]
    assert result["unmatched_detection_indices"] == [0]


def test_keeps_overlapping_class_conflict_as_ambiguity_not_match():
    dom = [{"selector": "#submit", "tag": "button", "role": "button", "visible": True, "bbox_xyxy": [0, 0, 50, 30]}]
    detections = [{"class_name": "icon", "confidence": 0.99, "bbox_xyxy": [0, 0, 50, 30]}]
    result = match_dom_to_detections(dom, detections)
    assert result["match_count"] == 0
    assert result["ambiguous_candidate_count"] == 1
    assert result["ambiguous_candidates"][0]["reason"] == "strong_geometry_but_incompatible_class"


@pytest.mark.parametrize("control_class", ["checkbox", "radio"])
def test_resolves_strong_icon_overlap_with_native_control_semantics(control_class):
    dom = [{
        "selector": f"#{control_class}", "tag": "input", "role": control_class,
        "input_type": control_class, "visible": True, "bbox_xyxy": [10, 10, 24, 34],
    }]
    detections = [{"class_name": "icon", "confidence": 0.8, "bbox_xyxy": [10, 10, 24, 34]}]

    result = match_dom_to_detections(dom, detections)

    assert result["match_count"] == 1
    assert result["ambiguous_candidate_count"] == 0
    match = result["matches"][0]
    assert match["vision_class"] == "icon"
    assert match["raw_vision_class"] == "icon"
    assert match["resolved_class"] == control_class
    assert match["semantic_resolution_applied"] is True
    assert match["evidence"]["resolution_source"] == "dom_native_control_semantics"


def test_does_not_resolve_icon_for_non_binary_control():
    dom = [{
        "selector": "#submit", "tag": "button", "role": "button", "input_type": "button",
        "visible": True, "bbox_xyxy": [0, 0, 50, 30],
    }]
    detections = [{"class_name": "icon", "confidence": 0.99, "bbox_xyxy": [0, 0, 50, 30]}]

    result = match_dom_to_detections(dom, detections)

    assert result["match_count"] == 0
    assert result["ambiguous_candidate_count"] == 1


def test_does_not_resolve_weak_icon_overlap_with_native_control():
    dom = [{
        "selector": "#agree", "tag": "input", "role": "checkbox", "input_type": "checkbox",
        "visible": True, "bbox_xyxy": [0, 0, 20, 20],
    }]
    detections = [{"class_name": "icon", "confidence": 0.99, "bbox_xyxy": [16, 16, 36, 36]}]

    result = match_dom_to_detections(dom, detections)

    assert result["match_count"] == 0
    assert result["ambiguous_candidate_count"] == 0
