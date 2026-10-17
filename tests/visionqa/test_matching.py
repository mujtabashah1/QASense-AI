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
