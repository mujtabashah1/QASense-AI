from src.visual_defects.baseline import classify_pair


def element(selector, box, **geometry):
    return {"selector": selector, "visible": True, "bbox_xyxy": box, "geometry": geometry}


def test_missing_element_is_detected():
    before = {"elements": [element("#x", [0, 0, 40, 30])], "document_size": {"width": 100, "height": 100}}
    after = {"elements": [], "document_size": {"width": 100, "height": 100}}
    result = classify_pair(before, after, "#x", None, {"width": 100, "height": 100}, {"changed_pixel_ratio_gt_10": 0.2})
    assert result["prediction"] == "missing_element"


def test_unchanged_pair_is_negative_control():
    document = {"elements": [element("#x", [0, 0, 40, 30], overflow_x="visible", z_index="auto")], "document_size": {"width": 100, "height": 100}}
    pixel = {"changed_pixel_ratio_gt_10": 0.0}
    result = classify_pair(document, document, "#x", None, {"width": 100, "height": 100}, pixel)
    assert result["prediction"] == "no_defect"
