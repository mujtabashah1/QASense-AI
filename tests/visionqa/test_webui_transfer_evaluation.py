import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_webui_eval_slice import canonical_hash
from scripts.evaluate_webui_transfer import match_screen
from src.visionqa.webui_evaluation import mapping_for_tag, prepare_boxes


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def test_fixed_manifest_integrity_and_hash():
    manifest = load("reports/visionqa/webui_eval_manifest.json")
    assert manifest["selection"]["screens"] == 399
    assert manifest["selection"]["source_sample_groups"] == 133
    content = dict(manifest); expected = content.pop("manifest_content_sha256")
    assert canonical_hash(content) == expected
    assert expected == "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"


def test_source_grouping_and_duplicate_exclusion():
    manifest = load("reports/visionqa/webui_eval_manifest.json")
    exclusions = load("reports/visionqa/webui_cross_split_duplicate_exclusions.json")
    groups = {}
    for screen in manifest["screens"]:
        groups.setdefault(screen["source_sample_id"], set()).add(screen["viewport"])
    assert len(groups) == 133
    assert all(value == {"desktop", "tablet", "mobile"} for value in groups.values())
    assert not set(groups) & set(exclusions["excluded_source_sample_ids"])


def test_mapped_boxes_are_valid_and_mapping_policy_is_explicit():
    manifest = load("reports/visionqa/webui_eval_manifest.json")
    for screen in manifest["screens"]:
        width, height = screen["image_identifier"]["width"], screen["image_identifier"]["height"]
        for box in screen["valid_ground_truth_boxes"]:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            assert box["mapping_type"] in {"exact", "compatible"}
    assert mapping_for_tag("button")["include_in_quantitative_scoring"]
    assert not mapping_for_tag("svg")["include_in_quantitative_scoring"]
    assert mapping_for_tag("div")["mapping_type"] == "ambiguous"


def test_invalid_and_ambiguous_boxes_are_excluded():
    boxes = {name: [] for name in ("tag", "x", "y", "width", "height", "text", "id", "class", "parent_index")}
    rows = [
        ("button", 1, 1, 20, 10, "", "", "", -1),
        ("button", 1, 1, 0, 10, "", "", "", -1),
        ("input", 2, 2, 20, 10, "", "newsletter-checkbox", "", -1),
    ]
    for row in rows:
        for name, value in zip(boxes, row): boxes[name].append(value)
    prepared = prepare_boxes(boxes, 100, 100)
    assert len(prepared["valid_ground_truth_boxes"]) == 1
    assert Counter(item["reason"] for item in prepared["excluded_invalid_or_ambiguous_boxes"]) == {"nonpositive_box": 1, "input_subtype_ambiguous": 1}


def test_evaluation_pipeline_matches_by_class_and_iou():
    screen = {"valid_ground_truth_boxes": [{"target_class": "button", "bbox_xyxy": [0, 0, 50, 30]}]}
    prediction = {"detections": [{"class_name": "button", "confidence": 0.9, "bbox_xyxy": [1, 1, 49, 29]}]}
    result = match_screen(screen, prediction)
    assert len(result["matches"]) == 1 and not result["false_negative_indices"] and not result["false_positive_indices"]


def test_responsive_groups_review_and_reports_are_complete():
    transfer = load("reports/visionqa/webui_transfer_metrics.json")
    reviews = load("reports/visionqa/webui_review_decisions.json")
    assert transfer["responsive_consistency"]["complete_triplets"] == 133
    assert transfer["human_review"]["completed_screens"] == 100
    assert reviews["review_status"] == "complete" and reviews["completed_review_count"] == 100
    assert (ROOT / "reports/visionqa/webui_review.html").stat().st_size > 1000
    for report in ("reports/integration/REAL_APPLICATION_STATE_VALIDATION.md", "reports/visionqa/WEBUI_TRANSFER_EVALUATION.md", "reports/visionqa/NEXT_MODEL_READINESS_DECISION.md"):
        assert (ROOT / report).stat().st_size > 500


def test_real_state_repeatability_and_protected_artifacts():
    real = load("reports/integration/real_state_evaluation.json")
    assert real["state_count"] == 6 and real["run_count"] == 12
    assert all(row["stable"] for row in real["repeatability"].values())
    protected = load("reports/state/protected_artifact_manifest.json")["artifacts"]
    expected = {
        "risknet_model": "b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c",
        "codelocator_v2_model": "9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288",
        "visionqa_model": "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57",
    }
    for key, digest in expected.items():
        path = ROOT / protected[key]["path"]
        assert file_sha256(path) == digest
