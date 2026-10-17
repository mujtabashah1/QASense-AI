import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "visionqa" / "v2"


def read_json(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_analysis_inputs_are_fixed_and_protected():
    evidence = read_json("analysis_input_verification.json")
    assert evidence["validation_images"] == 5214
    assert evidence["class_count"] == 14
    assert evidence["v1"]["unchanged"] is True
    assert evidence["v2"]["unchanged"] is True
    assert evidence["dataset_manifest"]["unchanged"] is True
    assert evidence["split_manifest"]["unchanged"] is True
    assert evidence["frozen_evaluation"]["rerun"] is False
    assert evidence["frozen_evaluation"]["labels_inspected"] is False
    assert evidence["frozen_evaluation"]["unchanged"] is True


def test_raw_predictions_use_identical_settings_and_all_validation_images():
    metadata = []
    for model in ("v1", "v2"):
        path = REPORT / "raw_predictions" / f"webui_val_{model}.jsonl.gz"
        meta = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text(encoding="utf-8"))
        assert meta["image_count"] == 5214
        assert meta["settings"] == {"imgsz": 640, "conf_floor": 0.001, "iou_nms": 0.7, "max_det": 1500, "batch": 4, "device": "0"}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["output_sha256"]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert sum(1 for _ in handle) == 5214
        metadata.append(meta)
    assert metadata[0]["validation_manifest_sha256"] == metadata[1]["validation_manifest_sha256"]


def test_threshold_analysis_is_complete_and_internally_consistent():
    payload = read_json("validation_threshold_analysis.json")
    assert payload["image_count"] == 5214
    assert payload["analysis_thresholds"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    expected_classes = {"button", "link", "input", "image", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table"}
    for model in ("visionqa_v1", "visionqa_v2"):
        rows = payload["models"][model]["threshold_metrics"]
        assert set(rows) == {f"{value:.2f}" for value in payload["analysis_thresholds"]}
        assert set(rows["0.25"]["per_class"]) == expected_classes
        for row in rows.values():
            assert row["tp"] + row["fn"] == 52104
            assert row["prediction_count"] == row["tp"] + row["fp"]
            assert 0 <= row["precision"] <= 1
            assert 0 <= row["recall"] <= 1


def test_analysis_conclusion_is_supported_and_recommendation_is_single():
    thresholds = read_json("validation_threshold_analysis.json")
    v2 = thresholds["models"]["visionqa_v2"]
    assert v2["precision_recall"]["overall"]["best_f1"]["threshold"] < 0.10
    assert v2["threshold_metrics"]["0.25"]["images_zero_predictions"] > 3000
    recommendation = read_json("next_experiment_recommendation.json")
    assert recommendation["model_training"] is False
    assert recommendation["frozen_399_screen_evaluation_used"] is False
    assert "source-grouped" in recommendation["decision"].lower()


def test_required_reports_and_machine_readable_outputs_exist():
    required = [
        "V2_FAILURE_ANALYSIS.md", "V2_CONFIDENCE_ANALYSIS.md", "V2_NEXT_EXPERIMENT_RECOMMENDATION.md",
        "validation_threshold_analysis.json", "confidence_distributions.json", "class_error_analysis.json",
        "object_size_analysis.json", "source_viewport_analysis.json", "calibration_analysis.json",
        "next_experiment_recommendation.json",
    ]
    for name in required:
        path = REPORT / name
        assert path.is_file() and path.stat().st_size > 0
