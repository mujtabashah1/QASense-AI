import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "visionqa" / "v2"


def load(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_five_group_preserving_folds_cover_validation_once():
    folds = load("fold_thresholds.json")
    manifest = json.loads((ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assignments = {row["group_id"]: row["fold"] for row in folds["group_assignments"]}
    validation = [row for row in manifest["screens"] if row["split"] == "val"]
    assert folds["fold_count"] == 5
    assert folds["group_count"] == 1738
    assert folds["screen_count"] == 5214
    assert sum(row["held_out"]["images"] for row in folds["folds"]) == 5214
    observed = defaultdict(set)
    for row in validation:
        observed[row["group_id"]].add(assignments[row["group_id"]])
    assert all(len(values) == 1 for values in observed.values())
    assert all({row["viewport"] for row in validation if row["group_id"] == group} == {"desktop", "tablet", "mobile"} for group in observed)


def test_threshold_selection_is_train_fold_based_and_stable():
    folds = load("fold_thresholds.json")
    metrics = load("cross_validated_metrics.json")
    selected = [row["selected_threshold"] for row in folds["folds"]]
    assert selected == [0.0675, 0.06, 0.0625, 0.07, 0.0625]
    assert all(0.005 <= value <= 0.25 for value in selected)
    assert metrics["selection_metric"] == "macro-F1"
    assert metrics["threshold_stability"]["stable"] is True
    assert metrics["threshold_stability"]["std"] <= 0.01
    assert metrics["threshold_stability"]["range"] <= 0.02
    assert metrics["recommended_global_threshold"] == 0.0625


def test_cross_validated_metrics_improve_f1_and_recall_with_measured_precision_cost():
    metrics = load("cross_validated_metrics.json")
    baseline = metrics["baseline_at_0_25"]
    calibrated = metrics["out_of_fold_aggregate"]
    comparison = metrics["comparison_to_0_25"]
    assert calibrated["recall"] > baseline["recall"]
    assert calibrated["f1"] > baseline["f1"]
    assert calibrated["macro_f1"] > baseline["macro_f1"]
    assert calibrated["precision"] < baseline["precision"]
    assert comparison["zero_prediction_screen_reduction"] > 0
    assert calibrated["prediction_count"] == sum(row["predictions"] for row in calibrated["per_class"].values())


def test_class_viewport_and_size_outputs_are_complete():
    metrics = load("cross_validated_metrics.json")
    classes = metrics["out_of_fold_aggregate"]["per_class"]
    assert set(classes) == {"button", "link", "input", "image", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table"}
    assert classes["select"]["tp"] == 0
    assert set(metrics["viewport_out_of_fold"]) == {"desktop", "tablet", "mobile"}
    assert set(metrics["object_size_out_of_fold"]["buckets"]) == {"tiny", "small", "medium", "large"}


def test_scope_decision_reports_and_protected_artifacts():
    decision = load("threshold_decision.json")
    protected = load("protected_artifacts_post_calibration.json")
    assert decision["decision_code"] == "B"
    assert decision["frozen_evaluation_used"] is False
    assert decision["gpu_inference_run"] is False
    assert decision["model_training_run"] is False
    assert protected["frozen_evaluation_rerun"] is False
    assert all(row["unchanged"] for row in protected["artifacts"].values())
    for name in ("V2_GLOBAL_THRESHOLD_CV.md", "V2_THRESHOLD_DECISION.md", "global_threshold_cv.json", "fold_thresholds.json", "cross_validated_metrics.json", "threshold_decision.json"):
        path = REPORT / name
        assert path.is_file() and path.stat().st_size > 0
