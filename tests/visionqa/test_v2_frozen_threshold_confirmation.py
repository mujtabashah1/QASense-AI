import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "visionqa" / "v2"


def load(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_preflight_fixed_hashes_threshold_and_screen_count():
    preflight = load("frozen_threshold_preflight.json")
    assert preflight["status"] == "passed"
    assert preflight["manifest_content_sha256"] == "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"
    assert preflight["candidate_sha256"] == "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d"
    assert preflight["threshold"] == 0.0625
    assert preflight["screens"] == 399 and preflight["groups"] == 133


def test_single_complete_prediction_pass_has_fixed_contract():
    manifest = json.loads((ROOT / "runs" / "visionqa" / "visionqa_v2_mixed14" / "frozen_inference_manifest_conf00625.json").read_text(encoding="utf-8"))
    outputs = [Path(item["output"]) for item in manifest["items"]]
    assert len(outputs) == len(set(outputs)) == 399
    assert all(path.is_file() for path in outputs)
    for path in outputs:
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["model"]["confidence_threshold"] == 0.0625
        assert row["model"]["class_count"] == 14
        assert row["detection_count"] == len(row["detections"])


def test_frozen_metrics_are_complete_and_consistent():
    result = load("frozen_threshold_confirmation.json")
    metrics = result["metrics"]
    assert result["evaluation_passes"] == 1
    assert result["screen_count"] == 399 and result["sample_group_count"] == 133
    assert result["confidence_threshold"] == 0.0625
    assert result["threshold_tuned_on_slice"] is False
    assert result["prediction_count"] == metrics["tp"] + metrics["fp"]
    assert metrics["tp"] + metrics["fn"] == 5808
    assert set(metrics["per_class"]) == {"button", "link", "input", "image", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table"}
    assert set(result["viewport_metrics"]) == {"desktop", "tablet", "mobile"}
    assert set(result["object_size_metrics"]["buckets"]) == {"tiny", "small", "medium", "large"}
    assert metrics["per_class"]["select"]["tp"] == 0


def test_tradeoff_and_gate_decision_follow_predeclared_rule():
    result = load("frozen_threshold_confirmation.json")
    gate = load("final_webui_promotion_gate.json")
    original = load("webui_evaluation.json")["metrics"]
    current = result["metrics"]
    assert current["recall"] > original["recall"]
    assert current["f1"] > original["f1"]
    assert current["macro_f1"] > original["macro_f1"]
    assert current["precision"] < original["precision"]
    assert gate["decision_code"] == "C"
    assert gate["promoted"] is False
    assert gate["unresolved_classes"] == ["select"]
    assert gate["training_performed"] is False
    assert gate["playwright_regression_performed"] is False


def test_protected_artifacts_and_reports():
    protected = load("protected_artifacts_post_frozen_confirmation.json")
    assert protected["inference_passes"] == 1
    assert protected["thresholds_tested"] == [0.0625]
    assert protected["training_performed"] is False
    assert protected["all_unchanged"] is True
    assert all(row["unchanged"] for row in protected["artifacts"].values())
    for name in ("V2_FROZEN_THRESHOLD_CONFIRMATION.md", "V2_FINAL_WEBUI_PROMOTION_GATE.md", "frozen_threshold_confirmation.json", "final_webui_promotion_gate.json"):
        path = REPORT / name
        assert path.is_file() and path.stat().st_size > 0
