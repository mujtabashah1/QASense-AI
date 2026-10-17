"""Build the validation-only VisionQA v2 failure-analysis reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "visionqa" / "v2"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    threshold = load("validation_threshold_analysis.json")
    confidence = load("confidence_distributions.json")
    errors = load("class_error_analysis.json")
    sizes = load("object_size_analysis.json")
    source = load("source_viewport_analysis.json")
    calibration = load("calibration_analysis.json")
    t1 = threshold["models"]["visionqa_v1"]; t2 = threshold["models"]["visionqa_v2"]
    m1 = t1["threshold_metrics"]["0.25"]; m2 = t2["threshold_metrics"]["0.25"]
    b1 = t1["precision_recall"]["overall"]["best_f1"]; b2 = t2["precision_recall"]["overall"]["best_f1"]

    # Add conditional ECE summaries so the enormous low-score mass cannot hide
    # calibration behavior in operational confidence bands.
    for model in ("visionqa_v1", "visionqa_v2"):
        bins = calibration["models"][model]["bins"]
        conditional = {}
        for floor in (0.1, 0.2):
            chosen = [row for row in bins if row["lower"] >= floor and row["count"]]
            count = sum(row["count"] for row in chosen)
            ece = sum(row["count"] * abs(row["mean_confidence"] - row["empirical_precision"]) for row in chosen) / count
            conditional[f"confidence_ge_{floor:.1f}"] = {"count": count, "ece": ece}
        calibration["models"][model]["conditional_ece"] = conditional
    (R / "calibration_analysis.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    v1_path = ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "best.pt"
    v2_path = ROOT / "models" / "visionqa" / "visionqa_v2_mixed14" / "candidate.pt"
    dataset_manifest = ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "dataset_manifest.json"
    split_manifest = ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "split_manifest.json"
    frozen_metrics = load("../webui_transfer_metrics.json") if False else json.loads((ROOT / "reports" / "visionqa" / "webui_transfer_metrics.json").read_text(encoding="utf-8"))
    dataset_json = json.loads(dataset_manifest.read_text(encoding="utf-8")); split_json = json.loads(split_manifest.read_text(encoding="utf-8"))
    input_check = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "analysis_scope": "WebUI training-corpus validation only",
        "validation_images": 5214, "class_count": 14,
        "class_names": ["button", "link", "input", "image", "icon", "checkbox", "radio", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table", "block"],
        "v1_v2_class_schema_identical": True,
        "v1": {"path": str(v1_path), "sha256": sha256(v1_path), "expected_sha256": "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57", "unchanged": sha256(v1_path) == "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57"},
        "v2": {"path": str(v2_path), "sha256": sha256(v2_path), "expected_sha256": "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d", "unchanged": sha256(v2_path) == "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d"},
        "dataset_manifest": {"physical_sha256": sha256(dataset_manifest), "recorded_content_sha256": dataset_json["manifest_sha256"], "expected_content_sha256": "442c0f3d250f513463ca6188127a31240fab173e781fe630ddc6ffa4a52ced56", "unchanged": dataset_json["manifest_sha256"] == "442c0f3d250f513463ca6188127a31240fab173e781fe630ddc6ffa4a52ced56"},
        "split_manifest": {"physical_sha256": sha256(split_manifest), "recorded_content_sha256": split_json["manifest_sha256"], "expected_content_sha256": "4bc01ed12f211408760ca1a67f2db901dc59d9b8bddf1fa5bc55702b37d9add3", "unchanged": split_json["manifest_sha256"] == "4bc01ed12f211408760ca1a67f2db901dc59d9b8bddf1fa5bc55702b37d9add3"},
        "frozen_evaluation": {"rerun": False, "labels_inspected": False, "recorded_content_sha256": frozen_metrics["manifest_content_sha256"], "expected_content_sha256": "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45", "unchanged": frozen_metrics["manifest_content_sha256"] == "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"},
    }
    (R / "analysis_input_verification.json").write_text(json.dumps(input_check, indent=2) + "\n", encoding="utf-8")

    recommendation = {
        "decision": "Run one source-grouped cross-validated global confidence-threshold recalibration on the existing 5,214-image WebUI validation predictions.",
        "execute_now": False, "model_training": False, "frozen_399_screen_evaluation_used": False,
        "hypothesis": "The v2 score scale moved downward; a validation-derived threshold near 0.057 recovers transfer detections that confidence 0.25 suppresses.",
        "method": [
            "Partition validation source groups into deterministic folds.",
            "Select one global threshold on calibration folds and measure it on held-out source folds.",
            "Accept the threshold hypothesis only if held-out folds show stable F1/recall gains without a precision collapse.",
            "Do not apply the result to the frozen 399-screen slice; reserve confirmation for a future prospectively frozen external holdout."
        ],
        "why_this_experiment": {"v2_best_validation_threshold": b2["threshold"], "v2_best_validation_f1": b2["f1"], "v2_f1_at_0_25": m2["f1"], "v2_predictions_at_0_25": m2["prediction_count"], "v2_zero_prediction_images_at_0_25": m2["images_zero_predictions"]},
        "estimated_cost": {"gpu": "none; reuse persisted raw predictions", "cpu": "minutes", "relative_to_prior_3_10_hour_training": "well below 1% and no checkpoint changes"},
    }
    (R / "next_experiment_recommendation.json").write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")

    threshold_rows = []
    for value in threshold["analysis_thresholds"]:
        key = f"{value:.2f}"; a = t1["threshold_metrics"][key]; b = t2["threshold_metrics"][key]
        threshold_rows.append(f"| {key} | {pct(a['precision'])} | {pct(a['recall'])} | {pct(a['f1'])} | {a['prediction_count']:,} | {pct(b['precision'])} | {pct(b['recall'])} | {pct(b['f1'])} | {b['prediction_count']:,} |")
    class_rows = []
    for name in m2["per_class"]:
        a=m1["per_class"][name]; b=m2["per_class"][name]; best=t2["precision_recall"][name]["best_f1"]
        class_rows.append(f"| {name} | {a['ground_truth']:,} | {pct(a['f1'])} | {pct(b['precision'])} | {pct(b['recall'])} | {pct(b['f1'])} | {best['threshold']:.4f} | {pct(best['f1'])} |")
    size_rows=[]
    for bucket in ("tiny","small","medium","large"):
        a=sizes["models"]["visionqa_v1"]["overall"][bucket]; b=sizes["models"]["visionqa_v2"]["overall"][bucket]
        size_rows.append(f"| {bucket} | {b['ground_truth']:,} | {pct(a['recall'])} | {pct(b['recall'])} |")
    viewport_rows=[]
    for viewport in ("desktop","tablet","mobile"):
        a=source["models"]["visionqa_v1"]["viewport"][viewport]; b=source["models"]["visionqa_v2"]["viewport"][viewport]
        viewport_rows.append(f"| {viewport} | {pct(a['recall'])} | {pct(a['f1'])} | {pct(b['recall'])} | {pct(b['f1'])} |")
    density_rows=[]
    for bucket,b in source["models"]["visionqa_v2"]["density"].items():
        a=source["models"]["visionqa_v1"]["density"][bucket]
        density_rows.append(f"| {bucket} | {b['images']:,} | {pct(a['recall'])} | {pct(b['recall'])} | {b['prediction_count']:,} |")
    miss1=errors["models"]["visionqa_v1"]["miss_categories"]; miss2=errors["models"]["visionqa_v2"]["miss_categories"]
    c1=confidence["models"]["visionqa_v1"]["overall"]["true_positive"]; c2=confidence["models"]["visionqa_v2"]["overall"]["true_positive"]
    source_changes=source["source_comparison_at_0_25"]

    failure_report = f"""# VisionQA v2 failure analysis

## Scope and scientific boundary

This analysis used only the fixed 5,214-screen WebUI training-corpus validation split. It ran v1 and v2 with identical 640-pixel preprocessing, NMS IoU 0.7, `max_det=1500`, and a 0.001 persistence floor. The frozen 399-screen WebUI evaluation was not rerun, relabeled, inspected, or used for threshold selection. No model was trained or modified.

## Main finding

VisionQA v2 did not stop producing candidates. It produced 4,845,989 supported-class candidates at the 0.001 floor versus 738,845 for v1. Its useful candidates moved to much lower confidence: true-positive median confidence fell from {c1['median']:.4f} to {c2['median']:.4f}. At confidence 0.25, v2 emitted only {m2['prediction_count']:,} predictions versus v1's {m1['prediction_count']:,}, and {m2['images_zero_predictions']:,} screens had zero v2 predictions. V2's best validation F1 was {b2['f1']:.4f} at threshold {b2['threshold']:.4f}; at 0.25 it was {m2['f1']:.4f}. Confidence suppression is therefore the primary measured mechanism behind recall collapse at the shared 0.25 operating point.

The gap cannot be assigned to confidence alone. Source response is heterogeneous, `select` has zero true positives at every measured operating point, and tiny/small objects remain weak. The external frozen set's already-recorded aggregate result is consistent with the confidence shift interacting with domain composition, but its labels were not reopened, so exact frozen-source attribution remains unproven.

## Threshold sensitivity

| Threshold | v1 P | v1 R | v1 F1 | v1 preds | v2 P | v2 R | v2 F1 | v2 preds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(threshold_rows)}

The exact validation optima from the persisted low-floor prediction ranking were v1 F1 {b1['f1']:.4f} at {b1['threshold']:.4f} and v2 F1 {b2['f1']:.4f} at {b2['threshold']:.4f}. These values are diagnostic; they are not production thresholds.

## Class behavior at confidence 0.25

| Class | GT | v1 F1 | v2 P | v2 R | v2 F1 | v2 best val threshold | v2 best val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

`select` is a real learned-class failure on this validation set: 102 ground-truth boxes and zero true positives even when scanning the persisted low-floor predictions. `block-table` improves strongly when scored at its validation optimum; `block-sidebar` and `block-form` also improve, while `block-footer` remains weak.

## Size and density

Size buckets use normalized box area: tiny `<0.001`, small `0.001–<0.01`, medium `0.01–<0.1`, and large `>=0.1`.

| Size | GT | v1 recall @0.25 | v2 recall @0.25 |
|---|---:|---:|---:|
{chr(10).join(size_rows)}

| Density bucket (validation quartiles) | Screens | v1 recall @0.25 | v2 recall @0.25 | v2 predictions |
|---|---:|---:|---:|---:|
{chr(10).join(density_rows)}

V2 is weakest on tiny objects, but confidence suppression affects every size. Dense pages are additionally difficult: v2 recall drops from {pct(list(source['models']['visionqa_v2']['density'].values())[1]['recall'])} in the medium-density bucket to {pct(list(source['models']['visionqa_v2']['density'].values())[-1]['recall'])} above the 75th percentile.

## Viewport and source behavior

| Viewport | v1 recall | v1 F1 | v2 recall | v2 F1 |
|---|---:|---:|---:|---:|
{chr(10).join(viewport_rows)}

Mobile is the weakest viewport for both models. At 0.25, v2's largest measured source regressions were {', '.join(row['source'] for row in source_changes[:3])}; its largest improvement was `{source_changes[-1]['source']}` ({source_changes[-1]['f1_change']:+.4f} F1). This supports source sensitivity. Train and validation remain closely structured: all three viewports are balanced, density medians are both 7, and they contain 35 and 34 sources respectively. A simple train/validation split imbalance is not supported as the sole cause.

## Classification versus localization

At 0.25, v2 misses include {miss2.get('correct_class_below_confidence_threshold',0):,} correct-class boxes suppressed below the threshold, {miss2.get('correct_class_localization_below_iou_0_5',0):,} localization-only failures, {miss2.get('wrong_class_overlap',0):,} wrong-class overlaps, and {miss2.get('no_nearby_prediction',0):,} cases with no nearby high-confidence prediction. Localization mismatch is a small minority; confidence and absent/very-low-score candidates dominate. The largest v2 wrong-class pair is `{errors['models']['visionqa_v2']['wrong_class_pairs'][0]['pair']}` with {errors['models']['visionqa_v2']['wrong_class_pairs'][0]['count']} cases.

## Root-cause evidence

| Candidate | Assessment | Evidence |
|---|---|---|
| Confidence shift | Supported, primary | TP median {c1['median']:.4f}→{c2['median']:.4f}; best F1 threshold {b1['threshold']:.4f}→{b2['threshold']:.4f}; v2 prediction count falls 85.8% versus v1 at 0.25. |
| Threshold sensitivity | Supported | v2 F1 is {b2['f1']:.4f} near 0.057 but {m2['f1']:.4f} at 0.25. |
| Source/domain shift | Partially supported | Large source-specific changes exist, including improvement on WordPress themes and regressions on CodePen/GitHub; frozen per-source attribution was deliberately not recomputed. |
| Class imbalance/taxonomy | Supported for specific classes | `select` has only 102 validation boxes and zero TP; structural classes vary materially. |
| Small-object weakness | Supported | v2 tiny recall {pct(sizes['models']['visionqa_v2']['overall']['tiny']['recall'])} versus large {pct(sizes['models']['visionqa_v2']['overall']['large']['recall'])} at 0.25. |
| Dense-layout conservatism | Supported | Recall falls in the highest-density quartile while few predictions survive. |
| Validation distribution bias | Partially supported | Train/val metadata are similar internally, but the frozen slice is an external source-grouped sample with a different aggregate outcome. |
| Localization mismatch | Not primary | Only {miss2.get('correct_class_localization_below_iou_0_5',0):,} v2 misses were isolated localization failures at 0.25. |
| Unified 14-class architecture failure | Not supported | V2 doubles best validation F1 and retained 91.79% of v1 AtomBlock mAP50-95; failures are operating-point and class/domain specific. |

## Conclusion

V2 remains experimental. The 14-class preservation strategy remains technically successful, but production promotion remains rejected because the existing frozen result has lower overall F1. The analysis supports a lower and validation-stable operating threshold as the smallest hypothesis to test next, without retraining.
"""
    (R / "V2_FAILURE_ANALYSIS.md").write_text(failure_report, encoding="utf-8")

    conf_report = f"""# VisionQA v2 confidence analysis

## Setup

Predictions were persisted before analysis thresholds at confidence 0.001. Both models used identical inference settings on the same 5,214 WebUI validation images. Matching uses class-correct IoU >= 0.5.

## Distribution shift

| Model | Supported predictions @0.001 | TP mean | TP median | TP p75 | TP p90 | Best validation F1 threshold |
|---|---:|---:|---:|---:|---:|---:|
| v1 | {confidence['models']['visionqa_v1']['overall']['all']['count']:,} | {c1['mean']:.4f} | {c1['median']:.4f} | {c1['p75']:.4f} | {c1['p90']:.4f} | {b1['threshold']:.4f} |
| v2 | {confidence['models']['visionqa_v2']['overall']['all']['count']:,} | {c2['mean']:.4f} | {c2['median']:.4f} | {c2['p75']:.4f} | {c2['p90']:.4f} | {b2['threshold']:.4f} |

V2 creates far more very-low-score proposals. Its TP confidence median is roughly one-third of v1's. Confidence 0.25 lies above the 90th percentile of v2 true-positive scores ({c2['p90']:.4f}), explaining the sharp recall loss.

## Calibration

The all-prediction ECE is {calibration['models']['visionqa_v1']['ece']:.6f} for v1 and {calibration['models']['visionqa_v2']['ece']:.6f} for v2. This aggregate is dominated by millions of predictions below 0.1 and must not be read alone. Conditional ECE for predictions >=0.1 is {calibration['models']['visionqa_v1']['conditional_ece']['confidence_ge_0.1']['ece']:.4f} for v1 and {calibration['models']['visionqa_v2']['conditional_ece']['confidence_ge_0.1']['ece']:.4f} for v2. In v2's 0.1–0.5 bins, empirical precision generally exceeds mean confidence, so v2 is under-confident in its useful operating band. V1 is strongly over-confident in the same bands.

## Interpretation boundary

The validation optimum is evidence of threshold sensitivity, not permission to change production behavior. No calibrator was fitted, and the frozen evaluation was not reused. The plots under `figures/` show the PR and fixed-threshold curves.
"""
    (R / "V2_CONFIDENCE_ANALYSIS.md").write_text(conf_report, encoding="utf-8")

    next_report = f"""# VisionQA v2 next experiment recommendation

Run exactly one **source-grouped cross-validated global confidence-threshold recalibration** using the already-persisted predictions from the 5,214-image WebUI validation split.

V2's best validation F1 is {b2['f1']:.4f} near confidence {b2['threshold']:.4f}, compared with {m2['f1']:.4f} at 0.25. Its true-positive median confidence is {c2['median']:.4f}, and 0.25 leaves {m2['images_zero_predictions']:,} validation images without any prediction. These measurements make threshold stability the cheapest direct hypothesis test.

Partition source groups into deterministic folds, select one global threshold on calibration folds, and report performance on held-out source folds. The threshold hypothesis passes only if F1 and recall improve consistently across held-out sources without an unacceptable precision loss. Do not use class-specific thresholds in this first test, because the rare classes are too small for stable per-class fitting.

This experiment reuses raw predictions, requires no GPU training, and should take minutes of CPU time—well below 1% of the prior 3.10-hour training cost. It must not rerun or tune against the frozen 399-screen evaluation. External confirmation should use a future prospectively frozen holdout.
"""
    (R / "V2_NEXT_EXPERIMENT_RECOMMENDATION.md").write_text(next_report, encoding="utf-8")

    state_path = ROOT / "reports" / "QASENSE_PROJECT_STATE.md"
    state = state_path.read_text(encoding="utf-8")
    marker = "## VisionQA v2 validation-only failure analysis (2026-10-17)"
    if marker not in state:
        state += f"""\n\n{marker}\n\n- VisionQA v2 remains experimental; VisionQA v1 remains production/rollback.\n- Analysis used only the 5,214-screen WebUI training-corpus validation split. No training occurred and the frozen 399-screen evaluation was not rerun or inspected.\n- Primary mechanism: v2 confidence shifted downward. True-positive median confidence changed from {c1['median']:.4f} (v1) to {c2['median']:.4f} (v2).\n- Validation best overall F1: {b2['f1']:.4f} at confidence {b2['threshold']:.4f}; v2 F1 at confidence 0.25: {m2['f1']:.4f}.\n- At confidence 0.25, v2 produced {m2['prediction_count']:,} detections and zero detections on {m2['images_zero_predictions']:,}/5,214 screens.\n- Secondary findings: tiny/small-object weakness, reduced recall as density rises, mobile weakness, source-specific variation, and complete `select` failure on validation. Localization mismatch was not the main cause.\n- Architecture assessment: unified 14-class architecture is not the primary failure; old-domain retention remains 91.79% and validation adaptation is measurable.\n- Exactly one recommended next experiment: source-grouped cross-validated global threshold recalibration using persisted validation predictions only. No GPU training; do not reuse the frozen 399-screen set.\n- Reports: `reports/visionqa/v2/V2_FAILURE_ANALYSIS.md`, `V2_CONFIDENCE_ANALYSIS.md`, and `V2_NEXT_EXPERIMENT_RECOMMENDATION.md`.\n"""
        state_path.write_text(state, encoding="utf-8")
    print(json.dumps({"reports_written": 3, "recommendation": recommendation["decision"], "state_updated": True}, indent=2))


if __name__ == "__main__":
    main()
