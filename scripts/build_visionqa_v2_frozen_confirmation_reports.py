"""Build final frozen-threshold confirmation and WebUI promotion-gate reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "visionqa" / "v2"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    confirmation_path = REPORT / "frozen_threshold_confirmation.json"
    confirmation = load(confirmation_path)
    original_v2 = load(REPORT / "webui_evaluation.json")
    v1_record = load(ROOT / "reports" / "visionqa" / "webui_transfer_metrics.json")
    atom = load(REPORT / "atomblock_regression.json")
    current = confirmation["metrics"]; old = original_v2["metrics"]; v1 = v1_record["all_mapped_screens"]
    if confirmation["confidence_threshold"] != 0.0625 or confirmation["screen_count"] != 399:
        raise RuntimeError("Frozen confirmation does not match the pre-registered evaluation")
    if confirmation["manifest_content_sha256"] != "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45":
        raise RuntimeError("Frozen manifest hash mismatch")
    if confirmation["checkpoint_sha256"] != "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d":
        raise RuntimeError("VisionQA v2 checkpoint hash mismatch")

    tradeoff = {
        "precision_change": current["precision"] - old["precision"],
        "recall_change": current["recall"] - old["recall"],
        "f1_change": current["f1"] - old["f1"],
        "macro_f1_change": current["macro_f1"] - old["macro_f1"],
        "prediction_count_change": confirmation["prediction_count"] - confirmation["baseline_at_0_25_recomputed_from_persisted_predictions"]["prediction_count"],
        "prediction_count_multiplier": confirmation["prediction_count"] / confirmation["baseline_at_0_25_recomputed_from_persisted_predictions"]["prediction_count"],
        "zero_prediction_screen_change": confirmation["zero_prediction_screens"] - confirmation["baseline_at_0_25_recomputed_from_persisted_predictions"]["zero_prediction_screens"],
        "zero_prediction_screen_reduction_fraction": (confirmation["baseline_at_0_25_recomputed_from_persisted_predictions"]["zero_prediction_screens"] - confirmation["zero_prediction_screens"]) / confirmation["baseline_at_0_25_recomputed_from_persisted_predictions"]["zero_prediction_screens"],
    }
    versus_v1 = {key: current[key] - v1[key] for key in ("precision", "recall", "f1", "macro_f1")}
    select = current["per_class"]["select"]
    source_summary = confirmation["source_group_summary"]
    decision = "C. WEBUI IMPROVED BUT REMAINS EXPERIMENTAL"
    reasons = [
        "Overall F1, macro-F1, recall, and precision exceed the VisionQA v1 frozen baseline.",
        "The calibrated threshold materially improves recall and F1 over v2 at 0.25.",
        "Select remains unresolved with zero true positives.",
        "Tiny-object and mobile recall remain weak.",
        "Improvement is source-sensitive: only 47 of 133 groups gain true positives and the top ten groups contribute 67.2% of added true positives.",
        "The scorable prediction count grows 10.68x, increasing false-positive operational load despite net F1 improvement.",
    ]
    next_milestone = "Run one select-focused label-quality and visual-separability audit on existing WebUI train/validation data; do not use the frozen evaluation set or retrain during that audit."
    gate = {
        "schema_version": "1.0", "decided_at": datetime.now(timezone.utc).isoformat(),
        "decision_code": "C", "decision": decision, "promoted": False,
        "production_model": str(ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "best.pt"),
        "experimental_model": str(ROOT / "models" / "visionqa" / "visionqa_v2_mixed14" / "candidate.pt"),
        "experimental_threshold": 0.0625, "reasons": reasons,
        "atomblock_retention_map50_95": 0.9179, "catastrophic_forgetting_gate": "passed previously",
        "tradeoff_vs_v2_at_0_25": tradeoff, "change_vs_v1": versus_v1,
        "unresolved_classes": ["select"], "next_single_milestone": next_milestone,
        "training_performed": False, "playwright_regression_performed": False,
    }
    (REPORT / "final_webui_promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    class_rows = []
    for name in ("button", "link", "input", "image", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table"):
        row = current["per_class"][name]
        class_rows.append(f"| {name} | {row['predictions']:,} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
    viewport_rows = []
    for name in ("desktop", "tablet", "mobile"):
        row = confirmation["viewport_metrics"][name]
        predictions = sum(value["predictions"] for value in row["per_class"].values())
        viewport_rows.append(f"| {name} | {predictions:,} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
    size_rows = []
    for name, row in confirmation["object_size_metrics"]["buckets"].items():
        size_rows.append(f"| {name} | {row['ground_truth']:,} | {row['matched']:,} | {row['recall']:.4f} |")

    confirmation_report = f"""# VisionQA v2 frozen-threshold confirmation

## Fixed protocol

VisionQA v2 `candidate.pt` was evaluated once on all 399 screens in the unchanged 133-group frozen WebUI manifest. The confidence threshold was the preselected source-grouped-CV value **0.0625**. Image size 640, NMS IoU 0.7, `max_det=1500`, class mapping, invalid-box filtering, and IoU 0.5 metric matching were unchanged. No threshold sweep, tuning, calibration fitting, training, or dataset modification occurred.

- Candidate SHA-256: `{confirmation['checkpoint_sha256']}`
- Frozen manifest canonical SHA-256: `{confirmation['manifest_content_sha256']}`
- Evaluation passes: {confirmation['evaluation_passes']}

## Overall comparison

| Model / threshold | Precision | Recall | F1 | Macro-F1 | Predictions | Zero screens |
|---|---:|---:|---:|---:|---:|---:|
| VisionQA v1 / 0.25 | {v1['precision']:.5f} | {v1['recall']:.5f} | {v1['f1']:.5f} | {v1['macro_f1']:.5f} | {v1['tp']+v1['fp']:,} | not persisted in the original aggregate |
| VisionQA v2 / 0.25 | {old['precision']:.5f} | {old['recall']:.5f} | {old['f1']:.5f} | {old['macro_f1']:.5f} | {confirmation['baseline_at_0_25_recomputed_from_persisted_predictions']['prediction_count']:,} | {confirmation['baseline_at_0_25_recomputed_from_persisted_predictions']['zero_prediction_screens']:,} |
| VisionQA v2 / 0.0625 | {current['precision']:.5f} | {current['recall']:.5f} | {current['f1']:.5f} | {current['macro_f1']:.5f} | {confirmation['prediction_count']:,} | {confirmation['zero_prediction_screens']:,} |

Against v2 at 0.25, precision changes **{tradeoff['precision_change']:+.5f}**, recall **{tradeoff['recall_change']:+.5f}**, F1 **{tradeoff['f1_change']:+.5f}**, and macro-F1 **{tradeoff['macro_f1_change']:+.5f}**. Scorable predictions increase by {tradeoff['prediction_count_change']:+,} ({tradeoff['prediction_count_multiplier']:.2f}×). Zero-prediction screens fall by {-tradeoff['zero_prediction_screen_change']:,}, from {confirmation['baseline_at_0_25_recomputed_from_persisted_predictions']['zero_prediction_rate']:.2%} to {confirmation['zero_prediction_rate']:.2%}.

## Per-class results

| Class | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

`select` remains unresolved: {select['predictions']} predictions, {select['tp']} true positives, and {select['fn']} false negatives. Confidence recalibration does not solve select detection.

## Viewport results

| Viewport | Predictions | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(viewport_rows)}

Mobile recall is {confirmation['viewport_metrics']['mobile']['recall']:.4f}. It is higher than the overall v2@0.25 recall, but remains substantially below desktop recall {confirmation['viewport_metrics']['desktop']['recall']:.4f}. Original per-viewport v2@0.25 metrics were not persisted, so no post-hoc threshold comparison is fabricated.

## Object-size results

| Size | Ground truth | Matched | Recall |
|---|---:|---:|---:|
{chr(10).join(size_rows)}

Tiny-object recall remains a major limitation at {confirmation['object_size_metrics']['buckets']['tiny']['recall']:.4f}.

## Source-group distribution

True positives improve in {source_summary['groups_tp_improved']} of {source_summary['groups_total']} groups, remain unchanged in {source_summary['groups_tp_unchanged']}, and regress in {source_summary['groups_tp_regressed']}. The ten most-improved groups contribute {source_summary['top_10_groups_share_of_added_true_positives']:.1%} of added true positives. Improvement is real but concentrated rather than uniformly distributed.

## AtomBlock reference

The existing unchanged v2 AtomBlock result is P 0.43786, R 0.38987, mAP50 0.35590, and mAP50-95 0.25541. Retention remains 91.79% of v1 mAP50-95; the catastrophic-forgetting gate remains passed. No AtomBlock evaluation was rerun.
"""
    (REPORT / "V2_FROZEN_THRESHOLD_CONFIRMATION.md").write_text(confirmation_report, encoding="utf-8")

    gate_report = f"""# VisionQA v2 final WebUI promotion gate

## Decision

**{decision}**

At the preselected threshold 0.0625, v2 materially exceeds v1 on the frozen WebUI aggregate: F1 {v1['f1']:.5f} → {current['f1']:.5f}, macro-F1 {v1['macro_f1']:.5f} → {current['macro_f1']:.5f}, recall {v1['recall']:.5f} → {current['recall']:.5f}, and precision {v1['precision']:.5f} → {current['precision']:.5f}. AtomBlock retention remains within the previously accepted gate.

Promotion is withheld because product behavior is not yet broad enough: `select` has zero true positives, tiny-object recall is {confirmation['object_size_metrics']['buckets']['tiny']['recall']:.4f}, mobile recall is {confirmation['viewport_metrics']['mobile']['recall']:.4f}, the scorable prediction volume increases {tradeoff['prediction_count_multiplier']:.2f}× relative to v2@0.25, and source-group gains are concentrated.

- Production/rollback model: `models/visionqa/atomblock_yolo11n/best.pt`
- Experimental model: `models/visionqa/visionqa_v2_mixed14/candidate.pt`
- Experimental calibrated threshold: `0.0625`
- Model promotion: **not performed**
- Playwright regression: **not performed**

## Next single milestone

{next_milestone}
"""
    (REPORT / "V2_FINAL_WEBUI_PROMOTION_GATE.md").write_text(gate_report, encoding="utf-8")

    state_path = ROOT / "reports" / "QASENSE_PROJECT_STATE.md"
    state = state_path.read_text(encoding="utf-8")
    marker = "## VisionQA v2 frozen threshold confirmation (2026-10-17)"
    if marker not in state:
        state += f"""\n\n{marker}\n\n- One authorized frozen confirmation evaluated 399 screens / 133 groups using VisionQA v2 candidate SHA-256 `{confirmation['checkpoint_sha256']}` at the preselected confidence threshold 0.0625.\n- Frozen manifest canonical SHA-256 remained `{confirmation['manifest_content_sha256']}`. No threshold sweep or tuning occurred.\n- Frozen v2@0.0625 P/R/F1/macro-F1: {current['precision']:.5f}/{current['recall']:.5f}/{current['f1']:.5f}/{current['macro_f1']:.5f}.\n- V2@0.25 → v2@0.0625: F1 {old['f1']:.5f} → {current['f1']:.5f}; recall {old['recall']:.5f} → {current['recall']:.5f}; precision {old['precision']:.5f} → {current['precision']:.5f}.\n- Scorable predictions: {confirmation['baseline_at_0_25_recomputed_from_persisted_predictions']['prediction_count']:,} → {confirmation['prediction_count']:,}; zero-prediction screens: {confirmation['baseline_at_0_25_recomputed_from_persisted_predictions']['zero_prediction_screens']:,} → {confirmation['zero_prediction_screens']:,}.\n- Remaining limitations: select TP=0, tiny recall {confirmation['object_size_metrics']['buckets']['tiny']['recall']:.4f}, mobile recall {confirmation['viewport_metrics']['mobile']['recall']:.4f}, increased false-positive volume, and source-concentrated gains.\n- AtomBlock mAP50-95 retention remains 91.79%; the 14-class architecture and catastrophic-forgetting gate remain intact.\n- Final gate: {decision}. VisionQA v1 remains production/rollback; v2 remains experimental and was not promoted.\n- Next milestone: {next_milestone}\n- No training, dataset modification, or Playwright promotion regression occurred.\n"""
        state_path.write_text(state, encoding="utf-8")
    print(json.dumps({"decision": decision, "next_milestone": next_milestone, "tradeoff": tradeoff, "versus_v1": versus_v1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
