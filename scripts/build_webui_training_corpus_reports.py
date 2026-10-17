"""Generate corpus report, training-readiness record, and future training spec."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/visionqa"


def main() -> None:
    metrics = json.loads((REPORTS / "webui_training_corpus_metrics.json").read_text(encoding="utf-8"))
    review = json.loads((REPORTS / "webui_training_review.json").read_text(encoding="utf-8"))
    exclusions = json.loads((REPORTS / "webui_training_exclusions.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    leakage_counts = {key: len(value) for key, value in metrics["leakage"].items()}
    weak = ("select", "block-nav", "block-sidebar", "block-form", "block-table")
    weak_counts = {name: {
        "train_boxes": metrics["distributions"]["train"]["classes"][name]["box_count"],
        "train_groups": metrics["distributions"]["train"]["classes"][name]["group_count"],
        "val_boxes": metrics["distributions"]["val"]["classes"][name]["box_count"],
        "val_groups": metrics["distributions"]["val"]["classes"][name]["group_count"],
    } for name in weak}
    conditions = {
        "frozen_evaluation_leakage_zero": leakage_counts["frozen_evaluation_groups"] == 0,
        "source_group_leakage_zero": leakage_counts["source_group_across_train_val"] == 0,
        "known_and_observed_exact_duplicate_leakage_zero": leakage_counts["known_duplicate_groups"] == 0 and leakage_counts["exact_image_hash_across_train_val"] == 0,
        "annotations_valid": metrics["audit_result"] == "PASS",
        "class_mapping_fixed": (REPORTS / "visionqa_v2_class_schema.json").exists(),
        "human_review_completed": review["status"] == "complete" and review["summary"]["reviewed_screens"] >= 300,
        "train_validation_groups_sufficient": metrics["distributions"]["train"]["groups"] > 1000 and metrics["distributions"]["val"]["groups"] > 200,
        "weak_classes_represented_in_both_splits": all(value["train_groups"] >= 100 and value["val_groups"] >= 20 for value in weak_counts.values()),
    }
    ready = all(conditions.values())
    readiness = {
        "schema_version": "1.0", "generated_at": now,
        "training_ready": ready, "training_performed": False,
        "conditions": conditions, "weak_class_evidence": weak_counts,
        "corpus": {
            "path": "datasets/processed/visionqa_webui_v2",
            "train_groups": metrics["distributions"]["train"]["groups"],
            "validation_groups": metrics["distributions"]["val"]["groups"],
            "train_screens": metrics["distributions"]["train"]["screens"],
            "validation_screens": metrics["distributions"]["val"]["screens"],
            "valid_boxes": metrics["counts"]["valid_boxes"],
            "dataset_manifest_sha256": metrics["manifest_sha256"]["dataset"],
            "split_manifest_sha256": metrics["manifest_sha256"]["split"],
        },
        "frozen_evaluation_manifest_sha256": "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45",
        "decision": "Ready to run the separately authorized VisionQA v2 fine-tuning milestone." if ready else "Not ready; resolve failed conditions before training.",
        "exact_next_milestone": "Benchmark and run resumable YOLO11n fine-tuning from VisionQA v1 weights on this fixed corpus, selecting by corpus validation metrics and evaluating once on both the frozen WebUI slice and the original AtomBlock held-out domain.",
    }
    (REPORTS / "visionqa_v2_readiness.json").write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    write_corpus_report(metrics, review, exclusions, readiness)
    write_training_spec(readiness)
    print(json.dumps({"training_ready": ready, "conditions": conditions, "corpus": readiness["corpus"], "training_performed": False}, indent=2))


def write_corpus_report(metrics: dict, review: dict, exclusions: dict, readiness: dict) -> None:
    reasons = {}
    for item in exclusions["group_exclusions"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    lines = [
        "# WebUI Training Corpus Report", "", "Generated for the VisionQA v2 data-preparation milestone. No model training occurred.", "",
        "## Corpus design", "",
        "The source WebUI dataset remains unchanged. The processed corpus uses complete desktop/tablet/mobile source groups, a deterministic seed of 42, a source-stratified split, and exact-image connected-component consolidation. The frozen 399-screen evaluation slice is external and absent from both train and validation.", "",
        "- Eligible complete candidate groups after frozen/duplicate exclusions: 12,042",
        "- Positive groups retained: 9,058",
        "- Deterministically retained negative-only groups: 453 (5% of positive groups)",
        f"- Train: {metrics['distributions']['train']['groups']} groups / {metrics['distributions']['train']['screens']} screens / {metrics['distributions']['train']['valid_boxes']} boxes",
        f"- Validation: {metrics['distributions']['val']['groups']} groups / {metrics['distributions']['val']['screens']} screens / {metrics['distributions']['val']['valid_boxes']} boxes",
        f"- Group exclusions: `{json.dumps(reasons, sort_keys=True)}`", "",
        "## Mapping and cleaning", "",
        "The ten-class schema is contiguous and maps back to the original AtomBlock IDs. Icon, checkbox, radio, and generic block are excluded. Partially visible boxes are clamped to the image boundary; fully offscreen, nonpositive, duplicate, ambiguous, and unmapped boxes are excluded without modifying the source annotations.", "",
        f"Box exclusions: `{json.dumps(metrics['counts']['excluded_box_reason_counts'], sort_keys=True)}`", "",
        "## Human review", "",
        f"{review['summary']['reviewed_screens']} screens from distinct groups were manually reviewed across {review['selection']['stratified_for']}. All were accepted for the supported mappings; no annotation was silently corrected. The review is screen-level mapping validation, not exhaustive relabeling of every DOM node.", "",
        "## Class representation", "",
        "| Class | Train boxes | Train screens | Train groups | Val boxes | Val screens | Val groups |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in metrics["distributions"]["train"]["classes"]:
        train = metrics["distributions"]["train"]["classes"][name]
        val = metrics["distributions"]["val"]["classes"][name]
        lines.append(f"| {name} | {train['box_count']} | {train['screen_count']} | {train['group_count']} | {val['box_count']} | {val['screen_count']} | {val['group_count']} |")
    lines += [
        "", "## Viewports and sources", "",
        f"- Train viewport screens: `{json.dumps(metrics['distributions']['train']['viewports'], sort_keys=True)}`; {metrics['distributions']['train']['unique_sources']} sources.",
        f"- Validation viewport screens: `{json.dumps(metrics['distributions']['val']['viewports'], sort_keys=True)}`; {metrics['distributions']['val']['unique_sources']} sources.",
        f"- Train negative screens/groups: {metrics['distributions']['train']['negative_screens']} / {metrics['distributions']['train']['negative_groups']}.",
        f"- Validation negative screens/groups: {metrics['distributions']['val']['negative_screens']} / {metrics['distributions']['val']['negative_groups']}.", "",
        "## Leakage and readiness", "",
        *[f"- {key}: {len(value)}" for key, value in metrics["leakage"].items()],
        f"- Dataset manifest SHA-256: `{metrics['manifest_sha256']['dataset']}`",
        f"- Split manifest SHA-256: `{metrics['manifest_sha256']['split']}`",
        f"- Corpus audit: **{metrics['audit_result']}**",
        f"- Training readiness: **{'YES' if readiness['training_ready'] else 'NO'}**", "",
        "Detailed per-source and per-viewport class distributions are in `webui_training_corpus_metrics.json`.",
    ]
    (REPORTS / "WEBUI_TRAINING_CORPUS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_spec(readiness: dict) -> None:
    content = f"""# VisionQA v2 Future Training Specification

No training was executed while creating this specification.

## Objective

The primary objective is to improve cross-domain WebUI transfer on the ten supported classes: button, link, input, image, select, block-nav, block-sidebar, block-footer, block-form, and block-table.

Secondary objectives are higher recall, fewer dense-layout and domain-shift misses, and better performance on select, form, table, sidebar, and navigation. This corpus does not support claims about icon, checkbox, radio, or generic block.

## Fixed baselines

- Frozen WebUI slice: precision 0.08805, recall 0.03564, F1 0.05074, macro-F1 0.04835.
- Original AtomBlock test: precision 0.47383, recall 0.40132, mAP50 0.37755, mAP50-95 0.27825.

## Proposed experiment — do not run in this milestone

1. Initialize YOLO11n from `models/visionqa/atomblock_yolo11n/best.pt` while creating a new ten-class detection head.
2. Train only on `datasets/processed/visionqa_webui_v2/data.yaml`; never use the frozen 399 screens for training, validation, augmentation, mining, mapping, or threshold tuning.
3. Start with image size 640, seed 42, validation each epoch, early stopping, and per-epoch resumable checkpoints. Benchmark safe batch sizes on the T1200 before choosing one.
4. Run an initial maximum of 30 epochs. Select checkpoints only from the new corpus validation metrics, including per-class recall and mAP.
5. After selection, evaluate once on the frozen WebUI transfer slice and regression-test the ten overlapping classes on the original AtomBlock held-out test. Preserve v1's separate 14-class metrics and unsupported-class capability.
6. Keep VisionQA v1 as the production rollback. A v2 checkpoint must not replace it automatically.

## Promotion principle

Promotion requires meaningful improvement over the frozen WebUI baseline, acceptable regression results on the overlapping AtomBlock classes, stable inference, no major supported-class regression, and successful Playwright integration. Final numeric promotion thresholds should be chosen during the training-design milestone before test evaluation, not retrofitted after seeing test results.

## Planned artifact path

`models/visionqa/visionqa_v2_webui_yolo11n/best.pt`

Training readiness recorded by this milestone: **{'YES' if readiness['training_ready'] else 'NO'}**.
"""
    (REPORTS / "VISIONQA_V2_TRAINING_SPEC.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
