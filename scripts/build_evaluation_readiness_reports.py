"""Build the real-state, WebUI transfer, and next-model readiness reports."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    real = load("reports/integration/real_state_evaluation.json")
    transfer = load("reports/visionqa/webui_transfer_metrics.json")
    errors = load("reports/visionqa/webui_error_analysis.json")
    manifest = load("reports/visionqa/webui_eval_manifest.json")
    review = load("reports/visionqa/webui_review_decisions.json")
    primary = Counter()
    state_rows = []
    seen = set()
    for run in real["runs"]:
        if run["state_id"] in seen: continue
        seen.add(run["state_id"]); counts = run["review"]["counts"]; primary.update({key: value for key, value in counts.items() if key != "match_rate"})
        state_rows.append((run["state_id"], run["review"]["visionqa"]["detection_count"], run["review"]["matching"]["match_count"], run["review"]["matching"]["semantic_conflicts"], counts["correctly_detected_elements"], counts["missed_detection"], counts["wrong_class"]))
    primary_match_rate = primary["correctly_detected_elements"] / primary["reviewed_elements"]
    real_lines = [
        "# Real Application-State Validation", "", "Generated: 2026-10-13 PDT", "",
        "## Scope", "", "No separate local application or configured deterministic QA target exists in the repository. This milestone therefore used the existing QASense runtime fixture, as required by the fallback rule. It is a realistic-fixture validation and is not presented as a production-application benchmark.", "",
        "Six distinct existing states were replayed twice: desktop default, empty email, unchecked terms, focused navigation, tablet default, and mobile default. Every run preserved screenshots, DOM, console/network/page errors, trace, VisionQA output, matching, and ambiguity evidence.", "",
        "## Results", "", "| State | Detections | Matches | Conflicts | Correct reviewed | Missed | Wrong class |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    real_lines.extend(f"| {state} | {det} | {matches} | {conflicts} | {correct} | {missed} | {wrong} |" for state, det, matches, conflicts, correct, missed, wrong in state_rows)
    real_lines += [
        "", f"Across the six primary runs, **{primary['reviewed_elements']} elements** were reviewed: **{primary['correctly_detected_elements']} correct**, **{primary['missed_detection']} missed**, **{primary['wrong_class']} wrong-class**, **{primary['semantic_conflicts']} semantic conflicts**, and **{primary['duplicate_detections']} duplicate detections**. The reviewed match rate was **{pct(primary_match_rate)}**.", "",
        "## Repeatability", "", f"All **{len(real['repeatability'])} states** were stable across two runs for screenshot hash, dimensions, DOM count, detection count, class counts, match count, semantic-conflict count, and normalized console/network/page-error evidence. UTC event timestamps were excluded from equality checks because timestamps are expected to differ.", "",
        "The fixture intentionally emits one page exception, console errors, and one failed request; those counts and messages were repeatable.", "",
        "## Findings", "", "Desktop states transferred better than smaller viewports. The mobile state retained 11 DOM candidates but produced only five detections and two semantic matches. Focus styling also changed the predicted class balance, demonstrating that state appearance affects detector output even when DOM structure is unchanged.", "",
        "Machine-readable manifest: `reports/integration/real_state_manifest.json`. Results: `reports/integration/real_state_evaluation.json`.", "",
    ]
    (ROOT / "reports/integration/REAL_APPLICATION_STATE_VALIDATION.md").write_text("\n".join(real_lines), encoding="utf-8")

    all_metrics = transfer["all_mapped_screens"]; reviewed = transfer["human_review"]["metrics_on_mapping_valid_reviewed_screens"]
    transfer_lines = [
        "# WebUI Transfer Evaluation", "", "Generated: 2026-10-13 PDT", "",
        "## Fixed slice", "", f"The frozen evaluation slice contains **{transfer['screen_count']} screens** from **{transfer['sample_group_count']} source-sample groups** and **{transfer['source_count']} sources**. Each group retains desktop, tablet, and mobile screenshots. Manifest content SHA-256: `{transfer['manifest_content_sha256']}`.", "",
        "All 15 known cross-split exact-image groups were excluded before selection. Nonpositive, fully off-viewport, duplicate mapped, and semantically ambiguous input boxes were excluded from scoring and retained in the manifest. The original dataset was not changed.", "",
        "## Class mapping", "", "Scored mappings are native `button→button`, `a→link`, `input/textarea→input`, `img→image`, `select→select`, `nav→block-nav`, `aside→block-sidebar`, `footer→block-footer`, `form→block-form`, and `table→block-table`. Generic SVG/i icons, generic containers, and checkbox/radio subtypes are not scored because WebUI does not provide sufficient semantics for reliable mapping.", "",
        "## Full-slice metrics at IoU 0.50", "", f"Precision: **{all_metrics['precision']:.5f}**  ", f"Recall: **{all_metrics['recall']:.5f}**  ", f"F1: **{all_metrics['f1']:.5f}**  ", f"Macro F1: **{all_metrics['macro_f1']:.5f}**", "",
        "| Class | GT | Predictions | Precision | Recall | F1 | AP50 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in all_metrics["per_class"].items(): transfer_lines.append(f"| {name} | {row['ground_truth']} | {row['predictions']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['ap50']:.4f} |")
    transfer_lines += [
        "", "AP50 is reported because mapped bounding boxes and fixed IoU=0.50 support it. AP50-95 is omitted: taxonomy conversion and DOM-box clipping do not support that stronger localization claim.", "",
        "## Human-reviewed subset", "", f"All **{review['completed_review_count']} selected screens** were visually inspected using ten annotated contact sheets. **{review['mapping_valid_screens']}** were accepted for reviewed quantitative scoring and **{review['annotation_questionable_screens']}** were isolated. Reviewed item decisions: {json.dumps(review['review_item_decisions'])}.", "",
        f"On the 96 mapping-valid reviewed screens: precision **{reviewed['precision']:.5f}**, recall **{reviewed['recall']:.5f}**, F1 **{reviewed['f1']:.5f}**. This confirms that the poor full-slice transfer is not only an automated-annotation artifact.", "",
        "Review workflow: `reports/visionqa/webui_review.html`. Decisions: `reports/visionqa/webui_review_decisions.json`.", "",
        "## Responsive consistency", "", f"All **{transfer['responsive_consistency']['complete_triplets']}** sample groups had complete viewport triplets. Mean Jaccard similarity of detected class sets across desktop/tablet/mobile was **{transfer['responsive_consistency']['mean_detection_class_jaccard']:.3f}**. There were **{transfer['responsive_consistency']['responsive_only_misses']}** class-level responsive-only miss patterns; missing occurrences were desktop {transfer['responsive_consistency']['newly_missing_by_viewport'].get('desktop',0)}, tablet {transfer['responsive_consistency']['newly_missing_by_viewport'].get('tablet',0)}, and mobile {transfer['responsive_consistency']['newly_missing_by_viewport'].get('mobile',0)}.", "",
        "These differences are consistency evidence only. Responsive changes can be intentional and are not labeled as defects.", "",
        "## Weak classes", "", "`select` had 121 mapped boxes and zero true positives; its median box occupied only 0.31% of the screen, supporting a small-object and domain-shift explanation. `block-form` had 164 boxes and 1.22% recall; missed forms were not generally small, and occurred on dense screens, supporting taxonomy/domain-shift rather than size alone. `block-table` had 164 boxes and zero true positives despite a median 9.17% screen area, strongly indicating domain/taxonomy mismatch rather than small-object size.", "",
        "Icon, checkbox, and radio were not scored. There were 1,698 SVG boxes and 132 i-tag boxes with ambiguous icon semantics, plus 120 checkbox/radio/switch-like input boxes excluded because input type is absent from WebUI bbox records.", "",
        "## Error buckets", "",
    ]
    transfer_lines.extend(f"- {name}: **{count:,}**" for name, count in sorted(errors["counts"].items(), key=lambda item: -item[1]))
    transfer_lines += ["", "Representative screen IDs are stored in `reports/visionqa/webui_error_analysis.json`.", ""]
    (ROOT / "reports/visionqa/WEBUI_TRANSFER_EVALUATION.md").write_text("\n".join(transfer_lines), encoding="utf-8")

    readiness = {
        "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": "B", "new_model_justified": True, "training_performed": False,
        "recommendation": "Prepare targeted VisionQA fine-tuning on a new curated WebUI-derived training/validation corpus; preserve the 399-screen evaluation slice as evaluation-only.",
        "evidence": {"full_slice_precision": all_metrics["precision"], "full_slice_recall": all_metrics["recall"], "full_slice_f1": all_metrics["f1"], "human_reviewed_precision": reviewed["precision"], "human_reviewed_recall": reviewed["recall"], "human_reviewed_f1": reviewed["f1"], "systematic_zero_recall_classes": [name for name, row in all_metrics["per_class"].items() if row["recall"] == 0], "mapped_ground_truth_boxes": sum(row["ground_truth"] for row in all_metrics["per_class"].values())},
        "supported_training_targets": ["button", "link", "input", "image", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table"],
        "excluded_targets": {"icon": "ambiguous svg/i semantics", "checkbox": "input type absent", "radio": "input type absent", "block": "generic containers incompatible"},
        "controlled_dataset": {"current": "pilot_v2", "changed": False, "reason": "Observed failures are element-detection domain/taxonomy transfer failures, not visual-defect mutation failures."},
        "exact_next_milestone": "Create a separate source-grouped, human-reviewed WebUI-derived training/validation corpus for the supported classes, excluding every source sample and exact-image group in webui_eval_manifest.json; benchmark a targeted fine-tune plan without touching the fixed evaluation slice.",
    }
    readiness_path = ROOT / "reports/visionqa/next_model_readiness.json"; readiness_path.write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    decision_lines = [
        "# Next-Model Readiness Decision", "", "Generated: 2026-10-13 PDT", "",
        "## Decision", "", "**B. Targeted VisionQA fine-tuning is justified. No model was trained in this milestone.**", "",
        f"The fixed WebUI slice shows systematic transfer weakness: full-slice precision {all_metrics['precision']:.5f}, recall {all_metrics['recall']:.5f}, and F1 {all_metrics['f1']:.5f}. Manual review of 100 screens retained 96 mapping-valid screens and confirmed precision {reviewed['precision']:.5f}, recall {reviewed['recall']:.5f}, and F1 {reviewed['f1']:.5f}. Select, table, and sidebar had zero recall; form and navigation recall were near zero. Real-state validation also degraded sharply on mobile.", "",
        "WebUI has sufficient candidate volume for the ten supported native-tag mappings, and those classes affect QASense DOM/visual grounding. This meets the readiness criteria for preparing a targeted fine-tuning dataset. It does not justify training directly from raw WebUI annotations.", "",
        "## Required next model path", "", readiness["recommendation"], "",
        "The next corpus must be source-grouped, filter invalid/off-viewport/nested boxes, undergo human mapping review, and exclude all 133 sample groups and cross-split duplicate groups in the fixed evaluation manifest. The frozen 399 screens must never become training data.", "",
        "Icon, checkbox, and radio require better semantic labels before inclusion. A separate auxiliary screen-understanding model remains a later option for SVG/container structure, but current evidence first supports targeted detector transfer for the ten defensible classes.", "",
        "## Controlled defect dataset", "", "`pilot_v2` remains unchanged. The observed failures concern element taxonomy, density, and domain transfer, so no evidence-based `pilot_v3` mutation cases were added.", "",
        "## Exact next milestone", "", readiness["exact_next_milestone"], "",
        "This milestone stops before that work, as required.", "",
    ]
    (ROOT / "reports/visionqa/NEXT_MODEL_READINESS_DECISION.md").write_text("\n".join(decision_lines), encoding="utf-8")
    print(json.dumps({"real_states": real["state_count"], "webui_screens": transfer["screen_count"], "reviewed": review["completed_review_count"], "decision": readiness["decision"], "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
