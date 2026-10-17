"""Evaluate the frozen WebUI transfer slice and generate review artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.webui_evaluation import SCORABLE_TARGET_CLASSES, box_iou  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def match_screen(screen: dict, prediction: dict, iou_threshold: float = 0.5) -> dict:
    gt = screen["valid_ground_truth_boxes"]
    # Every prediction in the globally mapped taxonomy is scorable. Restricting
    # predictions to classes present in a screen's GT would hide false positives.
    preds = [row for row in prediction["detections"] if row["class_name"] in SCORABLE_TARGET_CLASSES]
    candidates = []
    for gi, ground in enumerate(gt):
        for pi, pred in enumerate(preds):
            if ground["target_class"] == pred["class_name"]:
                iou = box_iou(ground["bbox_xyxy"], pred["bbox_xyxy"])
                if iou >= iou_threshold: candidates.append((iou, gi, pi))
    used_gt, used_pred, matches = set(), set(), []
    for iou, gi, pi in sorted(candidates, reverse=True):
        if gi in used_gt or pi in used_pred: continue
        used_gt.add(gi); used_pred.add(pi)
        matches.append({"ground_truth_index": gi, "prediction_index": pi, "iou": iou, "class_name": gt[gi]["target_class"], "confidence": preds[pi]["confidence"]})
    false_negatives = [index for index in range(len(gt)) if index not in used_gt]
    false_positives = [index for index in range(len(preds)) if index not in used_pred]
    wrong_class = []
    duplicates = []
    for pi in false_positives:
        pred = preds[pi]
        same_overlap = [m for m in matches if m["class_name"] == pred["class_name"] and box_iou(gt[m["ground_truth_index"]]["bbox_xyxy"], pred["bbox_xyxy"]) >= iou_threshold]
        if same_overlap:
            duplicates.append(pi); continue
        options = [(box_iou(ground["bbox_xyxy"], pred["bbox_xyxy"]), gi) for gi, ground in enumerate(gt) if ground["target_class"] != pred["class_name"]]
        if options and max(options)[0] >= iou_threshold:
            iou, gi = max(options); wrong_class.append({"prediction_index": pi, "ground_truth_index": gi, "iou": iou, "predicted": pred["class_name"], "expected": gt[gi]["target_class"]})
    review_items = []
    for match in matches:
        review_items.append({"item_id": f"match-{match['ground_truth_index']}-{match['prediction_index']}", "auto_decision": "model_true_positive", "class_name": match["class_name"], "iou": match["iou"], "confidence": match["confidence"], "allowed_reviewer_decisions": ["model_true_positive", "model_false_positive", "model_false_negative", "wrong_class", "annotation_questionable", "ignore"]})
    for gi in false_negatives:
        review_items.append({"item_id": f"fn-{gi}", "auto_decision": "model_false_negative", "class_name": gt[gi]["target_class"], "bbox_xyxy": gt[gi]["bbox_xyxy"], "allowed_reviewer_decisions": ["model_false_negative", "annotation_questionable", "mapping_invalid", "ignore"]})
    for pi in false_positives:
        review_items.append({"item_id": f"fp-{pi}", "auto_decision": "wrong_class" if any(item["prediction_index"] == pi for item in wrong_class) else "model_false_positive", "class_name": preds[pi]["class_name"], "bbox_xyxy": preds[pi]["bbox_xyxy"], "confidence": preds[pi]["confidence"], "allowed_reviewer_decisions": ["model_false_positive", "wrong_class", "model_true_positive", "annotation_questionable", "ignore"]})
    return {"ground_truth_count": len(gt), "scorable_prediction_count": len(preds), "matches": matches, "false_negative_indices": false_negatives, "false_positive_indices": false_positives, "wrong_class": wrong_class, "duplicate_prediction_indices": duplicates, "review_items": review_items, "predictions": preds}


def ap_from_records(records: list[tuple[float, bool]], gt_count: int) -> float:
    if gt_count == 0: return 0.0
    records.sort(key=lambda item: item[0], reverse=True)
    tp = fp = 0; points = []
    for confidence, correct in records:
        tp += int(correct); fp += int(not correct)
        points.append((tp / gt_count, tp / (tp + fp)))
    return sum(max((precision for recall, precision in points if recall >= threshold), default=0.0) for threshold in [i / 100 for i in range(101)]) / 101


def metrics(results: list[dict]) -> dict:
    classes = sorted({box["target_class"] for row in results for box in row["screen"]["valid_ground_truth_boxes"]})
    per_class = {}
    all_tp = all_fp = all_fn = 0
    for class_name in classes:
        gt_count = sum(sum(box["target_class"] == class_name for box in row["screen"]["valid_ground_truth_boxes"]) for row in results)
        matched_keys = {(row["screen"]["screen_id"], match["prediction_index"]) for row in results for match in row["evaluation"]["matches"] if match["class_name"] == class_name}
        prediction_records = []
        prediction_count = 0
        for row in results:
            for pi, pred in enumerate(row["evaluation"]["predictions"]):
                if pred["class_name"] == class_name:
                    prediction_count += 1; prediction_records.append((pred["confidence"], (row["screen"]["screen_id"], pi) in matched_keys))
        tp = len(matched_keys); fp = prediction_count - tp; fn = gt_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"ground_truth": gt_count, "predictions": prediction_count, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "ap50": ap_from_records(prediction_records, gt_count)}
        all_tp += tp; all_fp += fp; all_fn += fn
    precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0; recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    return {"iou_threshold": 0.5, "tp": all_tp, "fp": all_fp, "fn": all_fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "macro_f1": statistics.mean(row["f1"] for row in per_class.values()), "per_class": per_class}


def choose_review_subset(results: list[dict], count: int = 100) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    weak = {"select", "block-form", "block-table", "block-sidebar"}
    for row in results:
        mapped = set(row["screen"]["mapped_classes"])
        row["review_priority"] = 10 * len(mapped & weak) + len(row["screen"]["valid_ground_truth_boxes"])
        by_source[row["screen"]["source_name"]].append(row)
    for bucket in by_source.values(): bucket.sort(key=lambda row: (-row["review_priority"], row["screen"]["screen_id"]))
    selected = []
    while len(selected) < count:
        progressed = False
        for source in sorted(by_source):
            if by_source[source] and len(selected) < count:
                selected.append(by_source[source].pop(0)); progressed = True
        if not progressed: break
    return selected


def draw_overlay(row: dict, output: Path) -> None:
    screen, evaluation = row["screen"], row["evaluation"]
    image = Image.open(screen["image_identifier"]["extracted_path"]).convert("RGB")
    draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    matched_gt = {item["ground_truth_index"] for item in evaluation["matches"]}
    matched_pred = {item["prediction_index"] for item in evaluation["matches"]}
    for index, box in enumerate(screen["valid_ground_truth_boxes"]):
        color = "#00d060" if index in matched_gt else "#ffd000"
        draw.rectangle(box["bbox_xyxy"], outline=color, width=3)
        draw.text((box["bbox_xyxy"][0] + 2, box["bbox_xyxy"][1] + 2), "G:" + box["target_class"], fill=color, font=font, stroke_width=2, stroke_fill="black")
    for index, pred in enumerate(evaluation["predictions"]):
        color = "#00bfff" if index in matched_pred else "#ff3040"
        draw.rectangle(pred["bbox_xyxy"], outline=color, width=2)
        draw.text((pred["bbox_xyxy"][0] + 2, max(0, pred["bbox_xyxy"][1] - 12)), f"P:{pred['class_name']} {pred['confidence']:.2f}", fill=color, font=font, stroke_width=2, stroke_fill="black")
    output.parent.mkdir(parents=True, exist_ok=True); image.save(output)


def make_contact_sheets(review_rows: list[dict], overlay_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True); paths = []
    for page_index in range(math.ceil(len(review_rows) / 10)):
        subset = review_rows[page_index * 10:(page_index + 1) * 10]
        sheet = Image.new("RGB", (1200, 1750), "white"); draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
        for index, row in enumerate(subset):
            source = Image.open(overlay_dir / f"{row['screen']['screen_id']}.jpg").convert("RGB")
            source.thumbnail((580, 300))
            x = (index % 2) * 600 + 10; y = (index // 2) * 350 + 35
            sheet.paste(source, (x, y))
            ev = row["evaluation"]
            caption = f"{page_index*10+index+1:03d} {row['screen']['screen_id']} GT={ev['ground_truth_count']} TP={len(ev['matches'])} FP={len(ev['false_positive_indices'])} FN={len(ev['false_negative_indices'])}"
            draw.text((x, y - 25), caption[:95], fill="black", font=font)
        path = output_dir / f"review_contact_{page_index + 1:02d}.jpg"; sheet.save(path, quality=90); paths.append(str(path))
    return paths


def write_review_html(results: list[dict], review_ids: set[str], overlay_dir: Path, output: Path) -> None:
    cards = []
    options = "".join(f'<option value="{value}">{value}</option>' for value in ["mapping_valid", "mapping_invalid", "annotation_questionable", "ignore"])
    for row in results:
        screen, evaluation = row["screen"], row["evaluation"]
        item_rows = []
        for item in evaluation["review_items"]:
            item_options = "".join(f'<option value="{html.escape(value)}" {"selected" if value == item["auto_decision"] else ""}>{html.escape(value)}</option>' for value in item["allowed_reviewer_decisions"])
            item_rows.append(f'<label>{html.escape(item["item_id"])} {html.escape(item["class_name"])} <select data-item="{html.escape(item["item_id"])}">{item_options}</select></label>')
        cards.append(f'''<article data-screen="{screen['screen_id']}" class="{'required' if screen['screen_id'] in review_ids else ''}"><h3>{screen['screen_id']}</h3><img loading="lazy" src="{overlay_dir.name}/{screen['screen_id']}.jpg"><p>{screen['source_name']} / {screen['viewport']} — GT {evaluation['ground_truth_count']}, matches {len(evaluation['matches'])}, FP {len(evaluation['false_positive_indices'])}, FN {len(evaluation['false_negative_indices'])}</p><label>Mapping <select data-mapping>{options}</select></label><div class="items">{''.join(item_rows)}</div><textarea placeholder="review notes"></textarea></article>''')
    output.write_text(f'''<!doctype html><meta charset="utf-8"><title>QASense WebUI transfer review</title><style>body{{font:14px Arial;margin:20px}}article{{border:1px solid #bbb;padding:12px;margin:15px 0}}article.required{{border:3px solid #1769aa}}img{{max-width:100%;max-height:720px}}label{{display:block;margin:5px}}textarea{{width:100%;height:50px}}.items{{columns:3}}</style><h1>WebUI transfer review</h1><p>Green ground truth, yellow missed ground truth, cyan matched prediction, red unmatched prediction. Thick blue border marks the required 100-screen review subset.</p><button onclick="exportReview()">Export decisions</button>{''.join(cards)}<script>function exportReview(){{const rows=[...document.querySelectorAll('article')].map(a=>({{screen_id:a.dataset.screen,mapping_decision:a.querySelector('[data-mapping]').value,notes:a.querySelector('textarea').value,items:[...a.querySelectorAll('[data-item]')].map(s=>({{item_id:s.dataset.item,decision:s.value}}))}}));const blob=new Blob([JSON.stringify({{reviews:rows}},null,2)],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='webui_review_decisions.json';link.click()}}</script>''', encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, default=ROOT / "reports/visionqa/webui_eval_manifest.json")
    args = parser.parse_args(); manifest = load(args.manifest)
    prediction_dir = ROOT / "runs/visionqa/webui_transfer_eval/predictions"
    results = []
    for screen in manifest["screens"]:
        prediction = load(prediction_dir / f"{screen['screen_id']}.json")
        results.append({"screen": screen, "prediction_summary": {"detection_count": prediction["detection_count"], "counts_by_class": prediction["counts_by_class"], "inference_time_ms": prediction["inference_time_ms"]}, "evaluation": match_screen(screen, prediction)})
    all_metrics = metrics(results)
    review_rows = choose_review_subset(results, 100); review_ids = {row["screen"]["screen_id"] for row in review_rows}
    overlay_dir = ROOT / "reports/visionqa/webui_review_overlays"
    for row in results: draw_overlay(row, overlay_dir / f"{row['screen']['screen_id']}.jpg")
    contact_paths = make_contact_sheets(review_rows, overlay_dir, ROOT / "reports/visionqa/webui_review_contact_sheets")
    write_review_html(results, review_ids, overlay_dir, ROOT / "reports/visionqa/webui_review.html")
    pending = {"schema_version": "1.0", "review_method": "Manual visual inspection of annotated contact sheets; automatic IoU decisions are candidates and must be confirmed.", "required_review_count": 100, "reviews": [{"screen_id": row["screen"]["screen_id"], "source_name": row["screen"]["source_name"], "viewport": row["screen"]["viewport"], "review_status": "pending", "mapping_decision": None, "reviewer_items": [{"item_id": item["item_id"], "auto_decision": item["auto_decision"], "reviewer_decision": None} for item in row["evaluation"]["review_items"]]} for row in review_rows]}
    decisions_path = ROOT / "reports/visionqa/webui_review_decisions.json"; decisions_path.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")

    error_counts = Counter(); examples: dict[str, list[str]] = defaultdict(list); small_areas, missed_areas = [], []
    for row in results:
        screen, evaluation = row["screen"], row["evaluation"]; image_area = screen["image_identifier"]["width"] * screen["image_identifier"]["height"]
        for match in evaluation["matches"]:
            if match["confidence"] < 0.4: error_counts["low_confidence_correct"] += 1; examples["low_confidence_correct"].append(screen["screen_id"])
        for gi in evaluation["false_negative_indices"]:
            box = screen["valid_ground_truth_boxes"][gi]; area = (box["bbox_xyxy"][2]-box["bbox_xyxy"][0])*(box["bbox_xyxy"][3]-box["bbox_xyxy"][1])/image_area; missed_areas.append(area)
            bucket = "missed_small_element" if area < 0.0025 else "visual_domain_shift"
            error_counts[bucket] += 1; examples[bucket].append(screen["screen_id"])
            if evaluation["ground_truth_count"] >= 20: error_counts["dense_layout_miss"] += 1; examples["dense_layout_miss"].append(screen["screen_id"])
        error_counts["wrong_semantic_class"] += len(evaluation["wrong_class"]); error_counts["duplicate_detection"] += len(evaluation["duplicate_prediction_indices"])
        if evaluation["wrong_class"]: examples["wrong_semantic_class"].append(screen["screen_id"])
        if evaluation["duplicate_prediction_indices"]: examples["duplicate_detection"].append(screen["screen_id"])

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results: groups[row["screen"]["source_sample_id"]].append(row)
    responsive = {"sample_groups": len(groups), "complete_triplets": 0, "mean_detection_class_jaccard": 0.0, "classes_present_and_detected_all_viewports": Counter(), "newly_missing_by_viewport": Counter(), "responsive_only_misses": 0, "viewport_specific_false_positives": Counter()}
    jaccards = []
    for rows in groups.values():
        if {row["screen"]["viewport"] for row in rows} != {"desktop", "tablet", "mobile"}: continue
        responsive["complete_triplets"] += 1
        detected_sets = [set(row["prediction_summary"]["counts_by_class"]) for row in rows]
        union = set.union(*detected_sets); intersection = set.intersection(*detected_sets)
        jaccards.append(len(intersection) / len(union) if union else 1.0)
        class_state: dict[str, dict[str, bool]] = defaultdict(dict)
        for row in rows:
            viewport = row["screen"]["viewport"]
            matched_classes = {match["class_name"] for match in row["evaluation"]["matches"]}
            gt_classes = set(row["screen"]["mapped_classes"])
            for class_name in gt_classes: class_state[class_name][viewport] = class_name in matched_classes
            responsive["viewport_specific_false_positives"][viewport] += len(row["evaluation"]["false_positive_indices"])
        for class_name, states in class_state.items():
            if len(states) == 3 and all(states.values()): responsive["classes_present_and_detected_all_viewports"][class_name] += 1
            if len(states) == 3 and 0 < sum(states.values()) < 3:
                responsive["responsive_only_misses"] += 1
                for viewport, present in states.items():
                    if not present: responsive["newly_missing_by_viewport"][viewport] += 1
    responsive["mean_detection_class_jaccard"] = statistics.mean(jaccards)
    responsive["classes_present_and_detected_all_viewports"] = dict(responsive["classes_present_and_detected_all_viewports"])
    responsive["newly_missing_by_viewport"] = dict(responsive["newly_missing_by_viewport"])
    responsive["viewport_specific_false_positives"] = dict(responsive["viewport_specific_false_positives"])
    error_payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "counts": dict(error_counts), "representative_screen_ids": {key: list(dict.fromkeys(value))[:12] for key, value in examples.items()}, "responsive": responsive, "mapping_ambiguity": {"icon": "svg/i are ambiguous and excluded", "checkbox": "input type is absent; hinted inputs excluded", "radio": "input type is absent; hinted inputs excluded"}}
    (ROOT / "reports/visionqa/webui_error_analysis.json").write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
    output = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(), "manifest_content_sha256": manifest["manifest_content_sha256"], "screen_count": len(results), "sample_group_count": manifest["selection"]["source_sample_groups"], "source_count": len(manifest["source_screen_counts"]), "confidence_threshold": 0.25, "threshold_tuned_on_slice": False, "mean_inference_time_ms": statistics.mean(row["prediction_summary"]["inference_time_ms"] for row in results), "all_mapped_screens": all_metrics, "human_review": {"selected_screens": len(review_rows), "status": "pending_manual_visual_review", "decision_file": str(decisions_path), "review_html": str(ROOT / "reports/visionqa/webui_review.html"), "contact_sheets": contact_paths}, "responsive_consistency": responsive}
    (ROOT / "reports/visionqa/webui_transfer_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screens": len(results), "review_subset": len(review_rows), "metrics": all_metrics, "responsive": responsive, "errors": dict(error_counts)}, indent=2))


if __name__ == "__main__":
    main()
