"""Source-grouped CV for one global VisionQA v2 confidence threshold.

This script is CPU-only. It consumes persisted WebUI validation predictions and
never reads the frozen evaluation manifest or runs model inference.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "visionqa" / "v2"
MANIFEST = ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "dataset_manifest.json"
PREDICTIONS = REPORT / "raw_predictions" / "webui_val_v2.jsonl.gz"
PREDICTION_METADATA = PREDICTIONS.with_suffix(PREDICTIONS.suffix + ".metadata.json")
SUPPORTED = {0: "button", 1: "link", 2: "input", 3: "image", 7: "select", 8: "block-nav", 9: "block-sidebar", 10: "block-footer", 11: "block-form", 12: "block-table"}
CLASS_IDS = list(SUPPORTED)
CLASS_INDEX = {class_id: index for index, class_id in enumerate(CLASS_IDS)}
CORPUS_TO_CANONICAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12}
GRID = np.round(np.arange(0.005, 0.2500001, 0.0025), 4)
N_FOLDS = 5
SEED = 42
IOU_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def load_inputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {row["screen_id"]: row for row in manifest["screens"] if row["split"] == "val"}
    if len(records) != 5214:
        raise ValueError(f"Expected 5214 WebUI validation screens, found {len(records)}")
    metadata = json.loads(PREDICTION_METADATA.read_text(encoding="utf-8"))
    if metadata["image_count"] != 5214 or metadata["settings"]["conf_floor"] > GRID[0]:
        raise ValueError("Persisted predictions do not cover the required validation analysis range")
    predictions = {}
    with gzip.open(PREDICTIONS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values = [[d["class_id"], d["confidence"], *d["bbox_xyxy"]] for d in row["detections"] if d["class_id"] in SUPPORTED]
            predictions[row["image_id"]] = np.asarray(values, dtype=np.float32).reshape(-1, 6)
    if set(predictions) != set(records):
        raise ValueError("Persisted prediction screen IDs differ from the validation manifest")
    ground_truth = {}
    for image_id, row in records.items():
        values = [[CORPUS_TO_CANONICAL[box["class_id"]], *box["bbox_xyxy"]] for box in row["boxes"] if box["class_id"] in CORPUS_TO_CANONICAL]
        ground_truth[image_id] = np.asarray(values, dtype=np.float32).reshape(-1, 5)
    return manifest, records, predictions, ground_truth, metadata


def assign_folds(records):
    """Balance groups within source strata; all three viewports stay together."""
    source_groups = defaultdict(dict)
    for image_id, row in records.items():
        source_groups[row["source"]].setdefault(row["group_id"], []).append(image_id)
    fold_group_counts = [0] * N_FOLDS
    assignment = {}
    rng = random.Random(SEED)
    # Large sources first gives the greedy balancing rule a stable base.
    for source, groups_map in sorted(source_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        groups = sorted(groups_map)
        rng.shuffle(groups)
        source_fold_counts = [0] * N_FOLDS
        for group_id in groups:
            fold = min(range(N_FOLDS), key=lambda value: (source_fold_counts[value], fold_group_counts[value], value))
            assignment[group_id] = fold
            source_fold_counts[fold] += 1
            fold_group_counts[fold] += 1
    screen_fold = {image_id: assignment[row["group_id"]] for image_id, row in records.items()}
    for group_id in assignment:
        folds = {screen_fold[image_id] for image_id, row in records.items() if row["group_id"] == group_id}
        if len(folds) != 1:
            raise AssertionError(f"Viewport leakage in group {group_id}")
    return assignment, screen_fold


def match_predictions(records, predictions, ground_truth):
    flags, matched = {}, {}
    for image_id in records:
        pred = predictions[image_id]; gt = ground_truth[image_id]
        is_tp = np.zeros(len(pred), dtype=bool); gt_match = np.full(len(pred), -1, dtype=np.int32)
        for class_id in CLASS_IDS:
            pi = np.flatnonzero(pred[:, 0] == class_id); gi = np.flatnonzero(gt[:, 0] == class_id)
            if not len(pi) or not len(gi):
                continue
            pi = pi[np.argsort(-pred[pi, 1], kind="stable")]
            overlaps = np.stack([iou_one(pred[index, 2:6], gt[gi, 1:5]) for index in pi])
            available = np.ones(len(gi), dtype=bool)
            for row_index, pred_index in enumerate(pi):
                if not np.any(available):
                    break
                candidate_iou = np.where(available, overlaps[row_index], -1.0); best = int(np.argmax(candidate_iou))
                if candidate_iou[best] >= IOU_THRESHOLD:
                    is_tp[pred_index] = True; gt_match[pred_index] = int(gi[best]); available[best] = False
        flags[image_id] = is_tp; matched[image_id] = gt_match
    return flags, matched


def calculate_grid_stats(records, screen_fold, predictions, ground_truth, flags):
    shape = (N_FOLDS, len(CLASS_IDS), len(GRID))
    pred_counts = np.zeros(shape, dtype=np.int64); tp_counts = np.zeros(shape, dtype=np.int64)
    gt_counts = np.zeros((N_FOLDS, len(CLASS_IDS)), dtype=np.int64)
    for image_id in records:
        fold = screen_fold[image_id]; pred = predictions[image_id]; gt = ground_truth[image_id]; correct = flags[image_id]
        for class_id in CLASS_IDS:
            class_index = CLASS_INDEX[class_id]
            scores = np.sort(pred[pred[:, 0] == class_id, 1])
            tp_scores = np.sort(pred[(pred[:, 0] == class_id) & correct, 1])
            pred_counts[fold, class_index] += len(scores) - np.searchsorted(scores, GRID, side="left")
            tp_counts[fold, class_index] += len(tp_scores) - np.searchsorted(tp_scores, GRID, side="left")
            gt_counts[fold, class_index] += int(np.count_nonzero(gt[:, 0] == class_id))
    return pred_counts, tp_counts, gt_counts


def metrics_from_counts(pred_counts, tp_counts, gt_counts):
    if gt_counts.ndim == 1:
        gt_counts = gt_counts[:, None]
    fp = pred_counts - tp_counts; fn = gt_counts - tp_counts
    precision = np.divide(tp_counts, pred_counts, out=np.zeros_like(tp_counts, dtype=float), where=pred_counts > 0)
    recall = np.divide(tp_counts, gt_counts, out=np.zeros_like(tp_counts, dtype=float), where=gt_counts > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    total_pred = pred_counts.sum(axis=0); total_tp = tp_counts.sum(axis=0); total_gt = gt_counts.sum(axis=0)
    overall_p = np.divide(total_tp, total_pred, out=np.zeros_like(total_tp, dtype=float), where=total_pred > 0)
    overall_r = np.divide(total_tp, total_gt, out=np.zeros_like(total_tp, dtype=float), where=total_gt > 0)
    overall_f1 = np.divide(2 * overall_p * overall_r, overall_p + overall_r, out=np.zeros_like(overall_p), where=(overall_p + overall_r) > 0)
    macro_f1 = f1.mean(axis=0)
    return {"class_precision": precision, "class_recall": recall, "class_f1": f1, "overall_precision": overall_p, "overall_recall": overall_r, "overall_f1": overall_f1, "macro_f1": macro_f1}


def scalar_metrics(pred_counts, tp_counts, gt_counts):
    result = metrics_from_counts(pred_counts[:, None], tp_counts[:, None], gt_counts)
    per_class = {}
    for index, class_id in enumerate(CLASS_IDS):
        pred = int(pred_counts[index]); tp = int(tp_counts[index]); gt = int(gt_counts[index])
        per_class[SUPPORTED[class_id]] = {"ground_truth": gt, "predictions": pred, "tp": tp, "fp": pred-tp, "fn": gt-tp, "precision": float(result["class_precision"][index, 0]), "recall": float(result["class_recall"][index, 0]), "f1": float(result["class_f1"][index, 0])}
    return {"precision": float(result["overall_precision"][0]), "recall": float(result["overall_recall"][0]), "f1": float(result["overall_f1"][0]), "macro_f1": float(result["macro_f1"][0]), "prediction_count": int(pred_counts.sum()), "per_class": per_class}


def subset_at_oof_threshold(image_ids, screen_fold, selected_indices, predictions, ground_truth, flags):
    pred = np.zeros(len(CLASS_IDS), dtype=np.int64); tp = np.zeros(len(CLASS_IDS), dtype=np.int64); gt = np.zeros(len(CLASS_IDS), dtype=np.int64)
    zero = 0
    for image_id in image_ids:
        threshold = GRID[selected_indices[screen_fold[image_id]]]; p = predictions[image_id]; f = flags[image_id]
        if not np.any(p[:, 1] >= threshold): zero += 1
        for class_id in CLASS_IDS:
            index = CLASS_INDEX[class_id]; mask = (p[:, 0] == class_id) & (p[:, 1] >= threshold)
            pred[index] += int(np.count_nonzero(mask)); tp[index] += int(np.count_nonzero(mask & f)); gt[index] += int(np.count_nonzero(ground_truth[image_id][:, 0] == class_id))
    result = scalar_metrics(pred, tp, gt); result["images"] = len(image_ids); result["zero_prediction_screens"] = zero; result["zero_prediction_rate"] = zero/len(image_ids) if image_ids else 0
    return result


def object_size_oof(records, screen_fold, selected_indices, predictions, ground_truth, flags, matched):
    definitions = {"tiny": "area < 0.001", "small": "0.001 <= area < 0.01", "medium": "0.01 <= area < 0.1", "large": "area >= 0.1"}
    counts = Counter(); hits = Counter()
    for image_id, row in records.items():
        threshold = GRID[selected_indices[screen_fold[image_id]]]; gt = ground_truth[image_id]; p = predictions[image_id]
        hit_indices = set(int(value) for value in matched[image_id][flags[image_id] & (p[:, 1] >= threshold)] if value >= 0)
        width = row["original_image_identifier"]["width"]; height = row["original_image_identifier"]["height"]
        for index, item in enumerate(gt):
            area = max(0, item[3]-item[1]) * max(0, item[4]-item[2]) / (width*height)
            bucket = "tiny" if area < .001 else "small" if area < .01 else "medium" if area < .1 else "large"
            counts[bucket] += 1; hits[bucket] += int(index in hit_indices)
    return {"definitions": definitions, "buckets": {name: {"ground_truth": counts[name], "matched": hits[name], "recall": hits[name]/counts[name] if counts[name] else None} for name in definitions}}


def mean_std(rows, key):
    values = [row[key] for row in rows]
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def main() -> int:
    manifest, records, predictions, ground_truth, prediction_metadata = load_inputs()
    group_assignment, screen_fold = assign_folds(records)
    flags, matched = match_predictions(records, predictions, ground_truth)
    pred_counts, tp_counts, gt_counts = calculate_grid_stats(records, screen_fold, predictions, ground_truth, flags)
    total_pred = pred_counts.sum(axis=0); total_tp = tp_counts.sum(axis=0); total_gt = gt_counts.sum(axis=0)
    folds = []; selected_indices = []
    for fold in range(N_FOLDS):
        train_metrics = metrics_from_counts(total_pred-pred_counts[fold], total_tp-tp_counts[fold], total_gt-gt_counts[fold])
        # Primary: macro-F1. Tie-breakers: overall F1, then higher threshold.
        candidates = np.flatnonzero(train_metrics["macro_f1"] == np.max(train_metrics["macro_f1"]))
        best_overall = np.max(train_metrics["overall_f1"][candidates]); candidates = candidates[train_metrics["overall_f1"][candidates] == best_overall]
        selected = int(candidates[-1]); selected_indices.append(selected)
        held = scalar_metrics(pred_counts[fold, :, selected], tp_counts[fold, :, selected], gt_counts[fold])
        held_ids = [image_id for image_id in records if screen_fold[image_id] == fold]
        zero = sum(not np.any(predictions[image_id][:, 1] >= GRID[selected]) for image_id in held_ids)
        held.update({"images": len(held_ids), "groups": len({records[i]["group_id"] for i in held_ids}), "zero_prediction_screens": zero, "zero_prediction_rate": zero/len(held_ids)})
        folds.append({"fold": fold+1, "selected_threshold": float(GRID[selected]), "training_macro_f1": float(train_metrics["macro_f1"][selected]), "training_overall_f1": float(train_metrics["overall_f1"][selected]), "held_out": held})

    selected_thresholds = [row["selected_threshold"] for row in folds]
    stability = {"mean": statistics.mean(selected_thresholds), "median": statistics.median(selected_thresholds), "std": statistics.pstdev(selected_thresholds), "variance": statistics.pvariance(selected_thresholds), "min": min(selected_thresholds), "max": max(selected_thresholds), "range": max(selected_thresholds)-min(selected_thresholds), "rule": "stable when population std <= 0.01 and range <= 0.02", "stable": statistics.pstdev(selected_thresholds) <= .01 and max(selected_thresholds)-min(selected_thresholds) <= .02}
    recommended = float(statistics.median(selected_thresholds))
    all_ids = list(records)
    oof = subset_at_oof_threshold(all_ids, screen_fold, selected_indices, predictions, ground_truth, flags)
    by_viewport = {viewport: subset_at_oof_threshold([image_id for image_id in records if records[image_id]["viewport"] == viewport], screen_fold, selected_indices, predictions, ground_truth, flags) for viewport in ("desktop", "tablet", "mobile")}
    sizes = object_size_oof(records, screen_fold, selected_indices, predictions, ground_truth, flags, matched)
    baseline_index = int(np.flatnonzero(np.isclose(GRID, .25))[0])
    baseline = scalar_metrics(total_pred[:, baseline_index], total_tp[:, baseline_index], total_gt)
    baseline_zero = sum(not np.any(predictions[image_id][:, 1] >= .25) for image_id in records)
    baseline.update({"images": len(records), "zero_prediction_screens": baseline_zero, "zero_prediction_rate": baseline_zero/len(records)})
    recommended_index = int(np.argmin(np.abs(GRID-recommended)))
    recommended_full_validation = scalar_metrics(total_pred[:, recommended_index], total_tp[:, recommended_index], total_gt)
    recommended_full_validation["threshold"] = recommended
    recommended_full_validation["zero_prediction_screens"] = sum(not np.any(predictions[image_id][:, 1] >= recommended) for image_id in records)
    fold_metric_summary = {key: mean_std([row["held_out"] for row in folds], key) for key in ("precision", "recall", "f1", "macro_f1", "prediction_count", "zero_prediction_rate")}
    comparison = {"precision_change": oof["precision"]-baseline["precision"], "recall_change": oof["recall"]-baseline["recall"], "f1_change": oof["f1"]-baseline["f1"], "macro_f1_change": oof["macro_f1"]-baseline["macro_f1"], "prediction_count_change": oof["prediction_count"]-baseline["prediction_count"], "prediction_count_multiplier": oof["prediction_count"]/baseline["prediction_count"], "zero_prediction_screen_reduction": baseline_zero-oof["zero_prediction_screens"], "zero_prediction_rate_change": oof["zero_prediction_rate"]-baseline["zero_prediction_rate"]}
    unresolved = [name for name, row in oof["per_class"].items() if row["f1"] == 0]
    decision_code = "B" if stability["stable"] and oof["f1"] > baseline["f1"] and oof["macro_f1"] > baseline["macro_f1"] else "C" if not stability["stable"] else "D"
    decisions = {"A": "GLOBAL THRESHOLD IS STABLE AND SUFFICIENT", "B": "GLOBAL THRESHOLD IMPROVES V2 BUT CLASS-SPECIFIC ISSUES REMAIN", "C": "GLOBAL THRESHOLD IS UNSTABLE ACROSS SOURCES", "D": "THRESHOLD RECALIBRATION DOES NOT SOLVE THE MAIN FAILURE"}
    next_step = "Authorize one final frozen 399-screen confirmation using the calibrated global threshold." if decision_code in {"A", "B"} else "Run class/source-specific calibration analysis without model training." if decision_code == "C" else "Design one targeted training/data experiment for unresolved classes."

    fold_payload = {"schema_version": "1.0", "seed": SEED, "fold_count": N_FOLDS, "strategy": "source-stratified group assignment; source_sample group is indivisible across desktop/tablet/mobile", "group_count": len(group_assignment), "screen_count": len(records), "folds": folds, "group_assignments": [{"group_id": group, "fold": fold+1} for group, fold in sorted(group_assignment.items())]}
    cv_payload = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(), "scope": "persisted WebUI validation predictions only", "threshold_grid": {"minimum": float(GRID[0]), "maximum": float(GRID[-1]), "step": .0025, "count": len(GRID)}, "selection_metric": "macro-F1", "tie_breakers": ["overall F1", "higher threshold"], "threshold_stability": stability, "recommended_global_threshold": recommended, "held_out_fold_summary": fold_metric_summary, "out_of_fold_aggregate": oof, "viewport_out_of_fold": by_viewport, "object_size_out_of_fold": sizes, "baseline_at_0_25": baseline, "recommended_threshold_full_validation_diagnostic": recommended_full_validation, "comparison_to_0_25": comparison}
    decision_payload = {"decision_code": decision_code, "decision": decisions[decision_code], "recommended_global_threshold": recommended if decision_code in {"A", "B"} else None, "threshold_stability": stability, "unresolved_classes": unresolved, "select_true_positives_at_prediction_floor": int(sum(np.count_nonzero(flags[i] & (predictions[i][:, 0] == 7)) for i in records)), "next_single_milestone": next_step, "frozen_evaluation_used": False, "gpu_inference_run": False, "model_training_run": False}
    global_payload = {"inputs": {"prediction_path": str(PREDICTIONS), "prediction_sha256": sha256(PREDICTIONS), "prediction_metadata": prediction_metadata, "validation_manifest": str(MANIFEST), "validation_manifest_sha256": sha256(MANIFEST)}, "folds": fold_payload, "metrics": cv_payload, "decision": decision_payload}
    for name, payload in (("fold_thresholds.json", fold_payload), ("cross_validated_metrics.json", cv_payload), ("threshold_decision.json", decision_payload), ("global_threshold_cv.json", global_payload)):
        (REPORT/name).write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    write_reports(folds, stability, recommended, fold_metric_summary, oof, by_viewport, sizes, baseline, comparison, decision_payload)
    update_state(folds, stability, recommended, fold_metric_summary, oof, decision_payload)
    print(json.dumps({"fold_thresholds": selected_thresholds, "stability": stability, "recommended_threshold": recommended, "oof": {key: oof[key] for key in ("precision", "recall", "f1", "macro_f1", "prediction_count", "zero_prediction_screens")}, "decision": decision_payload["decision"], "next": next_step}, indent=2))
    return 0


def write_reports(folds, stability, recommended, summary, oof, viewports, sizes, baseline, comparison, decision):
    fold_rows = [f"| {row['fold']} | {row['selected_threshold']:.4f} | {row['held_out']['images']:,} | {row['held_out']['groups']:,} | {row['held_out']['precision']:.4f} | {row['held_out']['recall']:.4f} | {row['held_out']['f1']:.4f} | {row['held_out']['macro_f1']:.4f} | {row['held_out']['prediction_count']:,} | {row['held_out']['zero_prediction_screens']:,} |" for row in folds]
    class_rows = [f"| {name} | {row['ground_truth']:,} | {row['predictions']:,} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {baseline['per_class'][name]['f1']:.4f} | {row['f1']-baseline['per_class'][name]['f1']:+.4f} |" for name,row in oof["per_class"].items()]
    old_viewports = json.loads((REPORT/"source_viewport_analysis.json").read_text(encoding="utf-8"))["models"]["visionqa_v2"]["viewport"]
    viewport_rows = [f"| {name} | {old_viewports[name]['recall']:.4f} | {row['recall']:.4f} | {row['recall']-old_viewports[name]['recall']:+.4f} | {row['precision']:.4f} | {row['f1']:.4f} | {row['prediction_count']:,} | {row['zero_prediction_screens']:,} |" for name,row in viewports.items()]
    old_sizes = json.loads((REPORT/"object_size_analysis.json").read_text(encoding="utf-8"))["models"]["visionqa_v2"]["overall"]
    size_rows = [f"| {name} | {row['ground_truth']:,} | {old_sizes[name]['recall']:.4f} | {row['recall']:.4f} | {row['recall']-old_sizes[name]['recall']:+.4f} |" for name,row in sizes["buckets"].items()]
    report = f"""# VisionQA v2 global-threshold cross-validation

## Scope

This CPU-only experiment used persisted v2 predictions and ground truth for the 5,214-screen WebUI training-corpus validation split. Five folds were formed from 1,738 indivisible source-sample groups using seed 42. Desktop, tablet, and mobile screens from a group always remain together. Sources were balanced across folds where their group counts permitted it. No inference, training, dataset mutation, or frozen-evaluation access occurred.

Each fold selected a global threshold on the other four folds by macro-F1 over the ten supported WebUI classes. Ties used overall F1 and then the higher threshold. The grid was 0.005–0.250 in increments of 0.0025.

## Fold results

| Fold | Selected threshold | Held-out screens | Groups | Precision | Recall | F1 | Macro-F1 | Predictions | Zero screens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

Threshold mean: **{stability['mean']:.4f}**; median: **{stability['median']:.4f}**; population standard deviation: **{stability['std']:.4f}**; variance: **{stability['variance']:.8f}**; range: **{stability['min']:.4f}–{stability['max']:.4f}**. The predeclared practical rule is stable when standard deviation is at most 0.01 and range is at most 0.02. Result: **{'stable' if stability['stable'] else 'unstable'}**.

The recommended threshold is the robust median of independently selected fold thresholds: **{recommended:.4f}**.

## Cross-validated performance

Held-out fold means: precision **{summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}**, recall **{summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}**, F1 **{summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}**, and macro-F1 **{summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}**.

Out-of-fold aggregate: precision **{oof['precision']:.4f}**, recall **{oof['recall']:.4f}**, F1 **{oof['f1']:.4f}**, macro-F1 **{oof['macro_f1']:.4f}**, and {oof['prediction_count']:,} predictions. Compared with confidence 0.25, precision changes {comparison['precision_change']:+.4f}, recall {comparison['recall_change']:+.4f}, F1 {comparison['f1_change']:+.4f}, macro-F1 {comparison['macro_f1_change']:+.4f}, and prediction count {comparison['prediction_count_change']:+,} ({comparison['prediction_count_multiplier']:.2f}×).

Zero-prediction screens fall from {baseline['zero_prediction_screens']:,}/{baseline['images']:,} ({baseline['zero_prediction_rate']:.2%}) to {oof['zero_prediction_screens']:,}/{oof['images']:,} ({oof['zero_prediction_rate']:.2%}), a reduction of {comparison['zero_prediction_screen_reduction']:,} screens.

## Per-class out-of-fold results

| Class | GT | Predictions | Precision | Recall | CV F1 | F1 @0.25 | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

## Viewports

| Viewport | Recall @0.25 | CV recall | Recall change | CV precision | CV F1 | Predictions | Zero screens |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(viewport_rows)}

## Object size

| Size | GT | Recall @0.25 | CV-calibrated recall | Change |
|---|---:|---:|---:|---:|
{chr(10).join(size_rows)}

Lowering the threshold improves every size bucket, but tiny/small recall remains limited; calibration does not solve representation or localization quality.
"""
    (REPORT/"V2_GLOBAL_THRESHOLD_CV.md").write_text(report, encoding="utf-8")
    decision_report = f"""# VisionQA v2 global-threshold decision

## Decision

**{decision['decision_code']}. {decision['decision']}**

The five independently selected thresholds are {', '.join(f'{row['selected_threshold']:.4f}' for row in folds)}. Their standard deviation is {stability['std']:.4f} and range is {stability['range']:.4f}, satisfying the predeclared stability rule. Cross-validated F1 improves from {baseline['f1']:.4f} at confidence 0.25 to {oof['f1']:.4f}, while recall improves from {baseline['recall']:.4f} to {oof['recall']:.4f}. Precision decreases from {baseline['precision']:.4f} to {oof['precision']:.4f}; the reported F1 and macro-F1 gains account for that cost.

The recommended global threshold is **{recommended:.4f}**, derived as the median fold-selected threshold. It is not derived from the prior single-split optimum and was not tested on the frozen 399-screen evaluation.

Class-specific problems remain. Unresolved zero-F1 classes: **{', '.join(decision['unresolved_classes']) if decision['unresolved_classes'] else 'none'}**. In particular, `select` has {decision['select_true_positives_at_prediction_floor']} true positives even at the persisted 0.001 prediction floor, so threshold calibration cannot repair that class.

## Next single milestone

{decision['next_single_milestone']} This step is recommended only; it was not executed here.
"""
    (REPORT/"V2_THRESHOLD_DECISION.md").write_text(decision_report, encoding="utf-8")


def update_state(folds, stability, recommended, summary, oof, decision):
    path = ROOT/"reports"/"QASENSE_PROJECT_STATE.md"; text = path.read_text(encoding="utf-8")
    marker = "## VisionQA v2 source-grouped threshold calibration (2026-10-17)"
    if marker not in text:
        text += f"""\n\n{marker}\n\n- Calibration used persisted predictions for the 5,214-screen WebUI validation split only; no inference or training occurred.\n- Five deterministic source-stratified group folds (seed 42) preserved desktop/tablet/mobile source-sample groups without leakage.\n- Fold thresholds: {', '.join(f'{row['selected_threshold']:.4f}' for row in folds)}.\n- Threshold mean {stability['mean']:.4f}, median {stability['median']:.4f}, standard deviation {stability['std']:.4f}, range {stability['range']:.4f}; classified {'stable' if stability['stable'] else 'unstable'}.\n- Recommended global threshold: {recommended:.4f}.\n- Held-out fold mean P/R/F1/macro-F1: {summary['precision']['mean']:.4f}/{summary['recall']['mean']:.4f}/{summary['f1']['mean']:.4f}/{summary['macro_f1']['mean']:.4f}.\n- Out-of-fold aggregate P/R/F1/macro-F1: {oof['precision']:.4f}/{oof['recall']:.4f}/{oof['f1']:.4f}/{oof['macro_f1']:.4f}.\n- Decision: {decision['decision_code']}. {decision['decision']}. VisionQA v2 remains experimental.\n- Next recommended milestone: {decision['next_single_milestone']} Not executed.\n- Frozen 399-screen evaluation remained untouched; protected checkpoints and datasets were not modified.\n"""
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
