"""Diagnose VisionQA v2 on WebUI validation without touching frozen evaluation data."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "visionqa" / "v2"
MANIFEST = ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "dataset_manifest.json"
PREDICTIONS = {
    "visionqa_v1": REPORT / "raw_predictions" / "webui_val_v1.jsonl.gz",
    "visionqa_v2": REPORT / "raw_predictions" / "webui_val_v2.jsonl.gz",
}
SUPPORTED = {0: "button", 1: "link", 2: "input", 3: "image", 7: "select", 8: "block-nav", 9: "block-sidebar", 10: "block-footer", 11: "block-form", 12: "block-table"}
CORPUS_TO_CANONICAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12}
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
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
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-12)


def load_inputs():
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {row["screen_id"]: row for row in raw["screens"]}
    val_ids = sorted(key for key, row in rows.items() if row["split"] == "val")
    if len(val_ids) != 5214:
        raise ValueError(f"Expected 5214 validation images, found {len(val_ids)}")
    ground_truth = {}
    for image_id in val_ids:
        row = rows[image_id]
        # The curated corpus manifest retains its contiguous 10-class schema;
        # predictions use the canonical 14-class product IDs.
        ground_truth[image_id] = np.asarray(
            [[CORPUS_TO_CANONICAL[box["class_id"]], *box["bbox_xyxy"]] for box in row["boxes"] if box["class_id"] in CORPUS_TO_CANONICAL], dtype=np.float32
        ).reshape(-1, 5)
    predictions = {}
    for model, path in PREDICTIONS.items():
        result = {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                detections = [[d["class_id"], d["confidence"], *d["bbox_xyxy"]] for d in row["detections"] if d["class_id"] in SUPPORTED]
                result[row["image_id"]] = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        if set(result) != set(val_ids):
            raise ValueError(f"{model}: prediction IDs differ from validation manifest")
        predictions[model] = result
    return raw, rows, val_ids, ground_truth, predictions


def match_predictions(val_ids, ground_truth, predictions):
    """Score-descending one-to-one matching at IoU 0.5, preserving low-floor predictions."""
    flags, matched = {}, {}
    for image_id in val_ids:
        gt = ground_truth[image_id]; pred = predictions[image_id]
        is_tp = np.zeros(len(pred), dtype=bool)
        gt_match = np.full(len(pred), -1, dtype=np.int32)
        for class_id in SUPPORTED:
            pi = np.flatnonzero(pred[:, 0] == class_id)
            gi = np.flatnonzero(gt[:, 0] == class_id)
            if not len(pi) or not len(gi):
                continue
            pi = pi[np.argsort(-pred[pi, 1], kind="stable")]
            overlaps = np.stack([iou_one(pred[pidx, 2:6], gt[gi, 1:5]) for pidx in pi])
            available = np.ones(len(gi), dtype=bool)
            for row_index, pidx in enumerate(pi):
                if not np.any(available):
                    break
                candidate_iou = np.where(available, overlaps[row_index], -1.0)
                best_pos = int(np.argmax(candidate_iou))
                if float(candidate_iou[best_pos]) >= IOU_THRESHOLD:
                    gidx = int(gi[best_pos]); is_tp[pidx] = True; gt_match[pidx] = gidx; available[best_pos] = False
        flags[image_id] = is_tp; matched[image_id] = gt_match
    return flags, matched


def safe_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def aggregate(image_ids, gt, preds, flags, threshold):
    class_rows = {}
    total_tp = total_fp = total_gt = total_pred = 0
    counts_per_image = []
    for class_id, name in SUPPORTED.items():
        tp = fp = gt_count = pred_count = 0
        for image_id in image_ids:
            p = preds[image_id]; keep = (p[:, 0] == class_id) & (p[:, 1] >= threshold)
            current_tp = int(np.count_nonzero(flags[image_id] & keep)); current_pred = int(np.count_nonzero(keep))
            tp += current_tp; fp += current_pred - current_tp; pred_count += current_pred
            gt_count += int(np.count_nonzero(gt[image_id][:, 0] == class_id))
        class_rows[name] = {**safe_metrics(tp, fp, gt_count - tp), "ground_truth": gt_count, "predictions": pred_count}
        total_tp += tp; total_fp += fp; total_gt += gt_count; total_pred += pred_count
    for image_id in image_ids:
        counts_per_image.append(int(np.count_nonzero(preds[image_id][:, 1] >= threshold)))
    result = safe_metrics(total_tp, total_fp, total_gt - total_tp)
    result.update({
        "macro_f1": statistics.mean(row["f1"] for row in class_rows.values()),
        "prediction_count": total_pred,
        "images_zero_predictions": sum(value == 0 for value in counts_per_image),
        "images_one_prediction": sum(value == 1 for value in counts_per_image),
        "images_many_predictions_ge_20": sum(value >= 20 for value in counts_per_image),
        "predictions_per_image_mean": statistics.mean(counts_per_image),
        "predictions_per_image_median": statistics.median(counts_per_image),
        "predictions_per_image_p95": float(np.percentile(counts_per_image, 95)),
        "per_class": class_rows,
    })
    return result


def quantiles(values):
    if not len(values):
        return {key: None for key in ("mean", "median", "p10", "p25", "p50", "p75", "p90", "p95")}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90, 95)},
    }


def confidence_report(val_ids, preds, flags):
    output = {}
    for scope, class_id in [("overall", None), *[(name, cid) for cid, name in SUPPORTED.items()]]:
        all_values = []; tp_values = []; fp_values = []
        for image_id in val_ids:
            p = preds[image_id]; mask = np.ones(len(p), dtype=bool) if class_id is None else p[:, 0] == class_id
            values = p[mask, 1]; correct = flags[image_id][mask]
            all_values.extend(values.tolist()); tp_values.extend(values[correct].tolist()); fp_values.extend(values[~correct].tolist())
        output[scope] = {"all": quantiles(all_values), "true_positive": quantiles(tp_values), "false_positive": quantiles(fp_values)}
    return output


def best_threshold_and_curve(val_ids, gt, preds, flags, class_id=None):
    scores = []; correct = []; gt_count = 0
    for image_id in val_ids:
        p = preds[image_id]; mask = np.ones(len(p), dtype=bool) if class_id is None else p[:, 0] == class_id
        scores.extend(p[mask, 1].tolist()); correct.extend(flags[image_id][mask].tolist())
        gt_count += len(gt[image_id]) if class_id is None else int(np.count_nonzero(gt[image_id][:, 0] == class_id))
    scores = np.asarray(scores, dtype=np.float64); correct = np.asarray(correct, dtype=np.int8)
    order = np.argsort(-scores, kind="stable"); scores = scores[order]; correct = correct[order]
    tp = np.cumsum(correct); fp = np.arange(1, len(correct) + 1) - tp
    precision = tp / np.maximum(tp + fp, 1); recall = tp / max(gt_count, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1)) if len(f1) else 0
    reasonable = np.flatnonzero(precision >= 0.10)
    reasonable_best = int(reasonable[np.argmax(recall[reasonable])]) if len(reasonable) else None
    sample_indices = np.unique(np.linspace(0, max(len(scores) - 1, 0), min(201, max(len(scores), 1))).astype(int)) if len(scores) else []
    return {
        "ground_truth": gt_count,
        "best_f1": {"threshold": float(scores[best]) if len(scores) else None, "precision": float(precision[best]) if len(scores) else 0.0, "recall": float(recall[best]) if len(scores) else 0.0, "f1": float(f1[best]) if len(scores) else 0.0},
        "maximum_recall_at_precision_ge_0_10": None if reasonable_best is None else {"threshold": float(scores[reasonable_best]), "precision": float(precision[reasonable_best]), "recall": float(recall[reasonable_best]), "f1": float(f1[reasonable_best])},
        "curve": [{"threshold": float(scores[i]), "precision": float(precision[i]), "recall": float(recall[i])} for i in sample_indices],
    }


def calibration(val_ids, preds, flags):
    bins = np.linspace(0, 1, 11); rows = []; total = 0; weighted = 0.0
    scores = np.concatenate([preds[i][:, 1] for i in val_ids]); correct = np.concatenate([flags[i].astype(np.float32) for i in val_ids])
    for index in range(10):
        mask = (scores >= bins[index]) & (scores < bins[index + 1] if index < 9 else scores <= bins[index + 1])
        count = int(np.count_nonzero(mask))
        mean_conf = float(np.mean(scores[mask])) if count else None; empirical = float(np.mean(correct[mask])) if count else None
        rows.append({"lower": float(bins[index]), "upper": float(bins[index + 1]), "count": count, "mean_confidence": mean_conf, "empirical_precision": empirical})
        if count: weighted += count * abs(mean_conf - empirical); total += count
    return {"bins": rows, "ece": weighted / total if total else None, "definition": "Detection ECE over supported-class post-NMS predictions at confidence >=0.001; correctness is class-correct IoU>=0.5 one-to-one match."}


def object_size_report(val_ids, rows, gt, preds, flags, threshold=0.25):
    definitions = {"tiny": "area < 0.001", "small": "0.001 <= area < 0.01", "medium": "0.01 <= area < 0.1", "large": "area >= 0.1"}
    counts = Counter(); class_counts = defaultdict(Counter); hits = Counter(); class_hits = defaultdict(Counter)
    for image_id in val_ids:
        width = rows[image_id]["original_image_identifier"]["width"]; height = rows[image_id]["original_image_identifier"]["height"]
        current_gt = gt[image_id]
        matched_gt = set(int(v) for v in np.asarray([], dtype=int))
        p = preds[image_id]; eligible = p[:, 1] >= threshold
        # matched indices from score-descending flags are recomputed through the same matching order
        used = set()
        for pidx in np.flatnonzero(eligible)[np.argsort(-p[eligible, 1], kind="stable")] if np.any(eligible) else []:
            candidates = np.flatnonzero(current_gt[:, 0] == p[pidx, 0]); candidates = np.asarray([x for x in candidates if int(x) not in used])
            if len(candidates):
                ov = iou_one(p[pidx, 2:6], current_gt[candidates, 1:5]); pos = int(np.argmax(ov))
                if ov[pos] >= IOU_THRESHOLD: used.add(int(candidates[pos]))
        matched_gt = used
        for gidx, item in enumerate(current_gt):
            area = max(0, item[3] - item[1]) * max(0, item[4] - item[2]) / (width * height)
            bucket = "tiny" if area < .001 else "small" if area < .01 else "medium" if area < .1 else "large"
            name = SUPPORTED[int(item[0])]; counts[bucket] += 1; class_counts[name][bucket] += 1
            if gidx in matched_gt: hits[bucket] += 1; class_hits[name][bucket] += 1
    return {
        "threshold": threshold, "bucket_definitions_normalized_area": definitions,
        "overall": {bucket: {"ground_truth": counts[bucket], "matched": hits[bucket], "recall": hits[bucket] / counts[bucket] if counts[bucket] else None} for bucket in definitions},
        "per_class": {name: {bucket: {"ground_truth": class_counts[name][bucket], "recall": class_hits[name][bucket] / class_counts[name][bucket] if class_counts[name][bucket] else None} for bucket in definitions} for name in SUPPORTED.values()},
    }


def subset_reports(val_ids, rows, gt, preds, flags, threshold=0.25):
    densities = np.asarray([len(gt[i]) for i in val_ids]); q25, q50, q75 = [int(x) for x in np.percentile(densities, [25, 50, 75])]
    density_sets = {
        f"low_le_{q25}": [i for i in val_ids if len(gt[i]) <= q25],
        f"medium_{q25+1}_{q50}": [i for i in val_ids if q25 < len(gt[i]) <= q50],
        f"high_{q50+1}_{q75}": [i for i in val_ids if q50 < len(gt[i]) <= q75],
        f"very_high_gt_{q75}": [i for i in val_ids if len(gt[i]) > q75],
    }
    density = {key: {"images": len(ids), **aggregate(ids, gt, preds, flags, threshold)} for key, ids in density_sets.items()}
    viewport = {}
    for name in ("desktop", "tablet", "mobile"):
        ids = [i for i in val_ids if rows[i]["viewport"] == name]
        viewport[name] = {"images": len(ids), **aggregate(ids, gt, preds, flags, threshold)}
    source = {}
    for name in sorted({rows[i]["source"] for i in val_ids}):
        ids = [i for i in val_ids if rows[i]["source"] == name]
        if len(ids) >= 30:
            source[name] = {"images": len(ids), **aggregate(ids, gt, preds, flags, threshold)}
    return {"density_quantiles": {"q25": q25, "q50": q50, "q75": q75}, "density": density, "viewport": viewport, "source_min_30_images": source}


def failure_modes(val_ids, gt, preds, flags, threshold=0.25):
    confusion = Counter(); localization = Counter(); by_class = defaultdict(Counter)
    for image_id in val_ids:
        current_gt = gt[image_id]; p = preds[image_id]; f = flags[image_id]
        matched = set()
        for pidx in np.flatnonzero((p[:, 1] >= threshold) & f):
            same = np.flatnonzero(current_gt[:, 0] == p[pidx, 0]); overlaps = iou_one(p[pidx, 2:6], current_gt[same, 1:5])
            if len(overlaps) and overlaps.max() >= IOU_THRESHOLD: matched.add(int(same[int(np.argmax(overlaps))]))
        for gidx, item in enumerate(current_gt):
            if gidx in matched: continue
            name = SUPPORTED[int(item[0])]; low_same = p[(p[:, 0] == item[0]) & (p[:, 1] < threshold)]
            high_same = p[(p[:, 0] == item[0]) & (p[:, 1] >= threshold)]
            high_other = p[(p[:, 0] != item[0]) & (p[:, 1] >= threshold)]
            if len(low_same) and np.max(iou_one(item[1:5], low_same[:, 2:6])) >= IOU_THRESHOLD:
                mode = "correct_class_below_confidence_threshold"
            elif len(high_same) and np.max(iou_one(item[1:5], high_same[:, 2:6])) >= 0.1:
                mode = "correct_class_localization_below_iou_0_5"
            elif len(high_other) and np.max(iou_one(item[1:5], high_other[:, 2:6])) >= IOU_THRESHOLD:
                mode = "wrong_class_overlap"
                overlaps = iou_one(item[1:5], high_other[:, 2:6]); pred_name = SUPPORTED.get(int(high_other[int(np.argmax(overlaps)), 0]), f"class-{int(high_other[int(np.argmax(overlaps)), 0])}")
                confusion[f"{name} -> {pred_name}"] += 1
            else:
                all_high = p[p[:, 1] >= threshold]
                mode = "no_nearby_prediction" if not len(all_high) or np.max(iou_one(item[1:5], all_high[:, 2:6])) < 0.1 else "nearby_incompatible_or_duplicate_prediction"
            localization[mode] += 1; by_class[name][mode] += 1
    return {"threshold": threshold, "miss_categories": dict(localization), "per_class_miss_categories": {key: dict(value) for key, value in by_class.items()}, "wrong_class_pairs": [{"pair": key, "count": value} for key, value in confusion.most_common()]}


def distribution_summary(raw_manifest):
    output = {}
    for split in ("train", "val"):
        records = [row for row in raw_manifest["screens"] if row["split"] == split]
        density = [row["valid_mapped_box_count"] for row in records]
        class_counts = Counter(); size_counts = Counter(); sources = Counter(row["source"] for row in records); viewports = Counter(row["viewport"] for row in records)
        for row in records:
            width = row["original_image_identifier"]["width"]; height = row["original_image_identifier"]["height"]
            class_counts.update(row["class_counts"])
            for box in row["boxes"]:
                x1, y1, x2, y2 = box["bbox_xyxy"]; area = max(0, x2-x1)*max(0,y2-y1)/(width*height)
                size_counts["tiny" if area < .001 else "small" if area < .01 else "medium" if area < .1 else "large"] += 1
        output[split] = {"screens": len(records), "groups": len({r["group_id"] for r in records}), "sources": dict(sources), "viewports": dict(viewports), "class_boxes": dict(class_counts), "object_sizes": dict(size_counts), "negative_screens": sum(v == 0 for v in density), "density": quantiles(density)}
    return output


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    raw_manifest, rows, val_ids, gt, all_predictions = load_inputs()
    all_flags = {}; all_matches = {}
    threshold_output = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(), "scope": "WebUI training-corpus validation only", "image_count": len(val_ids), "iou_threshold": IOU_THRESHOLD, "analysis_thresholds": THRESHOLDS, "models": {}}
    confidence_output = {"scope": threshold_output["scope"], "raw_prediction_floor": 0.001, "models": {}}
    class_error_output = {"scope": threshold_output["scope"], "models": {}}
    size_output = {"scope": threshold_output["scope"], "size_definitions": "normalized box area", "models": {}}
    source_output = {"scope": threshold_output["scope"], "models": {}, "train_validation_distribution": distribution_summary(raw_manifest)}
    calibration_output = {"scope": threshold_output["scope"], "models": {}}
    for model, predictions in all_predictions.items():
        flags, matches = match_predictions(val_ids, gt, predictions); all_flags[model] = flags; all_matches[model] = matches
        threshold_output["models"][model] = {
            "threshold_metrics": {f"{threshold:.2f}": aggregate(val_ids, gt, predictions, flags, threshold) for threshold in THRESHOLDS},
            "precision_recall": {"overall": best_threshold_and_curve(val_ids, gt, predictions, flags), **{name: best_threshold_and_curve(val_ids, gt, predictions, flags, cid) for cid, name in SUPPORTED.items()}},
        }
        confidence_output["models"][model] = confidence_report(val_ids, predictions, flags)
        class_error_output["models"][model] = failure_modes(val_ids, gt, predictions, flags)
        size_output["models"][model] = object_size_report(val_ids, rows, gt, predictions, flags)
        source_output["models"][model] = subset_reports(val_ids, rows, gt, predictions, flags)
        calibration_output["models"][model] = calibration(val_ids, predictions, flags)

    # Comparative rankings and fixed, evidence-oriented root-cause statuses.
    source_comparison = []
    common_sources = set(source_output["models"]["visionqa_v1"]["source_min_30_images"]) & set(source_output["models"]["visionqa_v2"]["source_min_30_images"])
    for source in common_sources:
        v1 = source_output["models"]["visionqa_v1"]["source_min_30_images"][source]; v2 = source_output["models"]["visionqa_v2"]["source_min_30_images"][source]
        source_comparison.append({"source": source, "images": v1["images"], "v1_f1": v1["f1"], "v2_f1": v2["f1"], "f1_change": v2["f1"]-v1["f1"], "v1_recall": v1["recall"], "v2_recall": v2["recall"]})
    source_output["source_comparison_at_0_25"] = sorted(source_comparison, key=lambda row: row["f1_change"])

    outputs = {
        "validation_threshold_analysis.json": threshold_output,
        "confidence_distributions.json": confidence_output,
        "class_error_analysis.json": class_error_output,
        "object_size_analysis.json": size_output,
        "source_viewport_analysis.json": source_output,
        "calibration_analysis.json": calibration_output,
    }
    for name, payload in outputs.items():
        (REPORT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    create_plots(threshold_output, confidence_output)
    print(json.dumps({
        "validation_images": len(val_ids),
        "manifest_sha256": sha256(MANIFEST),
        "v1_at_0_25": threshold_output["models"]["visionqa_v1"]["threshold_metrics"]["0.25"],
        "v2_at_0_25": threshold_output["models"]["visionqa_v2"]["threshold_metrics"]["0.25"],
        "v1_best": threshold_output["models"]["visionqa_v1"]["precision_recall"]["overall"]["best_f1"],
        "v2_best": threshold_output["models"]["visionqa_v2"]["precision_recall"]["overall"]["best_f1"],
        "v1_ece": calibration_output["models"]["visionqa_v1"]["ece"],
        "v2_ece": calibration_output["models"]["visionqa_v2"]["ece"],
    }, indent=2))
    return 0


def create_plots(threshold_output, confidence_output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = REPORT / "figures"; figure_dir.mkdir(exist_ok=True)
    for model in ("visionqa_v1", "visionqa_v2"):
        curve = threshold_output["models"][model]["precision_recall"]["overall"]["curve"]
        plt.plot([r["recall"] for r in curve], [r["precision"] for r in curve], label=model)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("WebUI validation precision-recall (IoU 0.5)"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figure_dir / "validation_precision_recall.png", dpi=160); plt.close()
    figure, axes = plt.subplots(5, 2, figsize=(11, 18), sharex=False, sharey=False)
    for axis, class_name in zip(axes.flat, SUPPORTED.values(), strict=True):
        for model in ("visionqa_v1", "visionqa_v2"):
            curve = threshold_output["models"][model]["precision_recall"][class_name]["curve"]
            axis.plot([r["recall"] for r in curve], [r["precision"] for r in curve], label=model)
        axis.set_title(class_name); axis.set_xlabel("Recall"); axis.set_ylabel("Precision"); axis.grid(alpha=.25)
    axes.flat[0].legend(); figure.suptitle("WebUI validation per-class precision-recall (IoU 0.5)"); figure.tight_layout(); figure.savefig(figure_dir / "validation_precision_recall_per_class.png", dpi=160); plt.close(figure)
    thresholds = THRESHOLDS
    for metric in ("precision", "recall", "f1", "macro_f1"):
        for model in ("visionqa_v1", "visionqa_v2"):
            values = [threshold_output["models"][model]["threshold_metrics"][f"{t:.2f}"][metric] for t in thresholds]
            plt.plot(thresholds, values, marker="o", label=model)
        plt.xlabel("Confidence threshold"); plt.ylabel(metric); plt.title(f"WebUI validation {metric} sensitivity"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figure_dir / f"threshold_{metric}.png", dpi=160); plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
