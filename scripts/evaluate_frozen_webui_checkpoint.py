"""Score one fixed checkpoint prediction set on the frozen 399-screen WebUI slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_webui_transfer import load, match_screen, metrics


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--baseline-predictions", type=Path)
    args = parser.parse_args()
    manifest = load(ROOT / "reports" / "visionqa" / "webui_eval_manifest.json")
    expected_manifest_hash = "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"
    manifest_content = dict(manifest); recorded_manifest_hash = manifest_content.pop("manifest_content_sha256")
    if recorded_manifest_hash != expected_manifest_hash or canonical_hash(manifest_content) != expected_manifest_hash:
        raise RuntimeError("Frozen WebUI manifest hash changed")
    results, latencies, baseline_results = [], [], []
    checkpoint_path = checkpoint_hash = None
    for screen in manifest["screens"]:
        prediction = load(args.predictions / f"{screen['screen_id']}.json")
        if float(prediction["model"]["confidence_threshold"]) != args.confidence_threshold:
            raise RuntimeError(f"Prediction threshold mismatch for {screen['screen_id']}")
        if checkpoint_path is None:
            checkpoint_path = Path(prediction["model"]["checkpoint"])
            checkpoint_hash = file_sha256(checkpoint_path)
        latencies.append(float(prediction["inference_time_ms"]))
        results.append({"screen": screen, "prediction_summary": {"detection_count": prediction["detection_count"], "counts_by_class": prediction["counts_by_class"], "inference_time_ms": prediction["inference_time_ms"]}, "evaluation": match_screen(screen, prediction)})
        if args.baseline_predictions:
            baseline_prediction = load(args.baseline_predictions / f"{screen['screen_id']}.json")
            baseline_results.append({"screen": screen, "prediction_summary": {"detection_count": baseline_prediction["detection_count"], "counts_by_class": baseline_prediction["counts_by_class"], "inference_time_ms": baseline_prediction["inference_time_ms"]}, "evaluation": match_screen(screen, baseline_prediction)})
    score = metrics(results)
    error_counts, examples = Counter(), defaultdict(list)
    groups = defaultdict(list)
    for row in results:
        screen, evaluation = row["screen"], row["evaluation"]
        groups[screen["source_sample_id"]].append(row)
        area_total = screen["image_identifier"]["width"] * screen["image_identifier"]["height"]
        for match in evaluation["matches"]:
            if match["confidence"] < 0.4: error_counts["low_confidence_correct"] += 1; examples["low_confidence_correct"].append(screen["screen_id"])
        for gi in evaluation["false_negative_indices"]:
            box = screen["valid_ground_truth_boxes"][gi]["bbox_xyxy"]
            area = (box[2]-box[0])*(box[3]-box[1])/area_total
            bucket = "missed_small_element" if area < 0.0025 else "visual_domain_shift"
            error_counts[bucket] += 1; examples[bucket].append(screen["screen_id"])
            if evaluation["ground_truth_count"] >= 20: error_counts["dense_layout_miss"] += 1; examples["dense_layout_miss"].append(screen["screen_id"])
        error_counts["wrong_semantic_class"] += len(evaluation["wrong_class"])
        error_counts["duplicate_detection"] += len(evaluation["duplicate_prediction_indices"])
    jaccards, responsive_misses = [], 0
    for rows in groups.values():
        detected = [set(row["prediction_summary"]["counts_by_class"]) for row in rows]
        union = set.union(*detected); intersection = set.intersection(*detected)
        jaccards.append(len(intersection)/len(union) if union else 1.0)
        states = defaultdict(dict)
        for row in rows:
            matched = {item["class_name"] for item in row["evaluation"]["matches"]}
            for name in row["screen"]["mapped_classes"]: states[name][row["screen"]["viewport"]] = name in matched
        responsive_misses += sum(len(value)==3 and 0 < sum(value.values()) < 3 for value in states.values())
    latency_sorted = sorted(latencies)
    runtime = {"mean_ms": statistics.mean(latencies), "median_ms": statistics.median(latencies), "p95_ms": latency_sorted[max(0, int(len(latency_sorted)*0.95)-1)], "screens": len(latencies)}
    viewport_metrics = {viewport: metrics([row for row in results if row["screen"]["viewport"] == viewport]) for viewport in ("desktop", "tablet", "mobile")}
    size_definitions = {"tiny": "area < 0.001", "small": "0.001 <= area < 0.01", "medium": "0.01 <= area < 0.1", "large": "area >= 0.1"}
    size_counts, size_hits = Counter(), Counter()
    for row in results:
        screen, evaluation = row["screen"], row["evaluation"]
        matched_gt = {item["ground_truth_index"] for item in evaluation["matches"]}
        image_area = screen["image_identifier"]["width"] * screen["image_identifier"]["height"]
        for index, ground in enumerate(screen["valid_ground_truth_boxes"]):
            x1, y1, x2, y2 = ground["bbox_xyxy"]; area = (x2-x1)*(y2-y1)/image_area
            bucket = "tiny" if area < .001 else "small" if area < .01 else "medium" if area < .1 else "large"
            size_counts[bucket] += 1; size_hits[bucket] += int(index in matched_gt)
    object_size = {"definitions": size_definitions, "buckets": {name: {"ground_truth": size_counts[name], "matched": size_hits[name], "recall": size_hits[name]/size_counts[name] if size_counts[name] else None} for name in size_definitions}}
    zero_screens = sum(row["prediction_summary"]["detection_count"] == 0 for row in results)
    prediction_count = sum(row["evaluation"]["scorable_prediction_count"] for row in results)
    source_group_summary = None
    baseline_summary = None
    if baseline_results:
        baseline_score = metrics(baseline_results)
        baseline_zero = sum(row["prediction_summary"]["detection_count"] == 0 for row in baseline_results)
        baseline_count = sum(row["evaluation"]["scorable_prediction_count"] for row in baseline_results)
        baseline_summary = {"confidence_threshold": 0.25, "metrics": baseline_score, "prediction_count": baseline_count, "zero_prediction_screens": baseline_zero, "zero_prediction_rate": baseline_zero/len(baseline_results)}
        current_by_group, baseline_by_group = defaultdict(list), defaultdict(list)
        for row in results: current_by_group[row["screen"]["source_sample_id"]].append(row)
        for row in baseline_results: baseline_by_group[row["screen"]["source_sample_id"]].append(row)
        group_changes = []
        for group_id in sorted(current_by_group):
            current = metrics(current_by_group[group_id]); old = metrics(baseline_by_group[group_id])
            group_changes.append({"source_sample_id": group_id, "source": current_by_group[group_id][0]["screen"]["source_name"], "tp_at_0_25": old["tp"], "tp_at_calibrated": current["tp"], "tp_change": current["tp"]-old["tp"], "f1_at_0_25": old["f1"], "f1_at_calibrated": current["f1"], "f1_change": current["f1"]-old["f1"]})
        added_tp = sum(max(0, row["tp_change"]) for row in group_changes)
        top_ten_added = sum(max(0, row["tp_change"]) for row in sorted(group_changes, key=lambda value: value["tp_change"], reverse=True)[:10])
        source_group_summary = {"groups_total": len(group_changes), "groups_tp_improved": sum(row["tp_change"] > 0 for row in group_changes), "groups_tp_unchanged": sum(row["tp_change"] == 0 for row in group_changes), "groups_tp_regressed": sum(row["tp_change"] < 0 for row in group_changes), "added_true_positives": added_tp, "top_10_groups_share_of_added_true_positives": top_ten_added/added_tp if added_tp else None, "groups": group_changes}
    payload = {"schema_version": "1.0", "evaluated_at": datetime.now(timezone.utc).isoformat(), "manifest_content_sha256": recorded_manifest_hash, "screen_count": len(results), "sample_group_count": len(groups), "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash, "confidence_threshold": args.confidence_threshold, "threshold_tuned_on_slice": False, "evaluation_passes": 1, "inference_settings": {"imgsz": 640, "nms_iou": 0.7, "max_det": 1500, "matching_iou": 0.5}, "metrics": score, "prediction_count": prediction_count, "zero_prediction_screens": zero_screens, "zero_prediction_rate": zero_screens/len(results), "viewport_metrics": viewport_metrics, "object_size_metrics": object_size, "source_group_summary": source_group_summary, "baseline_at_0_25_recomputed_from_persisted_predictions": baseline_summary, "responsive_consistency": {"complete_triplets": len(groups), "mean_detection_class_jaccard": statistics.mean(jaccards), "responsive_only_misses": responsive_misses}, "runtime": runtime}
    error_payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "counts": dict(error_counts), "representative_screen_ids": {key: list(dict.fromkeys(value))[:12] for key,value in examples.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    args.errors.parent.mkdir(parents=True, exist_ok=True); args.errors.write_text(json.dumps(error_payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"metrics": score, "runtime": runtime, "errors": dict(error_counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
