"""Add evidence-based weak-class and mapping-ambiguity analysis."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_webui_transfer import load, match_screen  # noqa: E402


def main() -> None:
    manifest = load(ROOT / "reports/visionqa/webui_eval_manifest.json")
    evidence: dict[str, dict[str, list | Counter]] = defaultdict(lambda: {"areas": [], "tp_areas": [], "fn_areas": [], "viewport_gt": Counter(), "viewport_tp": Counter(), "screen_density": []})
    unmapped = Counter(); excluded = Counter()
    for screen in manifest["screens"]:
        prediction = load(ROOT / "runs/visionqa/webui_transfer_eval/predictions" / f"{screen['screen_id']}.json")
        evaluation = match_screen(screen, prediction)
        matched = {item["ground_truth_index"] for item in evaluation["matches"]}
        image_area = screen["image_identifier"]["width"] * screen["image_identifier"]["height"]
        for index, box in enumerate(screen["valid_ground_truth_boxes"]):
            class_name = box["target_class"]
            area = (box["bbox_xyxy"][2] - box["bbox_xyxy"][0]) * (box["bbox_xyxy"][3] - box["bbox_xyxy"][1]) / image_area
            evidence[class_name]["areas"].append(area); evidence[class_name]["viewport_gt"][screen["viewport"]] += 1
            if index in matched:
                evidence[class_name]["tp_areas"].append(area); evidence[class_name]["viewport_tp"][screen["viewport"]] += 1
            else: evidence[class_name]["fn_areas"].append(area)
            evidence[class_name]["screen_density"].append(evaluation["ground_truth_count"])
        unmapped.update(screen["unmapped_classes"])
        excluded.update(item["reason"] for item in screen["excluded_invalid_or_ambiguous_boxes"])
    result = {}
    for class_name, values in evidence.items():
        viewport_recall = {view: values["viewport_tp"][view] / values["viewport_gt"][view] if values["viewport_gt"][view] else None for view in ("desktop", "tablet", "mobile")}
        result[class_name] = {
            "ground_truth": len(values["areas"]), "median_relative_area": statistics.median(values["areas"]),
            "median_tp_relative_area": statistics.median(values["tp_areas"]) if values["tp_areas"] else None,
            "median_fn_relative_area": statistics.median(values["fn_areas"]) if values["fn_areas"] else None,
            "mean_ground_truth_objects_on_containing_screen": statistics.mean(values["screen_density"]),
            "viewport_recall": viewport_recall,
        }
    result["icon"] = {"supported": False, "reason": "WebUI svg/i tags are semantically ambiguous and excluded from scoring.", "unmapped_svg_boxes": unmapped["svg"], "unmapped_i_boxes": unmapped["i"]}
    result["checkbox"] = {"supported": False, "reason": "BBox schema omits input type; hinted checkbox/radio/switch inputs were excluded rather than relabeled.", "ambiguous_input_boxes": excluded["input_subtype_ambiguous"]}
    result["radio"] = {"supported": False, "reason": "BBox schema omits input type; hinted checkbox/radio/switch inputs were excluded rather than relabeled.", "ambiguous_input_boxes_shared": excluded["input_subtype_ambiguous"]}
    errors_path = ROOT / "reports/visionqa/webui_error_analysis.json"; errors = load(errors_path)
    errors["counts"]["responsive_only_miss"] = errors["responsive"]["responsive_only_misses"]
    errors["mapping_ambiguity_counts"] = {"unmapped_svg": unmapped["svg"], "unmapped_i": unmapped["i"], "input_subtype_ambiguous": excluded["input_subtype_ambiguous"]}
    errors["weak_class_evidence"] = {name: result[name] for name in ("icon", "checkbox", "radio", "select", "block-form", "block-table")}
    errors_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(errors["weak_class_evidence"], indent=2))


if __name__ == "__main__":
    main()
