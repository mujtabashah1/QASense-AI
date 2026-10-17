"""Run and report the controlled deterministic visual-defect benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visual_defects.baseline import DEFECT_LABELS, classify_pair, pixel_difference  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(rows: list[dict]) -> dict:
    labels = sorted({row["actual"] for row in rows} | {row["predicted"] for row in rows})
    per_class = {}
    for label in labels:
        tp = sum(row["actual"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["actual"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["actual"] == label and row["predicted"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": sum(row["actual"] == label for row in rows), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    defect_rows = [row for row in rows if row["actual"] != "no_defect"]
    defect_tp = sum(row["predicted"] == row["actual"] for row in defect_rows)
    macro_labels = [label for label in labels if label != "no_defect"]
    return {
        "evaluations": len(rows), "positive_cases": len(defect_rows),
        "exact_accuracy": sum(row["actual"] == row["predicted"] for row in rows) / len(rows),
        "positive_exact_recall": defect_tp / len(defect_rows),
        "macro_precision_defects": sum(per_class[label]["precision"] for label in macro_labels) / len(macro_labels),
        "macro_recall_defects": sum(per_class[label]["recall"] for label in macro_labels) / len(macro_labels),
        "macro_f1_defects": sum(per_class[label]["f1"] for label in macro_labels) / len(macro_labels),
        "per_class": per_class,
    }


def render(result: dict) -> str:
    metrics = result["metrics"]
    lines = [
        "# Controlled Visual Defect Benchmark", "", f"Generated: `{result['generated_at']}`", "",
        "## Scope", "", "This benchmark measures deterministic DOM/pixel/VisionQA rules on controlled CSS/DOM mutations. It does not estimate accuracy on real-world visual defects.", "",
        "## Results", "",
        f"- Mutated positive cases: **{metrics['positive_cases']}**",
        f"- Unchanged negative controls: **{metrics['evaluations'] - metrics['positive_cases']}**",
        f"- Exact multiclass accuracy: **{metrics['exact_accuracy']:.3f}**",
        f"- Positive exact recall: **{metrics['positive_exact_recall']:.3f}**",
        f"- Defect macro precision: **{metrics['macro_precision_defects']:.3f}**",
        f"- Defect macro recall: **{metrics['macro_recall_defects']:.3f}**",
        f"- Defect macro F1: **{metrics['macro_f1_defects']:.3f}**", "",
        "| Class | Support | Precision | Recall | F1 | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in metrics["per_class"].items():
        lines.append(f"| {label} | {row['support']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['fp']} | {row['fn']} |")
    lines += [
        "", "## Decision", "",
        result["model_decision"], "",
        "The pilot has one controlled page fixture and deterministic mutations. Add diverse real application states and human-reviewed failures before training a visual-defect model. Current evidence supports improving the rules and dataset breadth first.", "",
        "Machine-readable predictions and metrics: `reports/visual_defects/controlled_benchmark.json`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=ROOT / "datasets/visual_defects_generated/pilot_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/visual_defects/controlled_benchmark.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/visual_defects/CONTROLLED_VISUAL_DEFECT_BENCHMARK.md")
    args = parser.parse_args()
    manifest = load(args.pilot / "manifest.json")
    rows: list[dict] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for item in manifest["cases"]:
        case_dir = Path(item["case_dir"])
        truth = load(case_dir / "ground_truth.json")
        before, after = load(case_dir / "dom_before.json"), load(case_dir / "dom_after.json")
        vision_before, vision_after = load(case_dir / "visionqa_baseline.json"), load(case_dir / "visionqa_mutated.json")
        pixel = pixel_difference(case_dir / "baseline.png", case_dir / "mutated.png")
        prediction = classify_pair(before, after, truth["target_selector"], truth["affected_selector"], truth["viewport"], pixel, vision_before, vision_after)
        row = {"case_id": item["case_id"], "actual": truth["defect_type"], "predicted": prediction["prediction"], "confidence": prediction["confidence"], "evidence": prediction["evidence"]}
        rows.append(row); confusion[row["actual"]][row["predicted"]] += 1
        # Every captured baseline is also an unchanged negative control.
        control_pixel = pixel_difference(case_dir / "baseline.png", case_dir / "baseline.png")
        control_prediction = classify_pair(before, before, truth["target_selector"], truth["affected_selector"], truth["viewport"], control_pixel, vision_before, vision_before)
        control = {"case_id": item["case_id"] + "-control", "actual": "no_defect", "predicted": control_prediction["prediction"], "confidence": control_prediction["confidence"], "evidence": control_prediction["evidence"]}
        rows.append(control); confusion["no_defect"][control["predicted"]] += 1
    metrics = score(rows)
    result = {
        "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": "Controlled Visual Defect Benchmark", "pilot_manifest": str(args.pilot / "manifest.json"),
        "metrics": metrics, "confusion": {actual: dict(values) for actual, values in confusion.items()}, "predictions": rows,
        "model_decision": "Do not train a visual-defect ML model yet. The pilot is large enough to validate contracts but not diverse enough to support generalization; expand with real application page states and review hard cases first.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render(result), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
