"""Compare the active VisionQA learning curve with its clean benchmark."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT / "runs" / "visionqa" / "atomblock_yolo11n_main_b2" / "results.csv"
BENCHMARK_JSON = ROOT / "reports" / "visionqa" / "metrics" / "benchmark.json"
OUTPUT = ROOT / "reports" / "visionqa" / "metrics" / "training_progress_review.json"
METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
LOSSES = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
)


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key.strip(): float(value.strip()) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    main_rows = read_rows(MAIN_CSV)
    benchmark_rows = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))["epoch_metrics"]
    latest = main_rows[-1]
    first = main_rows[0]
    benchmark_same_epoch = next(
        (row for row in benchmark_rows if int(row["epoch"]) == int(latest["epoch"])),
        None,
    )
    finite = all(math.isfinite(value) for row in main_rows for value in row.values())
    best_map = max(row["metrics/mAP50-95(B)"] for row in main_rows)
    no_collapse = latest["metrics/mAP50-95(B)"] >= 0.7 * best_map
    learned_beyond_epoch_one = best_map > first["metrics/mAP50-95(B)"] if len(main_rows) > 1 else True
    status = "healthy" if finite and no_collapse and learned_beyond_epoch_one else "review_required"
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "rules": {
            "all_values_finite": finite,
            "latest_map50_95_at_least_70_percent_of_best": no_collapse,
            "best_map50_95_exceeds_epoch_one": learned_beyond_epoch_one,
            "note": "Small epoch-to-epoch metric decreases are normal; validation best controls selection.",
        },
        "completed_epochs": len(main_rows),
        "latest_epoch": int(latest["epoch"]),
        "latest": latest,
        "best_validation_epoch": max(
            main_rows, key=lambda row: row["metrics/mAP50-95(B)"]
        ),
        "change_from_epoch_one": {
            key: latest[key] - first[key] for key in (*METRICS, *LOSSES)
        },
        "same_epoch_benchmark": benchmark_same_epoch,
        "difference_from_same_epoch_benchmark": (
            {key: latest[key] - benchmark_same_epoch[key] for key in (*METRICS, *LOSSES)}
            if benchmark_same_epoch
            else None
        ),
    }
    write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))
    return 0 if status == "healthy" else 1


if __name__ == "__main__":
    raise SystemExit(main())
