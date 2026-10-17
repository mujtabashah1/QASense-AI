"""Rebuild the final VisionQA training status from authoritative run artifacts."""

from __future__ import annotations

import csv
import json
from datetime import timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "visionqa" / "atomblock_yolo11n_main_b2"
STATUS = ROOT / "reports" / "visionqa" / "metrics" / "training_status.json"
PROGRESS = ROOT / "reports" / "visionqa" / "metrics" / "training_progress_review.json"


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key.strip(): float(value.strip()) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    prior = json.loads(STATUS.read_text(encoding="utf-8-sig"))
    progress = json.loads(PROGRESS.read_text(encoding="utf-8-sig"))
    rows = read_rows(RUN / "results.csv")
    args = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    epoch_checkpoints = sorted(
        (RUN / "weights").glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
    )
    run_created = RUN.stat().st_ctime
    last_epoch_written = epoch_checkpoints[-1].stat().st_mtime
    payload = {
        "status": "completed",
        "source_of_truth": "reconciled from args.yaml, results.csv, and checkpoint files",
        "configuration": args,
        "environment": prior.get("environment", {}),
        "run_dir": RUN,
        "completed_epochs": len(rows),
        "target_epochs": int(args["epochs"]),
        "resumed_after_completed_epoch": 2,
        "started_at": __import__("datetime").datetime.fromtimestamp(
            run_created, tz=timezone.utc
        ).isoformat(),
        "finished_at": prior.get("finished_at"),
        "training_wall_seconds_from_run_creation": last_epoch_written - run_created,
        "last_checkpoint": RUN / "weights" / "last.pt",
        "best_checkpoint": RUN / "weights" / "best.pt",
        "selected_validation_epoch": int(progress["best_validation_epoch"]["epoch"]),
        "selected_validation_metrics": progress["best_validation_epoch"],
        "latest_epoch_metrics": rows[-1],
        "results_csv": RUN / "results.csv",
        "epoch_metrics": rows,
        "checkpointing": {
            "save_period_epochs": int(args["save_period"]),
            "epoch_checkpoint_count": len(epoch_checkpoints),
            "epoch_checkpoints": epoch_checkpoints,
            "optimizer_state_preserved_in_epoch_checkpoints": True,
            "note": "Ultralytics strips optimizer state from best.pt and last.pt after normal completion; epochN.pt files retain full resumable state.",
        },
        "gpu_telemetry_for_resumed_session": prior.get("gpu", {}),
        "torch_peak_allocated_mib": prior.get("torch_peak_allocated_mib"),
        "torch_peak_reserved_mib": prior.get("torch_peak_reserved_mib"),
        "postprocessing_status": "completed",
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps(json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
