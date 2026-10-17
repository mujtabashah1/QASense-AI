"""Create a durable, source-of-truth snapshot of the active VisionQA run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "visionqa" / "atomblock_yolo11n_main_b2"
DEFAULT_PROCESS = ROOT / "reports" / "visionqa" / "metrics" / "training_process.json"
DEFAULT_OUTPUT = ROOT / "reports" / "visionqa" / "metrics" / "training_live_snapshot.json"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def read_csv(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({key.strip(): float(value.strip()) for key, value in raw.items()})
    return rows


def epoch_durations(cumulative_times: list[float]) -> tuple[list[float], list[int]]:
    """Recover epoch durations when an Ultralytics resume resets its wall timer."""
    durations: list[float] = []
    reset_epochs: list[int] = []
    previous = 0.0
    for index, current in enumerate(cumulative_times):
        delta = current - previous
        reference = median(durations) if durations else current
        # A resume starts a fresh timer but appends to the existing CSV. If its
        # first cumulative value only slightly exceeds the old session's final
        # time, the raw delta is implausibly short; the current value is the
        # elapsed time in the new session.
        if index and (delta <= 0 or delta < 0.5 * reference):
            durations.append(current)
            reset_epochs.append(index + 1)
        else:
            durations.append(delta)
        previous = current
    return durations, reset_epochs


def nvidia_state() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip().split(",")
        return {
            "utilization_percent": float(values[0]),
            "memory_used_mib": float(values[1]),
            "memory_total_mib": float(values[2]),
            "temperature_c": float(values[3]),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def cuda_process_ids() -> list[int]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def last_progress_line(log_path: Path | None) -> str | None:
    if not log_path or not log_path.is_file():
        return None
    text = ANSI.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    candidates = [line.strip() for line in text.splitlines() if "/30" in line]
    return candidates[-1] if candidates else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--process-state", type=Path, default=DEFAULT_PROCESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    rows = read_csv(run_dir / "results.csv")
    process_state = read_json(args.process_state.resolve())
    train_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    weights_dir = run_dir / "weights"
    epoch_files = sorted(weights_dir.glob("epoch*.pt"))
    compute_pids = cuda_process_ids()
    completed_epochs = len(rows)
    target_epochs = int(train_args["epochs"])
    per_epoch_seconds: list[float] = []
    timer_reset_epochs: list[int] = []
    if rows:
        cumulative = [row["time"] for row in rows]
        per_epoch_seconds, timer_reset_epochs = epoch_durations(cumulative)
        per_epoch_seconds = [value for value in per_epoch_seconds if value > 0]
    typical_epoch_seconds = median(per_epoch_seconds) if per_epoch_seconds else None
    benchmark = read_json(ROOT / "reports" / "visionqa" / "metrics" / "benchmark.json")
    benchmark_rows = benchmark.get("epoch_metrics", [])
    benchmark_durations, _ = epoch_durations([row["time"] for row in benchmark_rows])
    benchmark_epoch_seconds = median(benchmark_durations) if benchmark_durations else None
    latest_epoch_seconds = per_epoch_seconds[-1] if per_epoch_seconds else None
    remaining_epochs = max(target_epochs - completed_epochs, 0)
    best_row = (
        max(rows, key=lambda row: row["metrics/mAP50-95(B)"]) if rows else None
    )
    stdout_log = Path(process_state["stdout_log"]) if process_state.get("stdout_log") else None

    snapshot: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "training" if compute_pids and completed_epochs < target_epochs else (
            "completed" if completed_epochs >= target_epochs else "not_running"
        ),
        "run_dir": str(run_dir),
        "detached_from_codex": bool(process_state.get("detached")),
        "launcher_pid": process_state.get("pid"),
        "cuda_process_ids": compute_pids,
        "log_paths": {
            "stdout": process_state.get("stdout_log"),
            "stderr": process_state.get("stderr_log"),
        },
        "last_console_progress": last_progress_line(stdout_log),
        "dataset": {
            "configured_training_pairs": 5422,
            "decodable_training_images_per_epoch": 5420,
            "validation_pairs_per_epoch": 1388,
            "held_out_test_pairs": 1389,
        },
        "training": {
            "completed_epochs": completed_epochs,
            "target_epochs": target_epochs,
            "remaining_epochs_at_maximum": remaining_epochs,
            "decodable_image_presentations_completed": completed_epochs * 5420,
            "maximum_decodable_image_presentations": target_epochs * 5420,
            "typical_completed_epoch_seconds": typical_epoch_seconds,
            "latest_completed_epoch_seconds": latest_epoch_seconds,
            "benchmark_reference_epoch_seconds": benchmark_epoch_seconds,
            "timer_reset_detected_at_epochs": timer_reset_epochs,
            "estimated_remaining_training_seconds": {
                "at_benchmark_pace": (
                    benchmark_epoch_seconds * remaining_epochs if benchmark_epoch_seconds else None
                ),
                "at_latest_observed_pace": (
                    latest_epoch_seconds * remaining_epochs if latest_epoch_seconds else None
                ),
            },
            "early_stopping_patience": int(train_args["patience"]),
            "batch": int(train_args["batch"]),
            "imgsz": int(train_args["imgsz"]),
            "device": str(train_args["device"]),
        },
        "latest_epoch": rows[-1] if rows else None,
        "best_validation_epoch": best_row,
        "checkpointing": {
            "save_period_epochs": int(train_args["save_period"]),
            "last": str(weights_dir / "last.pt"),
            "best": str(weights_dir / "best.pt"),
            "epoch_checkpoints": [str(path) for path in epoch_files],
        },
        "gpu": nvidia_state(),
    }
    write_json(args.output.resolve(), snapshot)
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
