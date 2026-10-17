"""Train or resume the QASense VisionQA AtomBlock YOLO detector.

Ultralytics writes ``last.pt`` and ``best.pt`` after each epoch. This wrapper
also enables epoch checkpoints and records telemetry/status as JSON so an
interrupted run can be inspected and resumed without losing completed epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "processed" / "visionqa_atomblock" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "visionqa"
DEFAULT_METRICS = ROOT / "reports" / "visionqa" / "metrics"
YOLO_CONFIG_DIR = DEFAULT_PROJECT
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import torch_safe_load  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class GpuMonitor:
    """Sample NVIDIA utilization, memory, and temperature during a run."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 2)

    def _sample_loop(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                values = [part.strip() for part in completed.stdout.strip().split(",")]
                if len(values) >= 5:
                    self.samples.append(
                        {
                            "timestamp": values[0],
                            "gpu_utilization_percent": float(values[1]),
                            "memory_used_mib": float(values[2]),
                            "memory_total_mib": float(values[3]),
                            "temperature_c": float(values[4]),
                        }
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}
        util = [s["gpu_utilization_percent"] for s in self.samples]
        memory = [s["memory_used_mib"] for s in self.samples]
        temperature = [s["temperature_c"] for s in self.samples]
        return {
            "sample_count": len(self.samples),
            "mean_gpu_utilization_percent": sum(util) / len(util),
            "peak_gpu_utilization_percent": max(util),
            "peak_memory_used_mib": max(memory),
            "peak_temperature_c": max(temperature),
        }


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            clean: dict[str, Any] = {}
            for key, value in row.items():
                key = key.strip()
                value = value.strip()
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            rows.append(clean)
        return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="atomblock_smoke")
    parser.add_argument("--resume", type=Path, help="Path to an Ultralytics last.pt checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-det", type=int, default=1500)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.metrics_output = (
        args.metrics_output.resolve()
        if args.metrics_output
        else (DEFAULT_METRICS / f"{args.name}.json").resolve()
    )

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. VisionQA training must not run on CPU.")
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

    args.project.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    resume_checkpoint: dict[str, Any] | None = None
    resume_train_args: dict[str, Any] = {}
    if args.resume:
        resume_checkpoint, _ = torch_safe_load(str(args.resume.resolve()))
        resume_train_args = dict(resume_checkpoint.get("train_args") or {})

    run_dir = Path(
        resume_train_args.get("save_dir", args.project / args.name)
    ).resolve()
    effective_configuration = (
        {
            **resume_train_args,
            "resume": str(args.resume.resolve()),
            "device": args.device,
            "workers": args.workers,
            "metrics_output": str(args.metrics_output),
        }
        if args.resume
        else vars(args)
    )
    started = time.perf_counter()
    status: dict[str, Any] = {
        "status": "starting",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "configuration": effective_configuration,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "run_dir": run_dir,
    }
    if resume_checkpoint is not None:
        status["resumed_from_epoch"] = int(resume_checkpoint.get("epoch", -1)) + 1
        status["resumed_checkpoint"] = args.resume.resolve()
    write_json(args.metrics_output, status)

    monitor = GpuMonitor()
    monitor.start()
    try:
        checkpoint = str(args.resume.resolve()) if args.resume else args.model
        model = YOLO(checkpoint)

        def on_epoch_end(trainer: Any) -> None:
            trainer_run_dir = Path(trainer.save_dir).resolve()
            status.update(
                {
                    "status": "training",
                    "updated_at": utc_now(),
                    "completed_epochs": int(trainer.epoch) + 1,
                    "last_checkpoint": Path(trainer.last),
                    "best_checkpoint": Path(trainer.best),
                    "run_dir": trainer_run_dir,
                    "latest_metrics": getattr(trainer, "metrics", {}),
                    "gpu": monitor.summary(),
                    "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                }
            )
            write_json(args.metrics_output, status)

        # on_fit_epoch_end runs after validation and Ultralytics checkpoint saving.
        model.add_callback("on_fit_epoch_end", on_epoch_end)

        if args.resume:
            results = model.train(resume=True, device=args.device, workers=args.workers)
        else:
            results = model.train(
                data=str(args.data),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                project=str(args.project),
                name=args.name,
                exist_ok=args.exist_ok,
                seed=args.seed,
                deterministic=True,
                amp=True,
                cache=False,
                val=True,
                plots=True,
                save=True,
                save_period=args.save_period,
                patience=args.patience,
                max_det=args.max_det,
                verbose=True,
            )

        elapsed = time.perf_counter() - started
        trainer = model.trainer
        actual_run_dir = Path(trainer.save_dir).resolve() if trainer else run_dir
        results_csv = actual_run_dir / "results.csv"
        rows = parse_results_csv(results_csv)
        completed_epochs = len(rows) or status.get("completed_epochs", 0)
        status.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "runtime_seconds": elapsed,
                "completed_epochs": completed_epochs,
                "seconds_per_epoch": elapsed / completed_epochs if completed_epochs else None,
                "results_csv": results_csv,
                "epoch_metrics": rows,
                "result_metrics": getattr(results, "results_dict", {}),
                "run_dir": actual_run_dir,
                "last_checkpoint": Path(trainer.last) if trainer else actual_run_dir / "weights" / "last.pt",
                "best_checkpoint": Path(trainer.best) if trainer else actual_run_dir / "weights" / "best.pt",
                "gpu": monitor.summary(),
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
        write_json(args.metrics_output, status)
        return 0
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "gpu": monitor.summary(),
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
        write_json(args.metrics_output, status)
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())

