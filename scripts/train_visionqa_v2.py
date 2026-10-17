"""Run one resumable stage of unified 14-class VisionQA v2 training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch
import ultralytics
from ultralytics import YOLO

from src.visionqa.v2_training import PartialLabelDetectionTrainer, WEBUI_SUPPORTED_IDS
from scripts.train_visionqa_yolo import GpuMonitor, json_safe, write_json

DEFAULT_DATA = ROOT / "datasets" / "processed" / "visionqa_v2_mixed14" / "data.yaml"
DEFAULT_MODEL = ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "best.pt"
DEFAULT_PROJECT = ROOT / "runs" / "visionqa"
DEFAULT_REPORTS = ROOT / "reports" / "visionqa" / "v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def per_class_metrics(trainer: Any) -> tuple[list[dict[str, Any]], float]:
    box = trainer.validator.metrics.box
    rows = []
    for result_index, class_value in enumerate(box.ap_class_index):
        class_id = int(class_value)
        precision = float(box.p[result_index]); recall = float(box.r[result_index])
        rows.append({
            "class_id": class_id, "class_name": trainer.model.names[class_id],
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "ap50": float(box.ap50[result_index]), "ap50_95": float(box.ap[result_index]),
        })
    supported = [row["f1"] for row in rows if row["class_id"] in WEBUI_SUPPORTED_IDS]
    return rows, statistics.mean(supported) if supported else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("benchmark", "stage_a", "stage_b"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--freeze", type=int, required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not args.data.is_file(): raise FileNotFoundError(args.data)
    if not args.model.is_file(): raise FileNotFoundError(args.model)
    if args.resume and not args.resume.is_file(): raise FileNotFoundError(args.resume)
    status_path = (args.status or DEFAULT_REPORTS / f"{args.stage}_training_status.json").resolve()
    config = {
        **vars(args), "seed": 42, "nc": 14, "amp": True, "save_period": 1,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "webui_supported_class_ids": sorted(WEBUI_SUPPORTED_IDS), "webui_masked_class_ids": [4, 5, 6, 13],
    }
    prior_status = None
    if args.resume and status_path.is_file():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
    status = {
        "status": "starting", "stage": args.stage,
        "started_at": prior_status.get("started_at", utc_now()) if prior_status else utc_now(),
        "configuration": json_safe(config),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory},
        "epoch_history": list(prior_status.get("epoch_history", [])) if prior_status else [],
        "resume_events": list(prior_status.get("resume_events", [])) if prior_status else [],
    }
    if args.resume:
        status["resume_events"].append({"resumed_at": utc_now(), "checkpoint": str(args.resume.resolve())})
    write_json(status_path, status)
    monitor = GpuMonitor(); monitor.start(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    try:
        checkpoint = args.resume.resolve() if args.resume else args.model.resolve()
        model = YOLO(str(checkpoint))

        def epoch_callback(trainer: Any) -> None:
            # Ultralytics invokes this callback once more during its final best.pt
            # validation, after incrementing trainer.epoch beyond the configured
            # training range. That pass is not a trained epoch and has no epochN.pt.
            if int(trainer.epoch) >= int(args.epochs):
                return
            rows, macro_f1 = per_class_metrics(trainer)
            entry = {
                "epoch_index": int(trainer.epoch), "epoch_number": int(trainer.epoch) + 1,
                "webui_supported_macro_f1": macro_f1, "per_class": rows,
                "metrics": json_safe(trainer.metrics),
                "checkpoint": str(Path(trainer.wdir) / f"epoch{trainer.epoch}.pt"),
                "epoch_runtime_seconds": float(trainer.epoch_time),
            }
            status["epoch_history"].append(entry)
            status.update({
                "status": "training", "updated_at": utc_now(), "completed_epochs": int(trainer.epoch) + 1,
                "latest_macro_f1": macro_f1, "last_checkpoint": str(trainer.last), "ultralytics_best_checkpoint": str(trainer.best),
                "gpu": monitor.summary(), "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            })
            write_json(status_path, status)

        model.add_callback("on_fit_epoch_end", epoch_callback)
        kwargs = dict(
            trainer=PartialLabelDetectionTrainer, data=str(args.data.resolve()), epochs=args.epochs,
            batch=args.batch, imgsz=args.imgsz, device=args.device, workers=args.workers,
            project=str(args.project.resolve()), name=args.name, exist_ok=False, seed=42, deterministic=True,
            amp=True, cache=False, val=True, plots=True, save=True, save_period=1, patience=args.patience,
            max_det=1500, verbose=True, freeze=args.freeze, lr0=args.lr0, lrf=0.1,
            optimizer="AdamW", nbs=16,
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
        )
        if args.resume: kwargs["resume"] = str(args.resume.resolve())
        model.train(**kwargs)
        history = status["epoch_history"]
        selected = max(history, key=lambda row: (row["webui_supported_macro_f1"], row["epoch_number"])) if history else None
        status.update({
            "status": "completed", "finished_at": utc_now(), "runtime_seconds": time.perf_counter() - started,
            "selected_by": "WebUI validation supported-class macro-F1", "selected_epoch": selected,
            "gpu": monitor.summary(), "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        })
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        status.update({"status": "failed", "finished_at": utc_now(), "runtime_seconds": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc), "gpu": monitor.summary()})
        write_json(status_path, status)
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())

