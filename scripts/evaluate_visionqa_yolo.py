"""Evaluate a VisionQA YOLO checkpoint and save overall/per-class test metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "datasets" / "processed" / "visionqa_atomblock" / "data.yaml",
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-det", type=int, default=1500)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "visionqa")
    parser.add_argument("--name", default="atomblock_test_evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "visionqa" / "metrics" / "evaluation.json",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. VisionQA evaluation must use the GPU.")

    started = time.perf_counter()
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )

    box = metrics.box
    class_indices = list(getattr(box, "ap_class_index", range(len(model.names))))
    per_class = []
    for result_index, class_id_value in enumerate(class_indices):
        class_id = int(class_id_value)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "precision": scalar(box.p[result_index]),
                "recall": scalar(box.r[result_index]),
                "ap50": scalar(box.ap50[result_index]),
                "ap50_95": scalar(box.ap[result_index]),
            }
        )

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "dataset": str(args.data.resolve()),
        "split": args.split,
        "configuration": {
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "max_det": args.max_det,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "runtime_seconds": time.perf_counter() - started,
        "overall": {
            "precision": scalar(box.mp),
            "recall": scalar(box.mr),
            "map50": scalar(box.map50),
            "map50_95": scalar(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {key: scalar(value) for key, value in metrics.speed.items()},
        "run_dir": str(metrics.save_dir),
        "confusion_matrix": str(Path(metrics.save_dir) / "confusion_matrix.png"),
        "confusion_matrix_normalized": str(
            Path(metrics.save_dir) / "confusion_matrix_normalized.png"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

