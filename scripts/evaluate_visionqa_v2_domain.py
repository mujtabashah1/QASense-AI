"""Evaluate a 14-class checkpoint on WebUI validation or AtomBlock test."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch
import ultralytics
from ultralytics import YOLO
from src.visionqa.v2_training import WebUISupportedClassValidator, WEBUI_SUPPORTED_IDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", choices=("webui_val", "atomblock_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    data = ROOT / "datasets" / "processed" / ("visionqa_v2_mixed14" if args.domain == "webui_val" else "visionqa_atomblock") / "data.yaml"
    split = "val" if args.domain == "webui_val" else "test"
    model = YOLO(str(args.checkpoint.resolve()))
    if len(model.names) != 14: raise ValueError(f"Expected 14 classes, got {len(model.names)}")
    started = time.perf_counter()
    kwargs = dict(data=str(data), split=split, batch=args.batch, imgsz=640, device="0", workers=2, max_det=1500, project=str(ROOT / "runs" / "visionqa"), name=args.name, exist_ok=False, plots=False, verbose=True)
    if args.domain == "webui_val": kwargs["validator"] = WebUISupportedClassValidator
    metrics = model.val(**kwargs)
    rows = []
    for index, class_value in enumerate(metrics.box.ap_class_index):
        class_id = int(class_value); p = float(metrics.box.p[index]); r = float(metrics.box.r[index])
        rows.append({"class_id": class_id, "class_name": model.names[class_id], "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0, "ap50": float(metrics.box.ap50[index]), "ap50_95": float(metrics.box.ap[index])})
    included = [row for row in rows if args.domain != "webui_val" or row["class_id"] in WEBUI_SUPPORTED_IDS]
    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(args.checkpoint.resolve()), "domain": args.domain,
        "dataset": str(data), "split": split, "runtime_seconds": time.perf_counter()-started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0)},
        "overall": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "macro_f1": statistics.mean(row["f1"] for row in included) if included else 0.0},
        "per_class": rows, "speed_ms_per_image": {key: float(value) for key,value in metrics.speed.items()}, "run_dir": str(metrics.save_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"domain": args.domain, "overall": payload["overall"], "runtime_seconds": payload["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
