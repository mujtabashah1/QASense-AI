"""Persist low-floor VisionQA predictions for the fixed WebUI corpus validation split."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation inference")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest_path = ROOT / "datasets" / "processed" / "visionqa_webui_v2" / "manifests" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sorted((row for row in manifest["screens"] if row["split"] == "val"), key=lambda row: row["screen_id"])
    if len(rows) != 5214:
        raise ValueError(f"Expected 5214 validation screens, found {len(rows)}")
    images = [str(Path(row["processed_image_path"]).resolve()) for row in rows]
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(image)

    # Passing a Python list makes Ultralytics treat all images as an in-memory
    # source. A text source preserves streaming and bounded VRAM on the 4 GB GPU.
    source_list = args.output.with_suffix(".images.txt")
    source_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    model = YOLO(str(checkpoint))
    if len(model.names) != 14:
        raise ValueError(f"Expected unified 14-class checkpoint, found {len(model.names)} classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = detections = 0
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as output:
        results = model.predict(
            source=str(source_list), stream=True, batch=args.batch, imgsz=args.imgsz,
            conf=args.conf_floor, iou=args.iou, max_det=args.max_det,
            device=args.device, verbose=False, save=False,
        )
        for row, result in zip(rows, results, strict=True):
            boxes = result.boxes
            prediction_rows = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().tolist()
                conf = boxes.conf.detach().cpu().tolist()
                cls = boxes.cls.detach().cpu().tolist()
                for coords, score, class_value in zip(xyxy, conf, cls, strict=True):
                    class_id = int(class_value)
                    prediction_rows.append({
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": float(score),
                        "bbox_xyxy": [float(value) for value in coords],
                    })
            payload = {
                "image_id": row["screen_id"],
                "image_path": str(Path(row["processed_image_path"]).resolve()),
                "width": int(row["original_image_identifier"]["width"]),
                "height": int(row["original_image_identifier"]["height"]),
                "model": args.model_name,
                "detections": prediction_rows,
            }
            output.write(json.dumps(payload, separators=(",", ":")) + "\n")
            count += 1
            detections += len(prediction_rows)
    elapsed = time.perf_counter() - started
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": sha256(manifest_path),
        "image_count": count,
        "detection_count_at_floor": detections,
        "settings": {"imgsz": args.imgsz, "conf_floor": args.conf_floor, "iou_nms": args.iou, "max_det": args.max_det, "batch": args.batch, "device": args.device},
        "runtime_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
