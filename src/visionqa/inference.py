"""Structured runtime inference for the trained AtomBlock detector."""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "best.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))


class VisionQADetector:
    """Load the stable detector once and return JSON-compatible evidence."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "0",
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.7,
        max_det: int = 1500,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"VisionQA checkpoint does not exist: {self.checkpoint}")
        if str(device).lower() != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = str(device)
        self.device_name = "cpu" if self.device.lower() == "cpu" else f"cuda:{self.device}"
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.max_det = max_det
        self.model = YOLO(str(self.checkpoint))

    def predict(self, screenshot: str | Path) -> dict[str, Any]:
        image = Path(screenshot).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Screenshot does not exist: {image}")
        started = time.perf_counter()
        result = self.model.predict(
            source=str(image),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - started) * 1000
        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                numeric_id = int(class_id)
                detections.append(
                    {
                        "class_id": numeric_id,
                        "class_name": result.names[numeric_id],
                        "confidence": float(confidence),
                        "bbox_xyxy": [float(coordinate) for coordinate in xyxy],
                    }
                )
        counts = Counter(item["class_name"] for item in detections)
        return {
            "image": {
                "path": str(image),
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
            },
            "model": {
                "name": self.checkpoint.parent.name,
                "checkpoint": str(self.checkpoint),
                "device": self.device_name,
                "imgsz": self.imgsz,
                "confidence_threshold": self.confidence,
                "iou_threshold": self.iou,
                "class_count": len(self.model.names),
                "class_names": {str(key): value for key, value in self.model.names.items()},
            },
            "detection_count": len(detections),
            "counts_by_class": dict(sorted(counts.items())),
            "inference_time_ms": inference_time_ms,
            "detections": detections,
        }
