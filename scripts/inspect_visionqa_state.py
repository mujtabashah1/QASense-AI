"""Capture reproducible VisionQA environment and AtomBlock dataset metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "visionqa_atomblock"
METRICS = ROOT / "reports" / "visionqa" / "metrics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch  # noqa: E402
import torchvision  # noqa: E402
import ultralytics  # noqa: E402
import yaml  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def nvidia_details() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        name, driver, memory, temperature = [value.strip() for value in output.split(",")]
        return {
            "name": name,
            "driver": driver,
            "vram_mib": int(memory),
            "temperature_c_at_capture": int(temperature),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def main() -> int:
    captured_at = datetime.now(timezone.utc).isoformat()
    environment = {
        "captured_at": captured_at,
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "virtual_environment": os.environ.get("VIRTUAL_ENV") or str(Path(sys.executable).parents[1]),
        "pip_version": subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": nvidia_details(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        environment["pytorch_gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }

    config_path = DATASET / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_details: dict[str, Any] = {}
    overall_classes: Counter[int] = Counter()
    valid = True
    for split in ("train", "val", "test"):
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        labels = list(label_dir.glob("*.txt"))
        class_counts: Counter[int] = Counter()
        boxes_per_image: list[int] = []
        empty = invalid_lines = invalid_classes = invalid_boxes = 0
        for label in labels:
            text = label.read_text(encoding="utf-8").strip()
            if not text:
                empty += 1
                boxes_per_image.append(0)
                continue
            count = 0
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x, y, width, height = (float(v) for v in parts[1:])
                except ValueError:
                    invalid_lines += 1
                    continue
                if class_id not in range(14):
                    invalid_classes += 1
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    invalid_boxes += 1
                    continue
                class_counts[class_id] += 1
                count += 1
            boxes_per_image.append(count)
        overall_classes.update(class_counts)
        split_valid = (
            len(images) == len(labels)
            and invalid_lines == 0
            and invalid_classes == 0
            and invalid_boxes == 0
        )
        valid = valid and split_valid
        split_details[split] = {
            "images": len(images),
            "labels": len(labels),
            "total_boxes": sum(class_counts.values()),
            "empty_labels": empty,
            "invalid_lines": invalid_lines,
            "invalid_class_ids": invalid_classes,
            "invalid_boxes": invalid_boxes,
            "class_counts": {config["names"][i]: class_counts[i] for i in range(14)},
            "boxes_per_image": {
                "mean": mean(boxes_per_image),
                "median": median(boxes_per_image),
                "p95": percentile(boxes_per_image, 0.95),
                "p99": percentile(boxes_per_image, 0.99),
                "max": max(boxes_per_image),
            },
            "passed": split_valid,
        }

    audit = {
        "captured_at": captured_at,
        "status": "PASS" if valid else "FAIL",
        "dataset_path": str(DATASET),
        "data_yaml": str(config_path),
        "nc": config["nc"],
        "names": config["names"],
        "splits": split_details,
        "total_boxes": sum(overall_classes.values()),
    }
    write_json(METRICS / "environment.json", environment)
    write_json(METRICS / "dataset_audit.json", audit)
    print(json.dumps({"environment": environment, "dataset_audit": audit}, indent=2))
    return 0 if valid and environment["cuda_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
