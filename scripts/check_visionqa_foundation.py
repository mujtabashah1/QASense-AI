"""Validate the AtomBlock dataset and, optionally, a VisionQA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import torch  # noqa: E402
import yaml  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "datasets" / "processed" / "visionqa_atomblock" / "data.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-image", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, object] = {}
    data_file = args.data.resolve()
    checks["data_yaml_exists"] = data_file.is_file()
    if not data_file.is_file():
        failures.append(f"Missing data YAML: {data_file}")
    else:
        config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
        names = config.get("names", {})
        checks["class_count"] = len(names)
        if config.get("nc") != 14 or len(names) != 14:
            failures.append("data.yaml must define exactly 14 classes")
        dataset_root = Path(config["path"])
        split_counts = {}
        for split in ("train", "val", "test"):
            image_dir = dataset_root / split / "images"
            label_dir = dataset_root / split / "labels"
            images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
            labels = list(label_dir.glob("*.txt"))
            split_counts[split] = {"images": len(images), "labels": len(labels)}
            if not image_dir.is_dir() or not label_dir.is_dir():
                failures.append(f"Missing {split} image or label directory")
            if len(images) != len(labels):
                failures.append(f"{split} image/label counts do not match")
        checks["splits"] = split_counts

    checks["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        checks["gpu"] = torch.cuda.get_device_name(0)

    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        checks["checkpoint_exists"] = checkpoint.is_file()
        if not checkpoint.is_file():
            failures.append(f"Missing checkpoint: {checkpoint}")
        else:
            from ultralytics import YOLO

            model = YOLO(str(checkpoint))
            checks["checkpoint_loaded"] = True
            sample = args.sample_image
            if sample is None and data_file.is_file():
                sample = next((Path(config["path"]) / "test" / "images").iterdir())
            results = model.predict(str(sample), device=0, verbose=False)
            checks["inference_result_count"] = len(results)
            checks["inference_has_boxes"] = hasattr(results[0], "boxes")
            if len(results) != 1 or not hasattr(results[0], "boxes"):
                failures.append("YOLO inference did not return structured boxes")

    checks["failures"] = failures
    print(json.dumps(checks, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
