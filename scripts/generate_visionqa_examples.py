"""Generate matched qualitative VisionQA examples from the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


@dataclass
class Candidate:
    score: float
    result: Any
    ground_truth: np.ndarray
    matched_gt: set[int]
    matched_predictions: set[int]
    tp: int
    fp: int
    fn: int


def load_ground_truth(label_path: Path, width: int, height: int) -> np.ndarray:
    rows = []
    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)
    seen = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip() in seen:
            continue
        seen.add(line.strip())
        class_id, x, y, box_width, box_height = (float(value) for value in line.split())
        x1 = (x - box_width / 2) * width
        y1 = (y - box_height / 2) * height
        x2 = (x + box_width / 2) * width
        y2 = (y + box_height / 2) * height
        rows.append([class_id, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def match_boxes(
    predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5
) -> tuple[set[int], set[int]]:
    matched_predictions: set[int] = set()
    matched_gt: set[int] = set()
    if not len(predictions) or not len(ground_truth):
        return matched_predictions, matched_gt
    order = np.argsort(-predictions[:, 4])
    for prediction_index in order:
        class_id = int(predictions[prediction_index, 5])
        choices = [
            gt_index
            for gt_index in range(len(ground_truth))
            if gt_index not in matched_gt and int(ground_truth[gt_index, 0]) == class_id
        ]
        if not choices:
            continue
        overlaps = [iou(predictions[prediction_index, :4], ground_truth[index, 1:]) for index in choices]
        best_position = int(np.argmax(overlaps))
        if overlaps[best_position] >= threshold:
            matched_predictions.add(int(prediction_index))
            matched_gt.add(choices[best_position])
    return matched_predictions, matched_gt


def draw_example(candidate: Candidate, destination: Path, category: str) -> None:
    image = candidate.result.orig_img.copy()
    predictions = (
        candidate.result.boxes.data.cpu().numpy()
        if candidate.result.boxes is not None
        else np.empty((0, 6))
    )
    names = candidate.result.names
    for index, row in enumerate(predictions):
        x1, y1, x2, y2, confidence, class_id = row[:6]
        color = (20, 180, 20) if index in candidate.matched_predictions else (0, 165, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            image,
            f"{names[int(class_id)]} {confidence:.2f}",
            (int(x1), max(14, int(y1) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    for index, row in enumerate(candidate.ground_truth):
        if index in candidate.matched_gt:
            continue
        class_id, x1, y1, x2, y2 = row
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"MISSED {names[int(class_id)]}",
            (int(x1), max(14, int(y1) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 720), 38), (255, 255, 255), -1)
    cv2.putText(
        image,
        f"{category}: TP={candidate.tp} FP={candidate.fp} FN={candidate.fn}",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Failed to write annotated image: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "datasets" / "processed" / "visionqa_atomblock" / "test" / "images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "datasets" / "processed" / "visionqa_atomblock" / "test" / "labels",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "visionqa" / "predictions")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=1500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Prediction generation must use the GPU.")
    model = YOLO(str(args.checkpoint.resolve()))
    selections: dict[str, Candidate] = {}
    category_classes = {
        "forms": {2, 5, 6, 7, 11},
        "navigation": {8, 9},
        "tables": {12},
        "rare_classes": {6, 9, 10, 11, 12},
    }

    results = model.predict(
        source=str(args.images.resolve()),
        stream=True,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.confidence,
        iou=0.7,
        max_det=args.max_det,
        verbose=False,
    )
    for result in results:
        height, width = result.orig_shape
        label_path = args.labels.resolve() / (Path(result.path).stem + ".txt")
        ground_truth = load_ground_truth(label_path, width, height)
        predictions = (
            result.boxes.data.cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 6), dtype=np.float32)
        )
        matched_predictions, matched_gt = match_boxes(predictions, ground_truth)
        tp = len(matched_predictions)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_gt)
        base = Candidate(
            score=0,
            result=result.cpu(),
            ground_truth=ground_truth,
            matched_gt=matched_gt,
            matched_predictions=matched_predictions,
            tp=tp,
            fp=fp,
            fn=fn,
        )
        scores = {
            "correct_detections": tp - 0.25 * (fp + fn),
            "missed_detections": fn,
            "false_positives": fp,
            "dense_page": len(ground_truth),
        }
        present = set(int(value) for value in ground_truth[:, 0]) if len(ground_truth) else set()
        for category, class_ids in category_classes.items():
            scores[category] = sum(int(row[0]) in class_ids for row in ground_truth)
        for category, score in scores.items():
            if score <= 0 and category in category_classes:
                continue
            if category not in selections or score > selections[category].score:
                base_copy = Candidate(**{**base.__dict__, "score": float(score)})
                selections[category] = base_copy

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "confidence": args.confidence,
        "iou_match_threshold": 0.5,
        "legend": {
            "green": "matched prediction",
            "orange": "false-positive prediction",
            "red": "unmatched ground-truth box (miss)",
        },
        "examples": {},
    }
    for category, candidate in selections.items():
        destination = args.output.resolve() / f"{category}.jpg"
        draw_example(candidate, destination, category)
        manifest["examples"][category] = {
            "source": candidate.result.path,
            "annotated_image": str(destination),
            "ground_truth_boxes": len(candidate.ground_truth),
            "predicted_boxes": len(candidate.result.boxes) if candidate.result.boxes is not None else 0,
            "true_positives": candidate.tp,
            "false_positives": candidate.fp,
            "false_negatives": candidate.fn,
        }
    manifest_path = args.output.resolve() / "visual_examples.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

