"""Audit the fixed mixed 14-class VisionQA v2 training corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed" / "visionqa_v2_mixed14"
MANIFEST = ROOT / "reports" / "visionqa" / "v2" / "mixed_dataset_manifest.json"
OUTPUT = ROOT / "reports" / "visionqa" / "v2" / "mixed_dataset_audit.json"
WEBUI_ALLOWED = {0, 1, 2, 3, 7, 8, 9, 10, 11, 12}
EXPECTED = {"webui_train": 23319, "webui_val": 5214, "atomblock_train": 5420}


def audit_partition(split: str, source: str) -> dict:
    images = sorted((DATA / split / "images" / source).glob("*"))
    labels = sorted((DATA / split / "labels" / source).glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    invalid_lines, invalid_classes, invalid_boxes = [], [], []
    class_counts = Counter()
    for label in labels:
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                invalid_lines.append(f"{label}:{line_number}")
                continue
            try:
                class_id = int(fields[0]); coords = [float(value) for value in fields[1:]]
            except ValueError:
                invalid_lines.append(f"{label}:{line_number}")
                continue
            if not 0 <= class_id < 14 or (source == "webui" and class_id not in WEBUI_ALLOWED):
                invalid_classes.append(f"{label}:{line_number}:{class_id}")
            x, y, width, height = coords
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                invalid_boxes.append(f"{label}:{line_number}:{coords}")
            class_counts[class_id] += 1
    return {
        "images": len(images), "labels": len(labels),
        "missing_labels": sorted(image_stems - label_stems), "orphan_labels": sorted(label_stems - image_stems),
        "invalid_lines": invalid_lines, "invalid_classes": invalid_classes, "invalid_boxes": invalid_boxes,
        "class_counts": {str(i): class_counts[i] for i in range(14)},
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    partitions = {
        "webui_train": audit_partition("train", "webui"),
        "webui_val": audit_partition("val", "webui"),
        "atomblock_train": audit_partition("train", "atomblock"),
    }
    failures = []
    for name, expected in EXPECTED.items():
        row = partitions[name]
        if row["images"] != expected or row["labels"] != expected:
            failures.append(f"{name} count mismatch")
        for field in ("missing_labels", "orphan_labels", "invalid_lines", "invalid_classes", "invalid_boxes"):
            if row[field]: failures.append(f"{name} {field}: {len(row[field])}")
    if any(partitions["webui_train"]["class_counts"][str(i)] or partitions["webui_val"]["class_counts"][str(i)] for i in (4, 5, 6, 13)):
        failures.append("WebUI contains labels for unknown classes")
    if any(partitions["atomblock_train"]["class_counts"][str(i)] == 0 for i in range(14)):
        failures.append("AtomBlock rehearsal does not cover all 14 classes")
    if manifest["frozen_eval_group_leakage"] != 0:
        failures.append("Frozen evaluation leakage")
    payload = {
        "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if not failures else "FAIL", "failures": failures, "partitions": partitions,
        "mixed_train_images": partitions["webui_train"]["images"] + partitions["atomblock_train"]["images"],
        "webui_fraction": partitions["webui_train"]["images"] / (partitions["webui_train"]["images"] + partitions["atomblock_train"]["images"]),
        "atomblock_fraction": partitions["atomblock_train"]["images"] / (partitions["webui_train"]["images"] + partitions["atomblock_train"]["images"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": payload["result"], "failures": failures, "mixed_train_images": payload["mixed_train_images"], "webui_fraction": payload["webui_fraction"], "atomblock_fraction": payload["atomblock_fraction"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
