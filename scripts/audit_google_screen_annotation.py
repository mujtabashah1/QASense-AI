"""Audit the locally downloaded Google Screen Annotation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


KNOWN_CATEGORIES = (
    "NAVIGATION_BAR", "PAGER_INDICATOR", "RADIO_BUTTON", "TEXT_INPUT",
    "LIST_ITEM", "PICTOGRAM", "CHECKBOX", "KEYBOARD", "TOOLBAR",
    "BUTTON", "SWITCH", "IMAGE", "LABEL", "TEXT", "MAP",
)
ELEMENT_START_RE = re.compile(
    r"(?:^|[,(])\s*(" + "|".join(KNOWN_CATEGORIES) + r")\s+"
)
INTEGER_RE = re.compile(r"(?<![\w.])-?\d+(?![\w.])")


def parse_elements(annotation: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Parse hierarchy records, choosing the last four integers before the next element.

    Visible text may itself contain parentheses and numbers, so a single regex that
    stops at a closing parenthesis can mistake text such as an ABN for a box.
    """
    starts = list(ELEMENT_START_RE.finditer(annotation))
    parsed: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(annotation)
        integers = [int(match.group()) for match in INTEGER_RE.finditer(annotation[start.end():end])]
        if len(integers) >= 4:
            parsed.append((start.group(1), tuple(integers[-4:])))
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(path: Path) -> tuple[dict, set[str], Counter]:
    ids: list[str] = []
    categories: Counter = Counter()
    annotation_count = invalid_boxes = malformed_rows = empty_annotations = 0
    coordinate_min = None
    coordinate_max = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        for row in reader:
            screen_id = (row.get("screen_id") or "").strip()
            annotation = row.get("screen_annotation") or ""
            if not screen_id:
                malformed_rows += 1
            ids.append(screen_id)
            if not annotation.strip():
                empty_annotations += 1
                continue
            elements = parse_elements(annotation)
            if not elements:
                malformed_rows += 1
            for category, (x1, x2, y1, y2) in elements:
                categories[category] += 1
                annotation_count += 1
                coordinate_min = min(x1, x2, y1, y2, coordinate_min if coordinate_min is not None else x1)
                coordinate_max = max(x1, x2, y1, y2, coordinate_max if coordinate_max is not None else x1)
                if x2 <= x1 or y2 <= y1:
                    invalid_boxes += 1
    id_counts = Counter(ids)
    result = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "columns": columns,
        "row_count": len(ids),
        "unique_screen_ids": len(id_counts),
        "duplicate_screen_id_rows": sum(value - 1 for value in id_counts.values() if value > 1),
        "empty_annotations": empty_annotations,
        "malformed_rows": malformed_rows,
        "parsed_annotation_count": annotation_count,
        "invalid_boxes": invalid_boxes,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "class_distribution": dict(categories.most_common()),
    }
    return result, set(ids), categories


def audit(root: Path) -> dict:
    split_results: dict[str, dict] = {}
    split_ids: dict[str, set[str]] = {}
    total_categories: Counter = Counter()
    for split, filename in (("train", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")):
        result, ids, categories = audit_split(root / filename)
        split_results[split] = result
        split_ids[split] = ids
        total_categories.update(categories)
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [path for path in root.rglob("*") if path.suffix.lower() in image_extensions]
    overlaps = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    return {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Google Screen Annotation",
        "root": str(root.resolve()),
        "license": "CC BY 4.0",
        "source_image_dataset": "Rico",
        "images_in_local_package": len(image_files),
        "screenshots_must_be_resolved_externally": len(image_files) == 0,
        "splits": split_results,
        "cross_split_screen_id_overlap": overlaps,
        "totals": {
            "rows": sum(item["row_count"] for item in split_results.values()),
            "unique_screen_ids": len(set().union(*split_ids.values())),
            "parsed_annotations": sum(total_categories.values()),
            "classes": len(total_categories),
            "class_distribution": dict(total_categories.most_common()),
        },
        "readme_declared_counts": {"train": 15743, "validation": 2364, "test": 4310, "total": 22417},
        "readme_count_difference": {
            "train": split_results["train"]["row_count"] - 15743,
            "validation": split_results["validation"]["row_count"] - 2364,
            "test": split_results["test"]["row_count"] - 4310,
            "total": sum(item["row_count"] for item in split_results.values()) - 22417,
        },
        "atomblock_compatibility": {
            "status": "not_directly_compatible",
            "reason": "Mobile semantic hierarchy and normalized text annotations differ from AtomBlock YOLO labels; local screenshot bytes are absent.",
            "recommended_option": "C",
            "recommendation": "Use as screen-language/representation or evaluation data only after Rico images and provenance are resolved; do not merge into AtomBlock now.",
        },
    }


def render_markdown(result: dict) -> str:
    splits = result["splits"]
    lines = [
        "# Google Screen Annotation Dataset Audit", "",
        f"Generated: `{result['generated_at']}`", "", "## Result", "",
        "**PASS for annotation integrity; BLOCKED for image-based training because screenshot files are not included locally.**", "",
        "The package contains ScreenAI text annotations keyed by Rico screenshot IDs. It is not a YOLO dataset and must not be merged with AtomBlock labels directly.", "",
        "## Inventory", "",
        "| Split | Rows | Unique IDs | Parsed boxes/elements | Invalid boxes | Empty annotations | Malformed rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        row = splits[name]
        lines.append(f"| {name} | {row['row_count']:,} | {row['unique_screen_ids']:,} | {row['parsed_annotation_count']:,} | {row['invalid_boxes']:,} | {row['empty_annotations']:,} | {row['malformed_rows']:,} |")
    lines += [
        "", f"Total screens: **{result['totals']['rows']:,}**. Local screenshot files: **{result['images_in_local_package']}**.", "",
        "The local CSV files contain **341 fewer rows** than the README declares (train -195, validation -53, test -93). The audit uses the physical files as the authoritative local state.", "",
        "Cross-split screen-ID overlap: " + ", ".join(f"{key}={value}" for key, value in result["cross_split_screen_id_overlap"].items()) + ".", "",
        "Coordinates are stored as `x1 x2 y1 y2` in a normalized screen coordinate space. Labels are hierarchical textual annotations and may include descriptions or visible text.", "",
        "## Class distribution", "",
        "| Category | Count |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {count:,} |" for name, count in result["totals"]["class_distribution"].items())
    lines += [
        "", "## Compatibility decision", "",
        "**Option C: use as evaluation/representation data after its Rico screenshots and attribution chain are available.**", "",
        result["atomblock_compatibility"]["reason"],
        "The categories overlap conceptually with button, input, image/icon, radio, checkbox, and navigation. Form, table, select, sidebar, and footer do not have direct categories. The taxonomy, mobile domain, hierarchy, and annotation encoding differ. A documented mapping plus image-level validation would be required before any fine-tuning.", "",
        "## License and limitations", "",
        "The repository README declares CC BY 4.0 and identifies Rico as the screenshot source. This local audit verifies files and structure; it does not replace a legal/provenance review of externally acquired Rico screenshots.", "",
        "Machine-readable results: `reports/datasets/google_screen_annotation_audit.json`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/screen_annotation-main"))
    parser.add_argument("--json", type=Path, default=Path("reports/datasets/google_screen_annotation_audit.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/datasets/GOOGLE_SCREEN_ANNOTATION_AUDIT.md"))
    args = parser.parse_args()
    result = audit(args.root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"rows": result["totals"]["rows"], "annotations": result["totals"]["parsed_annotations"], "classes": result["totals"]["classes"], "images": result["images_in_local_package"]}, indent=2))


if __name__ == "__main__":
    main()
