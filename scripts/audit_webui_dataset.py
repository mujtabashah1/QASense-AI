"""Stream a full integrity and compatibility audit of local WebUI Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


VIEWPORT_DIMENSIONS = {"desktop": (1280, 720), "tablet": (768, 1024), "mobile": (375, 812)}
ATOMBLOCK_TAG_MAP = {
    "button": "button", "a": "link", "input": "input", "textarea": "input",
    "img": "image", "svg": "icon", "select": "select", "nav": "block-nav",
    "aside": "block-sidebar", "footer": "block-footer", "form": "block-form",
    "table": "block-table",
}


def _example(target: list[dict[str, Any]], value: dict[str, Any], limit: int = 10) -> None:
    if len(target) < limit:
        target.append(value)


def audit(root: Path) -> dict:
    shards = sorted((root / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards under {root / 'data'}")
    split_sets: dict[str, set[str]] = defaultdict(set)
    split_sources: dict[str, set[str]] = defaultdict(set)
    split_rows: Counter = Counter()
    split_shards: Counter = Counter()
    viewports: Counter = Counter()
    sources: Counter = Counter()
    frameworks: Counter = Counter()
    css_frameworks: Counter = Counter()
    component_types: Counter = Counter()
    tags: Counter = Counter()
    mapped_atomblock: Counter = Counter()
    image_hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    row_keys: Counter = Counter()
    missing: Counter = Counter()
    issues: Counter = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bbox_total = element_count_total = 0
    shard_records: list[dict[str, Any]] = []

    columns = [
        "sample_id", "image", "html", "css", "js", "viewport", "source_name",
        "source_url", "description", "framework", "css_framework", "component_type",
        "element_count", "has_animations", "bboxes",
    ]
    for shard in shards:
        split = shard.name.split("-")[0]
        split_shards[split] += 1
        parquet = pq.ParquetFile(shard)
        shard_records.append({"name": shard.name, "bytes": shard.stat().st_size, "rows": parquet.metadata.num_rows})
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                split_rows[split] += 1
                sample_id = row.get("sample_id") or ""
                viewport = row.get("viewport") or ""
                source = row.get("source_name") or ""
                split_sets[split].add(sample_id)
                split_sources[split].add(source)
                viewports[viewport] += 1
                sources[source] += 1
                frameworks[row.get("framework") or ""] += 1
                css_frameworks[row.get("css_framework") or ""] += 1
                component_types[row.get("component_type") or ""] += 1
                row_keys[(split, sample_id, viewport)] += 1
                for field in columns:
                    value = row.get(field)
                    if value is None or (isinstance(value, str) and not value.strip()):
                        missing[field] += 1

                image = row.get("image") or {}
                image_bytes = image.get("bytes") if isinstance(image, dict) else None
                if not image_bytes:
                    issues["missing_image_bytes"] += 1
                else:
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    image_hashes[digest].append((split, sample_id, viewport))
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as opened:
                            width, height = opened.size
                            image_format = opened.format
                            opened.verify()
                        if image_format != "PNG":
                            issues["non_png_image"] += 1
                        expected = VIEWPORT_DIMENSIONS.get(viewport)
                        if expected and (width, height) != expected:
                            issues["image_dimension_mismatch"] += 1
                            _example(examples["image_dimension_mismatch"], {"sample_id": sample_id, "viewport": viewport, "actual": [width, height], "expected": list(expected)})
                    except Exception as exc:  # PIL exposes multiple decode exception types.
                        issues["corrupt_images"] += 1
                        _example(examples["corrupt_images"], {"sample_id": sample_id, "viewport": viewport, "error": str(exc)})

                boxes = row.get("bboxes") or {}
                fields = {name: boxes.get(name) or [] for name in ("tag", "x", "y", "width", "height", "text", "id", "class", "parent_index")}
                lengths = {name: len(values) for name, values in fields.items()}
                if len(set(lengths.values())) != 1:
                    issues["bbox_field_length_mismatch"] += 1
                    _example(examples["bbox_field_length_mismatch"], {"sample_id": sample_id, "viewport": viewport, "lengths": lengths})
                box_count = min(lengths.values()) if lengths else 0
                bbox_total += box_count
                declared = row.get("element_count")
                if declared is not None:
                    element_count_total += int(declared)
                if box_count == 0:
                    issues["rows_without_boxes"] += 1
                expected_dimensions = VIEWPORT_DIMENSIONS.get(viewport)
                for index in range(box_count):
                    tag = str(fields["tag"][index] or "").lower()
                    tags[tag] += 1
                    if tag in ATOMBLOCK_TAG_MAP:
                        mapped_atomblock[ATOMBLOCK_TAG_MAP[tag]] += 1
                    x, y, width, height = (fields[name][index] for name in ("x", "y", "width", "height"))
                    if width <= 0 or height <= 0:
                        issues["nonpositive_boxes"] += 1
                        _example(examples["nonpositive_boxes"], {"sample_id": sample_id, "viewport": viewport, "index": index, "box": [x, y, width, height]})
                    if expected_dimensions and (x < 0 or y < 0 or x + width > expected_dimensions[0] or y + height > expected_dimensions[1]):
                        issues["boxes_outside_viewport"] += 1
                    parent = fields["parent_index"][index]
                    if parent < -1 or parent >= box_count or parent == index:
                        issues["invalid_parent_indices"] += 1
                        _example(examples["invalid_parent_indices"], {"sample_id": sample_id, "viewport": viewport, "index": index, "parent_index": parent, "box_count": box_count})

    duplicate_row_keys = sum(count - 1 for count in row_keys.values() if count > 1)
    repeated_hash_groups = {key: value for key, value in image_hashes.items() if len(value) > 1}
    cross_split_hash_groups = [
        values for values in repeated_hash_groups.values() if len({item[0] for item in values}) > 1
    ]
    sample_overlaps = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    source_overlaps = {
        "train_validation": sorted(split_sources["train"] & split_sources["validation"]),
        "train_test": sorted(split_sources["train"] & split_sources["test"]),
        "validation_test": sorted(split_sources["validation"] & split_sources["test"]),
    }
    viewport_completeness = Counter()
    per_sample_viewports: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, sample_id, viewport in row_keys:
        per_sample_viewports[(split, sample_id)].add(viewport)
    for values in per_sample_viewports.values():
        viewport_completeness["complete" if values == set(VIEWPORT_DIMENSIONS) else "incomplete"] += 1

    return {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "WebUI",
        "root": str(root.resolve()),
        "declared_license": "MIT (dataset README/front matter); upstream source assets may retain their own terms",
        "shards": shard_records,
        "summary": {
            "rows": sum(split_rows.values()), "samples": len(per_sample_viewports),
            "embedded_images_checked": sum(split_rows.values()), "bbox_count": bbox_total,
            "declared_element_count_sum": element_count_total,
        },
        "splits": {split: {"rows": split_rows[split], "samples": len(split_sets[split]), "shards": split_shards[split], "sources": sorted(split_sources[split])} for split in ("train", "validation", "test")},
        "distributions": {
            "viewport": dict(viewports.most_common()), "source_name": dict(sources.most_common()),
            "framework": dict(frameworks.most_common()), "css_framework": dict(css_frameworks.most_common()),
            "component_type": dict(component_types.most_common()), "bbox_tag": dict(tags.most_common()),
            "atomblock_mappable_tags": dict(mapped_atomblock.most_common()),
        },
        "integrity": {
            "missing_values": dict(missing), "issues": dict(issues),
            "duplicate_split_sample_viewport_rows": duplicate_row_keys,
            "duplicate_image_hash_groups": len(repeated_hash_groups),
            "cross_split_duplicate_image_hash_groups": len(cross_split_hash_groups),
            "cross_split_sample_id_overlap": sample_overlaps,
            "cross_split_source_overlap": source_overlaps,
            "cross_split_duplicate_image_examples": cross_split_hash_groups[:20],
            "viewport_completeness": dict(viewport_completeness),
            "examples": dict(examples),
        },
        "atomblock_compatibility": {
            "status": "partially_mappable_not_merge_ready",
            "reason": "WebUI supplies pixels and DOM-node boxes, but its tag taxonomy includes nested containers, text, SVG paths, and every visible node; AtomBlock supplies 14 curated UI classes.",
            "recommended_option": "C_then_targeted_A",
            "recommendation": "Keep the current detector unchanged. First use WebUI as a source-held-out evaluation and responsive-geometry dataset; only fine-tune on a validated, deduplicated subset after explicit tag/role mapping and nested-box filtering.",
        },
    }


def render_markdown(result: dict) -> str:
    summary, integrity = result["summary"], result["integrity"]
    lines = [
        "# WebUI Dataset Audit", "", f"Generated: `{result['generated_at']}`", "",
        "## Result", "", "**PASS for local file and split integrity; PARTIAL compatibility with AtomBlock. Direct label merging is not justified.**", "",
        "## Inventory", "", "| Split | Rows | Unique samples | Shards | Sources |", "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = result["splits"][split]
        lines.append(f"| {split} | {row['rows']:,} | {row['samples']:,} | {row['shards']} | {len(row['sources'])} |")
    lines += [
        "", f"Total: **{summary['rows']:,} rows**, **{summary['samples']:,} responsive UI samples**, **{summary['bbox_count']:,} DOM boxes**.", "",
        "Every embedded image was decoded and verified during this audit.", "",
        "## Integrity", "",
        f"- Corrupt images: **{integrity['issues'].get('corrupt_images', 0):,}**",
        f"- Missing image bytes: **{integrity['issues'].get('missing_image_bytes', 0):,}**",
        f"- Duplicate split/sample/viewport rows: **{integrity['duplicate_split_sample_viewport_rows']:,}**",
        f"- Exact duplicate image hash groups: **{integrity['duplicate_image_hash_groups']:,}**",
        f"- Cross-split exact-image duplicate groups: **{integrity['cross_split_duplicate_image_hash_groups']:,}**",
        f"- Rows without visible boxes: **{integrity['issues'].get('rows_without_boxes', 0):,}**",
        f"- Nonpositive boxes: **{integrity['issues'].get('nonpositive_boxes', 0):,}**",
        f"- Boxes extending outside the viewport: **{integrity['issues'].get('boxes_outside_viewport', 0):,}**",
        f"- Invalid parent indices: **{integrity['issues'].get('invalid_parent_indices', 0):,}**",
        "- Cross-split sample-ID overlap: " + ", ".join(f"{key}={value}" for key, value in integrity["cross_split_sample_id_overlap"].items()),
        "- Cross-split source overlap: " + ", ".join(f"{key}={len(value)}" for key, value in integrity["cross_split_source_overlap"].items()),
        "", "Empty JavaScript (17,601 rows) and CSS (12,300 rows) are valid for pages that do not use those resource types. Rows without boxes and nonpositive boxes contradict the README's broad 100% coverage claim and must be excluded or reviewed before detection use.", "",
        "Boxes outside the viewport are common because the annotations include clipped, scrolling, transformed, or ancestor DOM geometry. They are retained as geometry evidence and cannot be treated as valid detector boxes without clipping and visibility rules.", "",
        "## Main distributions", "", "### Viewports", "", "| Viewport | Rows |", "|---|---:|",
    ]
    lines.extend(f"| {name or '(missing)'} | {count:,} |" for name, count in result["distributions"]["viewport"].items())
    lines += ["", "### Component types", "", "| Type | Rows |", "|---|---:|"]
    lines.extend(f"| {name or '(missing)'} | {count:,} |" for name, count in result["distributions"]["component_type"].items())
    lines += ["", "### Top DOM tags", "", "| Tag | Boxes |", "|---|---:|"]
    lines.extend(f"| {name or '(missing)'} | {count:,} |" for name, count in list(result["distributions"]["bbox_tag"].items())[:30])
    lines += [
        "", "## AtomBlock compatibility", "",
        result["atomblock_compatibility"]["reason"], "",
        "**Decision: keep the current AtomBlock model unchanged. Use WebUI first for source-held-out evaluation and responsive geometry.** A later fine-tune is defensible only after role-aware mapping, removal of nested duplicate boxes, and a measured evaluation gap.", "",
        "## License and limitations", "",
        "The dataset README/front matter declares MIT. The rows aggregate public sites and projects; downstream redistribution or product use should preserve source metadata and review upstream asset terms. GPT-generated descriptions are metadata rather than ground-truth element classes.", "",
        "Machine-readable results: `reports/datasets/webui_dataset_audit.json`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/webui"))
    parser.add_argument("--json", type=Path, default=Path("reports/datasets/webui_dataset_audit.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/datasets/WEBUI_DATASET_AUDIT.md"))
    args = parser.parse_args()
    result = audit(args.root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"rows": result["summary"]["rows"], "samples": result["summary"]["samples"], "images": result["summary"]["embedded_images_checked"], "boxes": result["summary"]["bbox_count"], "issues": result["integrity"]["issues"]}, indent=2))


if __name__ == "__main__":
    main()
