"""Create a fixed, source-grouped 399-screen WebUI transfer-evaluation slice."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.webui_evaluation import mapping_for_tag, prepare_boxes  # noqa: E402


def split_name(shard: Path) -> str:
    return shard.name.split("-")[0]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=133, help="Each sample contributes all three viewports")
    args = parser.parse_args()
    data_root = ROOT / "datasets/webui/data"
    shards = sorted(data_root.glob("*.parquet"))
    audit = json.loads((ROOT / "reports/datasets/webui_dataset_audit.json").read_text(encoding="utf-8"))
    duplicate_groups = audit["integrity"]["cross_split_duplicate_image_examples"]
    excluded_sample_ids = sorted({item[1] for group in duplicate_groups for item in group})
    exclusion_payload = {"reason": "Exact image hash occurs across nominal WebUI splits.", "group_count": len(duplicate_groups), "excluded_source_sample_ids": excluded_sample_ids, "groups": duplicate_groups}
    exclusion_path = ROOT / "reports/visionqa/webui_cross_split_duplicate_exclusions.json"
    exclusion_path.write_text(json.dumps(exclusion_payload, indent=2) + "\n", encoding="utf-8")

    groups: dict[tuple[str, str], dict] = {}
    all_tags: set[str] = set()
    weak_weights = {"block-table": 12, "select": 10, "block-sidebar": 9, "block-form": 8, "block-footer": 6, "block-nav": 5, "input": 3, "image": 2, "button": 1, "link": 1}
    columns = ["sample_id", "viewport", "source_name", "component_type", "bboxes"]
    for shard in shards:
        split = split_name(shard)
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                sample_id = row["sample_id"]
                if sample_id in excluded_sample_ids: continue
                key = (split, sample_id)
                group = groups.setdefault(key, {"split": split, "sample_id": sample_id, "source_name": row["source_name"], "component_type": row["component_type"], "viewports": set(), "class_counts": Counter(), "shards": set()})
                group["viewports"].add(row["viewport"]); group["shards"].add(shard.name)
                tags = row["bboxes"]["tag"] if row.get("bboxes") else []
                all_tags.update(str(tag or "").lower() for tag in tags)
                for tag in tags:
                    mapping = mapping_for_tag(str(tag or ""))
                    if mapping["include_in_quantitative_scoring"]:
                        group["class_counts"][mapping["qasense_target_class"]] += 1
    candidates = [group for group in groups.values() if group["viewports"] == {"desktop", "tablet", "mobile"} and sum(group["class_counts"].values()) > 0]
    for group in candidates:
        group["score"] = sum(min(group["class_counts"].get(name, 0), 12) * weight for name, weight in weak_weights.items())
        group["tie"] = canonical_hash([group["split"], group["sample_id"]])
    buckets: dict[str, list[dict]] = defaultdict(list)
    for group in candidates: buckets[group["source_name"]].append(group)
    for bucket in buckets.values(): bucket.sort(key=lambda item: (-item["score"], item["tie"]))
    selected: list[dict] = []
    source_names = sorted(buckets)
    while len(selected) < args.samples:
        progress = False
        for source in source_names:
            if buckets[source] and len(selected) < args.samples:
                selected.append(buckets[source].pop(0)); progress = True
        if not progress: break
    if len(selected) != args.samples:
        raise RuntimeError(f"Only selected {len(selected)} complete sample groups")
    selected_keys = {(row["split"], row["sample_id"]) for row in selected}

    mapping_entries = [mapping_for_tag(tag) for tag in sorted(all_tags)]
    mapping_payload = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(), "policy": "Only exact/compatible native DOM tags are scored. Generic containers/SVG and unavailable input subtypes are excluded.", "mappings": mapping_entries}
    mapping_path = ROOT / "reports/visionqa/webui_class_mapping.json"
    mapping_path.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")

    output_images = ROOT / "runs/visionqa/webui_transfer_eval/images"
    output_images.mkdir(parents=True, exist_ok=True)
    screens: list[dict] = []
    class_counts, exclusion_counts, source_counts = Counter(), Counter(), Counter()
    extract_columns = ["sample_id", "image", "viewport", "source_name", "component_type", "bboxes"]
    for shard in shards:
        split = split_name(shard)
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=extract_columns):
            for row_index, row in enumerate(batch.to_pylist()):
                key = (split, row["sample_id"])
                if key not in selected_keys: continue
                image_bytes = row["image"]["bytes"]
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                with Image.open(io.BytesIO(image_bytes)) as opened: width, height = opened.size
                screen_id = f"webui-{split}-{row['sample_id']}-{row['viewport']}"
                image_path = output_images / f"{screen_id}.png"
                image_path.write_bytes(image_bytes)
                prepared = prepare_boxes(row["bboxes"], width, height)
                for box in prepared["valid_ground_truth_boxes"]: class_counts[box["target_class"]] += 1
                for box in prepared["excluded_invalid_or_ambiguous_boxes"]: exclusion_counts[box["reason"]] += 1
                source_counts[row["source_name"]] += 1
                screens.append({
                    "screen_id": screen_id, "source_sample_id": row["sample_id"], "viewport": row["viewport"],
                    "split": split, "source_name": row["source_name"], "source_shard": shard.name,
                    "image_identifier": {"sha256": image_hash, "embedded_path": row["image"].get("path"), "extracted_path": str(image_path), "width": width, "height": height},
                    **prepared, "mapped_classes": sorted({box["target_class"] for box in prepared["valid_ground_truth_boxes"]}),
                    "review_status": "pending_external_review_record",
                })
    screens.sort(key=lambda row: (row["source_name"], row["source_sample_id"], {"desktop": 0, "tablet": 1, "mobile": 2}[row["viewport"]]))
    if len(screens) != args.samples * 3:
        raise RuntimeError(f"Expected {args.samples * 3} screens, extracted {len(screens)}")
    body = {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(), "fixed_slice": True,
        "selection": {"source_grouped": True, "source_sample_groups": args.samples, "screens": len(screens), "all_three_viewports_preserved": True, "deterministic_round_robin_by_source_then_weak_class_coverage": True},
        "cross_split_duplicate_exclusion_file": str(exclusion_path), "class_mapping_file": str(mapping_path),
        "class_box_counts": dict(class_counts), "box_exclusion_counts": dict(exclusion_counts), "source_screen_counts": dict(source_counts),
        "screens": screens,
    }
    body["manifest_content_sha256"] = canonical_hash(body)
    manifest_path = ROOT / "reports/visionqa/webui_eval_manifest.json"
    manifest_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screens": len(screens), "sample_groups": args.samples, "sources": len(source_counts), "classes": dict(class_counts), "exclusions": dict(exclusion_counts), "manifest_sha256": body["manifest_content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
