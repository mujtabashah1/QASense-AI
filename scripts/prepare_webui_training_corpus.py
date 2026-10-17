"""Build a leakage-safe WebUI-derived YOLO corpus without training a model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.webui_corpus import (  # noqa: E402
    SEED, V2_CLASSES, V2_CLASS_TO_ID, VIEWPORTS, assign_source_grouped_split,
    canonical_hash, class_schema, group_id, screen_id, stable_rank, xyxy_to_yolo,
)
from src.visionqa.webui_evaluation import prepare_boxes  # noqa: E402


def original_split(shard: Path) -> str:
    return shard.name.split("-")[0]


def load_frozen_groups() -> tuple[set[str], dict]:
    path = ROOT / "reports/visionqa/webui_eval_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"
    actual = payload.get("manifest_content_sha256")
    check = dict(payload); check.pop("manifest_content_sha256", None)
    if actual != expected or canonical_hash(check) != expected:
        raise RuntimeError("Frozen WebUI evaluation manifest hash changed")
    groups = {group_id(row["split"], row["source_sample_id"]) for row in payload["screens"]}
    return groups, payload


def load_known_duplicate_groups() -> tuple[set[str], dict]:
    path = ROOT / "reports/visionqa/webui_cross_split_duplicate_exclusions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {
        group_id(split, sample)
        for duplicate in payload["groups"] for split, sample, _viewport in duplicate
    }
    return groups, payload


def choose_review_screens(screens: list[dict], target: int = 300) -> list[str]:
    """Systematic stratified review: weak classes, density, exclusions, sources and viewports."""
    weak = {"select", "block-form", "block-table", "block-sidebar", "block-nav"}
    ranked = []
    for row in screens:
        classes = set(row["class_counts"])
        score = 100 * len(classes & weak) + min(row["valid_mapped_box_count"], 200)
        score += min(row["excluded_box_count"], 100)
        ranked.append((-
            score, stable_rank(row["source"], row["screen_id"]), row))
    ranked.sort(key=lambda value: (value[0], value[1]))
    selected: list[dict] = []
    seen_groups: set[str] = set()
    # First guarantee every class and viewport when available.
    for class_name in V2_CLASSES:
        for viewport in VIEWPORTS:
            candidate = next((r for _, _, r in ranked if r["viewport"] == viewport and class_name in r["class_counts"] and r["group_id"] not in seen_groups), None)
            if candidate:
                selected.append(candidate); seen_groups.add(candidate["group_id"])
    by_source: dict[str, list[dict]] = defaultdict(list)
    for _, _, row in ranked:
        if row["group_id"] not in seen_groups:
            by_source[row["source"]].append(row)
    while len(selected) < target:
        progressed = False
        for source in sorted(by_source):
            while by_source[source] and by_source[source][0]["group_id"] in seen_groups:
                by_source[source].pop(0)
            if by_source[source] and len(selected) < target:
                row = by_source[source].pop(0)
                selected.append(row); seen_groups.add(row["group_id"]); progressed = True
        if not progressed:
            break
    if len(selected) < target:
        for _, _, row in ranked:
            if row["screen_id"] not in {item["screen_id"] for item in selected}:
                selected.append(row)
                if len(selected) == target: break
    return [row["screen_id"] for row in selected[:target]]


def render_review_assets(rows: list[dict], output: Path) -> None:
    overlay_dir = output / "review_overlays"; sheet_dir = output / "review_contact_sheets"
    overlay_dir.mkdir(parents=True, exist_ok=True); sheet_dir.mkdir(parents=True, exist_ok=True)
    colors = [(225, 50, 50), (30, 150, 255), (40, 180, 90), (255, 150, 30), (170, 70, 220)]
    thumbnails = []
    font = ImageFont.load_default()
    for row in rows:
        image = Image.open(row["processed_image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in row["boxes"]:
            color = colors[box["class_id"] % len(colors)]
            draw.rectangle(box["bbox_xyxy"], outline=color, width=2)
            draw.text((box["bbox_xyxy"][0] + 2, box["bbox_xyxy"][1] + 2), box["class_name"], fill=color, font=font)
        overlay = overlay_dir / f"{row['screen_id']}.jpg"
        image.thumbnail((540, 720)); image.save(overlay, quality=88)
        canvas = Image.new("RGB", (560, 770), "white"); canvas.paste(image, ((560-image.width)//2, 35))
        title = f"{row['screen_id']} | {row['split']} | boxes={row['valid_mapped_box_count']} excluded={row['excluded_box_count']}"
        ImageDraw.Draw(canvas).text((8, 8), title[:100], fill="black", font=font)
        thumbnails.append(canvas)
    for start in range(0, len(thumbnails), 10):
        batch = thumbnails[start:start+10]
        sheet = Image.new("RGB", (1120, 3850), "white")
        for index, thumb in enumerate(batch): sheet.paste(thumb, ((index % 2) * 560, (index // 2) * 770))
        sheet.save(sheet_dir / f"review_contact_{start//10+1:02d}.jpg", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-group-fraction", type=float, default=0.05)
    parser.add_argument("--review-screens", type=int, default=300)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    data_root = ROOT / "datasets/webui/data"
    shards = sorted(data_root.glob("*.parquet"))
    output = ROOT / "datasets/processed/visionqa_webui_v2"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing corpus: {output}")
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(parents=True)
    reports = ROOT / "reports/visionqa"; reports.mkdir(parents=True, exist_ok=True)

    frozen_groups, frozen_manifest = load_frozen_groups()
    known_duplicate_groups, duplicate_payload = load_known_duplicate_groups()
    exclusions: list[dict] = []
    for gid in sorted(frozen_groups): exclusions.append({"group_id": gid, "reason": "frozen_eval_group"})
    for gid in sorted(known_duplicate_groups - frozen_groups): exclusions.append({"group_id": gid, "reason": "cross_split_exact_duplicate"})
    blocked = frozen_groups | known_duplicate_groups

    groups: dict[str, dict] = {}
    columns = ["sample_id", "image", "viewport", "source_name", "bboxes"]
    box_exclusions = Counter(); decode_failures = []
    original_box_count = 0
    for shard in shards:
        split = original_split(shard)
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                gid = group_id(split, row["sample_id"])
                if gid in blocked: continue
                sid = screen_id(split, row["sample_id"], row["viewport"])
                image_bytes = row["image"]["bytes"]
                try:
                    with Image.open(io.BytesIO(image_bytes)) as opened:
                        opened.load(); width, height = opened.size
                        if width <= 0 or height <= 0: raise ValueError("nonpositive dimensions")
                except Exception as exc:
                    decode_failures.append({"screen_id": sid, "group_id": gid, "reason": str(exc)})
                    continue
                bboxes = row.get("bboxes") or {}
                original_box_count += len(bboxes.get("tag") or [])
                prepared = prepare_boxes(bboxes, width, height)
                for excluded in prepared["excluded_invalid_or_ambiguous_boxes"]: box_exclusions[excluded["reason"]] += 1
                box_exclusions["ambiguous_or_unmapped"] += sum(prepared["unmapped_classes"].values())
                boxes = []
                for box in prepared["valid_ground_truth_boxes"]:
                    class_name = box["target_class"]
                    boxes.append({"class_id": V2_CLASS_TO_ID[class_name], "class_name": class_name, "bbox_xyxy": box["bbox_xyxy"], "source_label": box["source_label"], "was_clipped": box["was_clipped_to_viewport"]})
                group = groups.setdefault(gid, {"group_id": gid, "source": row["source_name"], "source_sample_id": row["sample_id"], "original_split": split, "screens": []})
                group["screens"].append({"screen_id": sid, "group_id": gid, "source": row["source_name"], "source_sample_id": row["sample_id"], "viewport": row["viewport"], "source_shard": shard.name, "width": width, "height": height, "image_bytes": image_bytes, "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "embedded_image_path": row["image"].get("path"), "boxes": boxes, "excluded_box_count": len(prepared["excluded_invalid_or_ambiguous_boxes"]) + sum(prepared["unmapped_classes"].values()), "class_counts": dict(Counter(box["class_name"] for box in boxes))})

    complete, invalid_groups = [], []
    for group in groups.values():
        viewports = {row["viewport"] for row in group["screens"]}
        if viewports != set(VIEWPORTS):
            invalid_groups.append(group["group_id"]); exclusions.append({"group_id": group["group_id"], "reason": "invalid_image"}); continue
        group["screens"].sort(key=lambda row: VIEWPORTS.index(row["viewport"]))
        group["valid_box_count"] = sum(len(row["boxes"]) for row in group["screens"])
        complete.append(group)

    positives = [group for group in complete if group["valid_box_count"] > 0]
    negatives = [group for group in complete if group["valid_box_count"] == 0]
    negative_target = min(len(negatives), max(1, round(len(positives) * args.negative_group_fraction)))
    negatives.sort(key=lambda group: stable_rank(group["source"], group["group_id"], "negative"))
    retained_negatives = negatives[:negative_target]
    for group in negatives[negative_target:]: exclusions.append({"group_id": group["group_id"], "reason": "invalid_box_only"})
    selected = positives + retained_negatives

    # Keep exact-image-connected groups in the same split. Start with source-balanced assignments,
    # then move every connected component to its majority split deterministically.
    assignments = assign_source_grouped_split(selected)
    hash_to_groups: dict[str, set[str]] = defaultdict(set)
    by_id = {group["group_id"]: group for group in selected}
    for group in selected:
        for row in group["screens"]: hash_to_groups[row["image_sha256"]].add(group["group_id"])
    adjacency: dict[str, set[str]] = defaultdict(set)
    for ids in hash_to_groups.values():
        if len(ids) > 1:
            for gid in ids: adjacency[gid].update(ids - {gid})
    visited = set()
    for gid in sorted(adjacency):
        if gid in visited: continue
        stack, component = [gid], []
        while stack:
            current = stack.pop()
            if current in visited: continue
            visited.add(current); component.append(current); stack.extend(adjacency[current] - visited)
        counts = Counter(assignments[item] for item in component)
        chosen = "val" if counts["val"] > counts["train"] else "train"
        for item in component: assignments[item] = chosen

    dataset_rows, split_groups = [], {"train": [], "val": []}
    for group in sorted(selected, key=lambda row: row["group_id"]):
        split = assignments[group["group_id"]]; split_groups[split].append(group)
        for row in group["screens"]:
            image_path = output / split / "images" / f"{row['screen_id']}.png"
            label_path = output / split / "labels" / f"{row['screen_id']}.txt"
            image_path.write_bytes(row.pop("image_bytes"))
            lines = []
            for box in row["boxes"]:
                yolo = xyxy_to_yolo(box["bbox_xyxy"], row["width"], row["height"])
                lines.append(f"{box['class_id']} " + " ".join(f"{value:.8f}" for value in yolo))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            dataset_rows.append({
                "screen_id": row["screen_id"], "group_id": group["group_id"], "source": group["source"],
                "source_sample_id": group["source_sample_id"], "viewport": row["viewport"],
                "original_split": group["original_split"], "source_shard": row["source_shard"],
                "original_image_identifier": {"embedded_path": row["embedded_image_path"], "sha256": row["image_sha256"], "width": row["width"], "height": row["height"]},
                "processed_image_path": str(image_path), "processed_label_path": str(label_path), "split": split,
                "valid_mapped_box_count": len(row["boxes"]), "excluded_box_count": row["excluded_box_count"],
                "class_counts": row["class_counts"], "review_status": "not_in_review_sample", "boxes": row["boxes"],
            })

    review_ids = set(choose_review_screens(dataset_rows, args.review_screens))
    review_rows = [row for row in dataset_rows if row["screen_id"] in review_ids]
    for row in review_rows: row["review_status"] = "pending_human_review"
    render_review_assets(review_rows, reports / "webui_training_review")
    review_payload = {
        "schema_version": "1.0", "created_at": now, "status": "pending_human_review",
        "selection": {"screens": len(review_rows), "stratified_for": ["all_supported_classes", "all_viewports", "multiple_sources", "weak_classes", "density", "exclusions", "small_elements", "large_structural_blocks"]},
        "reviews": [{"screen_id": row["screen_id"], "group_id": row["group_id"], "viewport": row["viewport"], "source": row["source"], "mapping_valid": None, "annotation_quality": None, "invalid_boxes_present": row["excluded_box_count"] > 0, "class_errors": [], "decision": "pending", "review_notes": ""} for row in review_rows],
    }
    (reports / "webui_training_review.json").write_text(json.dumps(review_payload, indent=2) + "\n", encoding="utf-8")

    frozen_payload = {
        "schema_version": "1.0", "created_at": now, "original_evaluation_manifest": "reports/visionqa/webui_eval_manifest.json",
        "original_evaluation_manifest_sha256": frozen_manifest["manifest_content_sha256"],
        "group_count": len(frozen_groups), "group_ids": sorted(frozen_groups),
        "screen_ids": sorted(row["screen_id"] for row in frozen_manifest["screens"]),
        "source_ids": sorted({row["source_name"] for row in frozen_manifest["screens"]}),
    }
    (reports / "webui_frozen_eval_groups.json").write_text(json.dumps(frozen_payload, indent=2) + "\n", encoding="utf-8")
    schema_payload = {"schema_version": "1.0", "created_at": now, "classes": class_schema(), "excluded_atomblock_classes": ["icon", "checkbox", "radio", "block"]}
    (reports / "visionqa_v2_class_schema.json").write_text(json.dumps(schema_payload, indent=2) + "\n", encoding="utf-8")
    exclusion_payload = {"schema_version": "1.0", "created_at": now, "authoritative_duplicate_artifact": "reports/visionqa/webui_cross_split_duplicate_exclusions.json", "known_duplicate_group_count": duplicate_payload["group_count"], "group_exclusions": exclusions, "box_exclusion_reason_counts": dict(box_exclusions), "decode_failures": decode_failures}
    (reports / "webui_training_exclusions.json").write_text(json.dumps(exclusion_payload, indent=2) + "\n", encoding="utf-8")

    split_records = [{"group_id": group["group_id"], "source": group["source"], "source_sample_id": group["source_sample_id"], "screen_ids": [screen_id(group["original_split"], group["source_sample_id"], viewport) for viewport in VIEWPORTS], "viewports": list(VIEWPORTS), "split": split} for split, rows in split_groups.items() for group in rows]
    split_records.sort(key=lambda row: row["group_id"])
    split_manifest = {"schema_version": "1.0", "created_at": now, "seed": SEED, "logic": "Source-stratified deterministic 80/20 group split; complete viewport triplets kept together; exact-image-connected groups consolidated into one split.", "groups": split_records}
    split_manifest["manifest_sha256"] = canonical_hash(split_manifest)
    (output / "manifests/split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    dataset_manifest = {"schema_version": "1.0", "created_at": now, "screens": dataset_rows}
    dataset_manifest["manifest_sha256"] = canonical_hash(dataset_manifest)
    (output / "manifests/dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    provenance = {"schema_version": "1.0", "created_at": now, "source_dataset": "datasets/webui", "source_paths": [str(path) for path in shards], "webui_audit_report": "reports/datasets/WEBUI_DATASET_AUDIT.md", "mapping_report": "reports/visionqa/webui_class_mapping.json", "class_schema": "reports/visionqa/visionqa_v2_class_schema.json", "selection_rules": "All complete positive groups plus a deterministic 5% negative-group subset after exclusions.", "exclusion_rules": ["frozen_eval_group", "cross_split_exact_duplicate", "invalid_image", "invalid_box_only", "ambiguous_mapping", "unusable_annotation", "manual_rejection"], "human_review_rules": "300 stratified screens covering classes, viewports, sources, density, exclusions, small controls, and structural blocks.", "seed": SEED, "script": "scripts/prepare_webui_training_corpus.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifests/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    yaml = "path: C:/QASense-AI/datasets/processed/visionqa_webui_v2\ntrain: train/images\nval: val/images\n\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(V2_CLASSES))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    print(json.dumps({"candidate_groups": len(complete), "positive_groups": len(positives), "negative_groups_retained": len(retained_negatives), "train_groups": len(split_groups["train"]), "val_groups": len(split_groups["val"]), "screens": len(dataset_rows), "review_screens": len(review_rows), "original_boxes": original_box_count, "valid_boxes": sum(row["valid_mapped_box_count"] for row in dataset_rows), "split_manifest_sha256": split_manifest["manifest_sha256"], "dataset_manifest_sha256": dataset_manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
