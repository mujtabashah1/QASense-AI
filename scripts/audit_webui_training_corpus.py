"""Audit the processed WebUI VisionQA v2 corpus without training."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.webui_corpus import V2_CLASSES, canonical_hash  # noqa: E402


def verified_manifest(path: Path) -> tuple[dict, bool, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256", "")
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return payload, recorded == canonical_hash(body), recorded


def main() -> None:
    corpus = ROOT / "datasets/processed/visionqa_webui_v2"
    dataset, dataset_hash_ok, dataset_hash = verified_manifest(corpus / "manifests/dataset_manifest.json")
    split_manifest, split_hash_ok, split_hash = verified_manifest(corpus / "manifests/split_manifest.json")
    frozen = json.loads((ROOT / "reports/visionqa/webui_frozen_eval_groups.json").read_text(encoding="utf-8"))
    duplicates = json.loads((ROOT / "reports/visionqa/webui_cross_split_duplicate_exclusions.json").read_text(encoding="utf-8"))
    exclusions = json.loads((ROOT / "reports/visionqa/webui_training_exclusions.json").read_text(encoding="utf-8"))

    rows = dataset["screens"]
    frozen_ids = set(frozen["group_ids"])
    duplicate_ids = {f"webui-{split}-{sample}" for group in duplicates["groups"] for split, sample, _ in group}
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    class_boxes = {split: Counter() for split in ("train", "val")}
    class_screens = {split: defaultdict(set) for split in ("train", "val")}
    class_groups = {split: defaultdict(set) for split in ("train", "val")}
    class_viewports = {split: defaultdict(Counter) for split in ("train", "val")}
    viewports = {split: Counter() for split in ("train", "val")}
    source_groups = {split: defaultdict(set) for split in ("train", "val")}
    source_screens = {split: Counter() for split in ("train", "val")}
    image_stems = {split: set() for split in ("train", "val")}
    label_stems = {split: set() for split in ("train", "val")}
    negative_screens = Counter()
    negative_groups = {split: set() for split in ("train", "val")}
    missing_images: list[str] = []
    missing_labels: list[str] = []
    corrupt_images: list[dict] = []
    invalid_lines: list[dict] = []

    for row in rows:
        split, gid, sid = row["split"], row["group_id"], row["screen_id"]
        group_splits[gid].add(split)
        hash_splits[row["original_image_identifier"]["sha256"]].add(split)
        viewports[split][row["viewport"]] += 1
        source_groups[split][row["source"]].add(gid)
        source_screens[split][row["source"]] += 1
        image_path, label_path = Path(row["processed_image_path"]), Path(row["processed_label_path"])
        if not image_path.exists():
            missing_images.append(sid)
        else:
            image_stems[split].add(image_path.stem)
            try:
                with Image.open(image_path) as opened:
                    opened.load()
                    expected = (row["original_image_identifier"]["width"], row["original_image_identifier"]["height"])
                    if opened.size != expected:
                        raise ValueError(f"dimensions {opened.size} != {expected}")
            except Exception as exc:
                corrupt_images.append({"screen_id": sid, "reason": str(exc)})
        if not label_path.exists():
            missing_labels.append(sid)
            continue
        label_stems[split].add(label_path.stem)
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            negative_screens[split] += 1
            negative_groups[split].add(gid)
        for number, line in enumerate(lines, 1):
            try:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("field_count")
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if class_id not in range(len(V2_CLASSES)):
                    raise ValueError("class_id")
                x, y, width, height = values
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError("normalized_box")
                name = V2_CLASSES[class_id]
                class_boxes[split][name] += 1
                class_screens[split][name].add(sid)
                class_groups[split][name].add(gid)
                class_viewports[split][name][row["viewport"]] += 1
            except Exception as exc:
                invalid_lines.append({"screen_id": sid, "line": number, "reason": str(exc), "value": line})

    dataset_groups = set(group_splits)
    leakage = {
        "frozen_evaluation_groups": sorted(dataset_groups & frozen_ids),
        "known_duplicate_groups": sorted(dataset_groups & duplicate_ids),
        "source_group_across_train_val": sorted(gid for gid, values in group_splits.items() if len(values) > 1),
        "exact_image_hash_across_train_val": sorted(value for value, splits in hash_splits.items() if len(splits) > 1),
    }
    integrity = {
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "orphan_labels": {split: sorted(label_stems[split] - image_stems[split]) for split in ("train", "val")},
        "images_missing_labels": {split: sorted(image_stems[split] - label_stems[split]) for split in ("train", "val")},
        "invalid_lines": invalid_lines,
        "corrupt_images": corrupt_images,
    }
    distributions = {}
    for split in ("train", "val"):
        total = sum(class_boxes[split].values())
        groups_in_split = {row["group_id"] for row in rows if row["split"] == split}
        distributions[split] = {
            "groups": len(groups_in_split),
            "screens": len(image_stems[split]),
            "labels": len(label_stems[split]),
            "valid_boxes": total,
            "viewports": dict(viewports[split]),
            "negative_screens": negative_screens[split],
            "negative_groups": len(negative_groups[split]),
            "unique_sources": len(source_groups[split]),
            "classes": {name: {
                "box_count": class_boxes[split][name],
                "screen_count": len(class_screens[split][name]),
                "group_count": len(class_groups[split][name]),
                "percentage_of_boxes": class_boxes[split][name] / total if total else 0,
                "viewport_box_counts": dict(class_viewports[split][name]),
            } for name in V2_CLASSES},
            "sources": {name: {"group_count": len(ids), "screen_count": source_screens[split][name]} for name, ids in sorted(source_groups[split].items())},
        }
    extreme_sources = {split: [
        {"source": source, **data, "fraction_of_groups": data["group_count"] / distributions[split]["groups"]}
        for source, data in distributions[split]["sources"].items()
        if data["group_count"] / distributions[split]["groups"] > 0.25
    ] for split in ("train", "val")}
    audit_pass = (
        dataset_hash_ok and split_hash_ok and not any(leakage.values())
        and not missing_images and not missing_labels and not corrupt_images and not invalid_lines
        and not any(integrity["orphan_labels"].values()) and not any(integrity["images_missing_labels"].values())
    )
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_result": "PASS" if audit_pass else "FAIL",
        "manifest_integrity": {"dataset": dataset_hash_ok, "split": split_hash_ok},
        "manifest_sha256": {"dataset": dataset_hash, "split": split_hash},
        "counts": {
            "candidate_groups": len(dataset_groups), "screens": len(rows),
            "valid_boxes": sum(value["valid_boxes"] for value in distributions.values()),
            "excluded_box_reason_counts": exclusions["box_exclusion_reason_counts"],
            "decode_failures": len(exclusions["decode_failures"]),
        },
        "distributions": distributions, "leakage": leakage, "integrity": integrity,
        "extreme_source_domination": extreme_sources,
    }
    metrics_path = ROOT / "reports/visionqa/webui_training_corpus_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, ROOT / "reports/visionqa/WEBUI_TRAINING_CORPUS_AUDIT.md")
    print(json.dumps({
        "audit_result": result["audit_result"], "counts": result["counts"],
        "splits": {split: {key: distributions[split][key] for key in ("groups", "screens", "labels", "valid_boxes", "viewports", "negative_screens", "unique_sources")} for split in ("train", "val")},
        "leakage_counts": {key: len(value) for key, value in leakage.items()},
        "invalid_lines": len(invalid_lines), "corrupt_images": len(corrupt_images),
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2))


def write_markdown(result: dict, path: Path) -> None:
    leakage, integrity = result["leakage"], result["integrity"]
    lines = [
        "# WebUI Training Corpus Audit", "", f"Generated: {result['generated_at']}", "",
        f"**AUDIT RESULT: {result['audit_result']}**", "", "## Integrity", "",
        f"- Dataset manifest hash valid: {result['manifest_integrity']['dataset']}",
        f"- Split manifest hash valid: {result['manifest_integrity']['split']}",
        f"- Frozen evaluation leakage: {len(leakage['frozen_evaluation_groups'])}",
        f"- Source-group leakage: {len(leakage['source_group_across_train_val'])}",
        f"- Known duplicate-group leakage: {len(leakage['known_duplicate_groups'])}",
        f"- Exact-image train/validation leakage: {len(leakage['exact_image_hash_across_train_val'])}",
        f"- Invalid label lines: {len(integrity['invalid_lines'])}",
        f"- Corrupt images: {len(integrity['corrupt_images'])}", "", "## Split counts", "",
    ]
    for split in ("train", "val"):
        data = result["distributions"][split]
        lines += [
            f"### {split}", "", f"- Groups: {data['groups']}",
            f"- Images/labels: {data['screens']} / {data['labels']}",
            f"- Valid boxes: {data['valid_boxes']}",
            f"- Negative screens/groups: {data['negative_screens']} / {data['negative_groups']}",
            f"- Viewports: `{json.dumps(data['viewports'], sort_keys=True)}`",
            f"- Unique sources: {data['unique_sources']}", "",
            "| Class | Boxes | Screens | Groups | Box % |", "|---|---:|---:|---:|---:|",
            *[f"| {name} | {row['box_count']} | {row['screen_count']} | {row['group_count']} | {100 * row['percentage_of_boxes']:.2f}% |" for name, row in data["classes"].items()], "",
        ]
    lines += ["## Source diversity", "", f"Extreme source domination flags (>25% of groups): `{json.dumps(result['extreme_source_domination'])}`", "", "Full source and per-class viewport distributions are stored in `webui_training_corpus_metrics.json`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
