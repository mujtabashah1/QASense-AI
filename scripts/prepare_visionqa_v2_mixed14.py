"""Build the source-tagged 14-class mixed WebUI + AtomBlock corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "datasets" / "processed" / "visionqa_webui_v2"
ATOMBLOCK = ROOT / "datasets" / "processed" / "visionqa_atomblock"
OUTPUT = ROOT / "datasets" / "processed" / "visionqa_v2_mixed14"
REPORT = ROOT / "reports" / "visionqa" / "v2" / "mixed_dataset_manifest.json"
WEBUI_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12}
NAMES = ["button", "link", "input", "image", "icon", "checkbox", "radio", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table", "block"]
FROZEN_SHA = "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_link(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy_fallback"


def remap_webui_label(source: Path, target: Path) -> Counter:
    rows, counts = [], Counter()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed label {source}:{line_number}")
        old_id = int(fields[0])
        if old_id not in WEBUI_REMAP:
            raise ValueError(f"Unsupported WebUI class {old_id} in {source}:{line_number}")
        new_id = WEBUI_REMAP[old_id]
        rows.append(" ".join([str(new_id), *fields[1:]]))
        counts[new_id] += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return counts


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def main() -> int:
    frozen_manifest = ROOT / "reports" / "visionqa" / "webui_eval_manifest.json"
    frozen_payload = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    recorded_frozen_hash = frozen_payload.pop("manifest_content_sha256")
    if recorded_frozen_hash != FROZEN_SHA or canonical_hash(frozen_payload) != FROZEN_SHA:
        raise RuntimeError("Frozen WebUI evaluation manifest hash changed")
    corpus_manifest = json.loads((WEBUI / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    frozen_groups = set(
        json.loads(
            (ROOT / "reports" / "visionqa" / "webui_frozen_eval_groups.json").read_text(encoding="utf-8")
        )["group_ids"]
    )
    leaked = sorted({row["group_id"] for row in corpus_manifest["screens"]} & frozen_groups)
    if leaked:
        raise RuntimeError(f"WebUI corpus contains frozen evaluation groups: {leaked[:5]}")

    records, class_counts, link_modes = [], Counter(), Counter()
    for split in ("train", "val"):
        for row in (item for item in corpus_manifest["screens"] if item["split"] == split):
            source_image = Path(row["processed_image_path"])
            source_label = Path(row["processed_label_path"])
            target_image = OUTPUT / split / "images" / "webui" / source_image.name
            target_label = OUTPUT / split / "labels" / "webui" / f"{source_image.stem}.txt"
            link_modes[safe_link(source_image, target_image)] += 1
            counts = remap_webui_label(source_label, target_label)
            class_counts.update(counts)
            records.append({"source": "webui", "split": split, "image": str(target_image), "label": str(target_label), "boxes": sum(counts.values()), "group_id": row["group_id"]})

    corrupt = []
    atom_images = sorted((ATOMBLOCK / "train" / "images").iterdir())
    for source_image in atom_images:
        if not valid_image(source_image):
            corrupt.append(str(source_image))
            continue
        source_label = ATOMBLOCK / "train" / "labels" / f"{source_image.stem}.txt"
        if not source_label.exists():
            raise FileNotFoundError(source_label)
        target_image = OUTPUT / "train" / "images" / "atomblock" / source_image.name
        target_label = OUTPUT / "train" / "labels" / "atomblock" / source_label.name
        link_modes[safe_link(source_image, target_image)] += 1
        link_modes[safe_link(source_label, target_label)] += 1
        counts = Counter(int(line.split()[0]) for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip())
        class_counts.update(counts)
        records.append({"source": "atomblock", "split": "train", "image": str(target_image), "label": str(target_label), "boxes": sum(counts.values())})

    atom_count = sum(row["source"] == "atomblock" for row in records)
    if atom_count != 5420:
        raise RuntimeError(f"Expected 5,420 decodable AtomBlock training images, got {atom_count}")
    data_yaml = OUTPUT / "data.yaml"
    yaml_lines = [f"path: {OUTPUT.as_posix()}", "train: train/images", "val: val/images", "", "nc: 14", "", "names:"]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(NAMES))
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": {str(index): name for index, name in enumerate(NAMES)},
        "webui_remap": {str(k): v for k, v in WEBUI_REMAP.items()},
        "unknown_on_webui": [4, 5, 6, 13],
        "counts": {
            "webui_train": sum(r["source"] == "webui" and r["split"] == "train" for r in records),
            "webui_val": sum(r["source"] == "webui" and r["split"] == "val" for r in records),
            "atomblock_train": atom_count,
            "mixed_train": sum(r["split"] == "train" for r in records),
            "webui_val_only": sum(r["split"] == "val" for r in records),
        },
        "class_box_counts": {NAMES[index]: class_counts[index] for index in range(14)},
        "corrupt_atomblock_images_excluded": corrupt,
        "storage_modes": dict(link_modes),
        "frozen_eval_manifest_sha256": recorded_frozen_hash,
        "frozen_eval_group_leakage": 0,
        "records": records,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

