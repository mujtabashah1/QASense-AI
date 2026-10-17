import hashlib
import json
from pathlib import Path

import pytest

from src.visionqa.webui_corpus import (
    V2_CLASSES, V2_CLASS_TO_ID, assign_source_grouped_split, canonical_hash,
    class_schema, group_id, xyxy_to_yolo,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_HASH = "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"


def test_frozen_evaluation_manifest_is_unchanged_and_has_133_groups():
    payload = json.loads((ROOT / "reports/visionqa/webui_eval_manifest.json").read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_content_sha256")
    assert recorded == FROZEN_HASH
    assert canonical_hash(payload) == FROZEN_HASH
    assert len({group_id(row["split"], row["source_sample_id"]) for row in payload["screens"]}) == 133


def test_v2_schema_is_contiguous_and_preserves_atomblock_ids():
    assert tuple(V2_CLASS_TO_ID) == V2_CLASSES
    assert list(V2_CLASS_TO_ID.values()) == list(range(10))
    entries = class_schema()
    assert [row["v2_class_id"] for row in entries] == list(range(10))
    assert {row["class_name"] for row in entries}.isdisjoint({"icon", "checkbox", "radio", "block"})
    assert all(row["original_atomblock_class_id"] >= 0 for row in entries)


def test_grouped_split_is_deterministic_and_source_representative():
    groups = [
        {"group_id": f"a-{i}", "source": "a"} for i in range(10)
    ] + [{"group_id": f"b-{i}", "source": "b"} for i in range(6)]
    first = assign_source_grouped_split(groups)
    second = assign_source_grouped_split(list(reversed(groups)))
    assert first == second
    assert {first[f"a-{i}"] for i in range(10)} == {"train", "val"}
    assert {first[f"b-{i}"] for i in range(6)} == {"train", "val"}


def test_xyxy_to_yolo_conversion_and_validation():
    assert xyxy_to_yolo([10, 20, 30, 60], 100, 100) == pytest.approx((0.2, 0.4, 0.2, 0.4))
    with pytest.raises(ValueError):
        xyxy_to_yolo([30, 20, 10, 60], 100, 100)
    with pytest.raises(ValueError):
        xyxy_to_yolo([-1, 20, 30, 60], 100, 100)


def test_authoritative_duplicate_artifact_has_15_groups():
    payload = json.loads((ROOT / "reports/visionqa/webui_cross_split_duplicate_exclusions.json").read_text(encoding="utf-8"))
    assert payload["group_count"] == 15
    assert len(payload["groups"]) == 15


def test_final_corpus_manifests_are_valid_and_disjoint():
    root = ROOT / "datasets/processed/visionqa_webui_v2/manifests"
    dataset = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    dataset_recorded = dataset.pop("manifest_sha256")
    split_recorded = split.pop("manifest_sha256")
    assert canonical_hash(dataset) == dataset_recorded == "442c0f3d250f513463ca6188127a31240fab173e781fe630ddc6ffa4a52ced56"
    assert canonical_hash(split) == split_recorded == "4bc01ed12f211408760ca1a67f2db901dc59d9b8bddf1fa5bc55702b37d9add3"
    by_group = {}
    for row in dataset["screens"]:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert len(by_group) == 9511
    assert all(len(splits) == 1 for splits in by_group.values())


def test_final_corpus_excludes_frozen_and_known_duplicate_groups():
    dataset = json.loads((ROOT / "datasets/processed/visionqa_webui_v2/manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "reports/visionqa/webui_frozen_eval_groups.json").read_text(encoding="utf-8"))
    duplicates = json.loads((ROOT / "reports/visionqa/webui_cross_split_duplicate_exclusions.json").read_text(encoding="utf-8"))
    corpus_groups = {row["group_id"] for row in dataset["screens"]}
    duplicate_groups = {group_id(split, sample) for group in duplicates["groups"] for split, sample, _ in group}
    assert corpus_groups.isdisjoint(frozen["group_ids"])
    assert corpus_groups.isdisjoint(duplicate_groups)


def test_final_audit_proves_label_image_and_leakage_integrity():
    metrics = json.loads((ROOT / "reports/visionqa/webui_training_corpus_metrics.json").read_text(encoding="utf-8"))
    assert metrics["audit_result"] == "PASS"
    assert metrics["counts"]["screens"] == 28533
    assert metrics["counts"]["valid_boxes"] == 274096
    assert metrics["counts"]["decode_failures"] == 0
    assert all(not values for values in metrics["leakage"].values())
    assert not metrics["integrity"]["invalid_lines"]
    assert not metrics["integrity"]["corrupt_images"]
    assert not metrics["integrity"]["missing_images"]
    assert not metrics["integrity"]["missing_labels"]


def test_review_is_complete_and_covers_all_classes_and_viewports():
    review = json.loads((ROOT / "reports/visionqa/webui_training_review.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "datasets/processed/visionqa_webui_v2/manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    rows = {row["screen_id"]: row for row in dataset["screens"]}
    selected = [rows[item["screen_id"]] for item in review["reviews"]]
    assert review["status"] == "complete"
    assert review["summary"]["reviewed_screens"] == 300
    assert all(item["decision"] == "include" and item["mapping_valid"] for item in review["reviews"])
    assert {row["viewport"] for row in selected} == {"desktop", "tablet", "mobile"}
    assert {name for row in selected for name in row["class_counts"]} == set(V2_CLASSES)


def test_readiness_and_reports_exist_without_training():
    readiness = json.loads((ROOT / "reports/visionqa/visionqa_v2_readiness.json").read_text(encoding="utf-8"))
    assert readiness["training_ready"] is True
    assert readiness["training_performed"] is False
    assert all(readiness["conditions"].values())
    for name in ("WEBUI_TRAINING_CORPUS_REPORT.md", "WEBUI_TRAINING_CORPUS_AUDIT.md", "VISIONQA_V2_TRAINING_SPEC.md"):
        assert (ROOT / "reports/visionqa" / name).is_file()


def test_protected_model_hashes_are_unchanged():
    expected = {
        "models/risknet/risknet_v1.joblib": "b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c",
        "models/codelocator_v2/line_localizer/model.safetensors": "9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288",
        "models/visionqa/atomblock_yolo11n/best.pt": "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57",
    }
    for relative, digest in expected.items():
        hasher = hashlib.sha256()
        with (ROOT / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == digest
