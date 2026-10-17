import hashlib
import json
from pathlib import Path

import yaml
from ultralytics.utils.patches import torch_load

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/processed/visionqa_v2_mixed14"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_mixed_data_yaml_preserves_all_14_classes():
    payload = yaml.safe_load((DATA / "data.yaml").read_text(encoding="utf-8"))
    assert payload["nc"] == 14
    assert list(payload["names"].values()) == ["button", "link", "input", "image", "icon", "checkbox", "radio", "select", "block-nav", "block-sidebar", "block-footer", "block-form", "block-table", "block"]


def test_mixed_corpus_counts_and_partial_labels_are_verified():
    audit = json.loads((ROOT / "reports/visionqa/v2/mixed_dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["result"] == "PASS"
    assert audit["mixed_train_images"] == 28739
    assert audit["partitions"]["webui_train"]["images"] == 23319
    assert audit["partitions"]["webui_val"]["images"] == 5214
    assert audit["partitions"]["atomblock_train"]["images"] == 5420
    for class_id in (4, 5, 6, 13):
        assert audit["partitions"]["webui_train"]["class_counts"][str(class_id)] == 0
        assert audit["partitions"]["atomblock_train"]["class_counts"][str(class_id)] > 0


def test_frozen_evaluation_manifest_content_hash_is_unchanged():
    payload = json.loads((ROOT / "reports/visionqa/webui_eval_manifest.json").read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_content_sha256")
    assert recorded == "674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45"
    assert canonical_hash(payload) == recorded


def test_protected_v1_checkpoint_is_unchanged():
    path = ROOT / "models/visionqa/atomblock_yolo11n/best.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57"


def test_training_configuration_disables_cross_domain_composites():
    config = json.loads((ROOT / "reports/visionqa/v2/training_config.json").read_text(encoding="utf-8"))
    assert config["partial_label_strategy"]["webui_masked_class_ids"] == [4, 5, 6, 13]
    assert set(config["partial_label_strategy"]["cross_image_augmentations_disabled"]) == {"mosaic", "mixup", "cutmix", "copy_paste"}
    assert config["main_experiment"]["maximum_total_epochs"] == 12


def test_benchmark_epoch_checkpoint_contains_resume_state():
    checkpoint = DATA.parents[2] / "runs/visionqa/visionqa_v2_mixed14_benchmark/weights/epoch0.pt"
    state = torch_load(checkpoint, map_location="cpu")
    assert state["epoch"] == 0
    assert state["optimizer"] is not None
    assert state["scaler"] is not None
    assert state["stopper"] is not None
    assert state["train_args"]
    assert state["train_results"]
