from pathlib import Path

import pytest
import torch

from src.visionqa.v2_training import (
    CANONICAL_NAMES,
    SOURCE_ATOMBLOCK,
    SOURCE_WEBUI,
    WEBUI_MASKED_IDS,
    WEBUI_SUPPORTED_IDS,
    class_supervision_mask,
    masked_classification_bce,
    source_id_from_path,
)


def test_canonical_schema_has_exactly_14_stable_classes():
    assert CANONICAL_NAMES == {
        0: "button", 1: "link", 2: "input", 3: "image", 4: "icon", 5: "checkbox", 6: "radio",
        7: "select", 8: "block-nav", 9: "block-sidebar", 10: "block-footer", 11: "block-form",
        12: "block-table", 13: "block",
    }


def test_source_tagging_requires_explicit_domain_directory():
    assert source_id_from_path(Path("train/images/webui/a.png")) == SOURCE_WEBUI
    assert source_id_from_path(Path("train/images/atomblock/a.png")) == SOURCE_ATOMBLOCK
    with pytest.raises(ValueError):
        source_id_from_path(Path("train/images/a.png"))


def test_webui_masks_four_unknown_classes_and_atomblock_masks_none():
    mask = class_supervision_mask([SOURCE_ATOMBLOCK, SOURCE_WEBUI])
    assert mask.shape == (2, 1, 14)
    assert torch.all(mask[0] == 1)
    for class_id in WEBUI_SUPPORTED_IDS:
        assert mask[1, 0, class_id] == 1
    for class_id in WEBUI_MASKED_IDS:
        assert mask[1, 0, class_id] == 0


def test_masked_webui_logits_do_not_change_classification_loss():
    base = torch.ones((1, 3, 14))
    changed = base.clone()
    changed[:, :, list(WEBUI_MASKED_IDS)] = 999
    denominator = torch.tensor(1.0)
    assert masked_classification_bce(base, [SOURCE_WEBUI], denominator) == masked_classification_bce(
        changed, [SOURCE_WEBUI], denominator
    )


def test_supported_webui_and_all_atomblock_logits_change_loss():
    base = torch.ones((1, 2, 14))
    supported_change = base.clone(); supported_change[:, :, 7] = 2
    masked_change = base.clone(); masked_change[:, :, 4] = 2
    denominator = torch.tensor(1.0)
    assert masked_classification_bce(base, [SOURCE_WEBUI], denominator) != masked_classification_bce(
        supported_change, [SOURCE_WEBUI], denominator
    )
    assert masked_classification_bce(base, [SOURCE_ATOMBLOCK], denominator) != masked_classification_bce(
        masked_change, [SOURCE_ATOMBLOCK], denominator
    )
