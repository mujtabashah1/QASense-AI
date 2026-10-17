"""Source-aware training components for the unified 14-class VisionQA v2 detector.

WebUI annotations cover ten of the fourteen canonical classes.  The remaining
classes may still be visible, so their negative classification loss is masked
for WebUI images.  AtomBlock images retain ordinary full-class supervision.
"""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

import numpy as np
import torch

from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


CANONICAL_NAMES = {
    0: "button",
    1: "link",
    2: "input",
    3: "image",
    4: "icon",
    5: "checkbox",
    6: "radio",
    7: "select",
    8: "block-nav",
    9: "block-sidebar",
    10: "block-footer",
    11: "block-form",
    12: "block-table",
    13: "block",
}
WEBUI_SUPPORTED_IDS = frozenset({0, 1, 2, 3, 7, 8, 9, 10, 11, 12})
WEBUI_MASKED_IDS = frozenset({4, 5, 6, 13})
SOURCE_ATOMBLOCK = 0
SOURCE_WEBUI = 1


def source_id_from_path(path: str | Path) -> int:
    """Return the explicit mixed-corpus source ID encoded in the path."""
    parts = {part.lower() for part in Path(path).parts}
    if "webui" in parts:
        return SOURCE_WEBUI
    if "atomblock" in parts:
        return SOURCE_ATOMBLOCK
    raise ValueError(f"Mixed-corpus image has no recognized source directory: {path}")


def class_supervision_mask(source_ids: Any, nc: int = 14, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Build a [batch, 1, classes] classification-supervision mask."""
    if nc != len(CANONICAL_NAMES):
        raise ValueError(f"VisionQA v2 requires exactly 14 classes, got {nc}")
    source = torch.as_tensor(source_ids, device=device, dtype=torch.long).reshape(-1)
    unknown = ~((source == SOURCE_ATOMBLOCK) | (source == SOURCE_WEBUI))
    if unknown.any():
        raise ValueError(f"Unknown source IDs: {source[unknown].detach().cpu().tolist()}")
    mask = torch.ones((source.numel(), 1, nc), device=device, dtype=dtype)
    webui_rows = source == SOURCE_WEBUI
    if webui_rows.any():
        for class_id in WEBUI_MASKED_IDS:
            mask[webui_rows, :, class_id] = 0
    return mask


def masked_classification_bce(
    bce_elements: torch.Tensor,
    source_ids: Any,
    target_scores_sum: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate elementwise BCE after applying source-aware class masking."""
    if bce_elements.ndim != 3:
        raise ValueError(f"Expected [batch, anchors, classes] BCE tensor, got {tuple(bce_elements.shape)}")
    if class_weights is not None:
        bce_elements = bce_elements * class_weights
    mask = class_supervision_mask(
        source_ids, bce_elements.shape[-1], device=bce_elements.device, dtype=bce_elements.dtype
    )
    return (bce_elements * mask).sum() / target_scores_sum


class SourceAwareYOLODataset(YOLODataset):
    """YOLO dataset that tags every item as AtomBlock or WebUI."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["source_id"] = torch.tensor(source_id_from_path(self.im_files[index]), dtype=torch.long)
        return item

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        collated = YOLODataset.collate_fn(batch)
        collated["source_id"] = torch.stack(tuple(collated["source_id"]), dim=0)
        return collated


class PartialLabelDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with WebUI-only masking of unknown classes."""

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        bce_elements = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = masked_classification_bce(
            bce_elements, batch["source_id"], target_scores_sum, self.class_weights
        )
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class PartialLabelDetectionModel(DetectionModel):
    """Detection model that installs the partial-label criterion."""

    def init_criterion(self):
        return PartialLabelDetectionLoss(self)


class WebUISupportedClassValidator(DetectionValidator):
    """Ignore unknown WebUI classes when computing adaptation metrics."""

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        filtered = []
        supported = torch.tensor(sorted(WEBUI_SUPPORTED_IDS), device=self.device)
        for pred in preds:
            keep = torch.isin(pred["cls"].long(), supported)
            filtered.append({key: value[keep] for key, value in pred.items()})
        super().update_metrics(filtered, batch)


class PartialLabelDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer for the mixed 14-class corpus."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode == "train":
            unsafe = {name: float(getattr(self.args, name, 0.0)) for name in ("mosaic", "mixup", "cutmix", "copy_paste")}
            if any(value > 0 for value in unsafe.values()):
                raise ValueError(f"Cross-image augmentation is unsafe for source-aware masking: {unsafe}")
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return SourceAwareYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        model = self.set_model_names_for_load(
            PartialLabelDetectionModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return WebUISupportedClassValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

