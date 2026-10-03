"""Extreme GT-box Oracle: preserve box interiors and black out everything else.

The rectangle boundaries expose ground-truth locations during validation. This
is an intentionally privileged stress test and is never a deployable detector.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from .core import yolo_xywh_to_boxes


@torch.no_grad()
def exact_box_union_mask(batch: dict[str, Any]) -> Tensor:
    """Return [B,1,H,W] bool mask for the union of exact GT rectangles."""

    images = batch["img"]
    batch_size, _, height, width = images.shape
    boxes_by_image = yolo_xywh_to_boxes(
        batch, height=height, width=width, batch_size=batch_size
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=images.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=images.device)
    for image_index, boxes in enumerate(boxes_by_image):
        boxes = boxes.to(device=images.device, dtype=torch.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in boxes:
            mask[image_index, 0] |= (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    return mask


@torch.no_grad()
def apply_oracle_blackout(batch: dict[str, Any]) -> dict[str, Any]:
    """Replace every pixel outside the union of GT boxes with exact zero."""

    mask = exact_box_union_mask(batch)
    batch["img"] = batch["img"] * mask.to(dtype=batch["img"].dtype)
    batch["oracle_blackout_mask"] = mask
    return batch


class OracleBlackoutValidator(DetectionValidator):
    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        return apply_oracle_blackout(super().preprocess(batch))


class OracleBlackoutTrainer(DetectionTrainer):
    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return apply_oracle_blackout(super().preprocess_batch(batch))

    def get_validator(self) -> OracleBlackoutValidator:
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return OracleBlackoutValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
        )

