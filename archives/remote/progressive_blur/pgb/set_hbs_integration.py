"""HBS-only adaptation of the published SET training branch to YOLOv8.

This is deliberately NOT a full SET reproduction: SET also includes API
perturbations and was originally evaluated with FCOS. The raw YOLO path is
unchanged at inference; GT masks exist only inside the training loss.
Source: https://github.com/HuixinSun/SET/blob/master/mmdet/models/detectors/fcos_set.py
"""

from __future__ import annotations

import math
from copy import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from .core import yolo_xywh_to_boxes


class SETDenoiser(nn.Module):
    """Official HBS DNResBlock shape, with YOLO feature-specific channels."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size, 1, padding, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size, 1, padding, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class HBSOnlyDetectionModel(DetectionModel):
    """Train-time second branch using HBS or its identity double-loss control."""

    def __init__(
        self, cfg: str = "yolov8n.yaml", ch: int = 3, nc: int | None = None,
        verbose: bool = True, *, hbs_mode: str = "hbs", hbs_aux_weight: float = 1.0,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if hbs_mode not in {"hbs", "identity"} or hbs_aux_weight <= 0:
            raise ValueError("invalid HBS mode or auxiliary weight")
        self.hbs_mode = hbs_mode
        self.hbs_aux_weight = float(hbs_aux_weight)
        detect = self.model[-1]
        self.hbs_blocks = nn.ModuleList()
        if hbs_mode == "hbs":
            for layer, stride in zip(detect.cv2, detect.stride.tolist()):
                channels = layer[0].conv.in_channels
                kernel = (int(math.log2(stride)) // 2 * 2) + 1
                self.hbs_blocks.append(SETDenoiser(channels, kernel))

    @staticmethod
    def _foreground_masks(batch: dict[str, Any], features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        images = batch["img"]
        height, width = images.shape[-2:]
        boxes_by_image = yolo_xywh_to_boxes(batch, height, width, images.shape[0])
        foreground = torch.zeros((images.shape[0], 1, height, width),
                                 device=images.device, dtype=features[0].dtype)
        for image_index, boxes in enumerate(boxes_by_image):
            for x1, y1, x2, y2 in boxes:
                xa = max(0, min(width, int(torch.floor(x1).item())))
                ya = max(0, min(height, int(torch.floor(y1).item())))
                xb = max(0, min(width, int(torch.ceil(x2).item())))
                yb = max(0, min(height, int(torch.ceil(y2).item())))
                if xb > xa and yb > ya:
                    foreground[image_index, 0, ya:yb, xa:xb] = 1
        return [F.interpolate(foreground, size=feature.shape[-2:], mode="nearest") for feature in features]

    def loss(self, batch: dict[str, Any], preds=None):
        if preds is not None or not self.training:
            return super().loss(batch, preds)
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        captured: list[tuple[torch.Tensor, ...]] = []

        def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            captured.append(tuple(inputs[0]))  # Detect mutates its input list in-place.

        hook = self.model[-1].register_forward_pre_hook(capture)
        try:
            raw_predictions = self.predict(batch["img"])
        finally:
            hook.remove()
        if len(captured) != 1 or len(captured[0]) != len(self.model[-1].stride):
            raise AssertionError("failed to capture YOLO detection-neck features")
        raw_loss, raw_items = self.criterion(raw_predictions, batch)
        features = captured[0]
        if self.hbs_mode == "hbs":
            masks = self._foreground_masks(batch, features)
            second_features = []
            for feature, fg, block in zip(features, masks, self.hbs_blocks):
                bg = 1 - fg
                second_features.append(block(feature * bg) * bg + feature * fg)
        else:
            second_features = list(features)
        second_predictions = self.model[-1](second_features)
        second_loss, second_items = self.criterion(second_predictions, batch)
        weighted = self.hbs_aux_weight
        items = {key: value + weighted * second_items[key] for key, value in raw_items.items()}
        items.update({f"aux_{key}": weighted * value.detach() for key, value in second_items.items()})
        return raw_loss + weighted * second_loss, items


class HBSOnlyDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer with frozen HBS/identity paired configurations."""

    _CUSTOM_KEYS = ("hbs_mode", "hbs_aux_weight")

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        options = dict(overrides or {})
        self.hbs_mode = str(options.pop("hbs_mode", "hbs"))
        self.hbs_aux_weight = float(options.pop("hbs_aux_weight", 1.0))
        super().__init__(cfg=cfg, overrides=options, _callbacks=_callbacks)
        for key in self._CUSTOM_KEYS:
            setattr(self.args, key, getattr(self, key))

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> HBSOnlyDetectionModel:
        model = self.set_model_names_for_load(HBSOnlyDetectionModel(
            cfg=cfg or "yolov8n.yaml", nc=self.data["nc"], ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            hbs_mode=self.hbs_mode, hbs_aux_weight=self.hbs_aux_weight,
        ))
        if weights:
            model.load(weights)
        return model

    def get_validator(self) -> DetectionValidator:
        validator_args = copy(self.args)
        for key in self._CUSTOM_KEYS:
            if hasattr(validator_args, key):
                delattr(validator_args, key)
        return DetectionValidator(self.test_loader, save_dir=self.save_dir,
                                  args=validator_args, _callbacks=self.callbacks)
