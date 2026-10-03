"""Ultralytics 8.4.146 integration for the safe-background gate.

The gate is a registered model submodule: training, validation, prediction,
EMA and checkpoints therefore use the same image-only front end. The module
reuses the already audited PGB trainer schedule and validation plumbing, but
has its own architecture, supervision and experiment configuration.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from torch import Tensor

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from .core import make_targets, yolo_xywh_to_boxes
from .safe_background import SafeBackgroundFrontEnd, safe_gate_loss
from .ultralytics_integration import (
    PGBDetectionModel,
    PGBDetectionTrainer,
    valid_pixels_from_batch,
)


class SafeBackgroundDetectionModel(PGBDetectionModel):
    """YOLOv8n plus an image-only conservative blur-permission front end."""

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        sbg_sigma: float = 8.0,
        sbg_far_threshold: float = 0.8,
        sbg_protect_weight: float = 4.0,
        sbg_far_weight: float = 0.25,
        sbg_map_only: bool = False,
        map_loss_weight: float = 0.25,
        eval_mix: float = 0.5,
        target_margin: float = 0.10,
        target_transition: float = 0.50,
        target_min_radius: float = 8.0,
        zero_mix_epochs: int = 0,
        ramp_epochs: int = 0,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            map_loss_weight=map_loss_weight,
            eval_mix=eval_mix,
            target_margin=target_margin,
            target_transition=target_transition,
            target_min_radius=target_min_radius,
            zero_mix_epochs=zero_mix_epochs,
            ramp_epochs=ramp_epochs,
        )
        if sbg_sigma <= 0:
            raise ValueError("sbg_sigma must be positive")
        if not 0 < sbg_far_threshold <= 1:
            raise ValueError("sbg_far_threshold must be in (0,1]")
        self.frontend = SafeBackgroundFrontEnd(sigma=sbg_sigma, mix=eval_mix)
        self.sbg_sigma = float(sbg_sigma)
        self.sbg_far_threshold = float(sbg_far_threshold)
        self.sbg_protect_weight = float(sbg_protect_weight)
        self.sbg_far_weight = float(sbg_far_weight)
        self.sbg_map_only = bool(sbg_map_only)

    def sbg_config(self) -> dict[str, float | bool]:
        return {
            "sbg_sigma": self.sbg_sigma,
            "sbg_far_threshold": self.sbg_far_threshold,
            "sbg_protect_weight": self.sbg_protect_weight,
            "sbg_far_weight": self.sbg_far_weight,
            "sbg_map_only": self.sbg_map_only,
        }

    def _gate_loss(self, batch: dict[str, Any], permission: Tensor) -> Tensor:
        height, width = batch["img"].shape[-2:]
        boxes = yolo_xywh_to_boxes(batch, height, width, batch["img"].shape[0])
        targets = make_targets(
            batch["img"],
            boxes,
            margin=self.target_margin,
            transition=self.target_transition,
            min_radius=self.target_min_radius,
            valid_pixels=valid_pixels_from_batch(
                batch, height, width, batch["img"].device
            ),
        )
        return safe_gate_loss(
            permission,
            targets,
            far_threshold=self.sbg_far_threshold,
            protect_weight=self.sbg_protect_weight,
            far_weight=self.sbg_far_weight,
        )

    def loss(self, batch: dict[str, Any], preds=None):
        images = batch["img"]
        if self.sbg_map_only:
            # Final-epoch validation supplies detector predictions even when
            # this stage updates only the gate. The predictions are ignored
            # for the auxiliary loss and used normally by the validator.
            # The Ultralytics trainer calls criterion.update() after each
            # epoch even when the detector forward is intentionally skipped.
            if getattr(self, "criterion", None) is None:
                self.criterion = self.init_criterion()
            permission = self.frontend.map_net(images)
            weighted = self._gate_loss(batch, permission) * self.map_loss_weight
            zero = weighted.detach().new_zeros(())
            return weighted * images.shape[0], {
                "box_loss": zero,
                "cls_loss": zero,
                "dfl_loss": zero,
                "map_loss": weighted.detach(),
            }

        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            processed, permission = self._frontend_forward(images)
            preds = DetectionModel.predict(self, processed)
            self._pgb_last_strength = permission
        else:
            permission = self._pgb_last_strength
            if permission is None or permission.shape != (images.shape[0], 1, *images.shape[-2:]):
                permission = self.frontend.map_net(images)
        detector_total, detector_items = self.criterion(preds, batch)
        weighted = self._gate_loss(batch, permission) * self.map_loss_weight
        total = detector_total + weighted * images.shape[0]
        items = dict(detector_items)
        items["map_loss"] = weighted.detach()
        return total, items


class SafeBackgroundTrainer(PGBDetectionTrainer):
    """Train a map-only, frozen-detector or joint stage with exact settings."""

    _SBG_DEFAULTS = {
        "sbg_sigma": 8.0,
        "sbg_far_threshold": 0.8,
        "sbg_protect_weight": 4.0,
        "sbg_far_weight": 0.25,
        "sbg_map_only": False,
    }

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ) -> None:
        options = dict(overrides or {})
        requested = {key: options.pop(key, default) for key, default in self._SBG_DEFAULTS.items()}
        self._set_sbg_options(requested)
        super().__init__(cfg=cfg, overrides=options, _callbacks=_callbacks)
        if self.resume:
            restored = {key: getattr(self.args, key, default) for key, default in self._SBG_DEFAULTS.items()}
            self._set_sbg_options(restored)
        for key in self._SBG_DEFAULTS:
            setattr(self.args, key, getattr(self, key))

    def _set_sbg_options(self, values: dict[str, Any]) -> None:
        self.sbg_sigma = float(values["sbg_sigma"])
        self.sbg_far_threshold = float(values["sbg_far_threshold"])
        self.sbg_protect_weight = float(values["sbg_protect_weight"])
        self.sbg_far_weight = float(values["sbg_far_weight"])
        self.sbg_map_only = bool(values["sbg_map_only"])
        if (
            self.sbg_sigma <= 0
            or not 0 < self.sbg_far_threshold <= 1
            or self.sbg_protect_weight <= 0
            or self.sbg_far_weight < 0
        ):
            raise ValueError("invalid safe-background gate configuration")

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ) -> SafeBackgroundDetectionModel:
        model = self.set_model_names_for_load(
            SafeBackgroundDetectionModel(
                cfg=cfg or "yolov8n.yaml",
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                sbg_sigma=self.sbg_sigma,
                sbg_far_threshold=self.sbg_far_threshold,
                sbg_protect_weight=self.sbg_protect_weight,
                sbg_far_weight=self.sbg_far_weight,
                sbg_map_only=self.sbg_map_only,
                map_loss_weight=self.pgb_map_loss_weight,
                eval_mix=self.pgb_eval_mix,
                target_margin=self.pgb_target_margin,
                target_transition=self.pgb_target_transition,
                target_min_radius=self.pgb_target_min_radius,
                zero_mix_epochs=self.pgb_zero_mix_epochs,
                ramp_epochs=self.pgb_ramp_epochs,
            )
        )
        if weights:
            model.load(weights)
            # A preceding map-only stage stores eval_mix=0 as a buffer. Loading
            # its weights must not silently disable blur in the next stage.
            model.set_eval_mix(self.pgb_eval_mix)
            model.set_train_mix(0.0)
        return model

    def get_validator(self) -> DetectionValidator:
        validator_args = copy(self.args)
        for key in (*self._PGB_DEFAULTS, *self._SBG_DEFAULTS):
            if hasattr(validator_args, key):
                delattr(validator_args, key)
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
        )
