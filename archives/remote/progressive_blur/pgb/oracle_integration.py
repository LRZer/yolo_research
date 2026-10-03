"""Ground-truth oracle control for progressive background blur.

The original Oracle control uses labels during training and validation.  The
train-only variant uses labels only for training preprocessing and validates
on unmodified images. Both are diagnostics, not deployable preprocessing.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from torch import Tensor

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from .core import GaussianBankRenderer, make_targets, yolo_xywh_to_boxes
from .ultralytics_integration import valid_pixels_from_batch


def oracle_strength_from_batch(
    batch: dict[str, Any],
    *,
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
) -> Tensor:
    """Construct the ideal continuous blur map from batch ground-truth boxes."""

    images = batch["img"]
    height, width = images.shape[-2:]
    boxes_by_image = yolo_xywh_to_boxes(
        batch, height=height, width=width, batch_size=images.shape[0]
    )
    targets = make_targets(
        images,
        boxes_by_image,
        margin=margin,
        transition=transition,
        min_radius=min_radius,
        valid_pixels=valid_pixels_from_batch(
            batch, height=height, width=width, device=images.device
        ),
    )
    return targets.strength


@torch.no_grad()
def apply_oracle_blur(
    batch: dict[str, Any],
    renderer: GaussianBankRenderer,
    *,
    mix: float,
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
) -> dict[str, Any]:
    """Replace normalized batch images with their ground-truth-guided version."""

    strength = oracle_strength_from_batch(
        batch,
        margin=margin,
        transition=transition,
        min_radius=min_radius,
    )
    batch["img"] = renderer(batch["img"], strength, mix=mix)
    # Retain the map for diagnostics without exposing it to the detector loss.
    batch["oracle_strength"] = strength
    return batch


@torch.no_grad()
def apply_stochastic_oracle_blur(
    batch: dict[str, Any],
    renderer: GaussianBankRenderer,
    *,
    mix: float,
    probability: float,
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
) -> dict[str, Any]:
    """Blur a random subset of training images; retain the rest verbatim."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    if mix == 0 or probability == 0:
        return batch
    if probability == 1:
        return apply_oracle_blur(
            batch, renderer, mix=mix, margin=margin,
            transition=transition, min_radius=min_radius,
        )
    original = batch["img"]
    apply_oracle_blur(
        batch, renderer, mix=mix, margin=margin,
        transition=transition, min_radius=min_radius,
    )
    chosen = torch.rand(original.shape[0], device=original.device) < probability
    batch["img"] = torch.where(chosen[:, None, None, None], batch["img"], original)
    batch["oracle_augmented"] = chosen
    return batch


class OracleBlurValidator(DetectionValidator):
    """Validation that intentionally uses labels to measure the oracle bound."""

    def __init__(
        self,
        *args,
        oracle_eval_mix: float = 0.5,
        oracle_target_margin: float = 0.10,
        oracle_target_transition: float = 0.50,
        oracle_target_min_radius: float = 8.0,
        oracle_sigma_levels: tuple[float, ...] = (0, 1, 2, 4, 8),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= oracle_eval_mix <= 1:
            raise ValueError("oracle_eval_mix must be in [0,1]")
        self.oracle_eval_mix = float(oracle_eval_mix)
        self.oracle_target_margin = float(oracle_target_margin)
        self.oracle_target_transition = float(oracle_target_transition)
        self.oracle_target_min_radius = float(oracle_target_min_radius)
        self.oracle_renderer = GaussianBankRenderer(
            sigma_levels=oracle_sigma_levels, mix=self.oracle_eval_mix
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        return apply_oracle_blur(
            batch,
            self.oracle_renderer,
            mix=self.oracle_eval_mix,
            margin=self.oracle_target_margin,
            transition=self.oracle_target_transition,
            min_radius=self.oracle_target_min_radius,
        )


class OracleBlurTrainer(DetectionTrainer):
    """Plain YOLO trainer whose batches receive perfect GT-guided blur maps."""

    _ORACLE_DEFAULTS = {
        "oracle_eval_mix": 0.5,
        "oracle_zero_mix_epochs": 5,
        "oracle_ramp_epochs": 5,
        "oracle_target_margin": 0.10,
        "oracle_target_transition": 0.50,
        "oracle_target_min_radius": 8.0,
        "oracle_sigma_levels": (0, 1, 2, 4, 8),
        "oracle_train_probability": 1.0,
    }

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ) -> None:
        options = dict(overrides or {})
        requested = {
            key: options.pop(key, default)
            for key, default in self._ORACLE_DEFAULTS.items()
        }
        self._set_oracle_options(requested)
        if self.oracle_zero_mix_epochs < 0 or self.oracle_ramp_epochs < 0:
            raise ValueError("oracle schedule epoch counts must be non-negative")
        if not 0 <= self.oracle_eval_mix <= 1:
            raise ValueError("oracle_eval_mix must be in [0,1]")
        if not 0 <= self.oracle_train_probability <= 1:
            raise ValueError("oracle_train_probability must be in [0,1]")
        self.oracle_renderer = GaussianBankRenderer(
            sigma_levels=self.oracle_sigma_levels, mix=self.oracle_eval_mix
        )
        super().__init__(cfg=cfg, overrides=options, _callbacks=_callbacks)
        if self.resume:
            restored = {
                key: getattr(self.args, key, default)
                for key, default in self._ORACLE_DEFAULTS.items()
            }
            self._set_oracle_options(restored)
            self.oracle_renderer = GaussianBankRenderer(
                sigma_levels=self.oracle_sigma_levels, mix=self.oracle_eval_mix
            )
        # Preserve custom values for DDP subprocess creation and checkpoints.
        for key in self._ORACLE_DEFAULTS:
            setattr(self.args, key, getattr(self, key))

    def _set_oracle_options(self, values: dict[str, Any]) -> None:
        self.oracle_eval_mix = float(values["oracle_eval_mix"])
        self.oracle_zero_mix_epochs = int(values["oracle_zero_mix_epochs"])
        self.oracle_ramp_epochs = int(values["oracle_ramp_epochs"])
        self.oracle_target_margin = float(values["oracle_target_margin"])
        self.oracle_target_transition = float(values["oracle_target_transition"])
        self.oracle_target_min_radius = float(values["oracle_target_min_radius"])
        self.oracle_sigma_levels = tuple(float(x) for x in values["oracle_sigma_levels"])
        self.oracle_train_probability = float(values["oracle_train_probability"])

    def mix_for_epoch(self, epoch: int) -> float:
        if epoch < self.oracle_zero_mix_epochs:
            return 0.0
        if self.oracle_ramp_epochs == 0:
            return self.oracle_eval_mix
        ramp_index = epoch - self.oracle_zero_mix_epochs + 1
        return self.oracle_eval_mix * min(
            max(ramp_index / self.oracle_ramp_epochs, 0.0), 1.0
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        return apply_stochastic_oracle_blur(
            batch,
            self.oracle_renderer,
            mix=self.mix_for_epoch(self.epoch),
            probability=self.oracle_train_probability,
            margin=self.oracle_target_margin,
            transition=self.oracle_target_transition,
            min_radius=self.oracle_target_min_radius,
        )

    def get_validator(self) -> OracleBlurValidator:
        validator_args = copy(self.args)
        for key in self._ORACLE_DEFAULTS:
            if hasattr(validator_args, key):
                delattr(validator_args, key)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return OracleBlurValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            oracle_eval_mix=self.oracle_eval_mix,
            oracle_target_margin=self.oracle_target_margin,
            oracle_target_transition=self.oracle_target_transition,
            oracle_target_min_radius=self.oracle_target_min_radius,
            oracle_sigma_levels=self.oracle_sigma_levels,
        )


class OracleTrainOnlyBlurTrainer(OracleBlurTrainer):
    """GT-guided training blur, with untouched images in every validation."""

    def get_validator(self) -> DetectionValidator:
        validator_args = copy(self.args)
        for key in self._ORACLE_DEFAULTS:
            if hasattr(validator_args, key):
                delattr(validator_args, key)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
        )
