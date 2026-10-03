"""Ultralytics 8.4.146 integration for the progressive-blur front end.

The integration keeps the front end inside the model so optimizer discovery,
EMA, checkpointing, validation and prediction all traverse the same module.
This file deliberately targets the recorded Ultralytics version; compatibility
with other releases must be re-audited rather than assumed.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from torch import Tensor

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .core import (
    ProgressiveBlurFrontEnd,
    balanced_map_loss,
    make_targets,
    yolo_xywh_to_boxes,
)


def valid_pixels_from_batch(
    batch: dict[str, Any], height: int, width: int, device: torch.device
) -> Tensor | None:
    """Reconstruct non-letterbox pixels from Ultralytics batch metadata.

    The current 8.4.146 training batch stores per-image original shape and the
    realized height/width resize gains. If metadata is absent (for example after
    a transform that no longer represents a single source image), return None
    rather than inventing an unreliable mask.
    """

    original_shapes = batch.get("ori_shape")
    ratio_pads = batch.get("ratio_pad")
    batch_size = batch["img"].shape[0]
    if original_shapes is None or ratio_pads is None:
        return None
    if len(original_shapes) != batch_size or len(ratio_pads) != batch_size:
        return None
    valid = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)
    for index, (original, ratio_pad) in enumerate(zip(original_shapes, ratio_pads)):
        try:
            original_height, original_width = (int(original[0]), int(original[1]))
            # Training batches normally provide (height_gain, width_gain).
            if len(ratio_pad) == 2 and all(
                isinstance(value, (int, float)) for value in ratio_pad
            ):
                height_gain, width_gain = float(ratio_pad[0]), float(ratio_pad[1])
                content_height = min(height, round(original_height * height_gain))
                content_width = min(width, round(original_width * width_gain))
                top = (height - content_height) // 2
                left = (width - content_width) // 2
            else:
                # Evaluation-style nested metadata:
                # ((height_gain, width_gain), (left_pad, top_pad)).
                gains, padding = ratio_pad
                height_gain, width_gain = float(gains[0]), float(gains[1])
                content_height = min(height, round(original_height * height_gain))
                content_width = min(width, round(original_width * width_gain))
                left, top = round(float(padding[0])), round(float(padding[1]))
            bottom = min(height, top + content_height)
            right = min(width, left + content_width)
            if top < 0 or left < 0 or bottom <= top or right <= left:
                return None
            valid[index, 0, top:bottom, left:right] = True
        except (TypeError, ValueError, IndexError):
            return None
    return valid


class PGBDetectionModel(DetectionModel):
    """YOLO detector whose image path always includes the learned front end."""

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        map_loss_weight: float = 1.0,
        eval_mix: float = 0.5,
        target_margin: float = 0.10,
        target_transition: float = 0.50,
        target_min_radius: float = 8.0,
        zero_mix_epochs: int = 5,
        ramp_epochs: int = 5,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if map_loss_weight < 0:
            raise ValueError("map_loss_weight must be non-negative")
        if not 0 <= eval_mix <= 1:
            raise ValueError("eval_mix must be in [0,1]")
        self.frontend = ProgressiveBlurFrontEnd(mix=eval_mix)
        self.map_loss_weight = float(map_loss_weight)
        self.target_margin = float(target_margin)
        self.target_transition = float(target_transition)
        self.target_min_radius = float(target_min_radius)
        self.zero_mix_epochs = int(zero_mix_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.register_buffer("pgb_train_mix", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("pgb_eval_mix", torch.tensor(float(eval_mix), dtype=torch.float32))
        self._pgb_last_strength: Tensor | None = None

    def pgb_config(self) -> dict[str, float | int]:
        """Serializable research configuration carried by model checkpoints."""

        return {
            "pgb_map_loss_weight": self.map_loss_weight,
            "pgb_eval_mix": float(self.pgb_eval_mix),
            "pgb_zero_mix_epochs": self.zero_mix_epochs,
            "pgb_ramp_epochs": self.ramp_epochs,
            "pgb_target_margin": self.target_margin,
            "pgb_target_transition": self.target_transition,
            "pgb_target_min_radius": self.target_min_radius,
        }

    def set_train_mix(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("training blur mix must be in [0,1]")
        self.pgb_train_mix.fill_(float(value))

    def set_eval_mix(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("evaluation blur mix must be in [0,1]")
        self.pgb_eval_mix.fill_(float(value))

    def current_mix(self) -> float:
        selected = self.pgb_train_mix if self.training else self.pgb_eval_mix
        return float(selected.detach())

    def _frontend_forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.frontend(images, mix=self.current_mix())

    def predict(self, x: Tensor, profile: bool = False, augment: bool = False, embed=None):
        """Run the image-only front end for every tensor inference path."""

        # DetectionModel.__init__ performs a dummy forward to infer strides
        # before this subclass can attach the front end.
        if not hasattr(self, "frontend"):
            return super().predict(x, profile=profile, augment=augment, embed=embed)
        processed, strength = self._frontend_forward(x)
        self._pgb_last_strength = strength
        return super().predict(processed, profile=profile, augment=augment, embed=embed)

    def loss(self, batch: dict[str, Any], preds=None):
        """Return native detector loss plus the weighted map-supervision loss."""

        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            processed, strength = self._frontend_forward(batch["img"])
            # Explicit parent dispatch prevents applying the front end twice.
            preds = super().predict(processed)
            self._pgb_last_strength = strength
        else:
            strength = self._pgb_last_strength
            expected = (
                batch["img"].shape[0],
                1,
                batch["img"].shape[2],
                batch["img"].shape[3],
            )
            if strength is None or strength.shape != expected:
                # This fallback supports externally supplied detector predictions.
                strength = self.frontend.map_net(batch["img"])

        detector_total, detector_items = self.criterion(preds, batch)
        height, width = batch["img"].shape[-2:]
        boxes_by_image = yolo_xywh_to_boxes(
            batch, height=height, width=width, batch_size=batch["img"].shape[0]
        )
        targets = make_targets(
            batch["img"],
            boxes_by_image,
            margin=self.target_margin,
            transition=self.target_transition,
            min_radius=self.target_min_radius,
            valid_pixels=valid_pixels_from_batch(
                batch, height=height, width=width, device=batch["img"].device
            ),
        )
        raw_map_loss = balanced_map_loss(strength, targets)
        weighted_map_loss = raw_map_loss * self.map_loss_weight

        # Ultralytics v8DetectionLoss returns detector loss multiplied by the
        # batch size. Match that reduction so lambda is independent of batch.
        total = detector_total + weighted_map_loss * batch["img"].shape[0]
        items = dict(detector_items)
        items["map_loss"] = weighted_map_loss.detach()
        return total, items


class PGBDetectionTrainer(DetectionTrainer):
    """Detection trainer with a reproducible rho warm-up schedule."""

    _PGB_DEFAULTS = {
        "pgb_map_loss_weight": 1.0,
        "pgb_eval_mix": 0.5,
        "pgb_zero_mix_epochs": 5,
        "pgb_ramp_epochs": 5,
        "pgb_target_margin": 0.10,
        "pgb_target_transition": 0.50,
        "pgb_target_min_radius": 8.0,
    }

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ):
        options = dict(overrides or {})
        requested = {
            key: options.pop(key, default) for key, default in self._PGB_DEFAULTS.items()
        }
        self._set_pgb_options(requested)
        if self.pgb_zero_mix_epochs < 0 or self.pgb_ramp_epochs < 0:
            raise ValueError("PGB schedule epoch counts must be non-negative")
        super().__init__(cfg=cfg, overrides=options, _callbacks=_callbacks)
        if self.resume:
            restored = {
                key: getattr(self.args, key, default)
                for key, default in self._PGB_DEFAULTS.items()
            }
            self._set_pgb_options(restored)
        # Keep these values in trainer args so DDP subprocess creation and
        # checkpoint train_args preserve the exact custom configuration.
        for key in self._PGB_DEFAULTS:
            setattr(self.args, key, getattr(self, key))

    def _set_pgb_options(self, values: dict[str, Any]) -> None:
        self.pgb_map_loss_weight = float(values["pgb_map_loss_weight"])
        self.pgb_eval_mix = float(values["pgb_eval_mix"])
        self.pgb_zero_mix_epochs = int(values["pgb_zero_mix_epochs"])
        self.pgb_ramp_epochs = int(values["pgb_ramp_epochs"])
        self.pgb_target_margin = float(values["pgb_target_margin"])
        self.pgb_target_transition = float(values["pgb_target_transition"])
        self.pgb_target_min_radius = float(values["pgb_target_min_radius"])

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ) -> PGBDetectionModel:
        if isinstance(weights, PGBDetectionModel):
            self._set_pgb_options(weights.pgb_config())
        model = self.set_model_names_for_load(
            PGBDetectionModel(
                cfg=cfg or "yolov8n.yaml",
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
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
        return model

    def mix_for_epoch(self, epoch: int) -> float:
        """Zero phase, then a linear ramp whose final ramp epoch reaches eval_mix."""

        if epoch < self.pgb_zero_mix_epochs:
            return 0.0
        if self.pgb_ramp_epochs == 0:
            return self.pgb_eval_mix
        ramp_index = epoch - self.pgb_zero_mix_epochs + 1
        return self.pgb_eval_mix * min(max(ramp_index / self.pgb_ramp_epochs, 0.0), 1.0)

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        if isinstance(model, PGBDetectionModel):
            model.set_train_mix(self.mix_for_epoch(self.epoch))
        return batch

    def get_validator(self) -> DetectionValidator:
        # DetectionValidator validates against the standard YOLO schema. Pass a
        # sanitized copy while retaining pgb_* keys on trainer.args for DDP and
        # checkpoint reproducibility.
        validator_args = copy(self.args)
        for key in self._PGB_DEFAULTS:
            if hasattr(validator_args, key):
                delattr(validator_args, key)
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
        )
