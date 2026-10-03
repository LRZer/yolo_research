"""Unconnected candidate: render stochastic Oracle blur only for chosen images.

The current formal trainer intentionally retains the registered original
implementation. This module exists solely for isolated equivalence/cost
tests before any separately registered follow-up experiment.
"""

from __future__ import annotations

from typing import Any

import torch

from .core import GaussianBankRenderer, make_targets, yolo_xywh_to_boxes
from .oracle_integration import apply_oracle_blur
from .ultralytics_integration import valid_pixels_from_batch


@torch.no_grad()
def apply_stochastic_oracle_blur_selected(
    batch: dict[str, Any],
    renderer: GaussianBankRenderer,
    *,
    mix: float,
    probability: float,
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
) -> dict[str, Any]:
    """Same image-level Bernoulli treatment, omitting unchosen map/render work.

    Diagnostic ``oracle_strength`` for the *unchosen* images is deliberately
    absent. It is not part of the detector loss. This has not been benchmarked
    as a complete trainer and is not enabled in any run.
    """

    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    if mix == 0 or probability == 0:
        return batch
    if probability == 1:
        return apply_oracle_blur(
            batch,
            renderer,
            mix=mix,
            margin=margin,
            transition=transition,
            min_radius=min_radius,
        )
    images = batch["img"]
    count, _, height, width = images.shape
    chosen = torch.rand(count, device=images.device) < probability
    batch["oracle_augmented"] = chosen
    if not bool(chosen.any()):
        return batch
    selected_indices = chosen.nonzero(as_tuple=False).flatten()
    selected_images = images[selected_indices]
    boxes_by_image = yolo_xywh_to_boxes(batch, height=height, width=width, batch_size=count)
    selected_boxes = [boxes_by_image[index] for index in selected_indices.tolist()]
    valid_pixels = valid_pixels_from_batch(batch, height=height, width=width, device=images.device)
    selected_valid = valid_pixels[selected_indices] if valid_pixels is not None else None
    targets = make_targets(
        selected_images,
        selected_boxes,
        margin=margin,
        transition=transition,
        min_radius=min_radius,
        valid_pixels=selected_valid,
    )
    processed = renderer(selected_images, targets.strength, mix=mix)
    result = images.clone()
    result[selected_indices] = processed
    batch["img"] = result
    batch["oracle_strength_selected"] = targets.strength
    return batch
