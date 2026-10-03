"""Bounded-error local construction of Oracle blur strength maps.

This is a candidate engineering optimization for crowded/high-resolution
images. It is not enabled in any completed experiment or formal config.
The maximum truncation error in strength is ``epsilon`` because locations
outside each box's influence rectangle have D^2 >= -2 log(epsilon), hence
``exp(-D^2/2) <= epsilon``. Box interiors are computed exactly.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from .core import BlurTargets


@torch.no_grad()
def make_targets_local(
    images: Tensor,
    boxes_by_image: Sequence[Tensor],
    *,
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
    epsilon: float = 1e-4,
    valid_pixels: Tensor | None = None,
    ignore_boxes_by_image: Sequence[Tensor] | None = None,
) -> BlurTargets:
    """Approximate ``core.make_targets`` with absolute error <= epsilon.

    All geometry is in pixels of the *already augmented* images. Ground truth
    remains training-only. This function has not been connected to a trainer.
    """

    if images.ndim != 4 or not torch.is_floating_point(images):
        raise ValueError("images must be floating point [B,C,H,W]")
    batch_size, _, height, width = images.shape
    if (len(boxes_by_image) != batch_size or margin < 0 or transition <= 0
            or min_radius <= 0 or not 0 < epsilon < 1):
        raise ValueError("invalid geometry, epsilon, or box list length")
    if ignore_boxes_by_image is not None and len(ignore_boxes_by_image) != batch_size:
        raise ValueError("ignore_boxes_by_image length must equal batch size")

    target = torch.zeros((batch_size, 1, height, width), device=images.device, dtype=torch.float32)
    protected = torch.zeros_like(target, dtype=torch.bool)
    valid = torch.ones_like(target, dtype=torch.bool)
    if valid_pixels is not None:
        if valid_pixels.shape != valid.shape:
            raise ValueError("valid_pixels must have shape [B,1,H,W]")
        valid &= valid_pixels.to(device=images.device, dtype=torch.bool)

    influence_radius = math.sqrt(-2.0 * math.log(epsilon))
    for image_index, raw_boxes in enumerate(boxes_by_image):
        boxes = torch.as_tensor(raw_boxes, device=images.device, dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() == 0:
            valid[image_index] = False
            continue
        if not torch.isfinite(boxes).all():
            raise ValueError("boxes contain non-finite values")
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
            raise ValueError("box width and height must remain positive after clipping")

        distance_squared = torch.full((height, width), float("inf"), device=images.device)
        # One device-to-host transfer per image instead of one sync per box.
        for x1, y1, x2, y2 in boxes.detach().cpu().tolist():
            box_width, box_height = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            half_x, half_y = (0.5 + margin) * box_width, (0.5 + margin) * box_height
            radius_x = max(transition * box_width, min_radius)
            radius_y = max(transition * box_height, min_radius)
            # Conservative integer bounds include every pixel center where
            # this box could change strength by more than epsilon.
            left = max(0, math.floor(center_x - half_x - influence_radius * radius_x) - 1)
            right = min(width, math.ceil(center_x + half_x + influence_radius * radius_x) + 1)
            top = max(0, math.floor(center_y - half_y - influence_radius * radius_y) - 1)
            bottom = min(height, math.ceil(center_y + half_y + influence_radius * radius_y) + 1)
            if left >= right or top >= bottom:
                continue
            xx = torch.arange(left, right, device=images.device, dtype=torch.float32) + 0.5
            yy = torch.arange(top, bottom, device=images.device, dtype=torch.float32) + 0.5
            dx2 = (((xx - center_x).abs() - half_x).clamp_min(0) / radius_x).square()
            dy2 = (((yy - center_y).abs() - half_y).clamp_min(0) / radius_y).square()
            patch = distance_squared[top:bottom, left:right]
            torch.minimum(patch, dy2[:, None] + dx2[None, :], out=patch)

        target[image_index, 0] = 1 - torch.exp(-0.5 * distance_squared)
        protected[image_index, 0] = distance_squared == 0
        if ignore_boxes_by_image is not None:
            ignored = torch.as_tensor(
                ignore_boxes_by_image[image_index], device=images.device,
                dtype=torch.float32,
            ).reshape(-1, 4)
            yy, xx = torch.meshgrid(
                torch.arange(height, device=images.device, dtype=torch.float32) + 0.5,
                torch.arange(width, device=images.device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            for x1, y1, x2, y2 in ignored:
                inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
                valid[image_index, 0] &= ~inside

    return BlurTargets(target, protected, valid)
