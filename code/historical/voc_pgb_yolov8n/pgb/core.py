"""Core tensors for object-protective progressive background blur.

Images are floating-point RGB tensors in ``[B, 3, H, W]``. Training boxes
must be absolute-pixel ``xyxy`` coordinates after all geometric transforms.
Ground-truth boxes are used only to construct an auxiliary training target;
the inference-time front end consumes images alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                cin,
                cout,
                kernel,
                stride,
                padding=dilation * (kernel // 2),
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=False),
        )


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, cin: int, cout: int, dilation: int = 1) -> None:
        super().__init__(
            ConvBNAct(cin, cin, groups=cin, dilation=dilation),
            ConvBNAct(cin, cout, kernel=1),
        )


class BlurMapNet(nn.Module):
    """Predict continuous blur strength: 0 preserves detail, 1 uses max blur."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 16, stride=2)
        self.enc4 = ConvBNAct(16, 32, stride=2)
        self.enc8 = ConvBNAct(32, 64, stride=2)
        self.context = nn.Sequential(
            DepthwiseSeparable(64, 64, dilation=2),
            DepthwiseSeparable(64, 64, dilation=4),
        )
        self.project = ConvBNAct(64, 32, kernel=1)
        self.fuse = DepthwiseSeparable(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        shallow = self.enc4(self.stem(images))
        deep = self.project(self.context(self.enc8(shallow)))
        deep = F.interpolate(
            deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.head(self.fuse(torch.cat((shallow, deep), dim=1)))
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits.sigmoid()


@dataclass
class BlurTargets:
    strength: Tensor
    protected: Tensor
    valid: Tensor


@torch.no_grad()
def make_targets(
    images: Tensor,
    boxes_by_image: Sequence[Tensor],
    margin: float = 0.10,
    transition: float = 0.50,
    min_radius: float = 8.0,
    valid_pixels: Tensor | None = None,
    ignore_boxes_by_image: Sequence[Tensor] | None = None,
) -> BlurTargets:
    """Build the scale-normalized continuous distance target from boxes.

    Images without usable boxes receive no map supervision rather than being
    assumed to contain only background. The detector loss may still use them.
    """

    if images.ndim != 4 or not torch.is_floating_point(images):
        raise ValueError("images must be floating point [B,C,H,W]")
    batch_size, _, height, width = images.shape
    if (
        len(boxes_by_image) != batch_size
        or margin < 0
        or transition <= 0
        or min_radius <= 0
    ):
        raise ValueError("invalid box list length or geometry parameters")
    if ignore_boxes_by_image is not None and len(ignore_boxes_by_image) != batch_size:
        raise ValueError("ignore_boxes_by_image length must equal batch size")

    target = torch.zeros(
        (batch_size, 1, height, width), device=images.device, dtype=torch.float32
    )
    protected = torch.zeros_like(target, dtype=torch.bool)
    valid = torch.ones_like(target, dtype=torch.bool)
    if valid_pixels is not None:
        if valid_pixels.shape != valid.shape:
            raise ValueError("valid_pixels must have shape [B,1,H,W]")
        valid &= valid_pixels.to(device=images.device, dtype=torch.bool)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=images.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    for image_index, boxes in enumerate(boxes_by_image):
        boxes = torch.as_tensor(
            boxes, device=images.device, dtype=torch.float32
        ).reshape(-1, 4)
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

        distance_squared = torch.full(
            (height, width), float("inf"), device=images.device
        )
        for x1, y1, x2, y2 in boxes:
            box_width, box_height = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            dx = (
                (xx - center_x).abs() - (0.5 + margin) * box_width
            ).clamp_min(0)
            dy = (
                (yy - center_y).abs() - (0.5 + margin) * box_height
            ).clamp_min(0)
            radius_x = (transition * box_width).clamp_min(min_radius)
            radius_y = (transition * box_height).clamp_min(min_radius)
            candidate = (dx / radius_x).square() + (dy / radius_y).square()
            distance_squared = torch.minimum(distance_squared, candidate)

        target[image_index, 0] = 1 - torch.exp(-0.5 * distance_squared)
        protected[image_index, 0] = distance_squared == 0

        if ignore_boxes_by_image is not None:
            ignored = torch.as_tensor(
                ignore_boxes_by_image[image_index],
                device=images.device,
                dtype=torch.float32,
            ).reshape(-1, 4)
            for x1, y1, x2, y2 in ignored:
                inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
                valid[image_index, 0] &= ~inside

    return BlurTargets(target, protected, valid)


class GaussianBankRenderer(nn.Module):
    """Fixed Gaussian bank with continuous interpolation between scales."""

    def __init__(
        self,
        sigma_levels: Sequence[float] = (0, 1, 2, 4, 8),
        mix: float = 0.5,
    ) -> None:
        super().__init__()
        levels = tuple(float(sigma) for sigma in sigma_levels)
        if (
            len(levels) < 2
            or levels[0] != 0
            or any(not math.isfinite(sigma) for sigma in levels)
        ):
            raise ValueError("sigma_levels must start at 0 and be finite")
        if any(right <= left for left, right in zip(levels, levels[1:])):
            raise ValueError("sigma_levels must be strictly increasing")
        if not 0 <= mix <= 1:
            raise ValueError("mix must be in [0,1]")
        self.sigma_levels = levels
        self.mix = float(mix)
        for index, sigma in enumerate(levels[1:], start=1):
            radius = int(math.ceil(3 * sigma))
            grid = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (grid / sigma).square())
            self.register_buffer(f"kernel_{index}", kernel / kernel.sum())

    def _blur(self, images: Tensor, index: int) -> Tensor:
        if index == 0:
            return images
        kernel = getattr(self, f"kernel_{index}").to(
            device=images.device, dtype=images.dtype
        )
        radius = kernel.numel() // 2
        channels = images.shape[1]
        padding_mode = "reflect" if min(images.shape[-2:]) > radius else "replicate"
        horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
        output = F.conv2d(
            F.pad(images, (radius, radius, 0, 0), mode=padding_mode),
            horizontal,
            groups=channels,
        )
        return F.conv2d(
            F.pad(output, (0, 0, radius, radius), mode=padding_mode),
            vertical,
            groups=channels,
        )

    def _weight(self, sigma: Tensor, index: int) -> Tensor:
        levels = self.sigma_levels
        if index == 0:
            return ((levels[1] - sigma) / (levels[1] - levels[0])).clamp(0, 1)
        if index == len(levels) - 1:
            return ((sigma - levels[-2]) / (levels[-1] - levels[-2])).clamp(0, 1)
        left = (sigma - levels[index - 1]) / (levels[index] - levels[index - 1])
        right = (levels[index + 1] - sigma) / (
            levels[index + 1] - levels[index]
        )
        return torch.minimum(left, right).clamp(0, 1)

    def forward(
        self, images: Tensor, strength: Tensor, mix: float | None = None
    ) -> Tensor:
        if images.ndim != 4 or not torch.is_floating_point(images):
            raise ValueError("images must be floating point [B,C,H,W]")
        expected = (images.shape[0], 1, images.shape[2], images.shape[3])
        if strength.shape != expected:
            raise ValueError(f"strength must have shape {expected}")
        amount = self.mix if mix is None else float(mix)
        if not 0 <= amount <= 1:
            raise ValueError("mix must be in [0,1]")
        if amount == 0:
            return images

        sigma = (
            strength.to(device=images.device, dtype=images.dtype).clamp(0, 1)
            * self.sigma_levels[-1]
        )
        blended = torch.zeros_like(images)
        for index in range(len(self.sigma_levels)):
            blended = blended + self._weight(sigma, index) * self._blur(images, index)
        return (1 - amount) * images + amount * blended


class ProgressiveBlurFrontEnd(nn.Module):
    def __init__(self, mix: float = 0.5) -> None:
        super().__init__()
        self.map_net = BlurMapNet()
        self.renderer = GaussianBankRenderer(mix=mix)

    def forward(
        self, images: Tensor, mix: float | None = None
    ) -> tuple[Tensor, Tensor]:
        strength = self.map_net(images)
        return self.renderer(images, strength, mix=mix), strength


def balanced_map_loss(prediction: Tensor, target: BlurTargets) -> Tensor:
    """Balance protected and outer-region Smooth-L1 losses per image."""

    if prediction.shape != target.strength.shape:
        raise ValueError("prediction and target strength shapes differ")
    error = F.smooth_l1_loss(
        prediction.float(), target.strength.float(), reduction="none", beta=0.1
    )
    valid = target.valid.bool()
    region_losses: list[tuple[Tensor, Tensor]] = []
    for mask in (valid & target.protected, valid & ~target.protected):
        area = mask.flatten(1).sum(1)
        average = (error * mask).flatten(1).sum(1) / area.clamp_min(1)
        region_losses.append((average, area > 0))
    total = sum(value for value, _ in region_losses)
    count = sum(exists.float() for _, exists in region_losses)
    image_loss = total / count.clamp_min(1)
    has_supervision = count > 0
    return image_loss.sum() / has_supervision.sum().clamp_min(1)


def yolo_xywh_to_boxes(
    batch: dict, height: int, width: int, batch_size: int
) -> list[Tensor]:
    """Convert an augmented Ultralytics batch from normalized xywh to xyxy."""

    boxes = batch["bboxes"]
    ids = batch["batch_idx"].reshape(-1).long()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(ids):
        raise ValueError("invalid bboxes or batch_idx shape")
    if len(ids) and ((ids < 0).any() or (ids >= batch_size).any()):
        raise ValueError("batch_idx outside batch range")
    scale = boxes.new_tensor([width, height, width, height])
    xywh = boxes * scale
    xyxy = torch.cat(
        (xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2),
        dim=1,
    )
    return [xyxy[ids == index] for index in range(batch_size)]

