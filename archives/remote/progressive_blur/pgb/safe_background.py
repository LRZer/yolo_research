"""Conservative image-only background gate for a YOLO detector.

Ground-truth boxes supervise the gate during training. They never enter the
front-end forward path, so validation and inference cannot use oracle boxes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .core import BlurTargets, ConvBNAct, DepthwiseSeparable, GaussianBankRenderer


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = ConvBNAct(channels, channels)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x + self.second(self.first(x)))


class EfficientResidualBlock(nn.Module):
    """Keep a high-resolution detail path without full 3x3 channel mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, groups=channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x + self.pointwise(self.depthwise(x)))


class SafeBackgroundGate(nn.Module):
    """Predict a soft *permission to blur* map, not a segmentation mask."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 24, stride=2)
        self.detail2 = EfficientResidualBlock(24)
        self.down4 = ConvBNAct(24, 48, stride=2)
        self.detail4 = EfficientResidualBlock(48)
        self.down8 = ConvBNAct(48, 96, stride=2)
        self.context8 = ResidualBlock(96)
        self.down16 = ConvBNAct(96, 128, stride=2)
        self.context16 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.fuse8 = DepthwiseSeparable(128 + 96, 96)
        self.fuse4 = DepthwiseSeparable(96 + 48, 48)
        self.fuse2 = DepthwiseSeparable(48 + 24, 24)
        self.head = nn.Conv2d(24, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, -4.0)

    @staticmethod
    def _up_to(x: Tensor, skip: Tensor) -> Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or not torch.is_floating_point(images):
            raise ValueError("images must be floating point [B,3,H,W]")
        x2 = self.detail2(self.stem(images))
        x4 = self.detail4(self.down4(x2))
        x8 = self.context8(self.down8(x4))
        x16 = self.context16(self.down16(x8))
        x8 = self.fuse8(torch.cat((self._up_to(x16, x8), x8), dim=1))
        x4 = self.fuse4(torch.cat((self._up_to(x8, x4), x4), dim=1))
        x2 = self.fuse2(torch.cat((self._up_to(x4, x2), x2), dim=1))
        logits = self.head(x2)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False).sigmoid()


class SafeBackgroundFrontEnd(nn.Module):
    """Softly mix a fixed Gaussian image only where the gate permits it."""

    def __init__(self, *, sigma: float = 8.0, mix: float = 0.5) -> None:
        super().__init__()
        self.map_net = SafeBackgroundGate()
        self.renderer = GaussianBankRenderer(sigma_levels=(0.0, sigma), mix=mix)

    def forward(self, images: Tensor, mix: float | None = None) -> tuple[Tensor, Tensor]:
        permission = self.map_net(images)
        return self.renderer(images, permission, mix=mix), permission


def safe_gate_loss(
    permission: Tensor,
    target: BlurTargets,
    *,
    far_threshold: float = 0.8,
    protect_weight: float = 4.0,
    far_weight: float = 0.25,
) -> Tensor:
    """Penalize blurring GT regions strongly; treat distant background weakly.

    Pixels between the expanded GT boxes and confidently distant background
    are ignored. Empty-label images also have no gate supervision, since their
    unlabeled contents cannot safely be assumed to be background.
    """

    if permission.shape != target.strength.shape:
        raise ValueError("permission and target shapes differ")
    if not 0 < far_threshold <= 1 or protect_weight <= 0 or far_weight < 0:
        raise ValueError("invalid loss weights or far threshold")
    protected = target.valid.bool() & target.protected.bool()
    far = target.valid.bool() & (target.strength >= far_threshold) & ~protected
    # PyTorch disallows BCELoss under autocast. The equivalent log expressions
    # stay in fp32 and preserve gradients back through the sigmoid gate.
    with torch.autocast(device_type=permission.device.type, enabled=False):
        probability = permission.float().clamp(1e-6, 1 - 1e-6)
        protect_error = -torch.log1p(-probability)
        far_error = -torch.log(probability)
    protected_count = protected.flatten(1).sum(1)
    far_count = far.flatten(1).sum(1)
    protected_mean = (protect_error * protected).flatten(1).sum(1) / protected_count.clamp_min(1)
    far_mean = (far_error * far).flatten(1).sum(1) / far_count.clamp_min(1)
    per_image = (
        protect_weight * protected_mean * (protected_count > 0)
        + far_weight * far_mean * (far_count > 0)
    )
    valid_image = (protected_count > 0) | (far_count > 0)
    return per_image.sum() / valid_image.sum().clamp_min(1)
