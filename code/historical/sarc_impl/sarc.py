"""Auditable CPU-first staging implementation for SARC.

This module is intentionally independent from Ultralytics.  It contains the
mathematics that must be validated before a repository-specific patch is made:

* 7x7 depthwise main branch plus 3x3 depthwise branches with dilation 1/2/3;
* exact Conv-BN and dilated-kernel fusion into one deploy-time 7x7 DWConv;
* assignment-histogram EMA and a detached-target calibration KL;
* R1/R2/R3 factories and analytical parameter/MAC budgets.

No CUDA operation or remote integration is performed here.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the eval-equivalent convolution kernel and bias.

    Folding is only valid with frozen BN running statistics, so both modules
    must be in eval mode.  This supports biased or bias-free convolutions.
    """

    if conv.training or bn.training:
        raise RuntimeError("Conv/BN fusion requires eval() with frozen BN statistics")
    if conv.out_channels != bn.num_features:
        raise ValueError("Conv output channels and BN features do not match")

    kernel = conv.weight
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, dtype=kernel.dtype, device=kernel.device)
    else:
        bias = conv.bias
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (bias - bn.running_mean) * scale
    return fused_kernel, fused_bias


def apply_bn_to_fused_conv(
    kernel: torch.Tensor, bias: torch.Tensor, bn: nn.BatchNorm2d
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold an eval BN following an already-fused convolution."""

    if bn.training:
        raise RuntimeError("BN fusion requires eval() with frozen running statistics")
    if kernel.shape[0] != bn.num_features or bias.shape != (bn.num_features,):
        raise ValueError("Kernel/bias and BN channel counts do not match")
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (
        kernel * scale.reshape(-1, 1, 1, 1),
        bn.bias + (bias - bn.running_mean) * scale,
    )


def embed_dilated_3x3_in_7x7(kernel: torch.Tensor, dilation: int) -> torch.Tensor:
    """Embed a 3x3 kernel with dilation 1/2/3 into a centered 7x7 kernel.

    For dilation ``d``, the source offsets ``{-d, 0, d}`` map to destination
    indices ``{3-d, 3, 3+d}``.  This is exact for stride 1 with source padding
    ``d`` and deploy padding 3.
    """

    if kernel.ndim != 4 or kernel.shape[-2:] != (3, 3):
        raise ValueError("Expected a [out, in/groups, 3, 3] kernel")
    if dilation not in (1, 2, 3):
        raise ValueError("Only dilation 1, 2, or 3 fits exactly in a 7x7 kernel")
    result = kernel.new_zeros((*kernel.shape[:-2], 7, 7))
    indices = (3 - dilation, 3, 3 + dilation)
    for source_y, target_y in enumerate(indices):
        for source_x, target_x in enumerate(indices):
            result[..., target_y, target_x] = kernel[..., source_y, source_x]
    return result


def assignment_relative_scale(
    widths: torch.Tensor, heights: torch.Tensor, stride: float, eps: float = 1e-9
) -> torch.Tensor:
    """Compute u=log2(sqrt(w*h)/stride) for assigned GT boxes."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    area = (widths * heights).clamp_min(eps)
    return torch.log2(torch.sqrt(area) / float(stride))


class AssignmentHistogramEMA(nn.Module):
    """Three-bin soft assignment histogram with a detached EMA target.

    The caller should all-reduce *soft counts and mass* across DDP ranks before
    calling :meth:`update_from_counts`.  Keeping distributed communication out
    of this class makes the state update explicit and auditable.
    """

    def __init__(
        self,
        centers: Sequence[float] = (2.25, 3.00, 3.75),
        temperature: float = 0.50,
        decay: float = 0.99,
        min_updates: int = 16,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if len(centers) != 3 or not all(a < b for a, b in zip(centers, centers[1:])):
            raise ValueError("centers must contain three strictly increasing values")
        if temperature <= 0 or not 0 <= decay < 1 or min_updates < 1:
            raise ValueError("invalid histogram hyperparameters")
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float64))
        self.register_buffer("ema_hist", torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_mass", torch.zeros((), dtype=torch.float64))
        self.temperature = float(temperature)
        self.decay = float(decay)
        self.min_updates = int(min_updates)
        self.eps = float(eps)

    @torch.no_grad()
    def soft_counts(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return three soft counts and valid sample mass for detached ``u``."""

        values = u.detach().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return self.ema_hist.new_zeros(3), self.ema_mass.new_zeros(())
        centers = self.centers.to(device=values.device, dtype=values.dtype)
        logits = -0.5 * ((values[:, None] - centers[None, :]) / self.temperature) ** 2
        membership = logits.softmax(dim=-1)
        return membership.sum(dim=0).to(self.ema_hist), self.ema_mass.new_tensor(values.numel())

    @torch.no_grad()
    def update_from_counts(self, counts: torch.Tensor, mass: torch.Tensor | float) -> bool:
        """Update from globally aggregated soft counts; return whether updated."""

        counts = counts.detach().to(device=self.ema_hist.device, dtype=self.ema_hist.dtype)
        mass_tensor = torch.as_tensor(mass, device=self.ema_mass.device, dtype=self.ema_mass.dtype)
        if counts.shape != (3,) or not torch.isfinite(counts).all() or not torch.isfinite(mass_tensor):
            raise ValueError("counts/mass must be finite and counts must have shape [3]")
        if mass_tensor.item() <= 0:
            return False
        probability = counts / counts.sum().clamp_min(self.eps)
        if self.ema_updates.item() == 0:
            self.ema_hist.copy_(probability)
            self.ema_mass.copy_(mass_tensor)
        else:
            self.ema_hist.mul_(self.decay).add_(probability, alpha=1.0 - self.decay)
            self.ema_hist.div_(self.ema_hist.sum().clamp_min(self.eps))
            self.ema_mass.mul_(self.decay).add_(mass_tensor, alpha=1.0 - self.decay)
        self.ema_updates.add_(1)
        return True

    @torch.no_grad()
    def update(self, u: torch.Tensor) -> bool:
        counts, mass = self.soft_counts(u)
        return self.update_from_counts(counts, mass)

    def calibration_loss(self, branch_logits: torch.Tensor) -> torch.Tensor:
        """Return KL(stopgrad(EMA histogram) || softmax(branch logits))."""

        if branch_logits.shape != (3,):
            raise ValueError("branch_logits must have shape [3]")
        if self.ema_updates.item() < self.min_updates:
            return branch_logits.sum() * 0.0
        target = self.ema_hist.detach().to(device=branch_logits.device, dtype=branch_logits.dtype)
        target = target / target.sum().clamp_min(self.eps)
        return F.kl_div(F.log_softmax(branch_logits, dim=0), target, reduction="sum")

    def snapshot(self) -> dict[str, object]:
        return {
            "histogram": self.ema_hist.detach().cpu().tolist(),
            "updates": int(self.ema_updates.item()),
            "effective_mass": float(self.ema_mass.item()),
            "ready": bool(self.ema_updates.item() >= self.min_updates),
        }


def _depthwise_conv(channels: int, kernel_size: int, dilation: int = 1) -> nn.Conv2d:
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(
        channels,
        channels,
        kernel_size,
        stride=1,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=False,
    )


class SARCDeployBlock(nn.Module):
    """Deployment form: one 7x7 DWConv, SiLU, gamma multiply, residual add."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=True)
        # gamma is learned during training, then frozen as a deploy constant.
        self.register_buffer("gamma", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(self.conv(x))


class SARCStagingBlock(nn.Module):
    """Training-time multi-branch SARC block with exact deployment fusion."""

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 3),
        gamma_init: float = 0.01,
        histogram_centers: Sequence[float] = (2.25, 3.00, 3.75),
        histogram_temperature: float = 0.50,
        histogram_decay: float = 0.99,
        histogram_min_updates: int = 16,
    ) -> None:
        super().__init__()
        if channels <= 0 or tuple(dilations) != (1, 2, 3):
            raise ValueError("SARC staging requires positive channels and dilations=(1,2,3)")
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.main_conv = _depthwise_conv(channels, 7)
        self.main_bn = nn.BatchNorm2d(channels)
        self.branch_convs = nn.ModuleList([_depthwise_conv(channels, 3, d) for d in self.dilations])
        self.branch_bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in self.dilations])
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.mix_bn = nn.BatchNorm2d(channels)
        # gamma is after SiLU by method definition.  It must remain at deploy.
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)]))
        self.histogram = AssignmentHistogramEMA(
            centers=histogram_centers,
            temperature=histogram_temperature,
            decay=histogram_decay,
            min_updates=histogram_min_updates,
        )

    @property
    def branch_weights(self) -> torch.Tensor:
        return self.branch_logits.softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.main_bn(self.main_conv(x))
        weights = self.branch_weights
        for weight, conv, bn in zip(weights, self.branch_convs, self.branch_bns):
            mixed = mixed + weight * bn(conv(x))
        mixed = self.mix_bn(mixed)
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(mixed)

    @torch.no_grad()
    def update_assignment_histogram(self, u: torch.Tensor) -> bool:
        return self.histogram.update(u)

    @torch.no_grad()
    def update_assignment_counts(self, global_counts: torch.Tensor, global_mass: torch.Tensor | float) -> bool:
        """DDP-safe interface after caller all-reduces counts and mass."""

        return self.histogram.update_from_counts(global_counts, global_mass)

    def calibration_loss(self) -> torch.Tensor:
        return self.histogram.calibration_loss(self.branch_logits)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse branches and all BNs into one 7x7 depthwise convolution."""

        if self.training:
            raise RuntimeError("Call eval() before deriving a deploy kernel")
        kernel, bias = fuse_conv_bn_eval(self.main_conv, self.main_bn)
        weights = self.branch_weights.detach()
        for weight, conv, bn, dilation in zip(weights, self.branch_convs, self.branch_bns, self.dilations):
            branch_kernel, branch_bias = fuse_conv_bn_eval(conv, bn)
            kernel = kernel + weight * embed_dilated_3x3_in_7x7(branch_kernel, dilation)
            bias = bias + weight * branch_bias
        return apply_bn_to_fused_conv(kernel, bias, self.mix_bn)

    def to_deploy(self) -> SARCDeployBlock:
        """Return a separate strictly equivalent deployment block."""

        kernel, bias = self.get_equivalent_kernel_bias()
        deploy = SARCDeployBlock(self.channels).to(device=kernel.device, dtype=kernel.dtype)
        deploy.conv.weight.data.copy_(kernel)
        deploy.conv.bias.data.copy_(bias)
        deploy.gamma.data.copy_(self.gamma.detach().to(kernel))
        deploy.eval()
        return deploy


class SingleDW7StagingBlock(nn.Module):
    """R1 fixed-cost control with the same deploy topology as SARC."""

    def __init__(self, channels: int, gamma_init: float = 0.01) -> None:
        super().__init__()
        self.channels = int(channels)
        self.main_conv = _depthwise_conv(channels, 7)
        self.main_bn = nn.BatchNorm2d(channels)
        self.mix_bn = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mix_bn(self.main_bn(self.main_conv(x)))
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(y)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("Call eval() before deriving a deploy kernel")
        kernel, bias = fuse_conv_bn_eval(self.main_conv, self.main_bn)
        return apply_bn_to_fused_conv(kernel, bias, self.mix_bn)

    def to_deploy(self) -> SARCDeployBlock:
        kernel, bias = self.get_equivalent_kernel_bias()
        deploy = SARCDeployBlock(self.channels).to(device=kernel.device, dtype=kernel.dtype)
        deploy.conv.weight.data.copy_(kernel)
        deploy.conv.bias.data.copy_(bias)
        deploy.gamma.data.copy_(self.gamma.detach().to(kernel))
        deploy.eval()
        return deploy


class SARCPyramidStaging(nn.Module):
    """Independent P3/P4/P5 wrapper for R1/R2/R3 staging."""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256),
        strides: Sequence[int] = (8, 16, 32),
        mode: str = "R3",
        calibration_gain: float = 0.01,
        **block_kwargs: object,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or len(strides) != 3 or mode not in {"R1", "R2", "R3"}:
            raise ValueError("Expected three levels and mode in {R1,R2,R3}")
        self.channels = tuple(int(c) for c in channels)
        self.strides = tuple(int(s) for s in strides)
        self.mode = mode
        self.calibration_gain = float(calibration_gain if mode == "R3" else 0.0)
        block_type = SingleDW7StagingBlock if mode == "R1" else SARCStagingBlock
        self.blocks = nn.ModuleList([block_type(c, **block_kwargs) for c in self.channels])

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError("Expected P3/P4/P5 features")
        return [block(feature) for block, feature in zip(self.blocks, features)]

    def update_assignment_histograms(self, u_by_level: Sequence[torch.Tensor]) -> None:
        if self.mode != "R3":
            return
        if len(u_by_level) != 3:
            raise ValueError("Expected one u tensor per level")
        for block, u in zip(self.blocks, u_by_level):
            assert isinstance(block, SARCStagingBlock)
            block.update_assignment_histogram(u)

    def calibration_loss(self) -> torch.Tensor:
        anchor = next(self.parameters())
        if self.mode != "R3":
            return anchor.sum() * 0.0
        raw = sum(block.calibration_loss() for block in self.blocks)
        return self.calibration_gain * raw

    def assignment_state(self) -> list[dict[str, object]]:
        if self.mode != "R3":
            return []
        return [block.histogram.snapshot() for block in self.blocks]


@dataclass(frozen=True)
class SARCBudget:
    deploy_conv_weights: int
    deploy_biases: int
    deploy_gamma_constants: int
    deploy_total_parameters: int
    deploy_total_stored_scalars: int
    deploy_macs: int
    deploy_flops_two_per_mac: int
    training_branch_parameters_r1: int
    training_branch_parameters_r2_r3: int
    training_conv_macs_r1: int
    training_conv_macs_r2_r3: int


def sarc_budget(
    channels: Sequence[int] = (64, 128, 256),
    spatial_shapes: Sequence[tuple[int, int]] = ((80, 80), (40, 40), (20, 20)),
) -> SARCBudget:
    """Analytical budget; excludes non-parameter activation/residual operations."""

    if len(channels) != len(spatial_shapes):
        raise ValueError("channels and spatial_shapes must have equal lengths")
    channel_sum = sum(channels)
    levels = len(channels)
    deploy_weights = 49 * channel_sum
    deploy_biases = channel_sum
    deploy_gamma_constants = levels
    deploy_macs = sum(c * h * w * 49 for c, (h, w) in zip(channels, spatial_shapes))
    train_macs_r23 = sum(c * h * w * (49 + 3 * 9) for c, (h, w) in zip(channels, spatial_shapes))
    # R1: conv 49C + two BNs 4C + scalar gamma.
    train_params_r1 = 53 * channel_sum + levels
    # R2/R3: main conv 49C, 3 small convs 27C, five BNs 10C,
    # three branch logits and one gamma per level. EMA buffers are not parameters.
    train_params_r23 = 86 * channel_sum + 4 * levels
    return SARCBudget(
        deploy_conv_weights=deploy_weights,
        deploy_biases=deploy_biases,
        deploy_gamma_constants=deploy_gamma_constants,
        deploy_total_parameters=deploy_weights + deploy_biases,
        deploy_total_stored_scalars=deploy_weights + deploy_biases + deploy_gamma_constants,
        deploy_macs=deploy_macs,
        deploy_flops_two_per_mac=2 * deploy_macs,
        training_branch_parameters_r1=train_params_r1,
        training_branch_parameters_r2_r3=train_params_r23,
        training_conv_macs_r1=deploy_macs,
        training_conv_macs_r2_r3=train_macs_r23,
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
