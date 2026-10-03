# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SARCBlock",
    "SARCDeployBlock",
    "SARCFixedBlock",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=None):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


def _fuse_sarc_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the eval-equivalent kernel and bias for one SARC Conv-BN branch."""
    if conv.training or bn.training:
        raise RuntimeError("SARC fusion requires eval() with frozen BatchNorm statistics")
    bias = (
        conv.bias
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=conv.weight.dtype, device=conv.weight.device)
    )
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * scale[:, None, None, None], bn.bias + (bias - bn.running_mean) * scale


def _fuse_sarc_output_bn(
    kernel: torch.Tensor, bias: torch.Tensor, bn: nn.BatchNorm2d
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the BatchNorm after the linear SARC branch sum."""
    if bn.training:
        raise RuntimeError("SARC fusion requires eval() with frozen BatchNorm statistics")
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return kernel * scale[:, None, None, None], bn.bias + (bias - bn.running_mean) * scale


def _embed_sarc_kernel(kernel: torch.Tensor, dilation: int) -> torch.Tensor:
    """Embed one dilation-1/2/3 3x3 kernel exactly into a centered 7x7 kernel."""
    if dilation not in {1, 2, 3} or kernel.shape[-2:] != (3, 3):
        raise ValueError("SARC only supports 3x3 branches with dilation 1, 2, or 3")
    result = kernel.new_zeros((*kernel.shape[:-2], 7, 7))
    indices = (3 - dilation, 3, 3 + dilation)
    for source_y, target_y in enumerate(indices):
        for source_x, target_x in enumerate(indices):
            result[..., target_y, target_x] = kernel[..., source_y, source_x]
    return result


def _sarc_depthwise(channels: int, kernel_size: int, dilation: int = 1) -> nn.Conv2d:
    """Build one bias-free stride-1 depthwise training branch."""
    return nn.Conv2d(
        channels,
        channels,
        kernel_size,
        stride=1,
        padding=dilation * (kernel_size - 1) // 2,
        dilation=dilation,
        groups=channels,
        bias=False,
    )


class SARCDeployBlock(nn.Module):
    """Fused SARC topology with gamma retained after SiLU as a frozen constant."""

    def __init__(self, channels: int):
        """Initialize a deploy-time 7x7 depthwise residual block."""
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=True)
        self.register_buffer("gamma", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused convolution, SiLU, frozen gamma multiplication, and residual addition."""
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(self.conv(x))


class _SARCBase(nn.Module):
    """Shared construction and in-place deploy installation for SARC controls."""

    def __init__(self, c1: int, c2: int, gamma_init: float):
        super().__init__()
        if c1 <= 0 or c1 != c2:
            raise ValueError(f"SARC must preserve a positive channel count, got c1={c1}, c2={c2}")
        self.channels = int(c1)
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)]))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Run the installed deploy convolution while retaining gamma after SiLU."""
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(self.reparam(x))

    def _install_deploy(self, deploy: SARCDeployBlock, training_names: tuple[str, ...]):
        """Replace training branches in place with their exact deploy representation."""
        self.reparam = deploy.conv
        gamma = deploy.gamma.detach().clone()
        for name in training_names:
            if hasattr(self, name):
                delattr(self, name)
        delattr(self, "gamma")
        self.register_buffer("gamma", gamma)
        self.forward = self.forward_fuse
        return self


class SARCFixedBlock(_SARCBase):
    """R1 fixed 7x7 depthwise control with the same deploy topology as full SARC."""

    sarc_mode = "R1"

    def __init__(self, c1: int, c2: int, gamma_init: float = 0.01):
        """Initialize the R1 control block."""
        super().__init__(c1, c2, gamma_init)
        self.main_conv = _sarc_depthwise(self.channels, 7)
        self.main_bn = nn.BatchNorm2d(self.channels)
        self.mix_bn = nn.BatchNorm2d(self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the single training branch and residual."""
        y = self.mix_bn(self.main_bn(self.main_conv(x)))
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(y)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the eval-equivalent 7x7 depthwise kernel and bias."""
        if self.training:
            raise RuntimeError("Call eval() before deriving a SARC deploy kernel")
        kernel, bias = _fuse_sarc_conv_bn(self.main_conv, self.main_bn)
        return _fuse_sarc_output_bn(kernel, bias, self.mix_bn)

    def to_deploy(self) -> SARCDeployBlock:
        """Create a separate, strictly equivalent deploy block."""
        kernel, bias = self.get_equivalent_kernel_bias()
        deploy = SARCDeployBlock(self.channels).to(device=kernel.device, dtype=kernel.dtype)
        deploy.conv.weight.data.copy_(kernel)
        deploy.conv.bias.data.copy_(bias)
        deploy.gamma.copy_(self.gamma.detach().to(kernel))
        return deploy.eval().requires_grad_(False)

    def fuse(self):
        """Convert this block in place for BaseModel.fuse()."""
        if hasattr(self, "reparam"):
            return self
        return self._install_deploy(self.to_deploy(), ("main_conv", "main_bn", "mix_bn"))


class SARCBlock(_SARCBase):
    """R2/R3 reparameterizable scale-aware receptive-field calibration block."""

    sarc_mode = "full"
    sarc_state_names = ("sarc_hist_ema", "sarc_hist_updates", "sarc_hist_mass")

    def __init__(
        self,
        c1: int,
        c2: int,
        gamma_init: float = 0.01,
        histogram_temperature: float = 0.50,
        histogram_decay: float = 0.99,
        histogram_min_updates: int = 16,
    ):
        """Initialize full SARC training branches and persistent assignment statistics."""
        super().__init__(c1, c2, gamma_init)
        if histogram_temperature <= 0 or not 0 <= histogram_decay < 1 or histogram_min_updates < 1:
            raise ValueError("Invalid SARC histogram temperature, decay, or warmup")
        self.dilations = (1, 2, 3)
        self.main_conv = _sarc_depthwise(self.channels, 7)
        self.main_bn = nn.BatchNorm2d(self.channels)
        self.branch_convs = nn.ModuleList([_sarc_depthwise(self.channels, 3, d) for d in self.dilations])
        self.branch_bns = nn.ModuleList([nn.BatchNorm2d(self.channels) for _ in self.dilations])
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.mix_bn = nn.BatchNorm2d(self.channels)
        self.register_buffer("sarc_hist_centers", torch.tensor([2.25, 3.00, 3.75]))
        self.register_buffer("sarc_hist_ema", torch.full((3,), 1.0 / 3.0))
        self.register_buffer("sarc_hist_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("sarc_hist_mass", torch.zeros(()))
        self.histogram_temperature = float(histogram_temperature)
        self.histogram_decay = float(histogram_decay)
        self.histogram_min_updates = int(histogram_min_updates)

    @property
    def branch_weights(self) -> torch.Tensor:
        """Return the input-independent branch mixture weights."""
        return self.branch_logits.softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all linear branches, output BN, SiLU, gamma, and residual."""
        mixed = self.main_bn(self.main_conv(x))
        for weight, conv, bn in zip(self.branch_weights, self.branch_convs, self.branch_bns):
            mixed = mixed + weight * bn(conv(x))
        mixed = self.mix_bn(mixed)
        return x + self.gamma.reshape(1, 1, 1, 1) * F.silu(mixed)

    @torch.no_grad()
    def assignment_soft_counts(self, relative_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached soft bin counts and matched-positive mass for one feature level."""
        values = relative_scale.detach().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return self.sarc_hist_ema.new_zeros(3), self.sarc_hist_mass.new_zeros(())
        centers = self.sarc_hist_centers.to(device=values.device, dtype=values.dtype)
        logits = -0.5 * ((values[:, None] - centers[None, :]) / self.histogram_temperature) ** 2
        counts = logits.softmax(-1).sum(0).to(self.sarc_hist_ema)
        return counts, self.sarc_hist_mass.new_tensor(values.numel())

    @torch.no_grad()
    def update_assignment_counts(self, global_counts: torch.Tensor, global_mass: torch.Tensor) -> None:
        """Update the EMA only after the caller has SUM-reduced counts and mass across DDP ranks."""
        counts = global_counts.detach().to(self.sarc_hist_ema)
        mass = global_mass.detach().to(self.sarc_hist_mass)
        if counts.shape != (3,):
            raise ValueError("SARC global counts must have shape [3]")
        valid = mass > 0
        probability = counts / counts.sum().clamp_min(1e-9)
        updated_histogram = self.sarc_hist_ema * self.histogram_decay + probability * (1 - self.histogram_decay)
        updated_histogram = updated_histogram / updated_histogram.sum().clamp_min(1e-9)
        first_update = self.sarc_hist_updates == 0
        updated_histogram = torch.where(first_update, probability, updated_histogram)
        updated_mass = torch.where(
            first_update, mass, self.sarc_hist_mass * self.histogram_decay + mass * (1 - self.histogram_decay)
        )
        self.sarc_hist_ema.copy_(torch.where(valid, updated_histogram, self.sarc_hist_ema))
        self.sarc_hist_mass.copy_(torch.where(valid, updated_mass, self.sarc_hist_mass))
        self.sarc_hist_updates.add_(valid.to(self.sarc_hist_updates))

    def calibration_loss(self, target: torch.Tensor | None = None) -> torch.Tensor:
        """Return KL(detached target || branch weights), or exact zero during histogram warmup."""
        target = self.sarc_hist_ema if target is None else target
        target = target.detach().to(self.branch_logits)
        target = target / target.sum().clamp_min(1e-9)
        loss = F.kl_div(F.log_softmax(self.branch_logits, 0), target, reduction="sum")
        return loss * (self.sarc_hist_updates >= self.histogram_min_updates).to(loss)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse all branches, input-independent weights, and BatchNorms into one 7x7 depthwise convolution."""
        if self.training:
            raise RuntimeError("Call eval() before deriving a SARC deploy kernel")
        kernel, bias = _fuse_sarc_conv_bn(self.main_conv, self.main_bn)
        for weight, conv, bn, dilation in zip(
            self.branch_weights.detach(), self.branch_convs, self.branch_bns, self.dilations
        ):
            branch_kernel, branch_bias = _fuse_sarc_conv_bn(conv, bn)
            kernel = kernel + weight * _embed_sarc_kernel(branch_kernel, dilation)
            bias = bias + weight * branch_bias
        return _fuse_sarc_output_bn(kernel, bias, self.mix_bn)

    def to_deploy(self) -> SARCDeployBlock:
        """Create a separate deploy block while preserving gamma after SiLU as a buffer."""
        kernel, bias = self.get_equivalent_kernel_bias()
        deploy = SARCDeployBlock(self.channels).to(device=kernel.device, dtype=kernel.dtype)
        deploy.conv.weight.data.copy_(kernel)
        deploy.conv.bias.data.copy_(bias)
        deploy.gamma.copy_(self.gamma.detach().to(kernel))
        return deploy.eval().requires_grad_(False)

    def fuse(self):
        """Convert this block in place for BaseModel.fuse()."""
        if hasattr(self, "reparam"):
            return self
        training_names = (
            "main_conv",
            "main_bn",
            "branch_convs",
            "branch_bns",
            "branch_logits",
            "mix_bn",
            "sarc_hist_centers",
            "sarc_hist_ema",
            "sarc_hist_updates",
            "sarc_hist_mass",
        )
        return self._install_deploy(self.to_deploy(), training_names)


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
