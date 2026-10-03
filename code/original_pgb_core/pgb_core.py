"""目标保护的渐变背景模糊：PyTorch 核心原型，不含 YOLO/VOC 训练器。

约定：images 为浮点 RGB [B,3,H,W]，范围 [0,1]。
boxes 为经过当前几何增强后的绝对像素 xyxy 坐标；不能传入原图旧坐标。
make_targets 仅用于训练监督/明确标注的真值诊断，不得参与正式推理。
ProgressiveBlurFrontEnd.forward 只接收图像，不读取任何真实框。
这是待验证的研究实现，不承诺提升检测精度，不含训练权重。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1, dilation: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride,
                      padding=dilation * (kernel // 2),
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=False),
        )


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__(ConvBNAct(cin, cin, groups=cin, dilation=dilation),
                         ConvBNAct(cin, cout, kernel=1))


class BlurMapNet(nn.Module):
    """预测连续模糊强度 S：0=清晰，1=最大模糊；不是精确分割概率。"""
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 16, stride=2)
        self.enc4 = ConvBNAct(16, 32, stride=2)
        self.enc8 = ConvBNAct(32, 64, stride=2)
        self.context = nn.Sequential(DepthwiseSeparable(64, 64, dilation=2),
                                     DepthwiseSeparable(64, 64, dilation=4))
        self.project = ConvBNAct(64, 32, kernel=1)
        self.fuse = DepthwiseSeparable(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, -3.0)  # 初始偏向弱处理，非保护保证。

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images 必须是 [B,3,H,W]")
        shallow = self.enc4(self.stem(images))
        deep = self.project(self.context(self.enc8(shallow)))
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(self.fuse(torch.cat((shallow, deep), dim=1)))
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits.sigmoid()


@dataclass
class BlurTargets:
    strength: Tensor   # [B,1,H,W] 连续几何监督 S*
    protected: Tensor  # bool [B,1,H,W] 扩张框并集
    valid: Tensor      # bool [B,1,H,W] 可监督区域；忽略 padding/difficult 等


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
    """从训练框构造尺度归一化距离场。

    每侧扩张 margin*w / margin*h；过渡半径为 max(transition*w, min_radius)。
    min_radius 的单位为当前输入图像像素，默认值面向约 640 大小的输入。
    空标注图不假定为纯背景：返回 valid=False，由检测损失独立处理。
    difficult 框可通过 ignore_boxes_by_image 排除出模糊监督。
    """
    if images.ndim != 4 or not torch.is_floating_point(images):
        raise ValueError("images 必须是浮点 [B,C,H,W]")
    b, _, h, w = images.shape
    if len(boxes_by_image) != b or margin < 0 or transition <= 0 or min_radius <= 0:
        raise ValueError("框列表长度或几何参数无效")
    if ignore_boxes_by_image is not None and len(ignore_boxes_by_image) != b:
        raise ValueError("ignore_boxes_by_image 长度必须等于 batch size")
    target = torch.zeros((b, 1, h, w), device=images.device, dtype=torch.float32)
    protected = torch.zeros_like(target, dtype=torch.bool)
    valid = torch.ones_like(target, dtype=torch.bool)
    if valid_pixels is not None:
        if valid_pixels.shape != valid.shape:
            raise ValueError("valid_pixels 必须是 [B,1,H,W]")
        valid &= valid_pixels.to(device=images.device, dtype=torch.bool)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=images.device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=images.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    for n, boxes in enumerate(boxes_by_image):
        boxes = torch.as_tensor(boxes, device=images.device, dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() == 0:
            valid[n] = False
            continue
        if not torch.isfinite(boxes).all():
            raise ValueError("框含非有限数值")
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
            raise ValueError("框裁剪到图像后宽高必须为正；请先过滤无效标注")
        d2 = torch.full((h, w), float("inf"), device=images.device)
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            dx = ((xx - cx).abs() - (0.5 + margin) * bw).clamp_min(0)
            dy = ((yy - cy).abs() - (0.5 + margin) * bh).clamp_min(0)
            rx = (transition * bw).clamp_min(min_radius)
            ry = (transition * bh).clamp_min(min_radius)
            d2 = torch.minimum(d2, (dx / rx).square() + (dy / ry).square())
        target[n, 0] = 1 - torch.exp(-0.5 * d2)
        protected[n, 0] = d2 == 0
        if ignore_boxes_by_image is not None:
            ignored = torch.as_tensor(ignore_boxes_by_image[n], device=images.device,
                                      dtype=torch.float32).reshape(-1, 4)
            for x1, y1, x2, y2 in ignored:
                inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
                valid[n, 0] &= ~inside
    return BlurTargets(target, protected, valid)


class GaussianBankRenderer(nn.Module):
    """固定高斯核 + 相邻尺度线性插值，近似空间变化的低通滤波。

    不是逐像素精确动态高斯核，也不模拟光学景深。
    核是 buffer，不是可学习参数；图像和强度图均可反向传播。
    sigma_levels 以实际输入像素计；改变输入尺寸时需明确是否同比缩放。
    """
    def __init__(self, sigma_levels: Sequence[float] = (0, 1, 2, 4, 8),
                 mix: float = 0.5):
        super().__init__()
        levels = tuple(float(s) for s in sigma_levels)
        if len(levels) < 2 or levels[0] != 0 or any(not math.isfinite(s) for s in levels):
            raise ValueError("sigma_levels 需以 0 开头，至少包含两个有限值")
        if any(b <= a for a, b in zip(levels, levels[1:])) or not 0 <= mix <= 1:
            raise ValueError("sigma_levels 需严格递增，mix 需在 [0,1]")
        self.sigma_levels, self.mix = levels, float(mix)
        for i, sigma in enumerate(levels[1:], start=1):
            radius = int(math.ceil(3 * sigma))
            grid = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (grid / sigma).square())
            self.register_buffer(f"kernel_{i}", kernel / kernel.sum())

    def _blur(self, images: Tensor, i: int) -> Tensor:
        if i == 0:
            return images
        kernel = getattr(self, f"kernel_{i}").to(device=images.device, dtype=images.dtype)
        radius, channels = kernel.numel() // 2, images.shape[1]
        # 小图无法 reflect 到大 padding 时使用 replicate，避免维度报错。
        mode = "reflect" if min(images.shape[-2:]) > radius else "replicate"
        horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
        out = F.conv2d(F.pad(images, (radius, radius, 0, 0), mode=mode),
                       horizontal, groups=channels)
        return F.conv2d(F.pad(out, (0, 0, radius, radius), mode=mode),
                        vertical, groups=channels)

    def _weight(self, sigma: Tensor, i: int) -> Tensor:
        levels = self.sigma_levels
        if i == 0:
            return ((levels[1] - sigma) / (levels[1] - levels[0])).clamp(0, 1)
        if i == len(levels) - 1:
            return ((sigma - levels[-2]) / (levels[-1] - levels[-2])).clamp(0, 1)
        left = (sigma - levels[i - 1]) / (levels[i] - levels[i - 1])
        right = (levels[i + 1] - sigma) / (levels[i + 1] - levels[i])
        return torch.minimum(left, right).clamp(0, 1)

    def forward(self, images: Tensor, strength: Tensor, mix: float | None = None) -> Tensor:
        if images.ndim != 4 or not torch.is_floating_point(images):
            raise ValueError("images 必须是浮点 [B,C,H,W]")
        expected = (images.shape[0], 1, images.shape[2], images.shape[3])
        if strength.shape != expected:
            raise ValueError(f"strength 必须是 {expected}")
        amount = self.mix if mix is None else float(mix)
        if not 0 <= amount <= 1:
            raise ValueError("mix 必须在 [0,1]")
        sigma = strength.to(device=images.device, dtype=images.dtype).clamp(0, 1) * self.sigma_levels[-1]
        blended = torch.zeros_like(images)
        for i in range(len(self.sigma_levels)):
            blended = blended + self._weight(sigma, i) * self._blur(images, i)
        return (1 - amount) * images + amount * blended


class ProgressiveBlurFrontEnd(nn.Module):
    def __init__(self, mix: float = 0.5):
        super().__init__()
        self.map_net = BlurMapNet()
        self.renderer = GaussianBankRenderer(mix=mix)

    def forward(self, images: Tensor, mix: float | None = None) -> tuple[Tensor, Tensor]:
        """推理接口只需要 images；返回处理图与预测强度图。"""
        strength = self.map_net(images)
        return self.renderer(images, strength, mix=mix), strength


def balanced_map_loss(prediction: Tensor, target: BlurTargets) -> Tensor:
    """按每张图的保护/外围区域分别求均值，再对存在的区域求平均。

    这是连续回归监督，不是精确分割损失；空监督返回可微的 0。
    返回按 batch 求平均的标量。与 YOLO 损失相加前须核对其 batch reduction。
    """
    if prediction.shape != target.strength.shape:
        raise ValueError("预测图与监督图形状不同")
    error = F.smooth_l1_loss(prediction.float(), target.strength.float(), reduction="none", beta=0.1)
    valid = target.valid.bool()
    losses = []
    for mask in (valid & target.protected, valid & ~target.protected):
        area = mask.flatten(1).sum(1)
        average = (error * mask).flatten(1).sum(1) / area.clamp_min(1)
        losses.append((average, area > 0))
    total = sum(value for value, _ in losses)
    count = sum(exists.float() for _, exists in losses)
    image_loss = total / count.clamp_min(1)
    has_supervision = count > 0
    return image_loss.sum() / has_supervision.sum().clamp_min(1)


def yolo_xywh_to_boxes(batch: dict, height: int, width: int, batch_size: int) -> list[Tensor]:
    """转换 Ultralytics batch 中归一化 xywh；在所有几何增强后调用。

    batch 需含 bboxes [N,4] 和 batch_idx [N]，与当前 batch['img'] 对齐。
    不提供数据加载、VOC 下载或训练器。
    """
    boxes = batch["bboxes"]
    ids = batch["batch_idx"].reshape(-1).long()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(ids):
        raise ValueError("bboxes 或 batch_idx 形状错误")
    if len(ids) and ((ids < 0).any() or (ids >= batch_size).any()):
        raise ValueError("batch_idx 超出 batch 范围")
    scale = boxes.new_tensor([width, height, width, height])
    xywh = boxes * scale
    xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2,
                      xywh[:, :2] + xywh[:, 2:] / 2), dim=1)
    return [xyxy[ids == i] for i in range(batch_size)]
