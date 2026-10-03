"""CPU 单元/烟雾测试；不是 VOC 检测实验。"""
import io
import json
import torch
from pgb_core import (BlurMapNet, GaussianBankRenderer, ProgressiveBlurFrontEnd,
                      make_targets, balanced_map_loss, yolo_xywh_to_boxes)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    x = torch.rand(2, 3, 64, 96)
    boxes = [torch.tensor([[20., 15., 40., 45.], [66., 22., 81., 39.]]), torch.empty(0, 4)]
    targets = make_targets(x, boxes, min_radius=2)
    assert torch.all(targets.strength[0][targets.protected[0]] == 0)
    assert torch.all((targets.strength >= 0) & (targets.strength <= 1))
    assert not targets.valid[1].any()
    renderer = GaussianBankRenderer(mix=1.)
    assert torch.equal(renderer(x, torch.zeros_like(targets.strength)), x)
    y = renderer(x, targets.strength)
    mask = targets.protected.expand_as(x)
    assert torch.equal(y[mask], x[mask])
    ones = torch.ones_like(x)
    assert torch.allclose(renderer(ones, targets.strength), ones, atol=2e-6)
    s = torch.rand_like(targets.strength, requires_grad=True)
    sigma = s * 8
    weights = sum(renderer._weight(sigma, i) for i in range(5))
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)
    loss = (renderer(x, s) - x).square().mean()
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
    model = ProgressiveBlurFrontEnd()
    out, pred = model(x)
    loss = balanced_map_loss(pred, targets) + (out - x).square().mean()
    loss.backward()
    assert out.shape == x.shape and pred.shape == (2, 1, 64, 96)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.map_net.stem[0].weight.grad.abs().sum() > 0
    # 只从图像处理分支反向传播，不依靠 map loss，确认渲染器与预测器之间未断图。
    model.zero_grad(set_to_none=True)
    out, pred = model(x)
    (out - x).square().mean().backward()
    assert model.map_net.head.weight.grad.abs().sum() > 0
    # xywh 转换、忽略区与空监督。
    batch = {"bboxes": torch.tensor([[.5, .5, .2, .4]]), "batch_idx": torch.tensor([0.])}
    converted = yolo_xywh_to_boxes(batch, 100, 200, 2)
    assert torch.allclose(converted[0], torch.tensor([[80., 30., 120., 70.]]))
    ignored = make_targets(x, boxes, ignore_boxes_by_image=[boxes[0], torch.empty(0, 4)])
    assert not ignored.valid[0, 0, 20:30, 25:35].any()
    empty = make_targets(x, [torch.empty(0, 4), torch.empty(0, 4)])
    empty_loss = balanced_map_loss(torch.rand_like(empty.strength, requires_grad=True), empty)
    assert empty_loss.item() == 0
    empty_loss.backward()
    model.eval()
    with torch.no_grad():
        original, _ = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = ProgressiveBlurFrontEnd().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        recovered, _ = restored(x)
        full, strength = restored(torch.rand(1, 3, 640, 640))
    assert torch.equal(original, recovered)
    assert full.shape == (1, 3, 640, 640) and strength.shape == (1, 1, 640, 640)
    result = {"status": "passed", "device": "cpu", "torch_version": torch.__version__,
              "trainable_frontend_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "scope": "核心张量运算、目标保护、凸组合、梯度、空标注、存取与640输入；未跑VOC/YOLO训练"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
