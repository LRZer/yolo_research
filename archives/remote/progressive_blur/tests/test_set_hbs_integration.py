"""CPU/GPU smoke for the HBS-only YOLOv8 training branch."""

import torch
from copy import copy

from pgb.set_hbs_integration import HBSOnlyDetectionModel
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG


def test_hbs_loss_gradients_and_inference_identity(checkpoint: str, device: str = "cpu") -> None:
    torch.manual_seed(2027)
    source = YOLO(checkpoint).model
    method = HBSOnlyDetectionModel(cfg="yolov8n.yaml", nc=20, verbose=False,
                                   hbs_mode="hbs", hbs_aux_weight=1.0)
    control = HBSOnlyDetectionModel(cfg="yolov8n.yaml", nc=20, verbose=False,
                                    hbs_mode="identity", hbs_aux_weight=1.0)
    method.names = source.names
    control.names = source.names
    method.args = copy(DEFAULT_CFG)
    control.args = copy(DEFAULT_CFG)
    method.load(source)
    control.load(source)
    for key, tensor in source.model.state_dict().items():
        assert torch.equal(method.model.state_dict()[key], tensor), key
        assert torch.equal(control.model.state_dict()[key], tensor), key
    method.to(device).eval()
    control.to(device).eval()
    images = torch.rand(2, 3, 128, 128, device=device)
    with torch.inference_mode():
        raw = source.to(device).eval().predict(images)[0]
        hbs = method.predict(images)[0]
        identity = control.predict(images)[0]
    assert torch.equal(raw, hbs)
    assert torch.equal(raw, identity)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.tensor([[0.0], [1.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]], device=device),
    }
    method.train()
    method.zero_grad(set_to_none=True)
    loss, items = method.loss(batch)
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value).all() for value in items.values())
    loss.sum().backward()
    gradients = [parameter.grad for parameter in method.hbs_blocks.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
    control.train()
    control.zero_grad(set_to_none=True)
    baseline_loss, baseline_items = control.loss(batch)
    assert torch.isfinite(baseline_loss).all()
    assert all(torch.isfinite(value).all() for value in baseline_items.values())
    baseline_loss.sum().backward()


def test_scratch_initialization_pair(model_yaml: str) -> None:
    torch.manual_seed(2027)
    method = HBSOnlyDetectionModel(cfg=model_yaml, nc=10, verbose=False, hbs_mode="hbs")
    torch.manual_seed(2027)
    control = HBSOnlyDetectionModel(cfg=model_yaml, nc=10, verbose=False, hbs_mode="identity")
    for key, tensor in method.model.state_dict().items():
        assert torch.equal(tensor, control.model.state_dict()[key]), key


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scratch-yaml")
    args = parser.parse_args()
    test_hbs_loss_gradients_and_inference_identity(args.checkpoint, args.device)
    if args.scratch_yaml:
        test_scratch_initialization_pair(args.scratch_yaml)
    print("HBS integration smoke passed")
