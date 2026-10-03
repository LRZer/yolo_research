"""Remote CUDA audit of the combined PGB + YOLOv8n loss graph."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from pgb.ultralytics_integration import PGBDetectionModel


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run with exactly one task-visible CUDA device")
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    device = torch.device("cuda:0")
    model = PGBDetectionModel("yolov8n.yaml", nc=2, verbose=False).to(device).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.set_train_mix(0.5)
    batch = {
        "img": torch.rand(2, 3, 128, 128, device=device),
        "cls": torch.tensor([[0.0], [1.0]], device=device),
        "bboxes": torch.tensor(
            [[0.35, 0.45, 0.25, 0.35], [0.65, 0.55, 0.20, 0.25]],
            device=device,
        ),
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
    }
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, items = model(batch)
    loss.sum().backward()
    torch.cuda.synchronize()
    frontend_grad = model.frontend.map_net.head.weight.grad
    detector_grad = model.model[0].conv.weight.grad
    if frontend_grad is None or detector_grad is None:
        raise AssertionError("front end or detector missing gradient")
    if not torch.isfinite(frontend_grad).all() or not torch.isfinite(detector_grad).all():
        raise AssertionError("front end or detector has non-finite gradient")

    state_keys = tuple(model.state_dict())
    result = {
        "status": "passed",
        "scope": "combined PGB+YOLOv8n CUDA/AMP loss and backward graph",
        "visible_gpu": torch.cuda.get_device_name(0),
        "loss_keys": list(items),
        "total_loss": float(loss.detach().sum()),
        "map_loss": float(items["map_loss"]),
        "frontend_gradient_l1": float(frontend_grad.abs().sum()),
        "detector_gradient_l1": float(detector_grad.abs().sum()),
        "checkpoint_contains_frontend": any(
            key.startswith("frontend.") for key in state_keys
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

