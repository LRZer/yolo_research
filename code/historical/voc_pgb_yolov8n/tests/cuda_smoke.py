"""Remote CUDA smoke test for the front end; this is not a training result."""

from __future__ import annotations

import json
import time

import torch

from pgb import ProgressiveBlurFrontEnd, balanced_map_loss, make_targets


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "The smoke test expects exactly one task-visible GPU after CUDA_VISIBLE_DEVICES"
        )

    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    device = torch.device("cuda:0")
    model = ProgressiveBlurFrontEnd(mix=0.5).to(device).train()
    images = torch.rand(2, 3, 128, 160, device=device)
    boxes = [
        torch.tensor([[20.0, 15.0, 70.0, 100.0]], device=device),
        torch.tensor([[80.0, 30.0, 145.0, 115.0]], device=device),
    ]
    targets = make_targets(images, boxes, min_radius=2)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, prediction = model(images)
        loss = (output - images).square().mean() + balanced_map_loss(
            prediction, targets
        )
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    gradients = [parameter.grad for parameter in model.parameters()]
    if not all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    ):
        raise AssertionError("missing or non-finite CUDA gradient")
    if model.map_net.head.weight.grad.abs().sum() == 0:
        raise AssertionError("map head received zero gradient")

    model.eval()
    full = torch.rand(1, 3, 640, 640, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        full_output, full_strength = model(full)
    torch.cuda.synchronize()
    if full_output.shape != full.shape or full_strength.shape != (1, 1, 640, 640):
        raise AssertionError("unexpected full-resolution output shape")

    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "CUDA/AMP forward-backward, finite gradients, 640 inference shape",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "visible_gpu_count": torch.cuda.device_count(),
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "trainable_frontend_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "smoke_step_seconds": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "loss": float(loss.detach()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

