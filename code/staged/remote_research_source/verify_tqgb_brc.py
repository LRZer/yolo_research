"""CPU-only verification for the TQGB Base-Referenced Clipping causal control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
from pathlib import Path

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ultralytics" / "utils" / "tqgb_clip.py"
SPEC = importlib.util.spec_from_file_location("tqgb_clip_standalone", MODULE_PATH)
TQGB_CLIP = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(TQGB_CLIP)
base_referenced_clip_grad_ = TQGB_CLIP.base_referenced_clip_grad_
distributed_sum_gradients_ = TQGB_CLIP.distributed_sum_gradients_


def assert_close(actual, expected, label, rtol=1e-6, atol=1e-7):
    """Assert tensor/scalar closeness with a compact failure message."""
    actual_tensor = torch.as_tensor(actual, dtype=torch.float64)
    expected_tensor = torch.as_tensor(expected, dtype=torch.float64)
    if not torch.allclose(actual_tensor, expected_tensor, rtol=rtol, atol=atol):
        raise AssertionError(f"{label}: actual={actual_tensor}, expected={expected_tensor}")


def parameter_with_grad(gradient):
    """Create a CPU parameter carrying a chosen synthetic gradient."""
    gradient = torch.as_tensor(gradient, dtype=torch.float32)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    parameter.grad = gradient.clone()
    return parameter


def test_default_off_config():
    """Confirm the library default is combined and only the formal control opts into base."""
    default_text = (ROOT / "ultralytics" / "cfg" / "default.yaml").read_text(encoding="utf-8")
    formal_text = (
        ROOT / "configs" / "research" / "tqgb_t1_brc_cls_m2o_proj_g008_e50_w4_pin_numa1.yaml"
    ).read_text(encoding="utf-8")
    assert "tqgb_clip_reference: combined" in default_text
    assert "tqgb_clip_reference: base" in formal_text


def test_route_off_equivalence():
    """BRC with zero route must equal PyTorch global clipping."""
    gradients = ([12.0, -5.0], [3.0, 4.0, -2.0])
    standard = [parameter_with_grad(value) for value in gradients]
    controlled = [parameter_with_grad(value) for value in gradients]
    torch.nn.utils.clip_grad_norm_(standard, max_norm=10.0)
    stats = base_referenced_clip_grad_(controlled, [None, None], max_norm=10.0)
    for index, (left, right) in enumerate(zip(controlled, standard)):
        assert_close(left.grad, right.grad, f"route-off parameter {index}")
    assert_close(stats["base_reference_norm"], stats["combined_norm"], "route-off norm identity")
    assert_close(
        stats["clip_scale_base_applied"],
        stats["clip_scale_combined_counterfactual"],
        "route-off scale identity",
    )


def test_route_active_reconstruction():
    """Active BRC must retain route gradients while removing them only from the norm reference."""
    base = (torch.tensor([6.0, 2.0]), torch.tensor([4.0, 8.0]))
    route = (torch.tensor([3.0, -1.0]), None)
    combined = (base[0] + route[0], base[1])
    parameters = [parameter_with_grad(value) for value in combined]
    stats = base_referenced_clip_grad_(parameters, list(route), max_norm=5.0)

    base_norm = math.sqrt(sum(float(value.square().sum()) for value in base))
    combined_norm = math.sqrt(sum(float(value.square().sum()) for value in combined))
    route_norm = float(route[0].norm())
    alpha_base = min(1.0, 5.0 / base_norm)
    alpha_combined = min(1.0, 5.0 / combined_norm)
    assert alpha_base != alpha_combined, "Synthetic active route failed to change the ordinary global scale"
    assert_close(stats["base_reference_norm"], base_norm, "active base norm")
    assert_close(stats["combined_norm"], combined_norm, "active combined norm")
    assert_close(stats["tqgb_route_o2o_cls_norm"], route_norm, "active route norm")
    assert_close(parameters[0].grad, combined[0] * alpha_base, "active receiving-head update")
    assert_close(parameters[1].grad, base[1] * alpha_base, "active non-route update")
    recovered_route = parameters[0].grad / alpha_base - base[0]
    assert_close(recovered_route, route[0], "route retained after BRC")


class ToyDualPath(torch.nn.Module):
    """Two-parameter-group model used to exercise real DDP reduction hooks on CPU."""

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
        self.o2o_cls = torch.nn.Parameter(torch.tensor([0.75, 1.25], dtype=torch.float32))

    def forward(self, multiplier):
        return self.shared * multiplier, self.o2o_cls * multiplier


def ddp_worker(rank, world_size, port):
    """Validate route capture + SUM against a closed-form two-rank global batch."""
    # Some CPU-only Windows PyTorch builds ship Gloo without libuv; this keeps the verifier portable and does not
    # affect the NCCL path used by the server-side formal training.
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # The real eager detector uses find_unused_parameters=True, so exercise the same reducer mode here.
        model = DistributedDataParallel(ToyDualPath(), find_unused_parameters=True)
        shared, head = model(torch.ones((), dtype=torch.float32))
        base_shared_coeff = torch.tensor([1.0 + rank, 2.0 + rank])
        base_head_coeff = torch.tensor([3.0 + rank, 4.0 + rank])
        route_coeff = torch.tensor([0.5 + rank, -0.25 - rank])
        base_loss = (shared * base_shared_coeff).sum() + (head * base_head_coeff).sum()
        route_loss = (head * route_coeff).sum()

        local_route = torch.autograd.grad(
            route_loss,
            [model.module.o2o_cls],
            retain_graph=True,
            allow_unused=False,
        )[0].detach().clone()
        distributed_sum_gradients_([local_route])
        # Mirror the trainer: local loss is multiplied by world_size, then DDP's averaged reduction produces a SUM
        # over the original local batch gradients.
        ((base_loss + route_loss) * world_size).backward()
        parameters = [model.module.shared, model.module.o2o_cls]
        stats = base_referenced_clip_grad_(parameters, [None, local_route], max_norm=5.0)

        expected_base_shared = sum(
            (torch.tensor([1.0 + other, 2.0 + other]) for other in range(world_size)),
            torch.zeros(2),
        )
        expected_base_head = sum(
            (torch.tensor([3.0 + other, 4.0 + other]) for other in range(world_size)),
            torch.zeros(2),
        )
        expected_route = sum(
            (torch.tensor([0.5 + other, -0.25 - other]) for other in range(world_size)),
            torch.zeros(2),
        )
        expected_base_norm = math.sqrt(
            float(expected_base_shared.square().sum() + expected_base_head.square().sum())
        )
        alpha_base = min(1.0, 5.0 / expected_base_norm)
        assert_close(local_route, expected_route, f"rank {rank} DDP route SUM", rtol=1e-5)
        assert_close(stats["base_reference_norm"], expected_base_norm, f"rank {rank} DDP base norm", rtol=1e-5)
        assert_close(
            model.module.shared.grad,
            expected_base_shared * alpha_base,
            f"rank {rank} DDP non-route update",
            rtol=1e-5,
        )
        assert_close(
            model.module.o2o_cls.grad,
            (expected_base_head + expected_route) * alpha_base,
            f"rank {rank} DDP route update",
            rtol=1e-5,
        )
    finally:
        dist.destroy_process_group()


def free_port():
    """Reserve and release a localhost port for the bounded synthetic DDP test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ddp", action="store_true")
    args = parser.parse_args()

    test_default_off_config()
    test_route_off_equivalence()
    test_route_active_reconstruction()
    tests = ["default_off_config", "route_off_equivalence", "route_active_reconstruction"]
    if not args.skip_ddp:
        mp.spawn(ddp_worker, args=(2, free_port()), nprocs=2, join=True)
        tests.append("two_rank_gloo_autograd_and_sum")
    print(json.dumps({"status": "PASS", "device": "cpu", "tests": tests}, ensure_ascii=False))


if __name__ == "__main__":
    # This verifier is intentionally CPU-only. CUDA visibility is not needed for any test above.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
