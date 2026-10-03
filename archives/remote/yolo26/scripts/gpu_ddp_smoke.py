#!/usr/bin/env python3
"""Minimal four-process CUDA/NCCL smoke test; performs no model or dataset writes."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(4200 + rank)
    x = torch.randn((16, 3, 640, 640), generator=generator, device=device, dtype=torch.float16)
    weight = torch.randn((32, 3, 3, 3), generator=generator, device=device, dtype=torch.float16)
    y = torch.nn.functional.conv2d(x, weight, padding=1)
    checksum = y.float().square().mean()
    dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    print(
        f"rank={rank} local_rank={local_rank} physical_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device={torch.cuda.get_device_name(local_rank)} checksum={checksum.item():.6f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
