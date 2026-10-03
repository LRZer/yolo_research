"""Reproducible synthetic CPU cost/equivalence check for Oracle targets.

This does not benchmark the full blur renderer/training pipeline and does not
use VisDrone images or any GPU. Never interpret it as end-to-end speed.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pgb.core import make_targets
from pgb.fast_oracle_target import make_targets_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--boxes", type=int, default=70)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(2027)
    center = torch.rand((args.boxes, 2), generator=generator) * torch.tensor([args.width, args.height])
    size = 4 + 40 * torch.rand((args.boxes, 2), generator=generator)
    boxes = torch.cat((center - size / 2, center + size / 2), dim=1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, args.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, args.height)
    boxes = boxes[(boxes[:, 2:] > boxes[:, :2]).all(dim=1)]
    images = torch.zeros((1, 3, args.height, args.width), dtype=torch.float32)

    results = {}
    outputs = {}
    for label, function in (("exact", make_targets), ("local_eps1e-4", make_targets_local)):
        durations = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            outputs[label] = function(images, [boxes])
            durations.append(time.perf_counter() - started)
        results[label] = {"seconds": durations, "minimum_seconds": min(durations)}
    results.update({
        "device": "CPU",
        "image_shape": list(images.shape),
        "boxes": len(boxes),
        "max_strength_abs_difference": float((outputs["exact"].strength - outputs["local_eps1e-4"].strength).abs().max()),
        "protected_exactly_equal": bool(torch.equal(outputs["exact"].protected, outputs["local_eps1e-4"].protected)),
        "valid_exactly_equal": bool(torch.equal(outputs["exact"].valid, outputs["local_eps1e-4"].valid)),
    })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
