"""CPU-only exact-vs-local Oracle target comparison on real VisDrone frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.core import make_targets
from pgb.fast_oracle_target import make_targets_local
from tools.export_oracle_visualizations import label_path, letterbox, read_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    for path in args.images:
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"cannot decode {path}")
        labels = read_labels(label_path(path))
        _, boxes, valid, _, _ = letterbox(source, labels, size=960)
        images = torch.zeros((1, 3, 960, 960), dtype=torch.float32)
        box_list = [torch.from_numpy(boxes)]
        times = {}
        outputs = {}
        for name, func in (("exact", make_targets), ("bounded_local_eps1e-4", make_targets_local)):
            func(images, box_list, valid_pixels=valid)
            durations = []
            for _ in range(3):
                started = time.perf_counter()
                outputs[name] = func(images, box_list, valid_pixels=valid)
                durations.append(time.perf_counter() - started)
            times[name] = {"seconds": durations, "min_seconds": min(durations)}
        a = outputs["exact"]
        b = outputs["bounded_local_eps1e-4"]
        delta = (a.strength - b.strength).abs()
        results.append({
            "image": str(path),
            "boxes": len(boxes),
            "times": times,
            "speedup_min_ratio": times["exact"]["min_seconds"] / times["bounded_local_eps1e-4"]["min_seconds"],
            "max_strength_absolute_error": float(delta.max()),
            "pixels_error_above_1e-4": int((delta > 1e-4).sum()),
            "protected_equal": bool(torch.equal(a.protected, b.protected)),
            "valid_equal": bool(torch.equal(a.valid, b.valid)),
        })
    print(json.dumps({"device": "CPU", "image_size": 960, "results": results}, indent=2))


if __name__ == "__main__":
    main()
