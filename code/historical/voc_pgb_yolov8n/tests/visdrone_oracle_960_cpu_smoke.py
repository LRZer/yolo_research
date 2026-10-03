"""One-image CPU functional smoke of the exact 960px Oracle blur path.

No detector, GPU, dataset images, or annotations are used. This checks only
that the unmodified training preprocessor remains finite at 960 pixels.
"""

from __future__ import annotations

import json
import time

import torch

from pgb.core import GaussianBankRenderer
from pgb.oracle_integration import apply_oracle_blur


def main() -> None:
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(2027)
    image = torch.rand((1, 3, 960, 960), generator=generator)
    center = 0.05 + 0.9 * torch.rand((70, 2), generator=generator)
    size = 0.01 + 0.03 * torch.rand((70, 2), generator=generator)
    batch = {
        "img": image.clone(),
        "bboxes": torch.cat((center, size), dim=1),
        "batch_idx": torch.zeros((70, 1), dtype=torch.long),
    }
    renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16), mix=0.5)
    start = time.perf_counter()
    result = apply_oracle_blur(batch, renderer, mix=0.5)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(result["img"]).all():
        raise AssertionError("non-finite processed image")
    if not torch.isfinite(result["oracle_strength"]).all():
        raise AssertionError("non-finite strength map")
    if (result["img"] < 0).any() or (result["img"] > 1).any():
        raise AssertionError("processed image outside [0, 1]")
    print(json.dumps({
        "device": "CPU",
        "image_shape": list(image.shape),
        "boxes": 70,
        "seconds_one_image": elapsed,
        "strength_min": float(result["oracle_strength"].min()),
        "strength_max": float(result["oracle_strength"].max()),
        "mean_abs_image_change": float((result["img"] - image).abs().mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
