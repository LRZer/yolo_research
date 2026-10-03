"""B/global-blur variant of the frozen A/original CUB ResNet50 trainer.

Only the input representation and task-specific config validation differ from A.
The blur matches the already-audited B preview: sigma=8*long_side/500,
scipy gaussian_filter(float32, reflect, truncate=3), then rounded uint8.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter

import train_cub_resnet50_a as engine


def global_blur_uint8(native: np.ndarray) -> np.ndarray:
    height, width = native.shape[:2]
    sigma = 8.0 * max(width, height) / 500.0
    filtered = gaussian_filter(native.astype(np.float32),
                               sigma=(sigma, sigma, 0.0), mode="reflect", truncate=3.0)
    return np.clip(np.rint(filtered), 0, 255).astype(np.uint8)


class CUBGlobalBlur(engine.CUBOriginal):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        identifier, path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if width > 500 or height > 500:
            raise ValueError(f"Image {identifier} exceeds 500x500")
        if self.train:
            rng = random.Random(self.seed * 1_000_003 + self.epoch * 100_003 + identifier)
            if rng.random() < 0.5:
                image = TF.hflip(image)
            image = TF.adjust_brightness(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, rng.uniform(0.9, 1.1))
        native = np.asarray(image, dtype=np.uint8)
        blurred = global_blur_uint8(native)
        canvas = Image.new("RGB", (500, 500), self.fill)
        canvas.paste(Image.fromarray(blurred, mode="RGB"),
                     ((500 - width) // 2, (500 - height) // 2))
        tensor = TF.normalize(TF.to_tensor(canvas), mean=self.mean, std=self.std)
        return tensor, label


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    expected = {
        "group": "B_global_blur", "architecture": "torchvision_resnet50_scratch",
        "epochs": 200, "seed": 2005, "canvas": 500,
        "effective_batch": 64, "microbatch": 16, "accumulation": 4, "workers": 4,
        "device_physical": 0, "numa_node": 0,
        "blur_sigma_at_long_side_500": 8.0,
        "blur_boundary": "reflect", "blur_truncate": 3.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected {key}: {config.get(key)!r}, expected {value!r}")
    if config["microbatch"] * config["accumulation"] != config["effective_batch"]:
        raise ValueError("Invalid batch accumulation configuration")
    return config, hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    engine.CUBOriginal = CUBGlobalBlur
    engine.load_config = load_config
    engine.main()
