"""E/grayscale variant of the frozen CUB ResNet50 scratch experiment.

After the same native-RGB training augmentations as A, convert the valid image
to PIL luminance (RGB -> L -> RGB). Test images undergo the same conversion.
Only then center-pad to 500x500 with a grayscale version of the common fill.
The model still receives three channels and uses the same A train-RGB stats.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import train_cub_resnet50_a as engine


class CUBGrayscale(engine.CUBOriginal):
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
        image = image.convert("L").convert("RGB")
        fill_l = Image.new("RGB", (1, 1), self.fill).convert("L").getpixel((0, 0))
        canvas = Image.new("RGB", (500, 500), (fill_l, fill_l, fill_l))
        canvas.paste(image, ((500 - width) // 2, (500 - height) // 2))
        tensor = TF.normalize(TF.to_tensor(canvas), mean=self.mean, std=self.std)
        return tensor, label


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    expected = {
        "group": "E_grayscale", "architecture": "torchvision_resnet50_scratch",
        "epochs": 200, "seed": 2005, "canvas": 500,
        "effective_batch": 64, "microbatch": 16, "accumulation": 4, "workers": 4,
        "device_physical": 0, "numa_node": 0,
        "grayscale_conversion": "PIL_RGB_to_L_to_RGB_after_common_augmentation",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected {key}: {config.get(key)!r}, expected {value!r}")
    if config["microbatch"] * config["accumulation"] != config["effective_batch"]:
        raise ValueError("Invalid batch accumulation configuration")
    return config, hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    engine.CUBOriginal = CUBGrayscale
    engine.load_config = load_config
    engine.main()
