"""F Scharr input; reuse the frozen A optimizer, model, augmentation and loop."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import train_cub_resnet50_a as engine

ROOT = Path("/home/ubuntu/rz/foveated_classification_research")


def scharr_rgb(image: Image.Image) -> Image.Image:
    """Exactly the approved preview's luminance, derivative, gain and rounding."""
    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1 / 32,
                   borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=1 / 32,
                   borderType=cv2.BORDER_REFLECT_101)
    magnitude = np.hypot(gx, gy)
    if not np.isfinite(magnitude).all():
        raise FloatingPointError("Non-finite Scharr response")
    response = np.rint(np.clip(4.0 * magnitude, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(response[..., None], 3, axis=2))


class CUBScharr(engine.CUBOriginal):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # One OpenCV thread per DataLoader worker avoids nested thread pools.
        cv2.setNumThreads(1)
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
        # Derivatives precede padding: there are no artificial padding edges.
        image = scharr_rgb(image)
        canvas = Image.new("RGB", (500, 500), (0, 0, 0))
        canvas.paste(image, ((500 - width) // 2, (500 - height) // 2))
        return TF.normalize(TF.to_tensor(canvas), mean=self.mean, std=self.std), label


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    baseline = json.loads((ROOT / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json").read_bytes())
    for key, value in baseline.items():
        if key not in ("group", "run_dir", "device_physical") and config.get(key) != value:
            raise ValueError(f"F differs from A's shared configuration: {key}")
    expected = {
        "group": "F_scharr",
        "device_physical": 6,
        "run_dir": str(ROOT / "runs/resnet50_F_scharr_native500_s2005_e200"),
        "scharr_conversion": "PIL_L_float32_Scharr_magnitude_after_common_RGB_augmentation",
        "scharr_derivative_scale": 1 / 32,
        "scharr_fixed_gain": 4.0,
        "scharr_border": "BORDER_REFLECT_101",
        "edge_padding_rgb": [0, 0, 0],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected F parameter {key}: {config.get(key)!r}")
    if config["microbatch"] * config["accumulation"] != config["effective_batch"]:
        raise ValueError("Invalid batch accumulation configuration")
    return config, hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    engine.CUBOriginal = CUBScharr
    engine.load_config = load_config
    engine.main()
