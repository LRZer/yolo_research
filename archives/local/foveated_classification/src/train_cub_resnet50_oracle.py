"""C/D GT-box Oracle representations for the frozen CUB ResNet50 trainer.

This intentionally reuses A's model, optimizer, augmentation RNG, train/test
loop, and original 200-epoch learning-rate schedule. Only the input image is
changed. Both training and official test use the dataset's true bounding box;
these variants therefore cannot be deployed without a box predictor.
"""

from __future__ import annotations

import argparse
import functools
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


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
A_CONFIG = ROOT / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json"
BOX_SHA256 = "c19d9f096c15564793408e5ba054b8d715a6805819d98f037c977f369ef4abd3"


@functools.lru_cache(maxsize=1)
def read_boxes(data_root: Path) -> dict[int, tuple[float, float, float, float]]:
    table = engine.parse_table(data_root / "bounding_boxes.txt")
    if len(table) != 11788:
        raise ValueError("Expected one CUB box for each of 11,788 images")
    boxes = {identifier: tuple(float(v) for v in value.split()) for identifier, value in table.items()}
    if any(len(box) != 4 or not all(np.isfinite(box)) for box in boxes.values()):
        raise ValueError("Invalid source bounding box")
    return boxes


def clipped_box(raw: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = raw
    result = (max(0.0, min(float(width), x)),
              max(0.0, min(float(height), y)),
              max(0.0, min(float(width), x + box_width)),
              max(0.0, min(float(height), y + box_height)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"Source box has empty clipped area: {raw}, image={width}x{height}")
    return result


def protection(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = box
    expand_x, expand_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    px1, py1 = max(0.0, x1 - expand_x), max(0.0, y1 - expand_y)
    px2, py2 = min(float(width), x2 + expand_x), min(float(height), y2 + expand_y)
    xx = np.arange(width, dtype=np.float32)[None, :] + 0.5
    yy = np.arange(height, dtype=np.float32)[:, None] + 0.5
    dist_x = np.maximum(np.maximum(px1 - xx, xx - px2), 0.0)
    dist_y = np.maximum(np.maximum(py1 - yy, yy - py2), 0.0)
    distance = np.sqrt(dist_x * dist_x + dist_y * dist_y)
    scale = max(width, height) / 500.0
    q = np.clip(distance / (70.0 * scale), 0.0, 1.0)
    smooth = q * q * (3.0 - 2.0 * q)
    return smooth, scale


def process_native(rgb: np.ndarray, box: tuple[float, float, float, float],
                   group: str, mean_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    smooth, scale = protection(width, height, box)
    source = rgb.astype(np.float32)
    if group == "C_context_removed":
        processed = source * (1.0 - smooth[:, :, None]) + mean_rgb[None, None, :] * smooth[:, :, None]
    elif group == "D_gradual_background":
        levels = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32) * scale
        filtered = [source] + [gaussian_filter(source, sigma=(float(level), float(level), 0.0),
                                               mode="reflect", truncate=3.0) for level in levels[1:]]
        sigma = np.clip(8.0 * scale * smooth, 0.0, float(levels[-1]))
        processed = np.empty_like(source)
        covered = np.zeros(sigma.shape, dtype=bool)
        for index in range(3):
            lo, hi = float(levels[index]), float(levels[index + 1])
            selected = (sigma >= lo) & (sigma <= hi) if index == 2 else (sigma >= lo) & (sigma < hi)
            covered |= selected
            blend = ((sigma[selected] - lo) / (hi - lo))[:, None]
            processed[selected] = filtered[index][selected] * (1.0 - blend) + filtered[index + 1][selected] * blend
        if not bool(np.all(covered)):
            raise AssertionError("D blur levels failed to cover every native pixel")
    else:
        raise ValueError(f"Unknown Oracle group: {group}")
    result = np.clip(np.rint(processed), 0, 255).astype(np.uint8)
    result[smooth == 0] = rgb[smooth == 0]
    if not np.array_equal(result[smooth == 0], rgb[smooth == 0]):
        raise AssertionError("Protected target pixels changed")
    return result, smooth


class CUBOracle(engine.CUBOriginal):
    group = ""

    def __init__(self, records, mean, std, fill, train, seed, epoch=0):
        super().__init__(records, mean, std, fill, train, seed, epoch)
        self.boxes = read_boxes(records[0][1].parents[2])
        self.mean_rgb = np.array(mean, dtype=np.float64) * 255.0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        identifier, path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if width > 500 or height > 500:
            raise ValueError(f"Image {identifier} exceeds 500x500")
        box = clipped_box(self.boxes[identifier], width, height)
        if self.train:
            rng = random.Random(self.seed * 1_000_003 + self.epoch * 100_003 + identifier)
            if rng.random() < 0.5:
                image = TF.hflip(image)
                box = (width - box[2], box[1], width - box[0], box[3])
            image = TF.adjust_brightness(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, rng.uniform(0.9, 1.1))
        rgb = np.asarray(image, dtype=np.uint8)
        processed, _ = process_native(rgb, box, self.group, self.mean_rgb)
        canvas = Image.new("RGB", (500, 500), self.fill)
        canvas.paste(Image.fromarray(processed, mode="RGB"), ((500 - width) // 2, (500 - height) // 2))
        return TF.normalize(TF.to_tensor(canvas), mean=self.mean, std=self.std), label


class CUBContextRemoved(CUBOracle):
    group = "C_context_removed"


class CUBGradualBackground(CUBOracle):
    group = "D_gradual_background"


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("group") not in ("C_context_removed", "D_gradual_background"):
        raise ValueError("This trainer only accepts C/D groups")
    a = json.loads(A_CONFIG.read_text(encoding="utf-8"))
    for key, value in a.items():
        if key not in ("group", "run_dir") and config.get(key) != value:
            raise ValueError(f"C/D common config differs from A: {key}")
    expected = {
        "bounding_boxes_sha256": BOX_SHA256,
        "box_protection_expand_fraction_each_side": 0.1,
        "transition_at_long_side_500": 70.0,
        "replacement_or_blur_boundary": "reflect",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid Oracle transform config: {key}")
    if config["group"] == "C_context_removed":
        if config.get("far_background") != "training_valid_rgb_mean":
            raise ValueError("Invalid C far-background fill")
    else:
        if config.get("blur_levels_at_long_side_500") != [0.0, 2.0, 4.0, 8.0] or config.get("blur_truncate") != 3.0:
            raise ValueError("Invalid D blur levels")
    box_path = Path(config["data_root"]) / "bounding_boxes.txt"
    if engine.sha256(box_path) != BOX_SHA256:
        raise ValueError("Official bounding-box file hash mismatch")
    return config, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()
    group = json.loads(args.config.read_text(encoding="utf-8"))["group"]
    engine.CUBOriginal = CUBContextRemoved if group == "C_context_removed" else CUBGradualBackground
    engine.load_config = load_config
    engine.main()


if __name__ == "__main__":
    main()
