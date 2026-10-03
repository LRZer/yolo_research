"""Compute valid-image RGB statistics from the official CUB training split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def table(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        identifier, value = line.split(maxsplit=1)
        result[int(identifier)] = value
    return result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(r"E:\datasets\CUB-200-2011\extracted\CUB_200_2011"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "CUB_OFFICIAL_TRAIN_RGB_STATS_2026-09-25.json")
    args = parser.parse_args()
    root = args.data.resolve()
    images = table(root / "images.txt")
    splits = table(root / "train_test_split.txt")
    labels = table(root / "image_class_labels.txt")
    if set(images) != set(splits) or set(images) != set(labels) or len(images) != 11788:
        raise ValueError("CUB image, split, and label IDs do not match")
    counts = np.zeros(3, dtype=np.uint64)
    squared = np.zeros(3, dtype=np.uint64)
    pixel_count = 0
    train_count = 0
    dimensions = {}
    for identifier in sorted(images):
        if splits[identifier] != "1":
            continue
        train_count += 1
        with Image.open(root / "images" / images[identifier]) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if width > 500 or height > 500:
            raise ValueError(f"Image {identifier} exceeds the native 500 canvas")
        dimensions[f"{width}x{height}"] = dimensions.get(f"{width}x{height}", 0) + 1
        values = rgb.astype(np.uint64)
        counts += values.sum(axis=(0, 1), dtype=np.uint64)
        squared += (values * values).sum(axis=(0, 1), dtype=np.uint64)
        pixel_count += width * height
    if train_count != 5994 or sum(value == "0" for value in splits.values()) != 5794:
        raise ValueError("Official train/test counts do not match expected values")
    mean_255 = counts.astype(np.float64) / pixel_count
    std_255 = np.sqrt(squared.astype(np.float64) / pixel_count - mean_255 * mean_255)
    result = {
        "source_data": str(root),
        "official_train_images": train_count,
        "official_test_images": 5794,
        "official_train_valid_pixels": pixel_count,
        "mean_rgb_0_255": mean_255.tolist(),
        "std_rgb_0_255": std_255.tolist(),
        "mean_rgb_0_1": (mean_255 / 255.0).tolist(),
        "std_rgb_0_1": (std_255 / 255.0).tolist(),
        "metadata_sha256": {name: file_sha256(root / name) for name in ("images.txt", "train_test_split.txt", "image_class_labels.txt")},
        "unique_training_dimensions": len(dimensions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
