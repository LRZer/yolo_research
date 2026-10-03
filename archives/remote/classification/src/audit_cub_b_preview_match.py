"""CPU-only pixel match of the B loader's blur against saved approved previews."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from train_cub_resnet50_a import parse_table
from train_cub_resnet50_b import CUBGlobalBlur, global_blur_uint8


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
DATA = ROOT / "data/CUB-200-2011/extracted/CUB_200_2011"
REFERENCES = ROOT / "_management/b_preview_reference"
STATS = ROOT / "configs/CUB_OFFICIAL_TRAIN_RGB_STATS_2026-09-25.json"


def main() -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    fill = tuple(int(np.rint(value)) for value in stats["mean_rgb_0_255"])
    images = parse_table(DATA / "images.txt")
    splits = parse_table(DATA / "train_test_split.txt")
    results = []
    for identifier in (10140, 1767):
        if splits[identifier] != "1":
            raise AssertionError("Approved preview is not from official train")
        with Image.open(DATA / "images" / images[identifier]) as source:
            native = np.asarray(source.convert("RGB"), dtype=np.uint8)
        blurred = global_blur_uint8(native)
        height, width = native.shape[:2]
        canvas = Image.new("RGB", (500, 500), fill)
        canvas.paste(Image.fromarray(blurred, mode="RGB"),
                     ((500 - width) // 2, (500 - height) // 2))
        with Image.open(REFERENCES / f"{identifier:05d}.png") as reference:
            expected = np.asarray(reference.convert("RGB"), dtype=np.uint8)
        actual = np.asarray(canvas, dtype=np.uint8)
        differing_values = int(np.count_nonzero(actual != expected))
        label = int(parse_table(DATA / "image_class_labels.txt")[identifier]) - 1
        dataset = CUBGlobalBlur([(identifier, DATA / "images" / images[identifier], label)],
                                stats["mean_rgb_0_1"], stats["std_rgb_0_1"], fill,
                                False, 2005)
        tensor, target = dataset[0]
        restored = ((tensor.permute(1, 2, 0).numpy() *
                     np.asarray(stats["std_rgb_0_1"], dtype=np.float32)) +
                    np.asarray(stats["mean_rgb_0_1"], dtype=np.float32)) * 255.0
        loader_different_values = int(np.count_nonzero(np.clip(np.rint(restored), 0, 255).astype(np.uint8)
                                                        != expected))
        results.append({"image_id": identifier, "size_wh": [width, height],
                        "different_channel_values": differing_values,
                        "loader_different_channel_values": loader_different_values,
                        "loader_shape": list(tensor.shape), "class_zero_based": target})
        if differing_values or loader_different_values or tuple(tensor.shape) != (3, 500, 500):
            raise AssertionError(f"B blur differs from approved preview for {identifier}")
    print(json.dumps({"status": "exact_match", "samples": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
