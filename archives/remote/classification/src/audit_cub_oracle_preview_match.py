"""CPU-only audit that C/D official loaders match the eight frozen preview PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import train_cub_resnet50_a as engine
import train_cub_resnet50_oracle as oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    root = oracle.ROOT
    stats = json.loads((root / "configs/CUB_OFFICIAL_TRAIN_RGB_STATS_2026-09-25.json").read_text(encoding="utf-8"))
    mean, std = stats["mean_rgb_0_1"], stats["std_rgb_0_1"]
    fill = tuple(int(np.rint(value)) for value in stats["mean_rgb_0_255"])
    mean_rgb = np.array(mean, dtype=np.float64) * 255.0
    source_root = root / "data/CUB-200-2011/extracted/CUB_200_2011"
    manifest = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["samples"]) != 8:
        raise ValueError("Expected eight preview cases")
    boxes = oracle.read_boxes(source_root)
    train_rows, test_rows = engine.load_metadata(source_root, stats)
    all_rows = train_rows + test_rows
    if len(all_rows) != 11788 or set(boxes) != {row[0] for row in all_rows}:
        raise ValueError("Official image and box IDs differ")
    clipped_outside = 0
    for identifier, image_path, _ in all_rows:
        with Image.open(image_path) as image:
            width, height = image.size
        x, y, box_width, box_height = boxes[identifier]
        clipped_outside += int(x < 0 or y < 0 or x + box_width > width or y + box_height > height)
        oracle.clipped_box(boxes[identifier], width, height)
    verified = []
    for sample in manifest["samples"]:
        identifier = sample["image_id"]
        if not sample["official_train"] or identifier not in boxes:
            raise ValueError("Preview sample is not a known training image")
        source_path = source_root / "images" / sample["relative_image_path"]
        if engine.sha256(source_path) != sample["source_sha256"]:
            raise ValueError(f"Source image changed: {identifier}")
        with Image.open(source_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        box = oracle.clipped_box(boxes[identifier], width, height)
        row = (identifier, source_path, sample["class_id"] - 1)
        for group, dataset_class in (("C_context_removed", oracle.CUBContextRemoved),
                                     ("D_gradual_background", oracle.CUBGradualBackground)):
            processed, smooth = oracle.process_native(rgb, box, group, mean_rgb)
            canvas = Image.new("RGB", (500, 500), fill)
            canvas.paste(Image.fromarray(processed, mode="RGB"), ((500 - width) // 2, (500 - height) // 2))
            expected_path = args.reference / f"{identifier:05d}_{sample['stage']}" / f"{group}.png"
            with Image.open(expected_path) as image:
                expected = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(np.asarray(canvas), expected):
                raise AssertionError(f"Transformation differs from frozen {group} preview: {identifier}")
            dataset = dataset_class([row], mean, std, fill, False, 2005)
            tensor, label = dataset[0]
            recovered = np.clip(np.rint((tensor.numpy().transpose(1, 2, 0) * np.array(std)
                                          + np.array(mean)) * 255), 0, 255).astype(np.uint8)
            if not np.array_equal(recovered, expected) or label != sample["class_id"] - 1:
                raise AssertionError(f"Formal loader differs from frozen {group} preview: {identifier}")
            if not np.array_equal(processed[smooth == 0], rgb[smooth == 0]):
                raise AssertionError(f"Protected native pixels changed: {identifier}")
            verified.append(f"{identifier}:{group}")
    print(json.dumps({"status": "preview_match", "all_clipped_boxes_valid": len(all_rows),
                      "raw_boxes_outside_image": clipped_outside,
                      "samples": 8, "comparisons": len(verified),
                      "cases": verified}, indent=2), flush=True)


if __name__ == "__main__":
    main()
