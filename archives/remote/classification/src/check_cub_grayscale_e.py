"""CPU-only E data and frozen-control check; creates no run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

import train_cub_resnet50_a as engine
from train_cub_resnet50_e import CUBGrayscale, load_config


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")


def main() -> None:
    e_path = ROOT / "configs/cub_resnet50_E_grayscale_native500_s2005_e200_gpu0.json"
    a_path = ROOT / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json"
    config, config_sha = load_config(e_path)
    a_config = json.loads(a_path.read_text(encoding="utf-8"))
    common = set(a_config) & set(config) - {"group", "run_dir"}
    if any(a_config[key] != config[key] for key in common):
        raise AssertionError("E differs from frozen A in a shared training field")
    if set(config) - set(a_config) != {"grayscale_conversion"}:
        raise AssertionError("E has unexpected extra configuration fields")
    stats = json.loads(Path(config["stats_json"]).read_text(encoding="utf-8"))
    train, test = engine.load_metadata(Path(config["data_root"]), stats)
    fill = tuple(round(x) for x in stats["mean_rgb_0_255"])
    max_channel_gap = 0.0
    checks = 0
    for rows, training in ((train, True), (test, False)):
        for record in (rows[0], rows[len(rows) // 2], rows[-1]):
            dataset = CUBGrayscale([record], stats["mean_rgb_0_1"],
                                   stats["std_rgb_0_1"], fill,
                                   training, config["seed"], epoch=1)
            tensor, label = dataset[0]
            if tuple(tensor.shape) != (3, 500, 500) or label != record[2]:
                raise AssertionError("Wrong E tensor shape or target")
            restored = tensor * torch.tensor(stats["std_rgb_0_1"])[:, None, None]
            restored += torch.tensor(stats["mean_rgb_0_1"])[:, None, None]
            gap = max((restored[0] - restored[1]).abs().max().item(),
                      (restored[0] - restored[2]).abs().max().item())
            max_channel_gap = max(max_channel_gap, gap)
            if gap > 2e-6:
                raise AssertionError(f"E input contains color: gap={gap}")
            if not training:
                with Image.open(record[1]) as source:
                    width, height = source.size
                center = restored[:, (500 - height) // 2:(500 + height) // 2,
                                  (500 - width) // 2:(500 + width) // 2]
                if center.numel() == 0:
                    raise AssertionError("Empty valid image region")
            if training:
                repeated, _ = dataset[0]
                if not torch.equal(tensor, repeated):
                    raise AssertionError("E train augmentation is not deterministic")
            checks += 1
    print(json.dumps({"status": "checked_grayscale_cpu", "config_sha256": config_sha,
                      "train_images": len(train), "test_images": len(test),
                      "images_checked": checks, "max_unnormalized_channel_gap": max_channel_gap}))


if __name__ == "__main__":
    main()
