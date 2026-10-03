"""CPU data/config audit and exact preview-to-loader equivalence for F."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image

import train_cub_resnet50_a as engine
from train_cub_resnet50_f import CUBScharr, load_config, scharr_rgb


def restore(tensor, mean, std):
    array = tensor.permute(1, 2, 0).numpy()
    return np.rint((array * np.array(std) + np.array(mean)) * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    config, config_sha = load_config(args.config)
    stats_path = Path(config["stats_json"])
    stats = json.loads(stats_path.read_bytes())
    train, test = engine.load_metadata(Path(config["data_root"]), stats)
    mean, std = stats["mean_rgb_0_1"], stats["std_rgb_0_1"]
    fill = tuple(int(np.rint(x)) for x in stats["mean_rgb_0_255"])
    root = Path("/home/ubuntu/rz/foveated_classification_research")
    preview_root = root / "previews/cub_edge_texture_v1_2026-09-30"
    preview = json.loads((preview_root / "manifest.json").read_bytes())
    train_lookup = {record[0]: record for record in train}
    checked = []
    for item in preview["samples"]:
        record = train_lookup[item["image_id"]]
        if engine.sha256(record[1]) != item["source_sha256"]:
            raise ValueError("Training source differs from approved preview")
        dataset = CUBScharr([record], mean, std, fill, False, config["seed"])
        tensor, label = dataset[0]
        actual = restore(tensor, mean, std)
        reference_path = preview_root / f"{item['image_id']:05d}_{item['stage']}" / "01_scharr.png"
        if engine.sha256(reference_path) != item["output_sha256"]["01_scharr.png"]:
            raise ValueError("Preview PNG hash changed")
        reference = np.asarray(Image.open(reference_path).convert("RGB"))
        if not np.array_equal(actual, reference):
            raise ValueError(f"F loader differs from approved preview for {item['image_id']}")
        if tensor.shape != (3, 500, 500) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Wrong normalized input shape or non-finite values")
        if not np.array_equal(actual[..., 0], actual[..., 1]) or not np.array_equal(actual[..., 1], actual[..., 2]):
            raise ValueError("Scharr input is not three equal raw channels")
        augmented = CUBScharr([record], mean, std, fill, True, config["seed"], 16)
        if not torch.equal(augmented[0][0], augmented[0][0]):
            raise ValueError("Common per-image augmentation is not deterministic")
        checked.append({"image_id": item["image_id"], "preview_pixels_equal": True,
                        "deterministic_train_augmentation": True, "label": label})
    for record in test[:3]:
        tensor, _ = CUBScharr([record], mean, std, fill, False, config["seed"])[0]
        if tensor.shape != (3, 500, 500) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Test input failed shape/finite check")
    constant = Image.fromarray(np.full((64, 64, 3), 127, np.uint8))
    if np.asarray(scharr_rgb(constant)).any():
        raise ValueError("Uniform RGB input unexpectedly produces edges")
    baseline = json.loads((root / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json").read_bytes())
    payload = {
        "status": "F_cpu_verified_preview_exact", "group": "F_scharr", "train_images": len(train),
        "test_images": len(test), "classes": 200, "config_sha256": config_sha,
        "stats_sha256": engine.sha256(stats_path), "baseline_engine_sha256": engine.sha256(Path(engine.__file__)),
        "trainer_sha256": engine.sha256(root / "src/train_cub_resnet50_f.py"),
        "preview_manifest_sha256": engine.sha256(preview_root / "manifest.json"),
        "shared_config_differences": {k: [baseline[k], config[k]] for k in baseline if baseline[k] != config[k]},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__,
                     "opencv": cv2.__version__, "pillow": PIL.__version__},
        "samples": checked, "uniform_input_zero_response": True,
        "training_started": False, "gpu_smoke_completed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine.atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
