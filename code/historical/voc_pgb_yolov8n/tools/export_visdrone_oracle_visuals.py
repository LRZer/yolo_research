"""CPU-only qualitative audit of the locked 960px train-only Oracle rule.

Uses official-val images only for visualization, never for training or
parameter selection. Outputs task-owned panels and a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.core import GaussianBankRenderer, make_targets
from tools.export_oracle_visualizations import label_path, letterbox, read_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"image directory not found: {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    records = [(p, read_labels(label_path(p))) for p in paths]
    records.sort(key=lambda item: (len(item[1]), item[0].name))
    positions = [int((len(records) - 1) * q) for q in (0.1, 0.5, 0.9)]
    selected = [records[i] for i in positions]
    renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16), mix=0.5).eval()
    results = []
    torch.set_num_threads(4)

    for index, (path, labels) in enumerate(selected, 1):
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"cannot decode {path}")
        original, boxes, valid, _, _ = letterbox(source, labels, size=960)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        target = make_targets(
            tensor,
            [torch.from_numpy(boxes)],
            margin=0.10,
            transition=0.50,
            min_radius=8.0,
            valid_pixels=valid,
        )
        with torch.no_grad():
            processed_tensor = renderer(tensor, target.strength, mix=0.5)
        processed_rgb = (processed_tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        processed = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        strength = target.strength[0, 0].numpy()
        heat = cv2.applyColorMap((strength * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
        diff = cv2.absdiff(original, processed)
        diff_heat = cv2.applyColorMap(
            cv2.normalize(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX),
            cv2.COLORMAP_INFERNO,
        )
        orig_boxes = original.copy()
        proc_boxes = processed.copy()
        for x1, y1, x2, y2 in boxes.round().astype(int):
            cv2.rectangle(orig_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(proc_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        panel = np.vstack((np.hstack((orig_boxes, heat)), np.hstack((proc_boxes, diff_heat))))
        filename = f"{index:02d}_{len(labels):03d}boxes_{path.stem}.jpg"
        if not cv2.imwrite(str(args.output / filename), panel, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {filename}")
        valid_array = valid[0, 0].numpy()
        results.append({
            "file": filename,
            "source": str(path),
            "boxes": len(labels),
            "mean_strength_valid": float(strength[valid_array].mean()),
            "fully_protected_fraction_valid": float(((strength == 0) & valid_array).sum() / valid_array.sum()),
            "mean_absolute_pixel_change_valid_0to255": float(diff[valid_array].mean()),
        })
    print(json.dumps({"sampling": "10th/50th/90th percentiles of official-val YOLO-label box count", "panels": results}, indent=2))


if __name__ == "__main__":
    main()
