"""Measure held-out GT protection and distant-background coverage of a gate.

Ground truth is used only by this diagnostic, never to create model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.core import make_targets  # noqa: E402
from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


def read_boxes(image_path: Path, height: int, width: int, gain: float, left: int, top: int) -> torch.Tensor:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    output: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = [float(value) for value in line.split()]
        if len(fields) != 5:
            raise ValueError(f"invalid label line in {label_path}")
        _, cx, cy, bw, bh = fields
        output.append([
            (cx - bw / 2) * width * gain + left,
            (cy - bh / 2) * height * gain + top,
            (cx + bw / 2) * width * gain + left,
            (cy + bh / 2) * height * gain + top,
        ])
    return torch.tensor(output, dtype=torch.float32).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, help="write the audit JSON to an existing project directory")
    args = parser.parse_args()
    if args.limit <= 0 or args.imgsz <= 0:
        raise ValueError("limit and imgsz must be positive")
    model, _ = load_checkpoint(args.checkpoint, device="cpu", fuse=False)
    if not isinstance(model, SafeBackgroundDetectionModel):
        raise AssertionError(f"wrong model class: {type(model)}")
    model.eval()
    torch.set_num_threads(2)
    paths = [Path(line.strip()) for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {name: 0 for name in ("protected", "far", "all_valid")}
    sums = {name: 0.0 for name in counts}
    thresholds = (0.1, 0.25, 0.5)
    above = {name: {threshold: 0 for threshold in thresholds} for name in counts}
    protected_instance_total = 0
    protected_instance_safe = 0
    instance_by_size = {
        name: {"total": 0, "safe_at_0_1": 0} for name in ("small", "medium", "large")
    }
    global_max = 0.0
    used_images = 0
    for path in paths[: args.limit]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        gain = min(args.imgsz / height, args.imgsz / width)
        resized_width, resized_height = round(width * gain), round(height * gain)
        left = round((args.imgsz - resized_width) / 2 - 0.1)
        top = round((args.imgsz - resized_height) / 2 - 0.1)
        processed = LetterBox(new_shape=(args.imgsz, args.imgsz))(image=image)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        boxes = read_boxes(path, height, width, gain, left, top)
        valid_pixels = torch.zeros((1, 1, args.imgsz, args.imgsz), dtype=torch.bool)
        valid_pixels[0, 0, top : top + resized_height, left : left + resized_width] = True
        targets = make_targets(tensor, [boxes], valid_pixels=valid_pixels)
        with torch.inference_mode():
            permission = model.frontend.map_net(tensor)
        global_max = max(global_max, float(permission.max()))
        masks = {
            "protected": targets.protected & targets.valid,
            "far": (targets.strength >= model.sbg_far_threshold) & targets.valid,
            "all_valid": targets.valid,
        }
        for name, mask in masks.items():
            counts[name] += int(mask.sum())
            sums[name] += float(permission[mask].sum())
            for threshold in thresholds:
                above[name][threshold] += int((permission[mask] > threshold).sum())
        for x1, y1, x2, y2 in boxes.tolist():
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(args.imgsz, int(x2 + 1)), min(args.imgsz, int(y2 + 1))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = permission[0, 0, y1:y2, x1:x2]
            protected_instance_total += 1
            safe = int(float((patch <= 0.1).float().mean()) >= 0.95)
            protected_instance_safe += safe
            instance_by_size[size]["total"] += 1
            instance_by_size[size]["safe_at_0_1"] += safe
        used_images += 1
    report = {
        "checkpoint": str(args.checkpoint),
        "images": used_images,
        "instances": protected_instance_total,
        "instance_safe_fraction_at_0_1": protected_instance_safe / max(1, protected_instance_total),
        "instance_by_size": {
            name: {
                **counts_by_size,
                "safe_fraction": counts_by_size["safe_at_0_1"] / max(1, counts_by_size["total"]),
            }
            for name, counts_by_size in instance_by_size.items()
        },
        "max_permission": global_max,
        "pixel_region": {
            name: {
                "pixels": counts[name],
                "mean_permission": sums[name] / max(1, counts[name]),
                "fraction_permission_above": {
                    str(threshold): above[name][threshold] / max(1, counts[name])
                    for threshold in thresholds
                },
            }
            for name in counts
        },
    }
    result = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        if not args.output.parent.is_dir():
            raise FileNotFoundError(args.output.parent)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
