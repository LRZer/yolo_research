"""CPU-only geometry audit of the exact Oracle target on a YOLO val view.

Uses original image dimensions and YOLO's square LetterBox geometry; reads
but never changes images/labels. This is a target-strength audit, not an
augmentation, rendered-image quality metric, or detector evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from pgb.core import make_targets


def letterbox_boxes_and_valid(
    labels: list[tuple[float, float, float, float]],
    original_width: int,
    original_height: int,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gain = min(size / original_height, size / original_width)
    new_width = round(original_width * gain)
    new_height = round(original_height * gain)
    left = round((size - new_width) / 2 - 0.1)
    top = round((size - new_height) / 2 - 0.1)
    right = left + new_width
    bottom = top + new_height
    valid = torch.zeros((1, 1, size, size), dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True

    boxes = []
    for xc, yc, width, height in labels:
        x1 = (xc - width / 2) * original_width * (new_width / original_width) + left
        x2 = (xc + width / 2) * original_width * (new_width / original_width) + left
        y1 = (yc - height / 2) * original_height * (new_height / original_height) + top
        y2 = (yc + height / 2) * original_height * (new_height / original_height) + top
        x1 = min(max(x1, 0), size)
        x2 = min(max(x2, 0), size)
        y1 = min(max(y1, 0), size)
        y2 = min(max(y2, 0), size)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), valid
    return torch.empty((0, 4), dtype=torch.float32), valid


def read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO columns")
        values = tuple(float(x) for x in parts[1:])
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"{path}:{line_number}: non-finite box")
        rows.append(values)
    return rows


@torch.no_grad()
def audit(view_root: Path, size: int) -> dict:
    image_dir = view_root / "images" / "val"
    label_dir = view_root / "labels" / "val"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"no validation images in {image_dir}")
    dummy_image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    histogram = torch.zeros(20, dtype=torch.long)
    total_valid = 0
    total_protected = 0
    total_boxes = 0
    thresholds = (0.1, 0.5, 0.9, 0.99)
    threshold_counts = {str(x): 0 for x in thresholds}
    means = []
    dimension_counts = Counter()
    per_image = []
    empty_images = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label: {label_path}")
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        dimension_counts[f"{original_width}x{original_height}"] += 1
        boxes, valid = letterbox_boxes_and_valid(
            read_labels(label_path), original_width, original_height, size
        )
        if len(boxes) == 0:
            empty_images += 1
            continue
        total_boxes += len(boxes)
        target = make_targets(
            dummy_image,
            [boxes],
            margin=0.1,
            transition=0.5,
            min_radius=8.0,
            valid_pixels=valid,
        )
        values = target.strength[target.valid]
        pixels = int(values.numel())
        if pixels == 0:
            raise AssertionError("nonempty image has no valid pixels")
        total_valid += pixels
        total_protected += int((target.protected & target.valid).sum())
        histogram += torch.histc(values, bins=20, min=0, max=1).long()
        for threshold in thresholds:
            threshold_counts[str(threshold)] += int((values >= threshold).sum())
        mean = float(values.mean())
        means.append(mean)
        per_image.append({
            "stem": image_path.stem,
            "boxes": len(boxes),
            "original_width": original_width,
            "original_height": original_height,
            "mean_strength": mean,
            "fraction_ge_0_5": float((values >= 0.5).float().mean()),
            "protected_fraction": float((target.protected & target.valid).sum()) / pixels,
        })
    if total_valid == 0:
        raise AssertionError("no usable labeled pixels")
    means.sort()
    def quantile(q: float) -> float:
        return means[round(q * (len(means) - 1))]
    return {
        "status": "cpu_only_exact_target_audit",
        "source_view": str(view_root),
        "split": "val",
        "image_count": len(images),
        "nonempty_images": len(means),
        "empty_images": empty_images,
        "usable_boxes_after_clipping": total_boxes,
        "imgsz": size,
        "target_margin": 0.1,
        "target_transition": 0.5,
        "target_min_radius": 8.0,
        "planned_sigma_levels": [0, 2, 4, 8, 16],
        "planned_final_mix": 0.5,
        "planned_train_probability": 0.5,
        "valid_pixels": total_valid,
        "protected_pixel_fraction": total_protected / total_valid,
        "fraction_strength_ge": {k: v / total_valid for k, v in threshold_counts.items()},
        "image_mean_strength": {
            "mean": sum(means) / len(means),
            "median": quantile(0.5),
            "p10": quantile(0.1),
            "p90": quantile(0.9),
        },
        "strength_histogram_20_bins": histogram.tolist(),
        "image_dimensions": dict(dimension_counts.most_common()),
        "caveat": "Val-box geometry after square LetterBox only. Training mosaic/perspective/scale transforms change geometry; no images were blurred and no detector AP or actual convolution time was measured.",
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.view_root, args.imgsz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
