"""Read-only, GT-only audit of geometric clipping by the pilot's 4x4 crops.

This does not select crops, run a detector, or infer AP. It measures the
idealized normalized geometry of crop_rect with a 10% halo per cell side.
Actual pixel rounding can differ by approximately one source-image pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
)


def crop_bounds(col: int, row: int) -> tuple[float, float, float, float]:
    return (
        max(0.0, col / 4 - 0.025), max(0.0, row / 4 - 0.025),
        min(1.0, (col + 1) / 4 + 0.025),
        min(1.0, (row + 1) / 4 + 0.025),
    )


def contains(crop: tuple[float, ...], box: tuple[float, ...]) -> bool:
    return all((crop[0] <= box[0], crop[1] <= box[1],
                box[2] <= crop[2], box[3] <= crop[3]))


def retained_fraction(crop: tuple[float, ...], box: tuple[float, ...]) -> float:
    width = max(0.0, min(crop[2], box[2]) - max(crop[0], box[0]))
    height = max(0.0, min(crop[3], box[3]) - max(crop[1], box[1]))
    return width * height / ((box[2] - box[0]) * (box[3] - box[1]))


def analyze(manifest: Path) -> dict:
    raw = manifest.read_bytes()
    paths = [Path(line.strip()) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("manifest is empty or contains duplicates")
    crops = [crop_bounds(col, row) for row in range(4) for col in range(4)]
    rows = {name: {"count": 0, "center_cut": 0, "no_full_crop": 0,
                   "source_box_oob": 0, "widths": [], "heights": [],
                   "center_retained": []} for name in NAMES}
    for image in paths:
        label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid label line: {label}: {line}")
            cls, x, y, w, h = map(float, fields)
            if not (cls.is_integer() and 0 <= cls < len(NAMES) and
                    0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                raise ValueError(f"invalid label value: {label}: {line}")
            raw_box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            box = (max(0.0, raw_box[0]), max(0.0, raw_box[1]),
                   min(1.0, raw_box[2]), min(1.0, raw_box[3]))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"zero visible box: {label}: {line}")
            center_cell = min(3, int(y * 4)) * 4 + min(3, int(x * 4))
            item = rows[NAMES[int(cls)]]
            item["count"] += 1
            item["source_box_oob"] += int(raw_box != box)
            item["center_cut"] += int(not contains(crops[center_cell], box))
            item["no_full_crop"] += int(not any(contains(crop, box) for crop in crops))
            item["widths"].append(box[2] - box[0])
            item["heights"].append(box[3] - box[1])
            item["center_retained"].append(retained_fraction(crops[center_cell], box))
    output = {}
    for name, item in rows.items():
        n = item["count"]
        output[name] = {
            "boxes": n,
            "center_cell_crop_cuts_fraction": item["center_cut"] / n if n else None,
            "none_of_16_crops_fully_contains_fraction": item["no_full_crop"] / n if n else None,
            "center_cell_crop_mean_visible_area_retained": sum(item["center_retained"]) / n if n else None,
            "cut_boxes_mean_visible_area_retained": (
                sum(value for value in item["center_retained"] if value < 1 - 1e-12)
                / item["center_cut"] if item["center_cut"] else None
            ),
            "center_cell_retains_below_80pct_fraction": (
                sum(value < 0.8 for value in item["center_retained"]) / n if n else None
            ),
            "center_cell_retains_below_50pct_fraction": (
                sum(value < 0.5 for value in item["center_retained"]) / n if n else None
            ),
            "visible_width_median_fraction_of_image": median(item["widths"]) if n else None,
            "visible_height_median_fraction_of_image": median(item["heights"]) if n else None,
            "boxes_extending_outside_image_in_source_label": item["source_box_oob"],
        }
    return {
        "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "images": len(paths), "total_boxes": sum(item["count"] for item in rows.values()),
        "grid": "4x4", "halo_each_side_fraction_of_cell": 0.1,
        "per_class": output,
        "caveat": "GT-only geometry. Assumes normalized 0.025 image-width/height halo, ignoring <=1 pixel rounding; no crop selection or detector inference. Center cut does not imply all selected overlapping crops cut. No-full-crop means even selecting all 16 available crops cannot fully contain the visible GT box.",
    }


def self_test() -> None:
    crop = crop_bounds(0, 0)
    assert crop == (0.0, 0.0, 0.275, 0.275)
    assert contains(crop, (0.01, 0.01, 0.2, 0.2))
    assert not contains(crop, (0.01, 0.01, 0.4, 0.4))
    assert abs(retained_fraction(crop, (0.0, 0.0, 0.4, 0.4)) - (0.275 / 0.4) ** 2) < 1e-12
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.manifest:
        parser.error("--manifest is required")
    print(json.dumps(analyze(args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
