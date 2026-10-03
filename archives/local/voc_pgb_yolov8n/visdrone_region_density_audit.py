"""Read-only upper-bound audit for GT center occupancy on a fixed grid.

GT ranks cells only for the explicitly labeled oracle upper bound. This is
not a deployable region selector and does not run a detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def audit(manifest: Path, grid: int, budgets: tuple[int, ...]) -> dict:
    paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("manifest must be nonempty with unique image paths")
    totals = {k: 0 for k in budgets}
    ratios = {k: [] for k in budgets}
    occupied = []
    center_total = 0
    all_boxes = 0
    empty = 0
    for image_path in paths:
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing image or label: {image_path} / {label_path}")
        cells = [0] * (grid * grid)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"bad YOLO label: {label_path}: {line}")
            class_id, x, y, width, height = map(float, fields)
            if not (0 <= class_id < 10 and class_id.is_integer()
                    and 0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0):
                raise ValueError(f"invalid YOLO label: {label_path}: {line}")
            column = min(int(x * grid), grid - 1)
            row = min(int(y * grid), grid - 1)
            cells[row * grid + column] += 1
        count = sum(cells)
        all_boxes += count
        if count == 0:
            empty += 1
            continue
        ranked = sorted(cells, reverse=True)
        occupied.append(sum(value > 0 for value in cells))
        for k in budgets:
            captured = sum(ranked[:k])
            totals[k] += captured
            ratios[k].append(captured / count)
        # Four central cells for the 4x4 case; central area is 25%.
        if grid == 4:
            center_total += sum(cells[row * grid + column] for row in (1, 2) for column in (1, 2))
    return {
        "manifest": str(manifest.resolve()),
        "images": len(paths),
        "empty_images": empty,
        "gt_boxes": all_boxes,
        "grid": f"{grid}x{grid}",
        "cell_area_fraction": 1 / (grid * grid),
        "occupied_cells_median": median(occupied),
        "occupied_cells_mean": sum(occupied) / len(occupied),
        "center_four_cells_gt_fraction": center_total / all_boxes if grid == 4 else None,
        "oracle_top_k": {
            str(k): {
                "max_gt_center_fraction_dataset": totals[k] / all_boxes,
                "median_per_image_fraction": median(ratios[k]),
                "selected_area_fraction": k / (grid * grid),
            }
            for k in budgets
        },
        "caveat": "GT-ranked cells are a label-leaking ceiling, not a deployable method or detector AP. Boxes crossing cell boundaries are assigned by center only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=4)
    args = parser.parse_args()
    if args.grid != 4:
        raise ValueError("this pre-registered pilot uses grid=4 only")
    print(json.dumps(audit(args.manifest, 4, (1, 2, 4, 8, 12, 16)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
