"""GT-oracle fixed-grid occupancy audit; read-only and GPU-free.

This tests whether selecting just a few high-resolution grid cells could
possibly cover most annotated object centers. It is an optimistic upper
bound requiring ground-truth labels, not a deployable selector or AP result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def image_counts(label_path: Path, grid: int) -> list[int]:
    counts = [0] * (grid * grid)
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO row")
        x, y = float(fields[1]), float(fields[2])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{label_path}:{line_number}: center out of image")
        column = min(int(x * grid), grid - 1)
        row = min(int(y * grid), grid - 1)
        counts[row * grid + column] += 1
    return counts


def summarize(label_paths: list[Path], grid: int) -> dict:
    occupied_hist: Counter[int] = Counter()
    total_images = len(label_paths)
    total_targets = 0
    per_k_covered = [0] * (grid * grid + 1)
    per_k_image_fraction = [0.0] * (grid * grid + 1)
    images_with_gt = 0
    for label_path in label_paths:
        counts = sorted(image_counts(label_path, grid), reverse=True)
        total = sum(counts)
        occupied_hist[sum(count > 0 for count in counts)] += 1
        total_targets += total
        if total:
            images_with_gt += 1
        for k in range(1, len(counts) + 1):
            covered = sum(counts[:k])
            per_k_covered[k] += covered
            per_k_image_fraction[k] += covered / total if total else 1.0
    return {
        "grid": f"{grid}x{grid}",
        "images": total_images,
        "images_with_gt": images_with_gt,
        "targets": total_targets,
        "occupied_cells_per_image_hist": {str(k): occupied_hist[k] for k in range(grid * grid + 1)},
        "top_k_oracle_cells": {
            str(k): {
                "image_area_fraction_if_nonoverlap": k / (grid * grid),
                "micro_gt_center_coverage": per_k_covered[k] / total_targets if total_targets else None,
                "mean_image_gt_center_coverage": per_k_image_fraction[k] / total_images if total_images else None,
            }
            for k in range(1, grid * grid + 1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = sorted(args.labels.glob("*.txt"))
    if len(labels) != 548:
        raise ValueError(f"expected 548 official-source val label files, got {len(labels)}")
    summary = {
        "metric": "gt_oracle_fixed_grid_center_coverage_not_AP_not_deployable",
        "source": str(args.labels.resolve()),
        "caveat": "Uses GT to choose top-k cells, ignores box extents, tile overlap, boundary context and ignored annotations",
        "grids": {str(grid): summarize(labels, grid) for grid in (2, 3)},
    }
    for result in summary["grids"].values():
        if result["targets"] != 38759:
            raise ValueError(f"GT count changed: {result['targets']} != 38,759")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({grid: {"images": value["images"], "targets": value["targets"]}
                      for grid, value in summary["grids"].items()}, indent=2))


if __name__ == "__main__":
    main()
