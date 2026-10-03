"""Exact GT-box union area on VisDrone val; read-only and GPU-free.

This measures potential protected versus transformable image area in the
training-only Oracle. It does not predict AP, foreground pixels, or a
deployable blur map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def parse_boxes(path: Path, margin: float) -> tuple[list[tuple[float, float, float, float]], int]:
    boxes = []
    clipped = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        x, y, w, h = map(float, fields[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized box")
        unbounded = (
            x - (0.5 + margin) * w,
            y - (0.5 + margin) * h,
            x + (0.5 + margin) * w,
            y + (0.5 + margin) * h,
        )
        bounded = tuple(min(1.0, max(0.0, z)) for z in unbounded)
        if bounded != unbounded:
            clipped += 1
        if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
            raise ValueError(f"{path}:{line_number}: empty box after clipping")
        boxes.append(bounded)
    return boxes, clipped


def union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Exact union of axis-aligned rectangles in a normalized unit square."""
    if not boxes:
        return 0.0
    edges = sorted({x for x0, _, x1, _ in boxes for x in (x0, x1)})
    area = 0.0
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            continue
        intervals = sorted((y0, y1) for x0, y0, x1, y1 in boxes if x0 < right and x1 > left)
        if not intervals:
            continue
        low, high = intervals[0]
        height = 0.0
        for next_low, next_high in intervals[1:]:
            if next_low > high:
                height += high - low
                low, high = next_low, next_high
            else:
                high = max(high, next_high)
        height += high - low
        area += (right - left) * height
    if not -1e-12 <= area <= 1 + 1e-12:
        raise AssertionError(f"bad union area: {area}")
    return min(1.0, max(0.0, area))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.labels.glob("*.txt"))
    if len(paths) != 548:
        raise ValueError(f"expected 548 official-source val labels, got {len(paths)}")
    rows = []
    for path in paths:
        original, original_clipped = parse_boxes(path, 0.0)
        protected, expanded_clipped = parse_boxes(path, 0.10)
        if len(original) != len(protected):
            raise AssertionError(path)
        raw_fraction = union_area(original)
        protected_fraction = union_area(protected)
        if protected_fraction + 1e-10 < raw_fraction:
            raise AssertionError(f"expanded union smaller for {path}")
        rows.append({
            "stem": path.stem,
            "boxes": len(original),
            "gt_box_union_fraction": raw_fraction,
            "margin10_protected_union_fraction": protected_fraction,
            "outside_margin10_fraction": 1 - protected_fraction,
            "original_clipped_boxes": original_clipped,
            "expanded_clipped_boxes": expanded_clipped,
        })
    total_boxes = sum(row["boxes"] for row in rows)
    if total_boxes != 38759:
        raise ValueError(f"GT count changed: {total_boxes} != 38759")
    summaries = {}
    for key in ("gt_box_union_fraction", "margin10_protected_union_fraction", "outside_margin10_fraction"):
        values = [row[key] for row in rows]
        summaries[key] = {
            "mean": mean(values),
            "median": median(values),
            "p10": quantile(values, 0.10),
            "p90": quantile(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    result = {
        "metric": "exact_gt_box_union_area_not_foreground_or_AP",
        "source_labels": str(args.labels.resolve()),
        "images": len(rows),
        "gt_boxes": total_boxes,
        "caveat": "Converted labels omit ignored regions; box union includes box-internal background and ignores distance transition outside the 10% margin; it is not a deployable mask or AP estimate.",
        "summaries": summaries,
        "images_with_margin10_protected_fraction_below_0_1": sum(row["margin10_protected_union_fraction"] < 0.1 for row in rows),
        "images_with_margin10_protected_fraction_below_0_25": sum(row["margin10_protected_union_fraction"] < 0.25 for row in rows),
        "clipped_original_boxes": sum(row["original_clipped_boxes"] for row in rows),
        "clipped_expanded_boxes": sum(row["expanded_clipped_boxes"] for row in rows),
        "per_image": rows,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
