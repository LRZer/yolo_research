"""Measure official-split target size after 640/960 letterbox scaling."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "data_splits" / "visdrone_official" / "view"


def summarize(widths: list[float], heights: list[float]) -> dict:
    areas = [w * h for w, h in zip(widths, heights)]
    mins = [min(w, h) for w, h in zip(widths, heights)]
    n = len(areas)
    if not n:
        raise ValueError("no objects")
    return {
        "boxes": n,
        "median_width_px": statistics.median(widths),
        "median_height_px": statistics.median(heights),
        "fraction_area_lt_32sq": sum(a < 32 * 32 for a in areas) / n,
        "fraction_min_edge_lt_8": sum(m < 8 for m in mins) / n,
        "fraction_min_edge_lt_4": sum(m < 4 for m in mins) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "source_yolo_labels_read_only", "splits": {}}
    for split in ("train", "val", "test"):
        source = VIEW / "images" / split
        images = sorted(path for path in source.iterdir() if path.is_file())
        dimensions = {640: ([], []), 960: ([], [])}
        for image in images:
            with Image.open(image) as opened:
                width, height = opened.size
            label = VIEW / "labels" / split / f"{image.stem}.txt"
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"malformed label: {label}")
                class_id = int(float(parts[0]))
                if not 0 <= class_id < 10:
                    raise ValueError(f"invalid class: {label}")
                box_width, box_height = float(parts[3]) * width, float(parts[4]) * height
                for imgsz in dimensions:
                    ratio = min(imgsz / width, imgsz / height)
                    dimensions[imgsz][0].append(box_width * ratio)
                    dimensions[imgsz][1].append(box_height * ratio)
        result["splits"][split] = {
            "images": len(images),
            "at_640": summarize(*dimensions[640]),
            "at_960": summarize(*dimensions[960]),
        }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
