"""Read-only count of VisDrone categories omitted by the YOLO conversion.

This reports annotation frequency and summed clipped box area, not official AP
or exact union-area coverage. No dataset file is modified.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="official VisDrone split root")
    args = parser.parse_args()
    image_dir = args.source / "images"
    annotation_dir = args.source / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise SystemExit("expected source/images and source/annotations")

    categories = Counter()
    images_with = Counter()
    area_sum = Counter()
    image_count = 0
    total_pixels = 0
    missing_annotation = []
    malformed_rows = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image_count += 1
        with Image.open(image_path) as image:
            width, height = image.size
        total_pixels += width * height
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            missing_annotation.append(image_path.name)
            continue
        present = set()
        for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                x, y, box_width, box_height, score, category, _, _ = map(float, line.split(",")[:8])
                category = int(category)
            except (ValueError, TypeError):
                malformed_rows.append(f"{annotation_path.name}:{line_number}")
                continue
            categories[category] += 1
            present.add(category)
            x1 = max(0.0, min(float(width), x))
            x2 = max(0.0, min(float(width), x + box_width))
            y1 = max(0.0, min(float(height), y))
            y2 = max(0.0, min(float(height), y + box_height))
            area_sum[category] += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        images_with.update(present)

    output = {
        "source": str(args.source.resolve()),
        "images": image_count,
        "total_image_pixels": total_pixels,
        "category_rows": dict(sorted(categories.items())),
        "images_with_category": dict(sorted(images_with.items())),
        "summed_clipped_box_area_fraction": {
            str(category): area / total_pixels for category, area in sorted(area_sum.items())
        },
        "missing_annotations": missing_annotation,
        "malformed_rows": malformed_rows,
        "note": "Area fractions sum individual clipped box areas, not exact unions; overlapping rectangles are counted more than once.",
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
