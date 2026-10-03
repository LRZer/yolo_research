from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(sys.argv[1])
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 640
NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]
SPLITS = {
    "train2007": "train",
    "val2007": "train",
    "train2012": "train",
    "val2012": "train",
    "test2007": "test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 6) if d else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {}
    return {
        "min": round(float(a.min()), 6),
        "p01": round(float(np.percentile(a, 1)), 6),
        "p05": round(float(np.percentile(a, 5)), 6),
        "p10": round(float(np.percentile(a, 10)), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "median": round(float(np.median(a)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "max": round(float(a.max()), 6),
        "mean": round(float(a.mean()), 6),
    }


def audit(group: str, split_names: list[str]) -> dict:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    side_min: list[float] = []
    side_max: list[float] = []
    per_class = Counter()
    per_class_stal = Counter()
    per_class_coco: dict[int, Counter] = {i: Counter() for i in range(len(NAMES))}
    image_dims = Counter()
    image_count = label_count = empty_label_count = object_count = 0
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    invalid_rows: list[str] = []
    image_with_small = 0
    image_with_objects = 0
    per_split: dict[str, dict] = {}

    for split in split_names:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        image_by_stem = {p.stem: p for p in images}
        label_by_stem = {p.stem: p for p in labels}
        split_missing = sorted(set(image_by_stem) - set(label_by_stem))
        split_orphans = sorted(set(label_by_stem) - set(image_by_stem))
        missing_labels.extend(f"{split}/{x}" for x in split_missing)
        orphan_labels.extend(f"{split}/{x}" for x in split_orphans)
        split_obj = 0
        split_empty = 0
        split_small = 0
        split_coco = Counter()

        for stem, image_path in image_by_stem.items():
            image_count += 1
            label_path = label_by_stem.get(stem)
            if label_path is None:
                continue
            label_count += 1
            with Image.open(image_path) as im:
                w0, h0 = im.size
            image_dims[(w0, h0)] += 1
            ratio = min(TARGET / w0, TARGET / h0)
            rows = [x.strip() for x in label_path.read_text().splitlines() if x.strip()]
            if not rows:
                empty_label_count += 1
                split_empty += 1
                continue
            image_with_objects += 1
            this_image_small = False
            for line_no, row in enumerate(rows, 1):
                fields = row.split()
                if len(fields) != 5:
                    invalid_rows.append(f"{split}/{stem}:{line_no}:columns={len(fields)}")
                    continue
                try:
                    c = int(float(fields[0]))
                    xc, yc, wn, hn = map(float, fields[1:])
                except ValueError:
                    invalid_rows.append(f"{split}/{stem}:{line_no}:parse")
                    continue
                if not 0 <= c < len(NAMES) or not all(math.isfinite(v) for v in (xc, yc, wn, hn)):
                    invalid_rows.append(f"{split}/{stem}:{line_no}:class_or_nan")
                    continue
                if (
                    wn <= 0
                    or hn <= 0
                    or wn > 1
                    or hn > 1
                    or not (0 <= xc <= 1 and 0 <= yc <= 1)
                    or xc - wn / 2 < -1e-5
                    or yc - hn / 2 < -1e-5
                    or xc + wn / 2 > 1 + 1e-5
                    or yc + hn / 2 > 1 + 1e-5
                ):
                    invalid_rows.append(f"{split}/{stem}:{line_no}:bbox_range")
                    continue
                wp = wn * w0 * ratio
                hp = hn * h0 * ratio
                area = wp * hp
                widths.append(wp)
                heights.append(hp)
                areas.append(area)
                side_min.append(min(wp, hp))
                side_max.append(max(wp, hp))
                per_class[c] += 1
                object_count += 1
                split_obj += 1
                if wp < 16 or hp < 16:
                    this_image_small = True
                    split_small += 1
                    per_class_stal[c] += 1
                if area < 32**2:
                    split_coco["small"] += 1
                    per_class_coco[c]["small"] += 1
                elif area < 96**2:
                    split_coco["medium"] += 1
                    per_class_coco[c]["medium"] += 1
                else:
                    split_coco["large"] += 1
                    per_class_coco[c]["large"] += 1
            if this_image_small:
                image_with_small += 1

        per_split[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": split_empty,
            "objects": split_obj,
            "stal_any_side_lt16_objects": split_small,
            "stal_any_side_lt16_percent": pct(split_small, split_obj),
            "coco_area": dict(split_coco),
            "missing_labels": len(split_missing),
            "orphan_labels": len(split_orphans),
        }

    w = np.asarray(widths)
    h = np.asarray(heights)
    a = np.asarray(areas)
    coco_small = int((a < 32**2).sum())
    coco_medium = int(((a >= 32**2) & (a < 96**2)).sum())
    coco_large = int((a >= 96**2).sum())
    metrics = {
        "group": group,
        "target_letterbox": TARGET,
        "images": image_count,
        "labels": label_count,
        "empty_labels": empty_label_count,
        "images_with_objects": image_with_objects,
        "objects": object_count,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "invalid_rows": invalid_rows,
        "stal_any_side_lt16_objects": int(((w < 16) | (h < 16)).sum()),
        "stal_any_side_lt16_percent": pct(int(((w < 16) | (h < 16)).sum()), object_count),
        "stal_both_sides_lt16_objects": int(((w < 16) & (h < 16)).sum()),
        "stal_both_sides_lt16_percent": pct(int(((w < 16) & (h < 16)).sum()), object_count),
        "width_lt16_objects": int((w < 16).sum()),
        "width_lt16_percent": pct(int((w < 16).sum()), object_count),
        "height_lt16_objects": int((h < 16).sum()),
        "height_lt16_percent": pct(int((h < 16).sum()), object_count),
        "min_side_thresholds": {
            str(t): {
                "count": int((np.minimum(w, h) < t).sum()),
                "percent": pct(int((np.minimum(w, h) < t).sum()), object_count),
            }
            for t in (8, 16, 24, 32, 48, 64, 96)
        },
        "images_with_stal_object": image_with_small,
        "images_with_stal_object_percent_all": pct(image_with_small, image_count),
        "coco_area": {
            "small_lt_32sq": {"count": coco_small, "percent": pct(coco_small, object_count)},
            "medium_32sq_to_96sq": {"count": coco_medium, "percent": pct(coco_medium, object_count)},
            "large_ge_96sq": {"count": coco_large, "percent": pct(coco_large, object_count)},
        },
        "scaled_width": summarize(widths),
        "scaled_height": summarize(heights),
        "scaled_min_side": summarize(side_min),
        "scaled_max_side": summarize(side_max),
        "scaled_area": summarize(areas),
        "class_counts": {f"{i}:{NAMES[i]}": per_class[i] for i in range(len(NAMES))},
        "class_scale_breakdown": {
            f"{i}:{NAMES[i]}": {
                "objects": per_class[i],
                "stal_any_side_lt16": per_class_stal[i],
                "stal_percent": pct(per_class_stal[i], per_class[i]),
                "coco_small": per_class_coco[i]["small"],
                "coco_small_percent": pct(per_class_coco[i]["small"], per_class[i]),
                "coco_medium": per_class_coco[i]["medium"],
                "coco_large": per_class_coco[i]["large"],
            }
            for i in range(len(NAMES))
        },
        "unique_native_image_dimensions": len(image_dims),
        "top_native_image_dimensions": [
            {"width": d[0], "height": d[1], "count": n} for d, n in image_dims.most_common(15)
        ],
        "per_split": per_split,
    }
    return metrics


train_splits = [s for s, group in SPLITS.items() if group == "train"]
test_splits = [s for s, group in SPLITS.items() if group == "test"]
print(json.dumps({
    "train": audit("train", train_splits),
    "test": audit("test", test_splits),
    "all": audit("all", list(SPLITS)),
}, ensure_ascii=False, indent=2))
