"""Read-only structure, annotation, and full image-decode audit of CUB."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:\datasets\CUB-200-2011\extracted")
MAIN = ROOT / "CUB_200_2011"
MASKS = ROOT / "segmentations"
ImageFile.LOAD_TRUNCATED_IMAGES = False


def keyed_rows(name: str, width: int | None = None) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for line in (MAIN / name).read_text(encoding="utf-8").splitlines():
        columns = line.split()
        key = int(columns[0])
        if key in result or (width is not None and len(columns) != width):
            raise ValueError(f"Malformed or duplicate record in {name}: {line}")
        result[key] = columns[1:]
    return result


def main() -> None:
    images = keyed_rows("images.txt", 2)
    labels = keyed_rows("image_class_labels.txt", 2)
    boxes = keyed_rows("bounding_boxes.txt", 5)
    split = keyed_rows("train_test_split.txt", 2)
    classes = keyed_rows("classes.txt", 2)
    ids = set(images)
    issues: list[str] = []
    for name, table in (("labels", labels), ("boxes", boxes), ("split", split)):
        if set(table) != ids:
            issues.append(f"{name} ID set differs from images.txt")
    if len(classes) != 200:
        issues.append(f"Expected 200 classes, found {len(classes)}")
    count_by_class: Counter[int] = Counter()
    count_by_split: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    bbox_outside: list[int] = []
    bbox_excess: list[tuple[int, float]] = []
    non_rgb_image_ids: list[int] = []
    non_l_mask_ids: list[int] = []
    mask_dimension_mismatch: list[int] = []
    missing: list[str] = []
    decode_errors: list[str] = []
    expected_mask_rel: set[Path] = set()
    for image_id, (relative,) in images.items():
        image_path = MAIN / "images" / relative
        mask_rel = Path(relative).with_suffix(".png")
        mask_path = MASKS / mask_rel
        expected_mask_rel.add(mask_rel)
        if not image_path.is_file():
            missing.append(str(image_path))
            continue
        if not mask_path.is_file():
            missing.append(str(mask_path))
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_modes[image.mode] += 1
                if image.mode != "RGB":
                    non_rgb_image_ids.append(image_id)
                dimensions[f"{width}x{height}"] += 1
            with Image.open(mask_path) as mask:
                mask.load()
                mask_modes[mask.mode] += 1
                if mask.mode != "L":
                    non_l_mask_ids.append(image_id)
                if mask.size != (width, height):
                    mask_dimension_mismatch.append(image_id)
        except Exception as exc:
            decode_errors.append(f"{image_id}: {type(exc).__name__}: {exc}")
            continue
        class_id = int(labels[image_id][0])
        count_by_class[class_id] += 1
        count_by_split[split[image_id][0]] += 1
        if int(Path(relative).parts[0].split(".")[0]) != class_id:
            issues.append(f"Image {image_id} class label and directory disagree")
        x, y, bw, bh = (float(value) for value in boxes[image_id])
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > width or y + bh > height:
            bbox_outside.append(image_id)
            bbox_excess.append((image_id, max(-x, -y, x + bw - width, y + bh - height, 0.0)))
    disk_images = set((MAIN / "images").rglob("*.jpg"))
    disk_masks = {path.relative_to(MASKS) for path in MASKS.rglob("*.png")}
    expected_images = {MAIN / "images" / relative[0] for relative in images.values()}
    if disk_images != expected_images:
        issues.append(f"Image file set differs: listed={len(expected_images)}, disk={len(disk_images)}")
    if disk_masks != expected_mask_rel:
        issues.append(f"Mask file set differs: listed={len(expected_mask_rel)}, disk={len(disk_masks)}")
    if set(split_value[0] for split_value in split.values()) != {"0", "1"}:
        issues.append("Unexpected train/test split values")
    result = {
        "official_image_count": len(images),
        "official_class_count": len(classes),
        "listed_bounding_box_count": len(boxes),
        "listed_split_count": len(split),
        "listed_class_label_count": len(labels),
        "disk_image_count": len(disk_images),
        "disk_mask_count": len(disk_masks),
        "full_decoded_image_count": sum(image_modes.values()),
        "full_decoded_mask_count": sum(mask_modes.values()),
        "train_test_counts": dict(count_by_split),
        "image_modes": dict(image_modes),
        "mask_modes": dict(mask_modes),
        "distinct_image_dimensions": len(dimensions),
        "top_image_dimensions": dimensions.most_common(12),
        "per_class_image_count_min": min(count_by_class.values(), default=0),
        "per_class_image_count_max": max(count_by_class.values(), default=0),
        "bbox_outside_count": len(bbox_outside),
        "bbox_outside_example_ids": bbox_outside[:20],
        "bbox_max_excess_pixels": max((value for _, value in bbox_excess), default=0.0),
        "bbox_largest_excess_examples": sorted(bbox_excess, key=lambda item: -item[1])[:10],
        "non_rgb_image_ids": non_rgb_image_ids,
        "non_l_mask_ids": non_l_mask_ids,
        "mask_dimension_mismatch_count": len(mask_dimension_mismatch),
        "mask_dimension_mismatch_example_ids": mask_dimension_mismatch[:20],
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "decode_error_count": len(decode_errors),
        "decode_error_examples": decode_errors[:20],
        "issues": issues[:20],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
