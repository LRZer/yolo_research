"""Diagnostic matched-box recall by fixed 640-reference object-size bucket.

This is not VisDrone's official evaluator and does not calculate AP. It reads
the task-owned symlink view, never edits the underlying images or labels, and
uses the same image IDs/buckets for any compared inference resolution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


BUCKETS = ("tiny_lt16", "small_16to32", "medium_32to96", "large_ge96")
THRESHOLDS = (0.5, 0.75)


def size_bucket(width: float, height: float, image_width: int, image_height: int) -> str:
    """Bucket by sqrt(box area) after hypothetical 640 letterbox resize."""
    side = np.sqrt(max(width, 0.0) * max(height, 0.0))
    side *= min(640 / image_width, 640 / image_height)
    if side < 16:
        return BUCKETS[0]
    if side < 32:
        return BUCKETS[1]
    if side < 96:
        return BUCKETS[2]
    return BUCKETS[3]


def read_yolo_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    buckets: list[str] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
            cls = int(fields[0])
            xc, yc, bw, bh = map(float, fields[1:])
            if cls not in range(10) or not np.isfinite([xc, yc, bw, bh]).all() or bw <= 0 or bh <= 0:
                raise ValueError(f"{path}:{line_number}: invalid class or box")
            x1 = (xc - bw / 2) * width
            x2 = (xc + bw / 2) * width
            y1 = (yc - bh / 2) * height
            y2 = (yc + bh / 2) * height
            classes.append(cls)
            boxes.append([x1, y1, x2, y2])
            buckets.append(size_bucket(bw * width, bh * height, width, height))
    return (
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        buckets,
    )


def iou_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(predictions) == 0 or len(targets) == 0:
        return np.zeros((len(predictions), len(targets)), dtype=np.float32)
    top_left = np.maximum(predictions[:, None, :2], targets[None, :, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], targets[None, :, 2:])
    overlap = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    pred_area = np.maximum(predictions[:, 2] - predictions[:, 0], 0) * np.maximum(
        predictions[:, 3] - predictions[:, 1], 0
    )
    target_area = np.maximum(targets[:, 2] - targets[:, 0], 0) * np.maximum(
        targets[:, 3] - targets[:, 1], 0
    )
    return overlap / np.maximum(pred_area[:, None] + target_area[None, :] - overlap, 1e-9)


def match_one_class(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, threshold: float
) -> np.ndarray:
    """Return one-to-one matched GT flags, greedily considering confidence."""
    matched = np.zeros(len(gt_boxes), dtype=bool)
    if not len(pred_boxes) or not len(gt_boxes):
        return matched
    order = np.argsort(-pred_scores, kind="stable")
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    for row in ious:
        available = np.where(~matched, row, -1.0)
        target = int(np.argmax(available))
        if available[target] >= threshold:
            matched[target] = True
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True, help="Task-owned split view root")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    image_dir = args.view / "images" / args.split
    label_dir = args.view / "labels" / args.split
    expected_images = {p.stem for p in image_dir.glob("*.jpg")}
    if not expected_images:
        raise ValueError(f"no JPG images in {image_dir}")
    if len(expected_images) != len(list(image_dir.glob("*.jpg"))):
        raise ValueError("duplicate image stems")
    if args.split == "val" and len(expected_images) != 548:
        raise ValueError(f"expected 548 official-source val images, got {len(expected_images)}")

    model = YOLO(str(args.weights))
    counter: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {str(threshold): [0, 0] for threshold in THRESHOLDS}
    )
    seen: set[str] = set()
    predictions = model.predict(
        source=str(image_dir),
        stream=True,
        batch=8,
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        device=args.device,
        verbose=False,
        save=False,
    )
    for result in predictions:
        image_path = Path(result.path)
        stem = image_path.stem
        if stem not in expected_images or stem in seen:
            raise ValueError(f"unexpected or repeated image result: {image_path}")
        seen.add(stem)
        with Image.open(image_path) as image:
            width, height = image.size
        gt_classes, gt_boxes, buckets = read_yolo_labels(label_dir / f"{stem}.txt", width, height)
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        pred_scores = result.boxes.conf.cpu().numpy()
        for cls in range(10):
            gt_indices = np.flatnonzero(gt_classes == cls)
            if not len(gt_indices):
                continue
            pred_indices = np.flatnonzero(pred_classes == cls)
            class_gt_boxes = gt_boxes[gt_indices]
            class_pred_boxes = pred_boxes[pred_indices]
            class_pred_scores = pred_scores[pred_indices]
            for threshold in THRESHOLDS:
                matched = match_one_class(class_pred_boxes, class_pred_scores, class_gt_boxes, threshold)
                key = str(threshold)
                for local_index, global_index in enumerate(gt_indices):
                    bucket = buckets[global_index]
                    for group in ("all", bucket, f"class_{cls}", f"class_{cls}/{bucket}"):
                        counter[group][key][0] += int(matched[local_index])
                        counter[group][key][1] += 1
    if seen != expected_images:
        raise ValueError(f"inference missed {len(expected_images - seen)} images")

    summary = {
        "metric": "diagnostic_one_to_one_box_recall_not_AP",
        "protocol": "YOLO_predict_conf0.001_iou0.7_max_det1000; GT buckets fixed at hypothetical 640 letterbox sqrt(area)",
        "caveat": "Converted YOLO labels omit official VisDrone ignore/others; this is not official toolkit scoring",
        "weights": str(args.weights.resolve()),
        "view": str(args.view.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "images": len(seen),
        "counts": {
            group: {
                threshold: {"matched": matched, "targets": total, "recall": matched / total if total else None}
                for threshold, (matched, total) in values.items()
            }
            for group, values in sorted(counter.items())
        },
    }
    if "all" not in summary["counts"] or summary["counts"]["all"]["0.5"]["targets"] == 0:
        raise ValueError("no labeled GT boxes counted")
    if args.split == "val" and summary["counts"]["all"]["0.5"]["targets"] != 38759:
        raise ValueError("official-source val GT count differs from the independently audited 38,759")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(seen), "all": summary["counts"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
