"""Paired GT-level recall diagnostic for two frozen VOC detector checkpoints.

Ground-truth boxes are read only after both models independently predict from
unmodified images. This uses a fixed confidence threshold, not VOC AP scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def gt_for(image_path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    rows = [list(map(float, line.split())) for line in label_path.read_text(encoding="utf-8").splitlines()]
    if any(len(row) != 5 for row in rows):
        raise ValueError(label_path)
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
    values = np.asarray(rows)
    classes = values[:, 0].astype(int)
    cx, cy, w, h = values[:, 1:].T
    boxes = np.column_stack(((cx - w / 2) * width, (cy - h / 2) * height,
                             (cx + w / 2) * width, (cy + h / 2) * height))
    gain = min(640 / height, 640 / width)
    areas_640 = w * width * gain * h * height * gain
    return boxes, classes, areas_640


def match_gt(gt_boxes: np.ndarray, gt_classes: np.ndarray, result, threshold: float) -> np.ndarray:
    matched = np.zeros(len(gt_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(result.boxes) == 0:
        return matched
    predicted_boxes = result.boxes.xyxy.cpu().numpy()
    predicted_classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    for index in np.argsort(-scores):
        compatible = np.flatnonzero((gt_classes == predicted_classes[index]) & ~matched)
        if len(compatible) == 0:
            continue
        gt = gt_boxes[compatible]
        pred = predicted_boxes[index]
        overlap_min = np.maximum(gt[:, :2], pred[:2])
        overlap_max = np.minimum(gt[:, 2:], pred[2:])
        intersection = np.maximum(0, overlap_max - overlap_min).prod(axis=1)
        area_gt = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(axis=1)
        area_pred = np.maximum(0, pred[2:] - pred[:2]).prod()
        ious = intersection / np.maximum(area_gt + area_pred - intersection, 1e-9)
        best = int(np.argmax(ious))
        if ious[best] >= threshold:
            matched[compatible[best]] = True
    return matched


def category(area: float) -> str:
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=1655)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(line.strip()) for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != 1655 or not 0 < args.limit <= len(paths):
        raise AssertionError(f"expected 1655 fixed dev images and valid limit, got {len(paths)}, {args.limit}")
    paths = paths[: args.limit]
    models = {"method": YOLO(str(args.method)), "baseline": YOLO(str(args.baseline))}
    if not isinstance(models["method"].model, SafeBackgroundDetectionModel):
        raise AssertionError("wrong method checkpoint")
    models["method"].model.set_eval_mix(0.5)
    if isinstance(models["baseline"].model, SafeBackgroundDetectionModel):
        raise AssertionError("wrong baseline checkpoint")
    # Order both predictions by the same fixed path list, never passing labels.
    results_by_model: dict[str, list] = {name: [] for name in models}
    for name, model in models.items():
        for start in range(0, len(paths), 64):
            chunk = paths[start : start + 64]
            results_by_model[name].extend(model.predict(
                source=[str(path) for path in chunk], imgsz=640, batch=16,
                device=args.device, half=False, rect=True, conf=args.conf,
                iou=0.7, max_det=300, verbose=False, save=False, stream=False,
            ))
        if len(results_by_model[name]) != len(paths):
            raise AssertionError(f"{name} predictions missing")
    counts = defaultdict(lambda: {"total": 0, "both": 0, "baseline_only": 0,
                                  "method_only": 0, "neither": 0})
    examples: dict[str, list] = {"baseline_only": [], "method_only": []}
    instances = 0
    for image_path, method, baseline in zip(paths, results_by_model["method"], results_by_model["baseline"]):
        height, width = method.orig_shape
        if baseline.orig_shape != method.orig_shape:
            raise AssertionError("two models saw different image sizes")
        gt_boxes, gt_classes, areas = gt_for(image_path, height, width)
        a = match_gt(gt_boxes, gt_classes, method, args.iou)
        b = match_gt(gt_boxes, gt_classes, baseline, args.iou)
        for index, (class_id, area) in enumerate(zip(gt_classes, areas)):
            status = "both" if a[index] and b[index] else (
                "method_only" if a[index] else ("baseline_only" if b[index] else "neither"))
            for group in ("all", f"size:{category(area)}", f"class:{int(class_id)}"):
                counts[group]["total"] += 1
                counts[group][status] += 1
            if status in examples and len(examples[status]) < 30:
                examples[status].append({"image": str(image_path), "class_id": int(class_id),
                                         "box_xyxy": [float(v) for v in gt_boxes[index]],
                                         "area_640": float(area)})
            instances += 1
    if args.limit == 1655 and instances != 3882:
        raise AssertionError(f"unexpected GT count: {instances}")
    report = {
        "method": str(args.method), "baseline": str(args.baseline),
        "images": len(paths), "instances": instances,
        "definition": "One-to-one greedy same-class IoU>=0.5 match at fixed conf>=0.25 after independent unmodified-image prediction; not AP or benchmark-standard recall",
        "confidence": args.conf, "gt_iou_threshold": args.iou,
        "nms_iou": 0.7, "imgsz": 640, "half": False,
        "counts": dict(counts), "examples": examples,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all": counts["all"]}))


if __name__ == "__main__":
    main()
