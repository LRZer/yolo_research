"""Frozen-detector proposal coverage; no GT is used to choose regions.

This is a GT-free region-selector diagnostic, NOT sliced detection or AP.
The GT labels enter only after scores and top-k cells have been finalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_boxes(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    classes = []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid label line in {label_path}: {line}")
        cls, x, y, w, h = map(float, fields)
        if not (cls.is_integer() and 0 <= cls < 10 and 0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
            raise ValueError(f"invalid label values in {label_path}: {line}")
        classes.append(int(cls))
        boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def any_same_class_iou_50(gt_cls: np.ndarray, gt_box: np.ndarray,
                          pred_cls: np.ndarray, pred_box: np.ndarray) -> np.ndarray:
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return np.zeros(len(gt_cls), dtype=bool)
    left_top = np.maximum(gt_box[:, None, :2], pred_box[None, :, :2])
    right_bottom = np.minimum(gt_box[:, None, 2:], pred_box[None, :, 2:])
    intersection = np.prod(np.maximum(right_bottom-left_top, 0), axis=2)
    gt_area = np.prod(np.maximum(gt_box[:, 2:]-gt_box[:, :2], 0), axis=1)
    pred_area = np.prod(np.maximum(pred_box[:, 2:]-pred_box[:, :2], 0), axis=1)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-7)
    return ((gt_cls[:, None] == pred_cls[None, :]) & (iou >= 0.5)).any(axis=1)


def selected_cells(pred_box: np.ndarray, confidence: np.ndarray,
                   width: int, height: int, grid: int, k: int) -> set[int]:
    scores = np.zeros(grid * grid, dtype=np.float64)
    if len(pred_box):
        centers = (pred_box[:, :2] + pred_box[:, 2:]) * 0.5
        columns = np.clip((centers[:, 0] / width * grid).astype(np.int64), 0, grid - 1)
        rows = np.clip((centers[:, 1] / height * grid).astype(np.int64), 0, grid - 1)
        np.add.at(scores, rows * grid + columns, confidence)
    # Sort by score, then deterministic cell index. No labels available here.
    return set(sorted(range(grid * grid), key=lambda cell: (-scores[cell], cell))[:k])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output already exists or parent missing")
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("checkpoint or manifest missing")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid image manifest")
    grid = 4
    budgets = (2, 4, 8)
    totals = {k: {"all": 0, "coarse_missed": 0} for k in budgets}
    boxes_all = boxes_missed = 0
    start = time.perf_counter()
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10-class detector")
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset+8]
        results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                batch=8, conf=0.001, iou=0.7, max_det=1000,
                                half=False, device=args.device, verbose=False,
                                save=False, stream=False)
        if len(results) != len(batch_paths):
            raise RuntimeError("prediction count mismatch")
        for image_path, result in zip(batch_paths, results):
            if Path(result.path).resolve() != image_path.resolve():
                raise RuntimeError(f"prediction/image order mismatch: {result.path} != {image_path}")
            height, width = result.orig_shape
            pred_box = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_cls = result.boxes.cls.cpu().numpy().astype(np.int64)
            confidence = result.boxes.conf.cpu().numpy().astype(np.float64)
            # Freeze GT-free selection before labels are opened below.
            selected = {k: selected_cells(pred_box, confidence, width, height, grid, k)
                        for k in budgets}
            gt_cls, gt_box = target_boxes(image_path, width, height)
            found = any_same_class_iou_50(gt_cls, gt_box, pred_cls, pred_box)
            centers = (gt_box[:, :2] + gt_box[:, 2:]) * 0.5
            columns = np.clip((centers[:, 0] / width * grid).astype(np.int64), 0, grid-1)
            rows = np.clip((centers[:, 1] / height * grid).astype(np.int64), 0, grid-1)
            gt_cells = rows * grid + columns
            boxes_all += len(gt_box)
            boxes_missed += int((~found).sum())
            for k in budgets:
                selected_gt = np.isin(gt_cells, list(selected[k]))
                totals[k]["all"] += int(selected_gt.sum())
                totals[k]["coarse_missed"] += int((selected_gt & ~found).sum())
        print(f"evaluated {min(offset+8, len(paths))}/{len(paths)}", flush=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "images": len(paths),
        "gt_boxes": boxes_all,
        "coarse_missed_gt_boxes": boxes_missed,
        "coarse_found_fraction_any_same_class_iou50": (boxes_all-boxes_missed)/boxes_all,
        "grid": "4x4",
        "selector": "sum of confidence of same-image coarse YOLO detections by cell, conf>=0.001, no GT; stable tie by cell ID",
        "inference": {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
                      "half": False, "device": args.device, "batch": 8},
        "top_k": {
            str(k): {"selected_area_fraction": k/16,
                     "all_gt_center_coverage": totals[k]["all"]/boxes_all,
                     "coarse_missed_gt_center_coverage":
                     totals[k]["coarse_missed"]/boxes_missed if boxes_missed else None}
            for k in budgets
        },
        "wall_seconds_including_model_load": time.perf_counter()-start,
        "caveat": "This measures GT-center coverage, not AP. Coarse-found matching is any same-class IoU>=0.5, not one-to-one detection recall. Selected cells are not run through a second detector. Full coarse inference cost must be counted in any future hybrid pipeline.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("FINAL_REPORT="+json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
