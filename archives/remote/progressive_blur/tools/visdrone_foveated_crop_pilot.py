"""No-training, GT-free selected-crop inference pilot on frozen YOLOv8n.

Methods: full, full+fixed central 4 cells, full+prediction-selected 4 cells.
GT is read only after candidate selection and all predictions are finalized.
This is converted-label Ultralytics-style AP, not VisDrone official toolkit AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils import nms as ultralytics_nms


# DetectionValidator uses multi_label=True, but DetectionPredictor leaves it
# at the NMS default False. Apply the validator choice to every predict call
# in this one-off process so the raw-image control has a comparable protocol.
_original_nms = ultralytics_nms.non_max_suppression


def _validator_style_nms(*args, **kwargs):
    kwargs["multi_label"] = True
    return _original_nms(*args, **kwargs)


ultralytics_nms.non_max_suppression = _validator_style_nms


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
FIXED_CELLS = (5, 6, 9, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, classes, scores


def top_cells(boxes: np.ndarray, scores: np.ndarray, width: int, height: int) -> tuple[int, ...]:
    heat = np.zeros(16, dtype=np.float64)
    if len(boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        cols = np.clip((centers[:, 0] / width * 4).astype(np.int64), 0, 3)
        rows = np.clip((centers[:, 1] / height * 4).astype(np.int64), 0, 3)
        np.add.at(heat, rows * 4 + cols, scores)
    return tuple(sorted(range(16), key=lambda cell: (-heat[cell], cell))[:4])


def crop_rect(cell: int, width: int, height: int) -> tuple[int, int, int, int]:
    col, row = cell % 4, cell // 4
    x0, x1 = round(col * width / 4), round((col + 1) * width / 4)
    y0, y1 = round(row * height / 4), round((row + 1) * height / 4)
    halo_x, halo_y = round((x1 - x0) * 0.1), round((y1 - y0) * 0.1)
    return max(0, x0 - halo_x), max(0, y0 - halo_y), min(width, x1 + halo_x), min(height, y1 + halo_y)


def merge_predictions(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      max_det: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonempty = [group for group in groups if len(group[0])]
    if not nonempty:
        return np.empty((0, 4), np.float32), np.empty(0, np.int64), np.empty(0, np.float32)
    boxes = np.concatenate([group[0] for group in nonempty])
    classes = np.concatenate([group[1] for group in nonempty])
    scores = np.concatenate([group[2] for group in nonempty])
    keep = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        local_keep = TorchNMS.nms(torch.from_numpy(boxes[indices]),
                                  torch.from_numpy(scores[indices]), 0.7).numpy()
        keep.extend(indices[local_keep].tolist())
    keep = sorted(keep, key=lambda idx: (-float(scores[idx]), int(idx)))[:max_det]
    return boxes[keep], classes[keep], scores[keep]


def gt_for_image(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    classes, boxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"bad label line: {label_path}: {line}")
        class_id, x, y, w, h = map(float, values)
        if not (class_id.is_integer() and 0 <= class_id < 10 and 0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
            raise ValueError(f"bad label value: {label_path}: {line}")
        classes.append(int(class_id))
        boxes.append([(x - w/2)*width, (y - h/2)*height,
                      (x + w/2)*width, (y + h/2)*height])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def match_tp(pred_boxes: np.ndarray, pred_classes: np.ndarray,
             gt_boxes: np.ndarray, gt_classes: np.ndarray) -> np.ndarray:
    tp = np.zeros((len(pred_boxes), 10), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    lt = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    rb = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    intersection = np.prod(np.maximum(rb - lt, 0), axis=2)
    gt_area = np.prod(np.maximum(gt_boxes[:, 2:] - gt_boxes[:, :2], 0), axis=1)
    pred_area = np.prod(np.maximum(pred_boxes[:, 2:] - pred_boxes[:, :2], 0), axis=1)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-7)
    iou *= (gt_classes[:, None] == pred_classes[None, :])
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if len(matches):
            if len(matches) > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp[matches[:, 1].astype(int), threshold_index] = True
    return tp


def evaluate(stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], names: dict) -> dict:
    tp = np.concatenate([row[0] for row in stats], axis=0)
    conf = np.concatenate([row[1] for row in stats], axis=0)
    pred_cls = np.concatenate([row[2] for row in stats], axis=0)
    target_cls = np.concatenate([row[3] for row in stats], axis=0)
    if len(target_cls) == 0:
        raise ValueError("no ground truth")
    _, _, p, r, _, ap, classes, *_ = ap_per_class(tp, conf, pred_cls, target_cls,
                                                  plot=False, names=names)
    return {
        "precision": float(p.mean()), "recall": float(r.mean()),
        "map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
        "predictions": int(len(conf)), "gt_boxes": int(len(target_cls)),
        "per_class": [
            {"id": int(class_id), "name": str(names[int(class_id)]),
             "ap50": float(ap[index, 0]), "ap50_95": float(ap[index].mean())}
            for index, class_id in enumerate(classes)
        ],
    }


def self_test() -> None:
    assert crop_rect(5, 400, 400) == (90, 90, 210, 210)
    boxes = np.array([[100, 100, 120, 120], [101, 101, 121, 121]], np.float32)
    cls = np.array([3, 3], np.int64)
    conf = np.array([0.9, 0.8], np.float32)
    merged = merge_predictions([(boxes, cls, conf)])
    assert len(merged[0]) == 1 and abs(float(merged[2][0]) - 0.9) < 1e-6
    assert match_tp(merged[0], merged[1], boxes[:1], cls[:1])[0, 0]
    assert top_cells(np.array([[140, 140, 160, 160]], np.float32),
                     np.array([0.9], np.float32), 400, 400)[0] == 5
    print("SELF_TEST_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.checkpoint or not args.checkpoint.is_file() or not args.manifest or not args.manifest.is_file():
        raise FileNotFoundError("checkpoint/manifest missing")
    if not args.output or args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output exists or parent is missing")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid image manifest")
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10 classes")
    names = dict(model.names)
    stats = {name: [] for name in ("full", "full_fixed4", "full_adaptive4")}
    timings = {"full_batch": 0.0, "fixed4_crop_and_predict": 0.0,
               "adaptive4_crop_and_predict": 0.0}
    total_start = time.perf_counter()
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset+8]
        t0 = time.perf_counter()
        base_results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                     batch=8, conf=0.001, iou=0.7, max_det=1000,
                                     half=False, device=args.device, verbose=False,
                                     save=False, stream=False)
        timings["full_batch"] += time.perf_counter() - t0
        if len(base_results) != len(batch_paths):
            raise RuntimeError("base result count mismatch")
        for image_path, base_result in zip(batch_paths, base_results):
            if Path(base_result.path).resolve() != image_path.resolve():
                raise RuntimeError("base image/prediction ordering mismatch")
            height, width = base_result.orig_shape
            base_pred = unpack(base_result)
            adaptive = top_cells(base_pred[0], base_pred[2], width, height)
            # Candidate cells are fixed before GT labels are read.
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"image decode or size mismatch: {image_path}")
            combined = {"full": base_pred}
            for method, cells, timing_key in (
                ("full_fixed4", FIXED_CELLS, "fixed4_crop_and_predict"),
                ("full_adaptive4", adaptive, "adaptive4_crop_and_predict"),
            ):
                t1 = time.perf_counter()
                rects = [crop_rect(cell, width, height) for cell in cells]
                crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
                crop_results = model.predict(source=crops, imgsz=960, batch=4,
                                             conf=0.001, iou=0.7, max_det=1000,
                                             half=False, device=args.device,
                                             verbose=False, save=False, stream=False)
                if len(crop_results) != 4:
                    raise RuntimeError("crop result count mismatch")
                groups = [base_pred]
                for result, (x0, y0, x1, y1) in zip(crop_results, rects):
                    boxes, classes, scores = unpack(result)
                    if len(boxes):
                        boxes[:, (0, 2)] += x0
                        boxes[:, (1, 3)] += y0
                        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
                        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
                    groups.append((boxes, classes, scores))
                combined[method] = merge_predictions(groups)
                timings[timing_key] += time.perf_counter() - t1
            # No GT has entered either crop selection or prediction path.
            gt_classes, gt_boxes = gt_for_image(image_path, width, height)
            for method, (pred_boxes, pred_classes, scores) in combined.items():
                tp = match_tp(pred_boxes, pred_classes, gt_boxes, gt_classes)
                stats[method].append((tp, scores, pred_classes, gt_classes))
        print(f"evaluated {min(offset+8, len(paths))}/{len(paths)}", flush=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "images": len(paths),
        "conditions": {method: evaluate(rows, names) for method, rows in stats.items()},
        "timing_seconds": {**timings, "total_script_wall_including_load": time.perf_counter()-total_start},
        "protocol": {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
                     "crop_grid": "4x4", "crop_halo_fraction_each_side": 0.1,
                     "selected_cells": 4, "fusion_nms_classwise_iou": 0.7,
                     "candidate_nms_multi_label": True,
                     "fixed_cells": list(FIXED_CELLS), "adaptive_score": "sum of coarse prediction confidences by cell",
                     "gt_used_in_candidate_selection": False,
                     "evaluator": "Ultralytics ap_per_class with validator-style greedy matching, converted YOLO labels; not official VisDrone"},
        "caveat": "Each crop is another 960 input; full+four crops uses at least 5x model input pixels. Stage timing excludes image read and model load and is not a stable deployment latency benchmark. Fixed-center4 is not a strong SAHI baseline.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("FINAL_REPORT="+json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
