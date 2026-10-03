"""One fixed-sigma crop-only blur diagnostic on a frozen VisDrone detector."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import visdrone_foveated_crop_pilot as base
from visdrone_crop_provenance_audit import matched_gt_to_pred, MATCH_CONF


SIGMA_AT_INPUT_960 = 3.0
TARGET_CLASSES = {0: "pedestrian", 1: "people", 5: "truck", 8: "bus"}


def blur_crop(crop: np.ndarray) -> tuple[np.ndarray, float]:
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
        raise ValueError("expected uint8 BGR crop")
    sigma_src = SIGMA_AT_INPUT_960 * max(crop.shape[:2]) / 960.0
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
    return blurred, float(sigma_src)


def remap(result, rect, width: int, height: int):
    boxes, classes, scores = base.unpack(result)
    x0, y0, _, _ = rect
    if len(boxes):
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)] + x0, 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)] + y0, 0, height)
    return boxes, classes, scores


def self_test():
    flat = np.full((100, 150, 3), 128, np.uint8)
    blurred, sigma = blur_crop(flat)
    assert np.array_equal(flat, blurred)
    assert abs(sigma - 3 * 150 / 960) < 1e-9
    impulse = flat.copy()
    impulse[50, 75] = 255
    changed, _ = blur_crop(impulse)
    assert changed[50, 75, 0] < 255
    assert not np.array_equal(impulse, changed)
    base.self_test()
    print("CROP_LIGHT_BLUR_SELF_TEST_OK", flush=True)


def run(args):
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("checkpoint or manifest missing")
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output exists or parent missing")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid image manifest")
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10 classes")
    names = dict(model.names)
    stats = {key: [] for key in ("full", "full_raw4", "full_blur4")}
    counters = {key: {name: Counter() for name in TARGET_CLASSES.values()} for key in stats}
    blur_abs_sum = 0
    blur_pixel_count = 0
    sigma_min = float("inf")
    sigma_max = 0.0
    started = time.perf_counter()
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset + 8]
        full_results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                     batch=8, conf=0.001, iou=0.7, max_det=1000,
                                     half=False, device=args.device, verbose=False,
                                     save=False, stream=False)
        if len(full_results) != len(batch_paths):
            raise RuntimeError("full-image count mismatch")
        for image_path, full_result in zip(batch_paths, full_results):
            if Path(full_result.path).resolve() != image_path.resolve():
                raise RuntimeError("image ordering mismatch")
            height, width = full_result.orig_shape
            full_pred = base.unpack(full_result)
            cells = base.top_cells(full_pred[0], full_pred[2], width, height)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"image decode/shape mismatch: {image_path}")
            rects = [base.crop_rect(cell, width, height) for cell in cells]
            crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
            blurred = []
            for crop in crops:
                item, sigma = blur_crop(crop)
                sigma_min = min(sigma_min, sigma)
                sigma_max = max(sigma_max, sigma)
                blur_abs_sum += int(np.abs(crop.astype(np.int16) - item.astype(np.int16)).sum())
                blur_pixel_count += int(crop.size)
                blurred.append(item)
            raw_results = model.predict(source=crops, imgsz=960, batch=4,
                                        conf=0.001, iou=0.7, max_det=1000,
                                        half=False, device=args.device, verbose=False,
                                        save=False, stream=False)
            blur_results = model.predict(source=blurred, imgsz=960, batch=4,
                                         conf=0.001, iou=0.7, max_det=1000,
                                         half=False, device=args.device, verbose=False,
                                         save=False, stream=False)
            if len(raw_results) != 4 or len(blur_results) != 4:
                raise RuntimeError("crop count mismatch")
            raw_groups = [full_pred]
            blur_groups = [full_pred]
            for result, rect in zip(raw_results, rects):
                raw_groups.append(remap(result, rect, width, height))
            for result, rect in zip(blur_results, rects):
                blur_groups.append(remap(result, rect, width, height))
            predictions = {
                "full": full_pred,
                "full_raw4": base.merge_predictions(raw_groups),
                "full_blur4": base.merge_predictions(blur_groups),
            }
            # Ground truth enters only after all crop selection and prediction.
            gt_classes, gt_boxes = base.gt_for_image(image_path, width, height)
            for key, (boxes, classes, scores) in predictions.items():
                tp = base.match_tp(boxes, classes, gt_boxes, gt_classes)
                stats[key].append((tp, scores, classes, gt_classes))
                for class_id, name in TARGET_CLASSES.items():
                    counter = counters[key][name]
                    counter["gt"] += int(np.count_nonzero(gt_classes == class_id))
                    counter["tp"] += len(matched_gt_to_pred(
                        boxes, classes, scores, gt_boxes, gt_classes, class_id
                    ))
                    counter["preds_at_conf"] += int(np.count_nonzero(
                        (classes == class_id) & (scores >= MATCH_CONF)
                    ))
        print(f"evaluated {min(offset + 8, len(paths))}/{len(paths)}", flush=True)
    fixed = {}
    for key, by_class in counters.items():
        fixed[key] = {}
        for name, counter in by_class.items():
            counter["fp"] = counter["preds_at_conf"] - counter["tp"]
            fixed[key][name] = dict(counter)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": base.sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base.sha256(args.manifest),
        "images": len(paths),
        "conditions": {key: base.evaluate(rows, names) for key, rows in stats.items()},
        "fixed_conf_025_iou_05": fixed,
        "blur_mean_abs_pixel_delta_255": blur_abs_sum / blur_pixel_count,
        "sigma_source_pixel_range": [sigma_min, sigma_max],
        "wall_seconds_all_conditions": time.perf_counter() - started,
        "protocol": {
            "imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
            "grid": "4x4", "halo_fraction_each_side": 0.1, "selected_cells": 4,
            "fusion_nms_classwise_iou": 0.7, "candidate_nms_multi_label": True,
            "blur_crop_sigma_at_input960": SIGMA_AT_INPUT_960,
            "full_image_branch_blurred": False,
            "real_gt_used_for_crop_selection_or_blur": False,
            "evaluator": "internal YOLO converted-label AP, not official VisDrone",
        },
        "caveat": "This is a single exploratory sigma on repeatedly inspected val548. Each crop method still requires five 960 detector calls per image. All-conditions wall time is not deployment latency.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
