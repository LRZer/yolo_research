"""Frozen VisDrone detector with GT-free, predicted-box-protected crop blur."""

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
from visdrone_predicted_protection_audit import (
    gt_protected_fraction, protection_mask, size_bin,
)


PROTECTION_CONF = 0.01
BLUR_SIGMA_AT_INPUT960 = 3.0
FEATHER_SIGMA_AT_INPUT960 = 8.0
TARGET_CLASSES = {0: "pedestrian", 1: "people", 5: "truck", 8: "bus"}


def predicted_background_blur(crop: np.ndarray, hard_mask: np.ndarray):
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
        raise ValueError("expected uint8 BGR crop")
    if hard_mask.shape != crop.shape[:2] or hard_mask.dtype != bool:
        raise ValueError("mask shape/dtype mismatch")
    scale = max(crop.shape[:2]) / 960.0
    sigma = BLUR_SIGMA_AT_INPUT960 * scale
    feather_sigma = FEATHER_SIGMA_AT_INPUT960 * scale
    hard = hard_mask.astype(np.float32)
    soft = np.maximum(np.clip(cv2.GaussianBlur(
        hard, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma
    ), 0.0, 1.0), hard)
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mixed = (crop.astype(np.float32) * soft[..., None]
             + blurred.astype(np.float32) * (1.0 - soft[..., None]))
    output = np.ascontiguousarray(np.clip(mixed, 0, 255).astype(np.uint8))
    if not np.array_equal(output[hard_mask], crop[hard_mask]):
        raise RuntimeError("hard-protected pixels changed")
    return output


def remap(result, rect, width: int, height: int):
    boxes, classes, scores = base.unpack(result)
    x0, y0, _, _ = rect
    if len(boxes):
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)] + x0, 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)] + y0, 0, height)
    return boxes, classes, scores


def self_test():
    rng = np.random.default_rng(2027)
    crop = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[25:75, 50:110] = True
    processed = predicted_background_blur(crop, mask)
    assert np.array_equal(processed[mask], crop[mask])
    assert not np.array_equal(processed[~mask], crop[~mask])
    assert np.array_equal(predicted_background_blur(crop, np.ones_like(mask)), crop)
    base.self_test()
    print("PREDICTED_BACKGROUND_CROP_BLUR_SELF_TEST_OK", flush=True)


def run(args):
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("checkpoint or manifest missing")
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output exists or parent missing")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid manifest")
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10 classes")
    names = dict(model.names)
    stats = {key: [] for key in ("full", "full_raw4", "full_predicted_background_blur4")}
    fixed = {key: {name: Counter() for name in TARGET_CLASSES.values()} for key in stats}
    audit = Counter()
    started = time.perf_counter()
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset + 8]
        full_results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                     batch=8, conf=0.001, iou=0.7, max_det=1000,
                                     half=False, device=args.device, verbose=False,
                                     save=False, stream=False)
        if len(full_results) != len(batch_paths):
            raise RuntimeError("full-image count mismatch")
        for path, result in zip(batch_paths, full_results):
            if Path(result.path).resolve() != path.resolve():
                raise RuntimeError("image ordering mismatch")
            height, width = result.orig_shape
            full_pred = base.unpack(result)
            rects = [base.crop_rect(cell, width, height) for cell in
                     base.top_cells(full_pred[0], full_pred[2], width, height)]
            # All spatial processing uses predictions only; GT is read later.
            hard_mask = protection_mask(full_pred[0], full_pred[2], width, height,
                                        PROTECTION_CONF)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"decode/size mismatch: {path}")
            crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
            protected_crops = [hard_mask[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
            processed = [predicted_background_blur(crop, mask)
                         for crop, mask in zip(crops, protected_crops)]
            audit["mask_boxes"] += int(np.count_nonzero(full_pred[2] >= PROTECTION_CONF))
            for raw_crop, new_crop, mask in zip(crops, processed, protected_crops):
                audit["crop_pixels"] += int(mask.size)
                audit["protected_pixels"] += int(mask.sum())
                audit["abs_channel_delta_255"] += int(np.abs(
                    raw_crop.astype(np.int16) - new_crop.astype(np.int16)).sum())
            raw_results = model.predict(source=crops, imgsz=960, batch=4,
                                        conf=0.001, iou=0.7, max_det=1000,
                                        half=False, device=args.device, verbose=False,
                                        save=False, stream=False)
            processed_results = model.predict(source=processed, imgsz=960, batch=4,
                                              conf=0.001, iou=0.7, max_det=1000,
                                              half=False, device=args.device, verbose=False,
                                              save=False, stream=False)
            if len(raw_results) != 4 or len(processed_results) != 4:
                raise RuntimeError("crop result count mismatch")
            raw_groups = [full_pred] + [remap(r, rect, width, height)
                                        for r, rect in zip(raw_results, rects)]
            processed_groups = [full_pred] + [remap(r, rect, width, height)
                                              for r, rect in zip(processed_results, rects)]
            predictions = {
                "full": full_pred,
                "full_raw4": base.merge_predictions(raw_groups),
                "full_predicted_background_blur4": base.merge_predictions(processed_groups),
            }
            # True boxes cannot affect the model inputs, mask, or selected cells.
            gt_classes, gt_boxes = base.gt_for_image(path, width, height)
            for box in gt_boxes:
                if not any(x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1
                           for x0, y0, x1, y1 in rects):
                    continue
                audit["gt_fully_in_selected_crop"] += 1
                if gt_protected_fraction(hard_mask, box) >= 0.95:
                    audit["gt_protected95"] += 1
                if size_bin(box, width, height) == "lt8":
                    audit["gt_lt8_fully_in_crop"] += 1
                    if gt_protected_fraction(hard_mask, box) >= 0.95:
                        audit["gt_lt8_protected95"] += 1
            for key, (boxes, classes, scores) in predictions.items():
                stats[key].append((base.match_tp(boxes, classes, gt_boxes, gt_classes),
                                   scores, classes, gt_classes))
                for class_id, name in TARGET_CLASSES.items():
                    counter = fixed[key][name]
                    counter["gt"] += int(np.count_nonzero(gt_classes == class_id))
                    counter["tp"] += len(matched_gt_to_pred(boxes, classes, scores,
                                                              gt_boxes, gt_classes, class_id))
                    counter["preds_at_conf"] += int(np.count_nonzero(
                        (classes == class_id) & (scores >= MATCH_CONF)))
        print(f"evaluated {min(offset + 8, len(paths))}/{len(paths)}", flush=True)
    fixed_report = {}
    for key, by_class in fixed.items():
        fixed_report[key] = {}
        for name, counter in by_class.items():
            counter["fp"] = counter["preds_at_conf"] - counter["tp"]
            fixed_report[key][name] = dict(counter)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": base.sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base.sha256(args.manifest),
        "images": len(paths),
        "conditions": {key: base.evaluate(rows, names) for key, rows in stats.items()},
        "fixed_conf_025_iou_05": fixed_report,
        "protection_audit": {
            **dict(audit),
            "mean_mask_boxes_per_image": audit["mask_boxes"] / len(paths),
            "protected_fraction_selected_crops": audit["protected_pixels"] / audit["crop_pixels"],
            "blur_permitted_fraction_selected_crops": 1 - audit["protected_pixels"] / audit["crop_pixels"],
            "mean_abs_pixel_delta_255": audit["abs_channel_delta_255"] / (audit["crop_pixels"] * 3),
            "gt_protected95_rate": audit["gt_protected95"] / audit["gt_fully_in_selected_crop"],
            "gt_lt8_protected95_rate": audit["gt_lt8_protected95"] / audit["gt_lt8_fully_in_crop"]
                if audit["gt_lt8_fully_in_crop"] else None,
        },
        "wall_seconds_all_conditions": time.perf_counter() - started,
        "protocol": {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
                     "candidate_nms_multi_label": True, "grid": "4x4",
                     "crop_halo_fraction_each_side": 0.1, "selected_cells": 4,
                     "fusion_classwise_nms_iou": 0.7,
                     "mask_prediction_conf": PROTECTION_CONF,
                     "mask_margin_fraction_each_side": 0.20,
                     "mask_minimum_margin_at_full960": 8,
                     "blur_sigma_at_crop_input960": BLUR_SIGMA_AT_INPUT960,
                     "feather_sigma_at_crop_input960": FEATHER_SIGMA_AT_INPUT960,
                     "gt_used_in_image_processing": False,
                     "evaluator": "internal converted-label YOLO AP, not official VisDrone"},
        "caveat": "test-dev was historically inspected. This one fixed method uses val GT mask-coverage audit to select geometry, then tests once on test-dev. The test-dev GT is read only after predictions for evaluation. Full+crop uses five 960 model calls per image."
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
