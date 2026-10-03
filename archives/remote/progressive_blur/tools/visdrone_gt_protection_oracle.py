"""Matched, frozen-detector diagnostic of predicted vs GT-protected crop blur.

The GT-protected condition intentionally leaks test annotations into its input.
It is an oracle upper-bound diagnostic and cannot be deployed or reported as AP
of an autonomous detector.
"""

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
from visdrone_predicted_background_crop_blur import (
    PROTECTION_CONF, predicted_background_blur, remap,
)
from visdrone_predicted_protection_audit import (
    gt_protected_fraction, protection_mask, size_bin,
)


METHODS = ("full", "full_raw4", "full_predicted_background_blur4",
           "full_gt_oracle_background_blur4")


def self_test() -> None:
    rng = np.random.default_rng(2027)
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    boxes = np.array([[30, 20, 45, 35], [90, 70, 105, 90]], np.float32)
    mask = protection_mask(boxes, np.ones(len(boxes), np.float32), 160, 120,
                           PROTECTION_CONF)
    for box in boxes:
        assert gt_protected_fraction(mask, box) == 1.0
    output = predicted_background_blur(image, mask)
    assert np.array_equal(output[mask], image[mask])
    assert np.any(output[~mask] != image[~mask])
    assert np.array_equal(predicted_background_blur(image, np.ones((120, 160), bool)), image)
    base.self_test()
    print("GT_PROTECTION_ORACLE_SELF_TEST_OK", flush=True)


def predict_crops(model, images: list[np.ndarray], args):
    results = model.predict(source=images, imgsz=960, batch=4, conf=0.001,
                            iou=0.7, max_det=1000, half=False, device=args.device,
                            verbose=False, save=False, stream=False)
    if len(results) != 4:
        raise RuntimeError("crop prediction count mismatch")
    return results


def audit_mask(counter: Counter, masks: list[np.ndarray],
               originals: list[np.ndarray], processed: list[np.ndarray],
               full_mask: np.ndarray, gt_boxes: np.ndarray,
               rects: list[tuple[int, int, int, int]], width: int, height: int):
    for raw_crop, new_crop, mask in zip(originals, processed, masks):
        counter["crop_pixels"] += int(mask.size)
        counter["protected_pixels"] += int(mask.sum())
        counter["abs_channel_delta_255"] += int(np.abs(
            raw_crop.astype(np.int16) - new_crop.astype(np.int16)).sum())
    for box in gt_boxes:
        if not any(x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1
                   for x0, y0, x1, y1 in rects):
            continue
        counter["eligible_gt"] += 1
        protected = gt_protected_fraction(full_mask, box) >= 0.95
        counter["protected95_gt"] += int(protected)
        if size_bin(box, width, height) == "lt8":
            counter["eligible_lt8_gt"] += 1
            counter["protected95_lt8_gt"] += int(protected)


def audit_summary(counter: Counter) -> dict:
    pixels = counter["crop_pixels"]
    eligible = counter["eligible_gt"]
    small = counter["eligible_lt8_gt"]
    return {
        **dict(counter),
        "protected_fraction_selected_crops": counter["protected_pixels"] / pixels,
        "blur_permitted_fraction_selected_crops": 1 - counter["protected_pixels"] / pixels,
        "mean_abs_pixel_delta_255": counter["abs_channel_delta_255"] / (3 * pixels),
        "gt_protected95_rate": counter["protected95_gt"] / eligible,
        "gt_lt8_protected95_rate": counter["protected95_lt8_gt"] / small if small else None,
    }


def run(args) -> None:
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("checkpoint or manifest missing")
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output exists or output parent missing")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid image manifest")
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10 classes")
    names = dict(model.names)
    stats = {key: [] for key in METHODS}
    fixed = {key: {str(names[i]): Counter() for i in range(10)} for key in METHODS}
    audit = {key: Counter() for key in ("predicted", "gt_oracle")}
    start = time.perf_counter()

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
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"image decode/size mismatch: {path}")
            crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
            predicted_mask = protection_mask(full_pred[0], full_pred[2],
                                             width, height, PROTECTION_CONF)

            # This read is the sole difference between predicted protection and
            # non-deployable oracle protection. It cannot affect crop selection.
            gt_classes, gt_boxes = base.gt_for_image(path, width, height)
            oracle_mask = protection_mask(gt_boxes,
                                          np.ones(len(gt_boxes), np.float32),
                                          width, height, PROTECTION_CONF)
            masks = {"predicted": predicted_mask, "gt_oracle": oracle_mask}
            processed = {}
            for key, full_mask in masks.items():
                crop_masks = [full_mask[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
                processed[key] = [predicted_background_blur(crop, mask)
                                  for crop, mask in zip(crops, crop_masks)]
                audit_mask(audit[key], crop_masks, crops, processed[key],
                           full_mask, gt_boxes, rects, width, height)
            audit["predicted"]["mask_boxes"] += int(np.count_nonzero(
                full_pred[2] >= PROTECTION_CONF))
            audit["gt_oracle"]["mask_boxes"] += len(gt_boxes)

            result_groups = {}
            for key, images in (("raw", crops), ("predicted", processed["predicted"]),
                                ("gt_oracle", processed["gt_oracle"])):
                crop_results = predict_crops(model, images, args)
                result_groups[key] = [full_pred] + [remap(r, rect, width, height)
                                                    for r, rect in zip(crop_results, rects)]
            predictions = {
                "full": full_pred,
                "full_raw4": base.merge_predictions(result_groups["raw"]),
                "full_predicted_background_blur4": base.merge_predictions(
                    result_groups["predicted"]),
                "full_gt_oracle_background_blur4": base.merge_predictions(
                    result_groups["gt_oracle"]),
            }
            for key, (boxes, classes, scores) in predictions.items():
                stats[key].append((base.match_tp(boxes, classes, gt_boxes, gt_classes),
                                   scores, classes, gt_classes))
                for class_id in range(10):
                    counter = fixed[key][str(names[class_id])]
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
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": base.sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base.sha256(args.manifest),
        "images": len(paths),
        "conditions": {key: base.evaluate(rows, names) for key, rows in stats.items()},
        "fixed_conf_025_iou_05": fixed_report,
        "mask_audit": {key: audit_summary(row) for key, row in audit.items()},
        "wall_seconds_all_conditions": time.perf_counter() - start,
        "protocol": {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
                     "candidate_nms_multi_label": True,
                     "selected_cells": 4, "crop_grid": "4x4",
                     "crop_halo_fraction_each_side": 0.1,
                     "fusion_classwise_nms_iou": 0.7,
                     "mask_prediction_conf": PROTECTION_CONF,
                     "mask_margin_fraction_each_side": 0.20,
                     "mask_minimum_margin_at_full960": 8,
                     "blur_sigma_at_crop_input960": 3.0,
                     "feather_sigma_at_crop_input960": 8.0,
                     "gt_used_in_oracle_mask_only": True,
                     "evaluator": "internal converted-label YOLO AP, not official VisDrone"},
        "caveat": "GT oracle leaks evaluation annotations into local image preprocessing. It is not a deployable detector or valid standalone AP claim. val548 has been repeatedly inspected; one fixed diagnostic only.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)


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
    else:
        run(args)


if __name__ == "__main__":
    main()
