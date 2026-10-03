"""One pre-registered frozen-model test of scale-gated crop fusion.

The original foveated pilot remains sealed; this imports its exact selection,
prediction parsing, merger and evaluator. Ground truth enters only after all
three candidate lists have been finalized.
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


SMALL_MAX_SIDE_AT_FULL_960 = 64.0


def small_only(group, width: int, height: int):
    """Keep crop boxes whose longest side at full-image 960 scale is <=64."""
    boxes, classes, scores = group
    scale = 960.0 / max(width, height)
    longest = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * scale
    keep = longest <= SMALL_MAX_SIDE_AT_FULL_960
    return (boxes[keep], classes[keep], scores[keep])


def self_test():
    boxes = np.array([[10, 10, 50, 50], [10, 10, 100, 50]], np.float32)
    classes = np.array([0, 8], np.int64)
    scores = np.array([0.8, 0.7], np.float32)
    filtered = small_only((boxes, classes, scores), 960, 540)
    assert len(filtered[0]) == 1 and int(filtered[1][0]) == 0
    filtered = small_only((boxes, classes, scores), 1920, 1080)
    assert len(filtered[0]) == 2
    assert base.top_cells(np.array([[140, 140, 160, 160]], np.float32),
                          np.array([0.9], np.float32), 400, 400)[0] == 5
    base.self_test()
    print("SMALL_FUSION_SELF_TEST_OK", flush=True)


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
        return
    if not args.checkpoint or not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.manifest or not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.output or args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("output exists or parent absent")
    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths or len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("invalid manifest")
    model = YOLO(str(args.checkpoint))
    if model.model.nc != 10:
        raise ValueError("expected 10 classes")
    names = dict(model.names)
    stats = {key: [] for key in ("full", "full_adaptive4", "full_adaptive4_small_only")}
    counts = Counter()
    started = time.perf_counter()
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset + 8]
        base_results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                     batch=8, conf=0.001, iou=0.7, max_det=1000,
                                     half=False, device=args.device, verbose=False,
                                     save=False, stream=False)
        if len(base_results) != len(batch_paths):
            raise RuntimeError("base batch mismatch")
        for image_path, base_result in zip(batch_paths, base_results):
            if Path(base_result.path).resolve() != image_path.resolve():
                raise RuntimeError("base ordering mismatch")
            height, width = base_result.orig_shape
            whole = base.unpack(base_result)
            cells = base.top_cells(whole[0], whole[2], width, height)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError("image decode/size mismatch")
            rects = [base.crop_rect(cell, width, height) for cell in cells]
            crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
            crop_results = model.predict(source=crops, imgsz=960, batch=4,
                                         conf=0.001, iou=0.7, max_det=1000,
                                         half=False, device=args.device,
                                         verbose=False, save=False, stream=False)
            if len(crop_results) != 4:
                raise RuntimeError("crop batch mismatch")
            groups = [whole]
            small_groups = [whole]
            for result, (x0, y0, x1, y1) in zip(crop_results, rects):
                boxes, classes, scores = base.unpack(result)
                if len(boxes):
                    boxes[:, (0, 2)] += x0
                    boxes[:, (1, 3)] += y0
                    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
                    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
                group = (boxes, classes, scores)
                filtered = small_only(group, width, height)
                groups.append(group)
                small_groups.append(filtered)
                counts["crop_candidates_before"] += len(boxes)
                counts["crop_candidates_kept"] += len(filtered[0])
                for class_id in np.unique(classes):
                    counts[f"class_{int(class_id)}_before"] += int(np.count_nonzero(classes == class_id))
                for class_id in np.unique(filtered[1]):
                    counts[f"class_{int(class_id)}_kept"] += int(np.count_nonzero(filtered[1] == class_id))
            predictions = {
                "full": whole,
                "full_adaptive4": base.merge_predictions(groups),
                "full_adaptive4_small_only": base.merge_predictions(small_groups),
            }
            gt_classes, gt_boxes = base.gt_for_image(image_path, width, height)
            for key, (pred_boxes, pred_classes, scores) in predictions.items():
                tp = base.match_tp(pred_boxes, pred_classes, gt_boxes, gt_classes)
                stats[key].append((tp, scores, pred_classes, gt_classes))
        print(f"evaluated {min(offset + 8, len(paths))}/{len(paths)}", flush=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": base.sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base.sha256(args.manifest),
        "images": len(paths),
        "conditions": {key: base.evaluate(rows, names) for key, rows in stats.items()},
        "candidate_counts": dict(counts),
        "wall_seconds_including_load": time.perf_counter() - started,
        "protocol": {
            "imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
            "crop_grid": "4x4", "crop_halo_fraction_each_side": 0.1,
            "selected_cells": 4, "fusion_nms_classwise_iou": 0.7,
            "candidate_nms_multi_label": True,
            "small_crop_candidate_max_side_at_full960": SMALL_MAX_SIDE_AT_FULL_960,
            "full_candidates_unfiltered": True,
            "gt_used_in_selection_or_filter": False,
            "evaluator": "internal Ultralytics ap_per_class on converted labels; not official VisDrone",
        },
        "caveat": "test-dev was historically inspected by earlier projects; this is not a blind result. All crop conditions invoke five 960 model inputs per image."
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
