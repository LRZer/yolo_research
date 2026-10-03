"""Read-only mask feasibility audit for GT-free target-protected crop blur."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

import visdrone_foveated_crop_pilot as base


CONF_LEVELS = (0.001, 0.01, 0.05)
MARGIN_FRACTION = 0.20
MIN_MARGIN_AT_INPUT960 = 8.0
SIZE_BINS = ("lt8", "8to16", "16to32", "ge32")


def protection_mask(boxes: np.ndarray, scores: np.ndarray, width: int, height: int,
                    minimum_conf: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    minimum_source_margin = MIN_MARGIN_AT_INPUT960 * max(width, height) / 960.0
    for box in boxes[scores >= minimum_conf]:
        x0, y0, x1, y1 = map(float, box)
        if not np.isfinite(box).all() or x1 <= x0 or y1 <= y0:
            continue
        mx = max(minimum_source_margin, (x1 - x0) * MARGIN_FRACTION)
        my = max(minimum_source_margin, (y1 - y0) * MARGIN_FRACTION)
        left = max(0, math.floor(x0 - mx))
        top = max(0, math.floor(y0 - my))
        right = min(width, math.ceil(x1 + mx))
        bottom = min(height, math.ceil(y1 + my))
        mask[top:bottom, left:right] = True
    return mask


def size_bin(box: np.ndarray, width: int, height: int) -> str:
    bw = max(0.0, float(box[2] - box[0]))
    bh = max(0.0, float(box[3] - box[1]))
    side = math.sqrt(bw * bh) * 960.0 / max(width, height)
    if side < 8:
        return "lt8"
    if side < 16:
        return "8to16"
    if side < 32:
        return "16to32"
    return "ge32"


def gt_protected_fraction(mask: np.ndarray, box: np.ndarray) -> float:
    height, width = mask.shape
    x0 = min(width, max(0, math.floor(float(box[0]))))
    y0 = min(height, max(0, math.floor(float(box[1]))))
    x1 = min(width, max(0, math.ceil(float(box[2]))))
    y1 = min(height, max(0, math.ceil(float(box[3]))))
    return float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0


def self_test():
    boxes = np.array([[100, 100, 120, 120], [400, 400, 420, 420]], np.float32)
    scores = np.array([0.06, 0.005], np.float32)
    mask = protection_mask(boxes, scores, 960, 960, 0.01)
    assert mask[110, 110] and not mask[410, 410]
    assert gt_protected_fraction(mask, boxes[0]) == 1.0
    assert size_bin(np.array([0, 0, 5, 5], np.float32), 960, 960) == "lt8"
    assert size_bin(np.array([0, 0, 40, 40], np.float32), 960, 960) == "ge32"
    base.self_test()
    print("PROTECTION_AUDIT_SELF_TEST_OK", flush=True)


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
    accum = {str(thr): {"gt": Counter(), "size": {b: Counter() for b in SIZE_BINS},
                        "class": {str(i): Counter() for i in range(10)},
                        "selected_crop_pixels": 0, "protected_crop_pixels": 0,
                        "mask_boxes": 0} for thr in CONF_LEVELS}
    total_gt = 0
    eligible_gt = 0
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset + 8]
        results = model.predict(source=[str(path) for path in batch_paths], imgsz=960,
                                batch=8, conf=0.001, iou=0.7, max_det=1000,
                                half=False, device=args.device, verbose=False,
                                save=False, stream=False)
        if len(results) != len(batch_paths):
            raise RuntimeError("batch mismatch")
        for path, result in zip(batch_paths, results):
            if Path(result.path).resolve() != path.resolve():
                raise RuntimeError("image ordering mismatch")
            height, width = result.orig_shape
            boxes, _classes, scores = base.unpack(result)
            cells = base.top_cells(boxes, scores, width, height)
            rects = [base.crop_rect(cell, width, height) for cell in cells]
            masks = {str(thr): protection_mask(boxes, scores, width, height, thr)
                     for thr in CONF_LEVELS}
            # Only now read the true boxes; they never select cells or masks.
            gt_classes, gt_boxes = base.gt_for_image(path, width, height)
            total_gt += len(gt_boxes)
            eligible = [i for i, box in enumerate(gt_boxes) if any(
                x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1
                for x0, y0, x1, y1 in rects
            )]
            eligible_gt += len(eligible)
            for thr in CONF_LEVELS:
                key = str(thr)
                data = accum[key]
                mask = masks[key]
                data["mask_boxes"] += int(np.count_nonzero(scores >= thr))
                for x0, y0, x1, y1 in rects:
                    selected = mask[y0:y1, x0:x1]
                    data["selected_crop_pixels"] += selected.size
                    data["protected_crop_pixels"] += int(selected.sum())
                for i in eligible:
                    box = gt_boxes[i]
                    bin_name = size_bin(box, width, height)
                    cls_name = str(int(gt_classes[i]))
                    fraction = gt_protected_fraction(mask, box)
                    safe = fraction >= 0.95
                    data["gt"]["eligible"] += 1
                    data["gt"]["protected95"] += int(safe)
                    data["gt"]["coverage_fraction_sum"] += fraction
                    data["size"][bin_name]["eligible"] += 1
                    data["size"][bin_name]["protected95"] += int(safe)
                    data["class"][cls_name]["eligible"] += 1
                    data["class"][cls_name]["protected95"] += int(safe)
        print(f"audited {min(offset + 8, len(paths))}/{len(paths)}", flush=True)
    summary = {}
    for key, data in accum.items():
        eligible = data["gt"]["eligible"]
        summary[key] = {
            "mean_mask_boxes_per_image": data["mask_boxes"] / len(paths),
            "protected_fraction_selected_crop_pixels": data["protected_crop_pixels"] / data["selected_crop_pixels"],
            "blur_permitted_fraction_selected_crop_pixels": 1 - data["protected_crop_pixels"] / data["selected_crop_pixels"],
            "eligible_gt": eligible,
            "protected95_gt": data["gt"]["protected95"],
            "protected95_rate": data["gt"]["protected95"] / eligible if eligible else None,
            "mean_gt_box_coverage": data["gt"]["coverage_fraction_sum"] / eligible if eligible else None,
            "size": {b: {"eligible": row["eligible"], "protected95": row["protected95"],
                         "protected95_rate": row["protected95"] / row["eligible"] if row["eligible"] else None}
                     for b, row in data["size"].items()},
            "class": {c: {"eligible": row["eligible"], "protected95": row["protected95"],
                          "protected95_rate": row["protected95"] / row["eligible"] if row["eligible"] else None}
                      for c, row in data["class"].items()},
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": base.sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base.sha256(args.manifest),
        "images": len(paths), "all_gt": total_gt, "eligible_gt": eligible_gt,
        "confidence_levels": list(CONF_LEVELS), "summary": summary,
        "protocol": {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 1000,
                     "candidate_nms_multi_label": True, "selected_cells": 4,
                     "crop_grid": "4x4", "halo_fraction_each_side": 0.1,
                     "margin_fraction_each_side": MARGIN_FRACTION,
                     "minimum_margin_at_full960": MIN_MARGIN_AT_INPUT960,
                     "gt_used_in_mask_or_selection": False,
                     "eligibility": "GT fully contained in any selected crop",
                     "safe": "at least 95 percent of integer-pixel GT box lies inside union predicted protection mask",
                     "size": "sqrt GT area at full960 letterbox scale; thresholds 8,16,32"},
        "caveat": "This audits a necessary coverage/area tradeoff only. No image is blurred and no new detection AP is measured. Full detector predictions are max_det1000 limited. Converted YOLO labels may omit ignored objects; val has been repeatedly inspected.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)


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
