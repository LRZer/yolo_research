"""Frozen-model source/provenance diagnostic for the existing adaptive crop pilot.

Runs the same full+prediction-selected four 960 crops as the sealed pilot,
without changing its result. GT is read only after selection and predictions.
This is a fixed-conf diagnostic, not a new AP evaluation or parameter sweep.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS

from visdrone_foveated_crop_pilot import (
    crop_rect, gt_for_image, merge_predictions, sha256, top_cells, unpack,
)


TARGET_CLASSES = {0: "pedestrian", 1: "people", 5: "truck", 8: "bus"}
MATCH_IOU = 0.5
MATCH_CONF = 0.25


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=2)
    a_area = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    b_area = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return inter / np.maximum(a_area[:, None] + b_area[None, :] - inter, 1e-7)


def matched_gt_to_pred(
    boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
    gt_boxes: np.ndarray, gt_classes: np.ndarray, class_id: int,
) -> dict[int, int]:
    pred_ids = np.flatnonzero((classes == class_id) & (scores >= MATCH_CONF))
    gt_ids = np.flatnonzero(gt_classes == class_id)
    if not len(pred_ids) or not len(gt_ids):
        return {}
    overlaps = iou_matrix(gt_boxes[gt_ids], boxes[pred_ids])
    pairs = np.array(np.nonzero(overlaps >= MATCH_IOU)).T
    if not len(pairs):
        return {}
    # Same IoU-first, unique-pred then unique-GT matcher as the sealed pilot.
    pairs = pairs[overlaps[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
    pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
    pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return {int(gt_ids[g]): int(pred_ids[p]) for g, p in pairs}


def merge_with_provenance(groups):
    boxes = np.concatenate([group[0] for group in groups])
    classes = np.concatenate([group[1] for group in groups])
    scores = np.concatenate([group[2] for group in groups])
    sources = np.concatenate([
        np.full(len(group[0]), source, dtype=np.int16)
        for source, group in enumerate(groups)
    ])
    keep = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        local_keep = TorchNMS.nms(
            torch.from_numpy(boxes[indices]), torch.from_numpy(scores[indices]), 0.7
        ).numpy()
        keep.extend(indices[local_keep].tolist())
    ranked = sorted(keep, key=lambda idx: (-float(scores[idx]), int(idx)))
    final = np.asarray(ranked[:1000], dtype=np.int64)
    return {
        "boxes": boxes[final], "classes": classes[final],
        "scores": scores[final], "sources": sources[final],
        "precap_keep": set(ranked), "final_keep": set(final.tolist()),
        "all_boxes": boxes, "all_classes": classes,
        "all_scores": scores, "all_sources": sources,
    }


def full_contained_by_selected(gt_box: np.ndarray, rects) -> bool:
    return any(
        x0 <= gt_box[0] and y0 <= gt_box[1]
        and gt_box[2] <= x1 and gt_box[3] <= y1
        for x0, y0, x1, y1 in rects
    )


def loss_reason(full_pred_index: int, merged) -> str:
    if full_pred_index in merged["final_keep"]:
        return "full_box_still_kept_match_conflict"
    if full_pred_index in merged["precap_keep"]:
        return "global_1000_cap"
    box = merged["all_boxes"][full_pred_index:full_pred_index + 1]
    same_class_kept = [
        idx for idx in merged["precap_keep"]
        if merged["all_classes"][idx] == merged["all_classes"][full_pred_index]
        and merged["all_scores"][idx] >= merged["all_scores"][full_pred_index]
    ]
    if not same_class_kept:
        return "nms_unattributed"
    overlaps = iou_matrix(box, merged["all_boxes"][same_class_kept])[0]
    candidates = np.flatnonzero(overlaps > 0.7)
    if not len(candidates):
        return "nms_unattributed"
    suppressor = same_class_kept[int(candidates[np.argmax(
        merged["all_scores"][np.asarray(same_class_kept)[candidates]]
    )])]
    return "nms_by_crop" if merged["all_sources"][suppressor] else "nms_by_full"


def run(args) -> dict:
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
        raise ValueError("expected 10-class checkpoint")
    counters = {name: Counter() for name in TARGET_CLASSES.values()}
    examples = {name: [] for name in TARGET_CLASSES.values()}
    checked_equal = 0
    for offset in range(0, len(paths), 8):
        batch_paths = paths[offset:offset + 8]
        base_results = model.predict(
            source=[str(path) for path in batch_paths], imgsz=960, batch=8,
            conf=0.001, iou=0.7, max_det=1000, half=False,
            device=args.device, verbose=False, save=False, stream=False,
        )
        if len(base_results) != len(batch_paths):
            raise RuntimeError("base count mismatch")
        for image_path, base_result in zip(batch_paths, base_results):
            if Path(base_result.path).resolve() != image_path.resolve():
                raise RuntimeError("base order mismatch")
            height, width = base_result.orig_shape
            base = unpack(base_result)
            selected = top_cells(base[0], base[2], width, height)
            rects = [crop_rect(cell, width, height) for cell in selected]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"decode mismatch: {image_path}")
            crop_results = model.predict(
                source=[image[y0:y1, x0:x1] for x0, y0, x1, y1 in rects],
                imgsz=960, batch=4, conf=0.001, iou=0.7, max_det=1000,
                half=False, device=args.device, verbose=False, save=False,
                stream=False,
            )
            if len(crop_results) != 4:
                raise RuntimeError("crop count mismatch")
            groups = [base]
            for result, (x0, y0, _x1, _y1) in zip(crop_results, rects):
                boxes, classes, scores = unpack(result)
                if len(boxes):
                    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)] + x0, 0, width)
                    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)] + y0, 0, height)
                groups.append((boxes, classes, scores))
            merged = merge_with_provenance(groups)
            if checked_equal < 16:
                reference = merge_predictions(groups)
                for key, result in zip(("boxes", "classes", "scores"), reference):
                    if not np.array_equal(merged[key], result):
                        raise RuntimeError(f"fusion differs from sealed pilot: {key}")
                checked_equal += 1
            gt_classes, gt_boxes = gt_for_image(image_path, width, height)
            for class_id, name in TARGET_CLASSES.items():
                counter = counters[name]
                gt_count = int(np.count_nonzero(gt_classes == class_id))
                counter["gt"] += gt_count
                full_matches = matched_gt_to_pred(*base, gt_boxes, gt_classes, class_id)
                fused_matches = matched_gt_to_pred(
                    merged["boxes"], merged["classes"], merged["scores"],
                    gt_boxes, gt_classes, class_id,
                )
                full_gt, fused_gt = set(full_matches), set(fused_matches)
                counter["full_tp"] += len(full_gt)
                counter["fused_tp"] += len(fused_gt)
                counter["lost_gt"] += len(full_gt - fused_gt)
                counter["gained_gt"] += len(fused_gt - full_gt)
                counter["stable_tp"] += len(full_gt & fused_gt)
                counter["stable_fn"] += gt_count - len(full_gt | fused_gt)
                counter["full_preds_at_conf"] += int(np.count_nonzero(
                    (base[1] == class_id) & (base[2] >= MATCH_CONF)
                ))
                counter["fused_preds_at_conf"] += int(np.count_nonzero(
                    (merged["classes"] == class_id) & (merged["scores"] >= MATCH_CONF)
                ))
                for gt_idx in full_gt - fused_gt:
                    reason = loss_reason(full_matches[gt_idx], merged)
                    counter[f"lost_{reason}"] += 1
                    contained = full_contained_by_selected(gt_boxes[gt_idx], rects)
                    counter["lost_fully_in_selected_crop"] += int(contained)
                    if len(examples[name]) < 12:
                        examples[name].append({
                            "image": str(image_path), "gt_xyxy": gt_boxes[gt_idx].tolist(),
                            "selected_cells": list(selected), "reason": reason,
                            "gt_fully_in_selected_crop": contained,
                        })
                for gt_idx in fused_gt - full_gt:
                    source = int(merged["sources"][fused_matches[gt_idx]])
                    counter["gained_by_crop" if source else "gained_by_full"] += 1
    summary = {}
    for name, counter in counters.items():
        counter["full_fp_at_conf"] = counter["full_preds_at_conf"] - counter["full_tp"]
        counter["fused_fp_at_conf"] = counter["fused_preds_at_conf"] - counter["fused_tp"]
        summary[name] = dict(counter)
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "images": len(paths), "reference_merge_exactly_equal_first_images": checked_equal,
        "fixed_diagnostic": {"min_conf": MATCH_CONF, "iou": MATCH_IOU,
                             "classes": TARGET_CLASSES},
        "summary": summary, "lost_examples_first_12_per_class": examples,
        "caveat": "One-to-one IoU-first matching at fixed confidence is a diagnostic, not AP or the entire PR curve. Lost reason is associated with the original matched full prediction; a different fused prediction may still cover an object at lower confidence. The experiment reuses the sealed pilot's selector and fusion without GT in candidate selection.",
    }


def self_test() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[1, 1, 9, 9]], dtype=np.float32)
    assert abs(float(iou_matrix(a, b)[0, 0]) - 0.64) < 1e-6
    cls = np.array([8], dtype=np.int64)
    scores = np.array([0.9], dtype=np.float32)
    assert matched_gt_to_pred(a, cls, scores, b, cls, 8) == {0: 0}
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.checkpoint, args.manifest, args.output)):
        parser.error("checkpoint, manifest and output required")
    report = run(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": report["images"], "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
