"""CPU-only diagnostic of VisDrone prediction confidence and false positives.

This uses task-owned YOLO labels rather than the official VisDrone ignore
protocol. Results are diagnostic TP/FP operating points, not official AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_visdrone_size_recall import iou_matrix, read_yolo_labels


IOU_THRESHOLDS = (0.5, 0.75)
SCORE_THRESHOLDS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_predictions(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-greedy, one-to-one classwise matching per image.

    Return prediction TP flags and matched GT flags. No official ignored
    regions, difficult handling or COCO max-detection averaging is applied.
    """
    tp = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    for cls in range(10):
        pred_idx = np.flatnonzero(pred_classes == cls)
        gt_idx = np.flatnonzero(gt_classes == cls)
        if not len(pred_idx) or not len(gt_idx):
            continue
        sorted_pred = pred_idx[np.argsort(-pred_scores[pred_idx], kind="stable")]
        ious = iou_matrix(pred_boxes[sorted_pred], gt_boxes[gt_idx])
        claimed = np.zeros(len(gt_idx), dtype=bool)
        for row_number, row in enumerate(ious):
            available = np.where(~claimed, row, -1.0)
            local_gt = int(np.argmax(available))
            if available[local_gt] >= iou_threshold:
                claimed[local_gt] = True
                tp[sorted_pred[row_number]] = True
                gt_matched[gt_idx[local_gt]] = True
    return tp, gt_matched


def add_counts(
    counts: dict[str, int],
    pred_scores: np.ndarray,
    tp: np.ndarray,
    gt_matched: np.ndarray,
    buckets: list[str],
    threshold: float,
) -> None:
    selected = pred_scores >= threshold
    counts["predictions"] += int(selected.sum())
    counts["true_positives"] += int((tp & selected).sum())
    counts["false_positives"] += int((~tp & selected).sum())
    # The matching was performed on all predictions, in confidence order.
    # For a stricter threshold, every previously matched GT whose own
    # prediction still passes remains a valid match.
    if len(gt_matched):
        counts["ground_truth"] += len(gt_matched)
    for bucket in ("tiny_lt16", "small_16to32", "medium_32to96", "large_ge96"):
        counts[f"gt_{bucket}"] += sum(value == bucket for value in buckets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    from ultralytics import YOLO

    image_dir = args.view / "images" / "val"
    label_dir = args.view / "labels" / "val"
    expected = {path.stem for path in image_dir.glob("*.jpg")}
    if len(expected) != 548:
        raise ValueError(f"expected 548 val images, found {len(expected)}")
    if not args.weights.is_file():
        raise ValueError(f"missing checkpoint: {args.weights}")

    aggregate = {
        str(iou): {str(score): defaultdict(int) for score in SCORE_THRESHOLDS}
        for iou in IOU_THRESHOLDS
    }
    seen: set[str] = set()
    all_gt = 0
    all_predictions = 0
    per_image = []
    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(image_dir),
        stream=True,
        batch=4,
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        device="cpu",
        half=False,
        verbose=False,
        save=False,
        project=str(args.output.parent),
        name=f"cpu_predict_{args.imgsz}",
        exist_ok=True,
    )
    for result in results:
        image_path = Path(result.path)
        stem = image_path.stem
        if stem not in expected or stem in seen:
            raise ValueError(f"unexpected or repeated image result: {image_path}")
        seen.add(stem)
        with Image.open(image_path) as image:
            width, height = image.size
        gt_classes, gt_boxes, buckets = read_yolo_labels(label_dir / f"{stem}.txt", width, height)
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        pred_scores = result.boxes.conf.cpu().numpy()
        if not np.isfinite(pred_boxes).all() or not np.isfinite(pred_scores).all():
            raise ValueError(f"non-finite prediction: {stem}")
        if np.any((pred_classes < 0) | (pred_classes >= 10)):
            raise ValueError(f"class out of range: {stem}")
        all_gt += len(gt_boxes)
        all_predictions += len(pred_boxes)
        image_row = {"stem": stem, "ground_truth": len(gt_boxes), "predictions": len(pred_boxes)}
        for iou_threshold in IOU_THRESHOLDS:
            tp, gt_matched = classify_predictions(
                pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_threshold
            )
            if int(tp.sum()) != int(gt_matched.sum()):
                raise AssertionError("TP and matched GT counts disagree")
            for score_threshold in SCORE_THRESHOLDS:
                row = aggregate[str(iou_threshold)][str(score_threshold)]
                add_counts(row, pred_scores, tp, gt_matched, buckets, score_threshold)
            image_row[f"tp_at_{iou_threshold}"] = int(tp.sum())
        per_image.append(image_row)
    if seen != expected or all_gt != 38759:
        raise ValueError(f"incomplete val inference: images={len(seen)}, GT={all_gt}")

    summary = {}
    for iou_threshold, score_rows in aggregate.items():
        summary[iou_threshold] = {}
        for score_threshold, counts in score_rows.items():
            total = counts["predictions"]
            ground_truth = counts["ground_truth"]
            summary[iou_threshold][score_threshold] = {
                **dict(counts),
                "precision": counts["true_positives"] / total if total else None,
                "recall": counts["true_positives"] / ground_truth if ground_truth else None,
            }
    payload = {
        "status": "cpu_only_diagnostic_prediction_quality_not_official_AP",
        "weights": str(args.weights.resolve()),
        "weights_sha256": file_sha256(args.weights),
        "view": str(args.view.resolve()),
        "split": "val",
        "imgsz": args.imgsz,
        "conf_floor": 0.001,
        "nms_iou": 0.7,
        "max_det": 1000,
        "device": "cpu",
        "images": len(seen),
        "ground_truth": all_gt,
        "predictions": all_predictions,
        "operating_points": summary,
        "per_image": per_image,
        "caveat": "Converted YOLO labels omit VisDrone ignore/other. These classwise greedy TP/FP rates are not official challenge AP or identical to Ultralytics mAP computation. CPU model.predict preprocessing may differ from model.val; compare only paired runs with this same script.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_image"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
