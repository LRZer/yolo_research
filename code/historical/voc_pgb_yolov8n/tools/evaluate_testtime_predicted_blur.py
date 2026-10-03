"""Evaluate predicted-box blur or privileged GT-box Oracle blur on VisDrone.

The selected box source protects image detail before the frozen detector runs.
The ground-truth mode deliberately uses evaluation labels for image processing;
its results are diagnostic only and are not deployable detection metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import nms, ops


STRENGTHS = {"raw": 0.0, "sham": -1.0, "light": 3.0, "medium": 8.0, "heavy": 16.0}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protect_and_blur(
    image: np.ndarray,
    boxes: np.ndarray,
    sigma: float,
    margin_fraction: float = 0.10,
    minimum_margin: int = 8,
    feather_sigma: float = 8.0,
) -> tuple[np.ndarray, float]:
    """Return blurred HWC RGB float image and protected-pixel fraction.

    Coordinates are in the *current detector input* (after LetterBox), not
    original source-image coordinates. Each selected box and optional halo
    remain exactly unchanged. The protection boundary fades into the blur.
    """
    if image.ndim != 3 or image.shape[2] != 3 or sigma <= 0:
        raise ValueError("expected RGB HWC image and positive sigma")
    height, width = image.shape[:2]
    protected = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            continue
        dx = max(minimum_margin, (x2 - x1) * margin_fraction)
        dy = max(minimum_margin, (y2 - y1) * margin_fraction)
        left = max(0, int(math.floor(x1 - dx)))
        right = min(width, int(math.ceil(x2 + dx)))
        top = max(0, int(math.floor(y1 - dy)))
        bottom = min(height, int(math.ceil(y2 + dy)))
        protected[top:bottom, left:right] = 1.0
    softened = cv2.GaussianBlur(protected, (0, 0), sigmaX=feather_sigma)
    softened = np.maximum(softened, protected)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    output = image * softened[..., None] + blurred * (1.0 - softened[..., None])
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0)), float(protected.mean())


def ground_truth_boxes_in_input(batch: dict, index: int) -> torch.Tensor:
    """Convert the validator's normalized xywh GT to current LetterBox xyxy.

    These boxes are privileged information and must never be used for a
    deployable or ordinary benchmark result.
    """
    selected = batch["bboxes"][batch["batch_idx"] == index]
    height, width = batch["img"].shape[-2:]
    scale = selected.new_tensor([width, height, width, height])
    return ops.xywh2xyxy(selected) * scale


class PredictedBackgroundBlurValidator(DetectionValidator):
    def __init__(
        self,
        *args,
        sigma: float,
        first_pass_weights: str,
        box_source: str,
        first_pass_conf: float,
        first_pass_iou: float,
        margin_fraction: float,
        minimum_margin: int,
        feather_sigma: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.first_pass_weights = first_pass_weights
        self.box_source = box_source
        self.first_pass_conf = first_pass_conf
        self.first_pass_iou = first_pass_iou
        self.margin_fraction = margin_fraction
        self.minimum_margin = minimum_margin
        self.feather_sigma = feather_sigma
        self.first_pass_model = None
        self.blur_audit = {
            "images": 0,
            "mask_boxes": 0,
            "images_without_mask_boxes": 0,
            "sum_protected_fraction": 0.0,
            "sum_mean_abs_pixel_change_255": 0.0,
        }

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.sigma == 0.0:
            return batch
        if self.box_source == "predicted" and self.first_pass_model is None:
            self.first_pass_model = YOLO(self.first_pass_weights, task="detect").model.to(self.device).float().eval()
            for parameter in self.first_pass_model.parameters():
                parameter.requires_grad_(False)
        if self.box_source == "predicted":
            with torch.inference_mode():
                raw_predictions = self.first_pass_model(batch["img"])
                detections = nms.non_max_suppression(
                    raw_predictions,
                    conf_thres=self.first_pass_conf,
                    iou_thres=self.first_pass_iou,
                    nc=self.nc,
                    multi_label=True,
                    max_det=self.args.max_det,
                )
            box_tensors = [detection[:, :4] for detection in detections]
        elif self.box_source == "ground_truth":
            box_tensors = [ground_truth_boxes_in_input(batch, i) for i in range(len(batch["img"]))]
        else:
            raise ValueError(f"unknown box source: {self.box_source}")
        processed = []
        for tensor, box_tensor in zip(batch["img"], box_tensors, strict=True):
            image = np.ascontiguousarray(tensor.permute(1, 2, 0).float().cpu().numpy())
            boxes = box_tensor.float().cpu().numpy()
            if self.sigma < 0:
                # Integrity control: build the selected mask and perform the
                # CPU tensor roundtrip, but return numerically unchanged pixels.
                output, protected_fraction = image.copy(), 0.0
            else:
                output, protected_fraction = protect_and_blur(
                    image, boxes, self.sigma, self.margin_fraction, self.minimum_margin, self.feather_sigma
                )
            processed.append(torch.from_numpy(output).permute(2, 0, 1))
            self.blur_audit["images"] += 1
            self.blur_audit["mask_boxes"] += len(boxes)
            self.blur_audit["images_without_mask_boxes"] += int(len(boxes) == 0)
            self.blur_audit["sum_protected_fraction"] += protected_fraction
            self.blur_audit["sum_mean_abs_pixel_change_255"] += float(np.abs(output - image).mean() * 255.0)
        batch["img"] = torch.stack(processed, dim=0).to(self.device, non_blocking=False)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--box-source", choices=("predicted", "ground_truth"), default="predicted")
    parser.add_argument("--conditions", nargs="+", choices=tuple(STRENGTHS), default=list(STRENGTHS))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-limit", type=int, default=None, help="Smoke-test image count; omit for all val images")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--nms-mode", choices=("auto", "false", "true"), default="auto",
                        help="Model head mode; false selects YOLO26 end-to-end O2O")
    parser.add_argument("--first-pass-conf", type=float, default=0.05)
    parser.add_argument("--first-pass-iou", type=float, default=0.7)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-margin", type=int, default=8)
    parser.add_argument("--feather-sigma", type=float, default=8.0)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError("weights or dataset YAML missing")
    if not (0 < args.first_pass_conf < 1) or not (0 < args.first_pass_iou < 1):
        raise ValueError("invalid first-pass threshold")
    if args.val_limit is not None and args.val_limit < 1:
        raise ValueError("val-limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_sha = file_sha256(args.weights)
    data_sha = file_sha256(args.data)
    for condition in args.conditions:
        result_path = args.output_dir / f"{condition}.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite completed result: {result_path}")
        sigma = STRENGTHS[condition]
        instances = []

        def validator_factory(**kwargs):
            validator = PredictedBackgroundBlurValidator(
                sigma=sigma,
                first_pass_weights=str(args.weights),
                box_source=args.box_source,
                first_pass_conf=args.first_pass_conf,
                first_pass_iou=args.first_pass_iou,
                margin_fraction=args.margin_fraction,
                minimum_margin=args.minimum_margin,
                feather_sigma=args.feather_sigma,
                **kwargs,
            )
            instances.append(validator)
            return validator

        model = YOLO(str(args.weights), task="detect")
        started = time.perf_counter()
        validation_kwargs = {"fraction": [1.0, args.val_limit, 1.0]} if args.val_limit is not None else {}
        if args.nms_mode != "auto":
            validation_kwargs["nms"] = args.nms_mode == "true"
        metrics = model.val(
            validator=validator_factory,
            data=str(args.data), split="val", imgsz=args.imgsz,
            batch=args.batch, workers=args.workers, device=args.device,
            conf=0.001, iou=0.7, max_det=args.max_det,
            half=False, rect=True, plots=False, save_json=False, save_txt=False,
            project=str(args.output_dir / "validator_runs"), name=condition,
            exist_ok=False, verbose=False,
            **validation_kwargs,
        )
        audit = instances[0].blur_audit
        if sigma != 0 and audit["images"]:
            audit["mean_mask_boxes_per_image"] = audit["mask_boxes"] / audit["images"]
            audit["mean_protected_fraction"] = audit["sum_protected_fraction"] / audit["images"]
            audit["mean_abs_pixel_change_255"] = audit["sum_mean_abs_pixel_change_255"] / audit["images"]
        box = metrics.box
        ids = [int(i) for i in box.ap_class_index]
        classes = [
            {"id": idx, "name": metrics.names[idx], "AP50": float(box.ap50[pos]), "AP50_95": float(box.ap[pos])}
            for pos, idx in enumerate(ids)
        ]
        result = {
            "status": "completed",
            "generated": datetime.now(timezone.utc).astimezone().isoformat(),
            "condition": condition,
            "box_source": args.box_source,
            "val_limit": args.val_limit,
            "sigma_input_pixels": sigma,
            "first_pass_uses_predictions_only": condition != "raw" and args.box_source == "predicted",
            "ground_truth_used_for_image_processing": condition != "raw" and args.box_source == "ground_truth",
            "weights": str(args.weights),
            "weights_sha256": weight_sha,
            "data": str(args.data),
            "data_sha256": data_sha,
            "validation": {"imgsz": args.imgsz, "batch": args.batch, "workers": args.workers, "conf": 0.001,
                           "iou": 0.7, "max_det": args.max_det, "half": False, "rect": True,
                           "nms_mode": args.nms_mode},
            "blur": {"first_pass_conf": args.first_pass_conf, "first_pass_iou": args.first_pass_iou,
                     "margin_fraction": args.margin_fraction, "minimum_margin": args.minimum_margin,
                     "feather_sigma": args.feather_sigma, "audit": audit},
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "classes": classes,
            "elapsed_wall_seconds": time.perf_counter() - started,
            "validator_speed_ms_per_image": metrics.speed,
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(result_path)
        print(json.dumps({"condition": condition, "mAP50_95": box.map, "elapsed": result["elapsed_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
