"""Save exact YOLO26/VOC validator-input GT Oracle blur previews.

This is a visualization of a deliberately privileged evaluation transform,
not an inference pipeline: evaluation ground-truth boxes are used in images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from evaluate_testtime_box_blur import ground_truth_boxes_in_input, protect_and_blur


STRENGTHS = {"light": 3.0, "medium": 8.0, "heavy": 16.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hard_protection_mask(height: int, width: int, boxes: np.ndarray) -> np.ndarray:
    """Match the no-margin integer rectangle rasterization in protect_and_blur."""
    mask = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in boxes.reshape(-1, 4):
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            continue
        left, right = max(0, math.floor(x1)), min(width, math.ceil(x2))
        top, bottom = max(0, math.floor(y1)), min(height, math.ceil(y2))
        mask[top:bottom, left:right] = 1.0
    return mask


def rgb_to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not save {path}")


def make_panel(images: dict[str, np.ndarray]) -> np.ndarray:
    tiles = []
    for label, image in images.items():
        height, width = image.shape[:2]
        tile = np.full((height + 36, width, 3), 248, dtype=np.uint8)
        tile[36:] = image
        cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.concatenate((np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)), axis=0)


class SaveVisualizationValidator(DetectionValidator):
    def __init__(self, *args, output_dir: Path, sample_ids: set[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.sample_ids = sample_ids
        self.saved: dict[str, dict] = {}

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        for index, source in enumerate(batch["im_file"]):
            sample_id = Path(source).stem
            if sample_id not in self.sample_ids or sample_id in self.saved:
                continue
            image = np.ascontiguousarray(batch["img"][index].permute(1, 2, 0).float().cpu().numpy())
            boxes = ground_truth_boxes_in_input(batch, index).float().cpu().numpy()
            height, width = image.shape[:2]
            hard_mask = hard_protection_mask(height, width, boxes)
            soft_mask = np.maximum(cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=8.0), hard_mask)

            sample_dir = self.output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            source_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if source_image is None:
                raise OSError(f"could not read original source image {source}")
            write_image(sample_dir / "source_original.jpg", source_image)
            raw = rgb_to_bgr_uint8(image)
            write_image(sample_dir / "input_raw.png", raw)
            write_image(sample_dir / "hard_protection_mask.png", np.rint(hard_mask * 255).astype(np.uint8))
            write_image(sample_dir / "soft_protection_mask.png", np.rint(soft_mask * 255).astype(np.uint8))

            outlined = raw.copy()
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(outlined, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            write_image(sample_dir / "input_with_gt_boxes.png", outlined)

            panel_images = {"Raw model input": raw}
            changes = {}
            for name, sigma in STRENGTHS.items():
                output, protected_fraction = protect_and_blur(
                    image, boxes, sigma, margin_fraction=0.0, minimum_margin=0, feather_sigma=8.0
                )
                if not np.array_equal(output[hard_mask == 1], image[hard_mask == 1]):
                    raise AssertionError(f"GT-box pixel changed in {sample_id}/{name}")
                if not math.isclose(protected_fraction, float(hard_mask.mean()), abs_tol=1e-7):
                    raise AssertionError(f"mask coverage disagrees in {sample_id}/{name}")
                rendered = rgb_to_bgr_uint8(output)
                write_image(sample_dir / f"{name}_sigma{int(sigma)}.png", rendered)
                panel_images[f"{name} blur / sigma {int(sigma)}"] = rendered
                changes[name] = float(np.abs(output - image).mean() * 255.0)
            write_image(sample_dir / "comparison_2x2.jpg", make_panel(panel_images))
            self.saved[sample_id] = {
                "source": str(source),
                "original_shape_hw": list(source_image.shape[:2]),
                "model_input_shape_hw": [height, width],
                "gt_box_count": len(boxes),
                "hard_protection_fraction": float(hard_mask.mean()),
                "mean_abs_pixel_change_255": changes,
                "files": sorted(path.name for path in sample_dir.iterdir()),
            }
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError("weights or data YAML missing")
    samples = set(args.sample_ids)
    if len(samples) != len(args.sample_ids):
        raise ValueError("duplicate sample id")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    validators = []

    def validator_factory(**kwargs):
        validator = SaveVisualizationValidator(output_dir=args.output_dir, sample_ids=samples, **kwargs)
        validators.append(validator)
        return validator

    metrics = YOLO(str(args.weights), task="detect").val(
        validator=validator_factory,
        data=str(args.data), split="val", imgsz=640, batch=16, workers=4, device=args.device,
        conf=0.001, iou=0.7, max_det=300, nms=False, half=False, rect=True,
        plots=False, save_json=False, save_txt=False, verbose=False,
        project=str(args.output_dir), name="raw_validator_check", exist_ok=False,
    )
    saved = validators[0].saved
    if set(saved) != samples:
        raise RuntimeError(f"missing requested samples: {sorted(samples - set(saved))}")
    result = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        "warning": "privileged GT Oracle visualization; true evaluation boxes were used in image processing",
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "model": "YOLO26n VOC150 last.pt, O2O nms=false",
        "transform": "current LetterBox model input; GT rectangles unchanged; no margin; feather sigma=8; Gaussian background sigma=3/8/16",
        "raw_validation_mAP50_95": float(metrics.box.map),
        "samples": saved,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": sorted(saved), "raw_mAP50_95": result["raw_validation_mAP50_95"]}), flush=True)


if __name__ == "__main__":
    main()
