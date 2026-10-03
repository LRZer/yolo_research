"""Compare complete YOLO.predict wall time on the same preloaded VOC images.

This is an inference-cost diagnostic, not AP evaluation or a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def measured_predict(model: YOLO, images: list, batch: int, device: str) -> tuple[float, int]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = model.predict(
        source=images,
        imgsz=640,
        batch=batch,
        device=device,
        half=False,
        rect=True,
        conf=0.25,
        iou=0.7,
        max_det=300,
        save=False,
        verbose=False,
        stream=False,
    )
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    if len(results) != len(images):
        raise AssertionError(f"predicted {len(results)} rather than {len(images)} images")
    return duration, sum(len(result.boxes) for result in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(line.strip()) for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) < 128:
        raise ValueError("need at least 128 dev images")
    # Deterministic coverage of the dev set without involving its labels.
    selected = [paths[index] for index in range(0, len(paths), max(1, len(paths) // 128))][:128]
    images = [cv2.imread(str(path)) for path in selected]
    if any(image is None for image in images):
        raise FileNotFoundError("a benchmark image could not be decoded")
    models = {"method": YOLO(str(args.method)), "baseline": YOLO(str(args.baseline))}
    if not isinstance(models["method"].model, SafeBackgroundDetectionModel):
        raise AssertionError("method checkpoint lacks expected frontend")
    models["method"].model.set_eval_mix(0.5)
    if isinstance(models["baseline"].model, SafeBackgroundDetectionModel):
        raise AssertionError("baseline unexpectedly has frontend")
    records: dict[str, object] = {}
    for batch, count in ((1, 64), (16, 128)):
        subset = images[:count]
        for name in ("baseline", "method"):
            model = models[name]
            measured_predict(model, subset[: min(batch, 16)], batch, args.device)
            elapsed, detections = [], []
            for _ in range(3):
                seconds, boxes = measured_predict(model, subset, batch, args.device)
                elapsed.append(seconds)
                detections.append(boxes)
            records[f"{name}_batch{batch}"] = {
                "images": count,
                "seconds_repeats": elapsed,
                "median_ms_per_image": statistics.median(elapsed) * 1000 / count,
                "total_predicted_boxes_repeats": detections,
            }
    report = {
        "scope": "YOLO.predict on preloaded OpenCV image arrays; includes preprocessing, model, and postprocessing but excludes disk image decoding",
        "image_count": len(images),
        "selected_paths_sha256": hashlib.sha256("\n".join(map(str, selected)).encode()).hexdigest(),
        "method": str(args.method),
        "baseline": str(args.baseline),
        "device": args.device,
        "imgsz": 640,
        "half": False,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "warmup": True,
        "records": records,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": records}))


if __name__ == "__main__":
    main()
