"""Evaluate one YOLO checkpoint through a locked O2O or O2M inference path and emit JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        parser.error(f"dataset YAML not found: {args.data}")
    return args


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    """Convert NumPy scalar values from the per-class summary to JSON-native scalars."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    results = YOLO(str(ARGS.checkpoint.resolve())).val(
        data=str(ARGS.data.resolve()),
        split="test",
        imgsz=ARGS.imgsz,
        batch=ARGS.batch,
        device="0",
        workers=ARGS.workers,
        nms=ARGS.nms,
        plots=False,
        save_json=False,
        project=str(ARGS.project.resolve()),
        name=ARGS.name,
        exist_ok=False,
    )
    report = {
        "checkpoint": str(ARGS.checkpoint.resolve()),
        "checkpoint_sha256": sha256(ARGS.checkpoint),
        "data": str(ARGS.data.resolve()),
        "split": "test",
        "path": "O2M+NMS" if ARGS.nms else "O2O NMS-free",
        "physical_device": ARGS.physical_device,
        "batch": ARGS.batch,
        "imgsz": ARGS.imgsz,
        "workers": ARGS.workers,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "per_class": results.summary(decimals=8),
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, default=json_default)
    ARGS.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
