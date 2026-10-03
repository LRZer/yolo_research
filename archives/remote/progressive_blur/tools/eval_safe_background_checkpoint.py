"""Evaluate a saved detector or image-only safe-background model.

This never reads validation labels to construct the input image; labels are
used only by Ultralytics' metric computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mix", type=float, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, help="save metrics JSON in an existing project directory")
    args = parser.parse_args()
    if not 0 <= args.mix <= 1:
        raise ValueError("mix must be in [0,1]")
    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError("checkpoint or data YAML is missing")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    model = YOLO(str(args.checkpoint))
    has_frontend = isinstance(model.model, SafeBackgroundDetectionModel)
    if has_frontend:
        model.model.set_eval_mix(args.mix)
    elif args.mix != 0:
        raise ValueError("a plain detector has no blur mix; use --mix 0")
    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        rect=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = metrics.box
    class_ids = [int(value) for value in box.ap_class_index]
    if len(class_ids) != len(model.names):
        raise RuntimeError(f"expected {len(model.names)} evaluated classes, got {len(class_ids)}")
    classes = [
        {
            "id": class_id,
            "name": str(model.names[class_id]),
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "ap50": float(box.ap50[position]),
            "ap50_95": float(box.ap[position]),
        }
        for position, class_id in enumerate(class_ids)
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "mix": args.mix,
        "has_frontend": has_frontend,
        "validation": {
            "split": "val", "imgsz": 640, "batch": args.batch,
            "workers": args.workers, "conf": 0.001, "iou": 0.7,
            "max_det": 300, "half": False, "rect": True,
        },
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": classes,
        "per_class_map50_95": {
            str(model.names[index]): float(value)
            for index, value in enumerate(box.maps)
        },
        "speed_ms_per_image": metrics.speed,
        "speed_scope_note": "Ultralytics validator timing; verify full frontend latency separately",
        "save_dir": str(metrics.save_dir),
    }
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        if not args.output.parent.is_dir():
            raise FileNotFoundError(args.output.parent)
        args.output.write_text(result + "\n", encoding="utf-8")
    print("SAFE_BACKGROUND_EVAL_JSON=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
