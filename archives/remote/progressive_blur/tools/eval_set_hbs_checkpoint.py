"""Independent, matched VisDrone validation of a saved HBS/identity checkpoint.

Ground-truth boxes are used only by the evaluator, never by model inference.
This is an Ultralytics/converted-YOLO-label result, not official VisDrone scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.set_hbs_integration import HBSOnlyDetectionModel  # noqa: E402
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
    parser.add_argument("--mode", choices=("hbs", "identity"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="load and verify without validation")
    args = parser.parse_args()

    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError("checkpoint or data YAML is missing")
    if not args.project.is_dir() or not args.output.parent.is_dir():
        raise FileNotFoundError("project or output parent is missing")
    if args.output.exists() or (args.project / args.name).exists():
        raise FileExistsError("refusing to overwrite an existing evaluation")
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("invalid batch or workers")

    model = YOLO(str(args.checkpoint))
    if not isinstance(model.model, HBSOnlyDetectionModel):
        raise TypeError("checkpoint is not an HBS-only detection model")
    if model.model.hbs_mode != args.mode or model.model.nc != 10:
        raise ValueError("checkpoint mode or class count differs from requested evaluation")
    if args.check_only:
        print(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
                          "model_type": type(model.model).__name__,
                          "mode": model.model.hbs_mode, "classes": model.model.nc}))
        return

    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=960,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        half=False,
        rect=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = metrics.box
    class_ids = [int(value) for value in box.ap_class_index]
    if len(class_ids) != 10 or set(class_ids) != set(range(10)):
        raise RuntimeError(f"expected all 10 classes, got {class_ids}")
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
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "mode": args.mode,
        "evaluation": {
            "split": "val", "imgsz": 960, "batch": args.batch,
            "workers": args.workers, "conf": 0.001, "iou": 0.7,
            "max_det": 1000, "half": False, "rect": True,
            "protocol": "Ultralytics YOLO.val on converted labels, not official VisDrone toolkit",
        },
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": classes,
        "speed_ms_per_image": metrics.speed,
        "speed_scope_note": "Ultralytics validator timing; same path/config for both paired models",
        "save_dir": str(metrics.save_dir),
    }
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(result, encoding="utf-8")
    print("SET_HBS_EVAL_JSON=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
