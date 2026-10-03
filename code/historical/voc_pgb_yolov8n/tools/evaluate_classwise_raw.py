"""Re-evaluate saved VOC checkpoints on raw development images by class.

This is an analysis aid, not a way to select a new checkpoint after seeing test
results. It uses the same validation settings as evaluate_blackout_raw.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_splits" / "voc_strict_seed2027.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    output = {
        "status": "raw_development_images_no_oracle_mask",
        "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "selection": "existing best.pt; no new checkpoint selection",
        "runs": [],
    }
    for run in args.run:
        path = run / "weights" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        model = YOLO(str(path), task="detect")
        metrics = model.val(
            data=str(args.data), split=args.split, imgsz=args.imgsz, batch=64, workers=4,
            device=args.device, conf=0.001, iou=0.7, max_det=args.max_det,
            half=False, plots=False, save_json=False, save_txt=False,
            project=str(ROOT / "runs" / "classwise_raw_validation"),
            name=run.name, exist_ok=True, verbose=False,
        )
        box = metrics.box
        class_ids = [int(value) for value in box.ap_class_index]
        ap50 = [float(value) for value in box.ap50]
        ap = [float(value) for value in box.ap]
        names = metrics.names
        classes = [
            {
                "id": class_id,
                "name": names[class_id],
                "AP50": ap50[index],
                "AP50_95": ap[index],
            }
            for index, class_id in enumerate(class_ids)
        ]
        if len(classes) != len(names):
            raise RuntimeError(f"expected {len(names)} annotated classes, got {len(classes)}")
        output["runs"].append({
            "run": str(run),
            "checkpoint": str(path),
            "checkpoint_sha256": digest(path),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "classes": classes,
        })
        print(json.dumps({"run": str(run), "mAP50_95": box.map}), flush=True)

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
