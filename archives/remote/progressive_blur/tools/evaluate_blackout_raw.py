"""Evaluate blackout-trained YOLO checkpoints on untouched VOC dev images."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "voc_p1" / "oracle_blackout_scratch_e20_b64_s2027"
DATA = ROOT / "data_splits" / "voc_strict_seed2027.yaml"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path: Path) -> int | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch")
        return int(epoch) + 1 if isinstance(epoch, int) and epoch >= 0 else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "p1_blackout_raw_validation.json")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = args.run / "weights"
    checkpoints = [weights / "best.pt", weights / "last.pt"]
    checkpoints += sorted(
        weights.glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
    )
    if not checkpoints[:2] or not all(path.is_file() for path in checkpoints[:2]):
        raise FileNotFoundError("best.pt and last.pt are required")

    output = {
        "status": "raw_development_images_no_oracle_mask",
        "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        "data": str(DATA),
        "split": "val",
        "model_input": "unaltered 640-pixel letterboxed images",
        "checkpoints": [],
    }
    for path in checkpoints:
        model = YOLO(str(path), task="detect")
        metrics = model.val(
            data=str(DATA),
            split="val",
            imgsz=640,
            batch=64,
            workers=4,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            half=False,
            plots=False,
            save_json=False,
            save_txt=False,
            project=str(ROOT / "runs" / "blackout_raw_validation"),
            name=f"{args.run.name}_{path.stem}",
            exist_ok=True,
            verbose=False,
        )
        row = {
            "checkpoint": str(path),
            "checkpoint_sha256": checksum(path),
            "checkpoint_epoch": checkpoint_epoch(path),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        output["checkpoints"].append(row)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(row), flush=True)

    print(json.dumps({"status": "complete", "evaluated": len(output["checkpoints"])}))


if __name__ == "__main__":
    main()
