"""Two-stage remote audit of train/val/EMA/checkpoint/resume integration.

Stage 1 intentionally exits immediately after the first checkpoint to emulate an
interruption before Ultralytics strips optimizer state. Stage 2 resumes that
checkpoint and finishes the originally configured two epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from pgb.ultralytics_integration import PGBDetectionModel, PGBDetectionTrainer
from ultralytics.nn.tasks import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "tiny_trainer_audit"
RUN = RUNS / "resume_audit"
LAST = RUN / "weights" / "last.pt"


def initial_overrides() -> dict:
    return {
        "model": "yolov8n.yaml",
        "data": str(ROOT / "audit_data" / "tiny.yaml"),
        "epochs": 2,
        "batch": 4,
        "imgsz": 128,
        "workers": 0,
        "device": "0",
        "project": str(RUNS),
        "name": "resume_audit",
        "exist_ok": True,
        "pretrained": False,
        "optimizer": "SGD",
        "lr0": 0.001,
        "deterministic": True,
        "seed": 2027,
        "amp": True,
        "val": True,
        "save": True,
        "plots": False,
        "cache": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
        "pgb_map_loss_weight": 1.0,
        "pgb_eval_mix": 0.5,
        "pgb_zero_mix_epochs": 0,
        "pgb_ramp_epochs": 1,
    }


def stage1() -> None:
    trainer = PGBDetectionTrainer(overrides=initial_overrides())

    def stop_after_first_checkpoint(current: PGBDetectionTrainer) -> None:
        if current.epoch == 0:
            marker = {
                "status": "intentional_interruption_after_checkpoint",
                "epoch": current.epoch,
                "last": str(current.last),
            }
            (ROOT / "research" / "tiny_stage1.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
            os._exit(0)

    trainer.add_callback("on_model_save", stop_after_first_checkpoint)
    trainer.train()
    raise RuntimeError("stage1 did not interrupt after its first checkpoint")


def stage2() -> None:
    if not LAST.is_file():
        raise FileNotFoundError(LAST)
    trainer = PGBDetectionTrainer(
        overrides={
            "resume": str(LAST),
            "device": "0",
            "workers": 0,
            "plots": False,
            "val": True,
            "cache": False,
        }
    )
    trainer.train()
    if trainer.pgb_eval_mix != 0.5 or trainer.pgb_ramp_epochs != 1:
        raise AssertionError("PGB-specific options were not restored from checkpoint")


def audit() -> None:
    results_path = RUN / "results.csv"
    best = RUN / "weights" / "best.pt"
    if not LAST.is_file() or not best.is_file() or not results_path.is_file():
        raise AssertionError("missing expected final artifacts")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise AssertionError(f"expected 2 result rows, got {len(rows)}")
    columns = set(rows[0])
    required = {"train/map_loss", "val/map_loss"}
    if not required.issubset(columns):
        raise AssertionError(f"missing loss columns: {required - columns}")

    model, checkpoint = load_checkpoint(LAST, device="cpu", fuse=False)
    if not isinstance(model, PGBDetectionModel):
        raise AssertionError(f"unexpected checkpoint model class: {type(model)}")
    state = model.state_dict()
    if not any(key.startswith("frontend.") for key in state):
        raise AssertionError("checkpoint omits the front end")
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.rand(1, 3, 128, 128))
    prediction_tensor = prediction[0] if isinstance(prediction, tuple) else prediction
    if not torch.isfinite(prediction_tensor).all():
        raise AssertionError("checkpoint inference produced non-finite values")

    summary = {
        "status": "verified_complete",
        "epochs": len(rows),
        "results_columns": list(rows[0]),
        "last_readable": True,
        "best_readable": load_checkpoint(best, device="cpu", fuse=False)[0] is not None,
        "model_class": type(model).__name__,
        "checkpoint_contains_frontend": True,
        "restored_pgb_eval_mix": float(model.pgb_eval_mix),
        "prediction_shape": list(prediction_tensor.shape),
        "checkpoint_version": checkpoint.get("version"),
    }
    output = ROOT / "research" / "tiny_trainer_audit.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage1", "stage2", "audit"))
    args = parser.parse_args()
    {"stage1": stage1, "stage2": stage2, "audit": audit}[args.stage]()


if __name__ == "__main__":
    main()
