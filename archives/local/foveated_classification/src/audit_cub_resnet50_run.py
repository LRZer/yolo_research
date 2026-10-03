"""Strict CPU-only audit for a completed CUB run, including revised E125."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, choices=(125, 200), default=200)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["group"] not in ("A_original", "B_global_blur", "C_context_removed", "D_gradual_background", "E_grayscale", "F_scharr") or config["epochs"] != 200:
        raise ValueError("Not a frozen A/B/C/D/E/F 200-epoch-schedule CUB config")
    if config["group"] in ("C_context_removed", "D_gradual_background"):
        box_path = Path(config["data_root"]) / "bounding_boxes.txt"
        if sha256(box_path) != config["bounding_boxes_sha256"]:
            raise ValueError("Official bounding-box SHA differs from C/D config")
    target = args.target_epochs
    run = Path(config["run_dir"])
    csv_path = run / "results.csv"
    checkpoint_path = run / "last.pt"
    state_path = run / "state.json"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != target or [int(row["epoch"]) for row in rows] != list(range(1, target + 1)):
        raise ValueError(f"CSV must contain exactly {target} continuous epochs")
    for row in rows:
        epoch = int(row["epoch"])
        if epoch <= config["warmup_epochs"]:
            expected_lr = config["base_lr"] * epoch / config["warmup_epochs"]
        else:
            progress = (epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
            expected_lr = config["min_lr"] + (config["base_lr"] - config["min_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))
        if not math.isclose(float(row["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Wrong frozen 200-epoch-schedule learning rate at epoch {epoch}")
        for key, value in row.items():
            if key == "epoch":
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite {key} at epoch {row['epoch']}")
        if int(row["train_images"]) != 5994 or int(row["test_images"]) != 5794:
            raise ValueError(f"Wrong train/test image count at epoch {row['epoch']}")
        if int(row["optimizer_steps"]) != 94:
            raise ValueError(f"Wrong optimizer step count at epoch {row['epoch']}")
        if float(row["epoch_seconds"]) <= 0:
            raise ValueError(f"Invalid epoch time at epoch {row['epoch']}")
        for key in ("train_top1", "train_top5", "test_top1", "test_top5"):
            if not 0 <= float(row[key]) <= 100:
                raise ValueError(f"Invalid accuracy {key} at epoch {row['epoch']}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != target:
        raise ValueError(f"Checkpoint is not epoch {target}")
    if checkpoint["config_sha256"] != sha256(args.config):
        raise ValueError("Checkpoint config hash differs from frozen config")
    stats_path = Path(config["stats_json"])
    if checkpoint["stats_sha256"] != sha256(stats_path):
        raise ValueError("Checkpoint RGB statistics hash differs")
    for key, value in rows[-1].items():
        actual = checkpoint["last_row"][key]
        if key == "epoch" or key in ("train_images", "test_images", "optimizer_steps"):
            equal = int(value) == int(actual)
        else:
            equal = math.isclose(float(value), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"Checkpoint last_row mismatch: {key}")
    weights = checkpoint["model"]
    if tuple(weights["fc.weight"].shape) != (200, 2048) or tuple(weights["fc.bias"].shape) != (200,):
        raise ValueError("Classifier head is not CUB 200-class ResNet50")
    if any(not bool(torch.isfinite(value).all()) for value in weights.values() if torch.is_tensor(value)):
        raise ValueError("Non-finite model parameter or buffer")
    if not checkpoint.get("optimizer") or not checkpoint.get("scaler"):
        raise ValueError("Missing optimizer or AMP state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["epoch"] != target:
        raise ValueError("State/checkpoint epoch mismatch")
    if target == 200:
        if state["status"] != "training_complete_pending_audit":
            raise ValueError("State file does not indicate 200 epochs finished")
    else:
        if state["status"] != "running" or state["total_epochs"] != 200:
            raise ValueError("Revised E125 must retain the original 200-epoch config/state")
        stop_path = run / "revised_e125_stop.json"
        stop = json.loads(stop_path.read_text(encoding="utf-8"))
        if (stop.get("status") != "sigterm_sent_after_completed_target"
                or stop.get("target_epoch") != target
                or stop.get("config_sha256") != sha256(args.config)
                or stop.get("observed_csv_rows") != target):
            raise ValueError("Missing or inconsistent authorized E125 stop record")
        proc_cmdline = Path("/proc") / str(stop["pid"]) / "cmdline"
        if proc_cmdline.exists() and str(args.config).encode() in proc_cmdline.read_bytes():
            raise ValueError("Trainer still running; E125 stop not yet complete")
    result = {
        "status": "verified_complete" if target == 200 else "verified_revised_e125",
        "group": config["group"], "epochs": target,
        "original_config_epochs": config["epochs"],
        "learning_rate_schedule_total_epochs": config["epochs"],
        "last_test_top1_percent": float(rows[-1]["test_top1"]),
        "last_test_top5_percent": float(rows[-1]["test_top5"]),
        "mean_epoch_seconds": sum(float(row["epoch_seconds"]) for row in rows) / target,
        "final_epoch_seconds": float(rows[-1]["epoch_seconds"]),
        "config_sha256": sha256(args.config), "stats_sha256": sha256(stats_path),
        "results_sha256": sha256(csv_path), "last_checkpoint_sha256": sha256(checkpoint_path),
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    destination = run / "audit.json"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
