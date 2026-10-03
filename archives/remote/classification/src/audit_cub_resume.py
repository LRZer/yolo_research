"""Read-only checkpoint/CSV identity audit before resuming an interrupted CUB run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    parser.add_argument("--target-epochs", type=int, default=125)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["epochs"] != 200 or args.target_epochs != 125:
        raise ValueError("Unexpected frozen schedule or target")
    run = Path(config["run_dir"])
    if (run / "revised_e125_stop.json").exists():
        raise ValueError("Run already has a stop record")
    with (run / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    if not epochs or epochs != list(range(1, len(rows) + 1)) or epochs[-1] >= args.target_epochs:
        raise ValueError("CSV epochs are invalid for a pre-target resume")
    if any(not math.isfinite(float(value)) for row in rows for key, value in row.items() if key != "epoch"):
        raise ValueError("CSV has non-finite metrics")
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != epochs[-1]:
        raise ValueError("Checkpoint/CSV epoch mismatch")
    if checkpoint["config_sha256"] != sha256(args.config):
        raise ValueError("Checkpoint/config SHA mismatch")
    if checkpoint["stats_sha256"] != sha256(Path(config["stats_json"])):
        raise ValueError("Checkpoint/training statistics SHA mismatch")
    for key, value in rows[-1].items():
        recorded = checkpoint["last_row"][key]
        if key in ("epoch", "train_images", "test_images", "optimizer_steps"):
            equal = int(value) == int(recorded)
        else:
            equal = math.isclose(float(value), float(recorded), rel_tol=1e-12, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"Checkpoint last_row mismatch: {key}")
    if any(int(row["train_images"]) != 5994 or int(row["test_images"]) != 5794
           or int(row["optimizer_steps"]) != 94 for row in rows):
        raise ValueError("Unexpected image or optimizer-step counts")
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    if state["epoch"] != epochs[-1] or state["config_sha256"] != sha256(args.config):
        raise ValueError("State/config/CSV mismatch")
    weights = checkpoint["model"]
    if tuple(weights["fc.weight"].shape) != (200, 2048):
        raise ValueError("Wrong classifier shape")
    if any(not bool(torch.isfinite(value).all()) for value in weights.values() if torch.is_tensor(value)):
        raise ValueError("Non-finite model parameter or buffer")
    if not checkpoint.get("optimizer") or not checkpoint.get("scaler"):
        raise ValueError("Missing optimizer or AMP scaler state")
    print(json.dumps({"status": "verified_resumable_checkpoint", "group": config["group"],
                      "epoch": epochs[-1], "target": args.target_epochs,
                      "config_sha256": sha256(args.config),
                      "results_sha256": sha256(run / "results.csv"),
                      "last_checkpoint_sha256": sha256(run / "last.pt")}), flush=True)


if __name__ == "__main__":
    main()
