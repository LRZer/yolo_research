"""Read-only audit of a map-only VOC checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=1)
    args = parser.parse_args()

    run_model, _ = load_checkpoint(args.run / "weights" / "last.pt", device="cpu", fuse=False)
    best_model, _ = load_checkpoint(args.run / "weights" / "best.pt", device="cpu", fuse=False)
    baseline, _ = load_checkpoint(args.baseline, device="cpu", fuse=False)
    if not isinstance(run_model, SafeBackgroundDetectionModel):
        raise AssertionError(f"wrong checkpoint class: {type(run_model)}")
    if not isinstance(best_model, SafeBackgroundDetectionModel):
        raise AssertionError(f"wrong best checkpoint class: {type(best_model)}")
    if not run_model.sbg_map_only or float(run_model.pgb_eval_mix) != 0:
        raise AssertionError("map-only checkpoint has unexpected inference configuration")

    current_detector = run_model.model.state_dict()
    starting_detector = baseline.model.state_dict()
    if current_detector.keys() != starting_detector.keys():
        raise AssertionError("detector state keys changed")
    changed = [key for key in current_detector if not torch.equal(current_detector[key], starting_detector[key])]
    if changed:
        raise AssertionError(f"frozen detector changed: {changed[:5]}")

    with (args.run / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != args.expected_epochs:
        raise AssertionError(f"expected {args.expected_epochs} epochs, got {len(rows)}")
    if [int(float(row["epoch"])) for row in rows] != list(range(1, args.expected_epochs + 1)):
        raise AssertionError("results.csv epochs are not consecutive")

    image = cv2.imread(str(args.sample))
    if image is None:
        raise FileNotFoundError(args.sample)
    letterboxed = LetterBox(new_shape=(640, 640))(image=image)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    run_model.eval()
    baseline.eval()
    with torch.inference_mode():
        permission = run_model.frontend.map_net(tensor)
        wrapped = run_model(tensor)
        native = baseline(tensor)
    wrapped_tensor = wrapped[0] if isinstance(wrapped, tuple) else wrapped
    native_tensor = native[0] if isinstance(native, tuple) else native
    if not torch.equal(wrapped_tensor, native_tensor):
        raise AssertionError("zero-mix prediction differs from frozen baseline")

    summary = {
        "status": "verified_map_only",
        "run": str(args.run),
        "epochs": len(rows),
        "baseline_detector_bitwise_unchanged": True,
        "zero_mix_prediction_bitwise_equal": True,
        "best_and_last_readable": True,
        "gate_parameters": sum(x.numel() for x in run_model.frontend.map_net.parameters()),
        "gate_min": float(permission.min()),
        "gate_mean": float(permission.mean()),
        "gate_max": float(permission.max()),
        "gate_fraction_above_0_1": float((permission > 0.1).float().mean()),
        "final_map_loss_logged": rows[-1].get("train/map_loss"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
