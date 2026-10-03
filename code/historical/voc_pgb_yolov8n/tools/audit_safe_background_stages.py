"""Verify stage transitions, checkpoints, and detector freezing on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


def epoch_count(run: Path, expected: int) -> int:
    with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual_epochs = [int(float(row["epoch"])) for row in rows]
    if actual_epochs != list(range(1, expected + 1)):
        raise AssertionError(f"wrong epoch sequence in {run}: {actual_epochs}")
    return len(rows)


def load_model(path: Path):
    model, _ = load_checkpoint(path, device="cpu", fuse=False)
    return model


def changed_tensors(left: dict, right: dict) -> int:
    if left.keys() != right.keys():
        raise AssertionError("state_dict keys differ")
    return sum(not torch.equal(left[key], right[key]) for key in left)


def check_stage(run: Path, expected_epochs: int, expected_mix: float) -> SafeBackgroundDetectionModel:
    latest = load_model(run / "weights" / "last.pt")
    best = load_model(run / "weights" / "best.pt")
    if not isinstance(latest, SafeBackgroundDetectionModel) or not isinstance(best, SafeBackgroundDetectionModel):
        raise AssertionError(f"wrong model class in {run}")
    if not torch.isclose(latest.pgb_eval_mix, torch.tensor(expected_mix)):
        raise AssertionError(f"wrong inference blur mix in {run}")
    epoch_count(run, expected_epochs)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--map-run", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--joint-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_model(args.baseline_checkpoint)
    map_model = check_stage(args.map_run, 5, 0.0)
    frozen = check_stage(args.frozen_run, 2, 0.5)
    joint = check_stage(args.joint_run, 3, 0.5)
    control = load_model(args.control_run / "weights" / "last.pt")
    load_model(args.control_run / "weights" / "best.pt")
    epoch_count(args.control_run, 3)
    if not map_model.sbg_map_only or frozen.sbg_map_only or joint.sbg_map_only:
        raise AssertionError("stage mode mismatch")

    reference = baseline.model.state_dict()
    detector_changes = {
        "map": changed_tensors(reference, map_model.model.state_dict()),
        "frozen_feedback": changed_tensors(reference, frozen.model.state_dict()),
        "joint": changed_tensors(reference, joint.model.state_dict()),
        "control": changed_tensors(reference, control.model.state_dict()),
    }
    gate_changes = {
        "map_to_frozen": changed_tensors(
            map_model.frontend.map_net.state_dict(), frozen.frontend.map_net.state_dict()
        ),
        "frozen_to_joint": changed_tensors(
            frozen.frontend.map_net.state_dict(), joint.frontend.map_net.state_dict()
        ),
    }
    if detector_changes["map"] != 0 or detector_changes["frozen_feedback"] != 0:
        raise AssertionError("frozen detector changed")
    if detector_changes["joint"] == 0 or detector_changes["control"] == 0:
        raise AssertionError("trainable detector did not change")
    if any(count == 0 for count in gate_changes.values()):
        raise AssertionError("gate failed to update in a training stage")
    print(json.dumps({
        "status": "verified_stages",
        "epoch_counts": {"map": 5, "frozen_feedback": 2, "joint": 3, "control": 3},
        "detector_changed_tensors_vs_original": detector_changes,
        "gate_changed_tensors": gate_changes,
        "inference_mix": {"map": 0.0, "frozen_feedback": 0.5, "joint": 0.5},
        "best_and_last_readable": True,
    }, indent=2))


if __name__ == "__main__":
    main()
