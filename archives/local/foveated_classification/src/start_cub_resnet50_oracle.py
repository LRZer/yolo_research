"""Safely launch C then D after the preceding E125 run is strictly audited."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import start_cub_resnet50_a as launcher


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
ORACLE_TRAINER = ROOT / "src/train_cub_resnet50_oracle.py"
CONFIGS = {
    "B": ROOT / "configs/cub_resnet50_B_global_blur_native500_s2005_e200_gpu0.json",
    "C": ROOT / "configs/cub_resnet50_C_context_removed_native500_s2005_e200_gpu0.json",
    "D": ROOT / "configs/cub_resnet50_D_gradual_background_native500_s2005_e200_gpu0.json",
}
RUNS = {
    "B": ROOT / "runs/resnet50_B_global_blur_native500_s2005_e200",
    "C": ROOT / "runs/resnet50_C_context_removed_native500_s2005_e200",
    "D": ROOT / "runs/resnet50_D_gradual_background_native500_s2005_e200",
}
GROUPS = {"B": "B_global_blur", "C": "C_context_removed", "D": "D_gradual_background"}
TRAINER_NAMES = {"train_cub_resnet50_a.py", "train_cub_resnet50_b.py", "train_cub_resnet50_oracle.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_task_trainers() -> list[int]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [arg.decode("utf-8", "replace") for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
        except (OSError, PermissionError):
            continue
        if "--train" in args and any(Path(arg).name in TRAINER_NAMES and str(ROOT / "src") in arg for arg in args):
            matches.append(int(proc.name))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("C", "D"), required=True)
    args = parser.parse_args()
    previous = "B" if args.group == "C" else "C"
    previous_run, previous_config = RUNS[previous], CONFIGS[previous]
    audit_path = previous_run / "audit.json"
    if not audit_path.is_file():
        raise RuntimeError(f"Previous group {previous} has no strict E125 audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "verified_revised_e125" or audit.get("group") != GROUPS[previous]
            or audit.get("epochs") != 125 or audit.get("original_config_epochs") != 200
            or audit.get("learning_rate_schedule_total_epochs") != 200
            or audit.get("config_sha256") != sha256(previous_config)
            or audit.get("results_sha256") != sha256(previous_run / "results.csv")
            or audit.get("last_checkpoint_sha256") != sha256(previous_run / "last.pt")):
        raise RuntimeError(f"Previous group {previous} audit is stale or inconsistent")
    if active_task_trainers():
        raise RuntimeError(f"Other CUB task trainers still active: {active_task_trainers()}")
    target_run, target_config = RUNS[args.group], CONFIGS[args.group]
    if any((target_run / name).exists() for name in ("launch.json", "results.csv", "last.pt", "audit.json")):
        raise RuntimeError(f"Group {args.group} already has run artifacts; refusing duplicate launch")
    config = json.loads(target_config.read_text(encoding="utf-8"))
    if config["group"] != GROUPS[args.group] or Path(config["run_dir"]) != target_run:
        raise ValueError("Target group/config/run mismatch")
    launcher.RUN = target_run
    launcher.CONFIG = target_config
    launcher.TRAINER = ORACLE_TRAINER
    launcher.LOG = target_run / "train.log"
    launcher.main()


if __name__ == "__main__":
    main()
