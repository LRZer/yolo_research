"""Launch grayscale E only after D's strict 125-epoch audit is current."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import start_cub_resnet50_a as launcher


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
PREVIOUS_CONFIG = ROOT / "configs/cub_resnet50_D_gradual_background_native500_s2005_e200_gpu0.json"
PREVIOUS_RUN = ROOT / "runs/resnet50_D_gradual_background_native500_s2005_e200"
CONFIG = ROOT / "configs/cub_resnet50_E_grayscale_native500_s2005_e200_gpu0.json"
RUN = ROOT / "runs/resnet50_E_grayscale_native500_s2005_e200"
TRAINER = ROOT / "src/train_cub_resnet50_e.py"
TRAINER_NAMES = {
    "train_cub_resnet50_a.py", "train_cub_resnet50_b.py",
    "train_cub_resnet50_oracle.py", "train_cub_resnet50_e.py",
}


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
    audit_path = PREVIOUS_RUN / "audit.json"
    if not audit_path.is_file():
        raise RuntimeError("D has no strict E125 audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "verified_revised_e125"
            or audit.get("group") != "D_gradual_background"
            or audit.get("epochs") != 125
            or audit.get("original_config_epochs") != 200
            or audit.get("learning_rate_schedule_total_epochs") != 200
            or audit.get("config_sha256") != sha256(PREVIOUS_CONFIG)
            or audit.get("results_sha256") != sha256(PREVIOUS_RUN / "results.csv")
            or audit.get("last_checkpoint_sha256") != sha256(PREVIOUS_RUN / "last.pt")):
        raise RuntimeError("D audit is stale or inconsistent")
    active = active_task_trainers()
    if active:
        raise RuntimeError(f"Other CUB task trainers still active: {active}")
    if any((RUN / name).exists() for name in ("launch.json", "results.csv", "last.pt", "audit.json")):
        raise RuntimeError("E already has run artifacts; refusing duplicate launch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["group"] != "E_grayscale" or Path(config["run_dir"]) != RUN:
        raise ValueError("E config/run mismatch")
    launcher.RUN = RUN
    launcher.CONFIG = CONFIG
    launcher.TRAINER = TRAINER
    launcher.LOG = RUN / "train.log"
    launcher.main()


if __name__ == "__main__":
    main()
