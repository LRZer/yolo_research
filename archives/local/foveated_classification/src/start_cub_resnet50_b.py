"""Launch the B/global-blur run only after the A run passes strict audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import start_cub_resnet50_a as launcher


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
A_RUN = ROOT / "runs/resnet50_A_native500_s2005_e200"
A_CONFIG = ROOT / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json"
B_RUN = ROOT / "runs/resnet50_B_global_blur_native500_s2005_e200"
B_CONFIG = ROOT / "configs/cub_resnet50_B_global_blur_native500_s2005_e200_gpu0.json"
B_TRAINER = ROOT / "src/train_cub_resnet50_b.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit_path = A_RUN / "audit.json"
    if not audit_path.is_file():
        raise RuntimeError("A run has no strict audit; refusing B launch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "verified_revised_e125" or audit.get("group") != "A_original"
            or audit.get("epochs") != 125 or audit.get("original_config_epochs") != 200
            or audit.get("learning_rate_schedule_total_epochs") != 200
            or audit.get("config_sha256") != sha256(A_CONFIG)
            or audit.get("results_sha256") != sha256(A_RUN / "results.csv")
            or audit.get("last_checkpoint_sha256") != sha256(A_RUN / "last.pt")):
        raise RuntimeError("A audit is stale or inconsistent; refusing B launch")
    launcher.TRAINER = ROOT / "src/train_cub_resnet50_a.py"
    if launcher.active_trainers():
        raise RuntimeError("A trainer is still active; refusing parallel B launch")
    launcher.RUN = B_RUN
    launcher.CONFIG = B_CONFIG
    launcher.TRAINER = B_TRAINER
    launcher.LOG = B_RUN / "train.log"
    launcher.main()


if __name__ == "__main__":
    main()
