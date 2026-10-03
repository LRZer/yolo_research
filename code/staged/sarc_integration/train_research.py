"""Run a recorded YOLO26/VOC research experiment from a YAML specification."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import ultralytics
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = Path("/home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate").resolve()
EXPECTED_BASE_COMMIT = "9d45ee4613e79d7bceef7a39e42c15c278b0d4e8"
ALLOWED_GPUS = "4,5,6,7"


def git_output(*args: str) -> str:
    """Return a read-only Git query from the isolated research checkout."""
    return subprocess.run(
        ["git", "-C", str(EXPECTED_SOURCE), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    """Validate provenance and launch the configured experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("train"), dict):
        raise ValueError("Research config must contain a 'train' mapping.")
    train = spec["train"]

    imported_source = Path(ultralytics.__file__).resolve().parents[1]
    if imported_source != EXPECTED_SOURCE:
        raise RuntimeError(f"Expected research source {EXPECTED_SOURCE}, imported {imported_source}")
    if git_output("rev-parse", "HEAD") != EXPECTED_BASE_COMMIT:
        raise RuntimeError("Research checkout is not based on the locked YOLO26 commit.")
    changed = set(git_output("status", "--porcelain").splitlines())
    allowed_suffixes = {
        "ultralytics/cfg/__init__.py",
        "ultralytics/cfg/default.yaml",
        "ultralytics/cfg/models/26/yolo26n-sarc-r1.yaml",
        "ultralytics/cfg/models/26/yolo26n-sarc-r2.yaml",
        "ultralytics/cfg/models/26/yolo26n-sarc-r3.yaml",
        "ultralytics/data/build.py",
        "ultralytics/engine/trainer.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/modules/conv.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/utils/loss.py",
    }
    unexpected = [line for line in changed if not any(line.endswith(path) for path in allowed_suffixes)]
    if unexpected:
        raise RuntimeError(f"Unexpected research-source changes: {unexpected}")

    if str(train.get("device")) != ALLOWED_GPUS:
        raise ValueError(f"All research runs must use only physical GPUs {ALLOWED_GPUS}.")
    if train.get("pretrained") is not False:
        raise ValueError("Current comparison protocol requires pretrained=False.")
    model_path = Path(train["model"]).resolve()
    data_path = Path(train["data"]).resolve()
    if model_path.suffix != ".yaml" or not model_path.is_file():
        raise FileNotFoundError(f"Locked model YAML not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    sarc_model = "yolo26n-sarc-" in model_path.name
    sarc_gain = float(train.get("sarc_cal_gain", 0.0))
    sarc_target = str(train.get("sarc_cal_target", "native")).lower()
    sarc_permutation = str(train.get("sarc_cal_level_permutation", "none")).lower().replace(" ", "")
    if sarc_model and any(
        float(train.get(key, 0.0)) > 0 for key in ("o2o_kd_gain", "tqgb_cls_m2o_gain", "tqgb_cls_o2m_gain")
    ):
        raise ValueError("SARC R1/R2/R3 must not be combined with KD or TQGB in the causal matrix.")
    if model_path.name.endswith("sarc-r1.yaml") and sarc_gain != 0:
        raise ValueError("SARC R1 requires sarc_cal_gain=0 exactly.")
    if model_path.name.endswith("sarc-r2.yaml") and sarc_gain != 0:
        raise ValueError("SARC R2 requires sarc_cal_gain=0 exactly.")
    if model_path.name.endswith("sarc-r3.yaml") and sarc_gain <= 0:
        raise ValueError("SARC R3 requires a positive sarc_cal_gain.")
    if not sarc_model and (sarc_gain != 0 or sarc_target != "native" or sarc_permutation != "none"):
        raise ValueError("SARC calibration options require an explicit yolo26-sarc model YAML.")

    print(f"Research ID: {spec.get('id')}")
    print(f"Hypothesis: {spec.get('hypothesis')}")
    print(f"SARC: model={sarc_model}, gain={sarc_gain:g}, target={sarc_target}, permutation={sarc_permutation}")
    print(f"Research diff: {git_output('diff', '--stat')}")
    YOLO(model_path).train(**train)


if __name__ == "__main__":
    main()
