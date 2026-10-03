"""Strict read-only audit of the paired 125-epoch VOC long training runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.safe_background_integration import SafeBackgroundDetectionModel  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


RUN_NAMES = {
    "method": "sbg_joint_voc_train90_e125_s2027_g0123_numa0",
    "baseline": "baseline_voc_train90_e125_s2027_g0123_numa0",
}
CONFIG_NAMES = {
    "method": "safe_background_voc_long_e125_s2027.json",
    "baseline": "baseline_voc_long_e125_s2027.json",
}
EXPECTED = {"epochs": 125, "batch": 64, "imgsz": 640, "seed": 2027}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_csv(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 125:
        raise AssertionError(f"{path}: expected 125 rows, got {len(rows)}")
    for index, row in enumerate(rows, start=1):
        if int(float(row["epoch"])) != index:
            raise AssertionError(f"{path}: nonconsecutive epoch at row {index}")
        for key, value in row.items():
            if value is None or not math.isfinite(float(value)):
                raise AssertionError(f"{path}: nonfinite {key} at epoch {index}")
    return {
        "rows": len(rows),
        "last_epoch": int(float(rows[-1]["epoch"])),
        "last_time_seconds": float(rows[-1]["time"]),
        "last_map50": float(rows[-1]["metrics/mAP50(B)"]),
        "last_map50_95": float(rows[-1]["metrics/mAP50-95(B)"]),
        "sha256": digest(path),
    }


def audit_weight(path: Path, config: dict[str, object], has_frontend: bool) -> dict[str, object]:
    model, checkpoint = load_checkpoint(path, device="cpu", fuse=False)
    if getattr(model, "nc", None) != 20:
        raise AssertionError(f"{path}: nc != 20")
    if isinstance(model, SafeBackgroundDetectionModel) != has_frontend:
        raise AssertionError(f"{path}: unexpected model class {type(model)}")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise AssertionError(f"{path}: missing train_args")
    for key, expected in EXPECTED.items():
        if train_args.get(key) != expected:
            raise AssertionError(f"{path}: train_args[{key}]={train_args.get(key)!r} != {expected!r}")
    for key in ("data", "device"):
        if str(train_args.get(key)) != str(config[key]):
            raise AssertionError(f"{path}: train_args[{key}] differs from frozen config")
    # Ultralytics omits custom trainer keys from checkpoint train_args. They
    # are instead persisted as attributes on this project's model subclass.
    if has_frontend:
        observed = {
            "pgb_map_loss_weight": model.map_loss_weight,
            "pgb_eval_mix": float(model.pgb_eval_mix),
            "sbg_sigma": model.sbg_sigma,
            "sbg_far_threshold": model.sbg_far_threshold,
            "sbg_protect_weight": model.sbg_protect_weight,
            "sbg_far_weight": model.sbg_far_weight,
            "sbg_map_only": model.sbg_map_only,
        }
        for key, value in observed.items():
            if value != config[key]:
                raise AssertionError(f"{path}: saved model {key}={value!r} differs from frozen config")
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "model_class": type(model).__name__,
        "nc": model.nc,
        "checkpoint_epoch_field": checkpoint.get("epoch"),
        "train_args_checked": sorted((*EXPECTED, "data", "device")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configs = {
        kind: json.loads((args.root / "configs" / name).read_text(encoding="utf-8"))
        for kind, name in CONFIG_NAMES.items()
    }
    shared_keys = (set(configs["method"]) & set(configs["baseline"])) - {"model", "name"}
    disagreements = [key for key in sorted(shared_keys) if configs["method"][key] != configs["baseline"][key]]
    if disagreements:
        raise AssertionError(f"shared training configuration differs: {disagreements}")
    for kind, config in configs.items():
        if any(config.get(key) != value for key, value in EXPECTED.items()):
            raise AssertionError(f"{kind}: unexpected frozen training hyperparameters")
    report: dict[str, object] = {"status": "verified_complete", "shared_config_fields": sorted(shared_keys), "runs": {}}
    for kind, name in RUN_NAMES.items():
        run = args.root / "runs" / "sbg_long_e125" / name
        config = configs[kind]
        saved_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        for key in (*EXPECTED, "data", "device"):
            if str(saved_args.get(key)) != str(config[key]):
                raise AssertionError(f"{kind}: args.yaml[{key}] differs from frozen config")
        weights = {
            filename: audit_weight(run / "weights" / filename, config, kind == "method")
            for filename in ("best.pt", "last.pt")
        }
        report["runs"][kind] = {  # type: ignore[index]
            "run": str(run),
            "config_sha256": digest(args.root / "configs" / CONFIG_NAMES[kind]),
            "args_yaml_sha256": digest(run / "args.yaml"),
            "csv": audit_csv(run / "results.csv"),
            "weights": weights,
        }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
