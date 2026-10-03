"""Run a baseline or PGB experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.oracle_integration import OracleBlurTrainer, OracleTrainOnlyBlurTrainer
from pgb.blackout_integration import OracleBlackoutTrainer
from pgb.ultralytics_integration import PGBDetectionTrainer
from pgb.safe_background_integration import SafeBackgroundTrainer
from pgb.set_hbs_integration import HBSOnlyDetectionTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind", choices=("baseline", "pgb", "safe_background", "hbs_only", "oracle", "oracle_train_only", "oracle_blackout")
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--name", help="unique new run name; never overwrite a prior run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.name:
        if args.resume is not None:
            parser.error("--name cannot be combined with --resume")
        config["name"] = args.name
    trainer_class = {
        "baseline": DetectionTrainer,
        "pgb": PGBDetectionTrainer,
        "safe_background": SafeBackgroundTrainer,
        "hbs_only": HBSOnlyDetectionTrainer,
        "oracle": OracleBlurTrainer,
        "oracle_train_only": OracleTrainOnlyBlurTrainer,
        "oracle_blackout": OracleBlackoutTrainer,
    }[args.kind]
    if args.resume is not None:
        config = {
            "resume": str(args.resume),
            "device": config["device"],
            "workers": config["workers"],
            "plots": config.get("plots", True),
            "val": config.get("val", True),
            "cache": config.get("cache", False),
        }
    trainer_class(overrides=config).train()


if __name__ == "__main__":
    main()
