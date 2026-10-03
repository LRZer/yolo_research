"""Stop only one identified CUB trainer after its requested completed epoch.

The original configurations remain frozen at 200 epochs so their checkpoint
hashes and first-125-epoch cosine learning rates remain comparable. This
watcher implements the user's revised 125-epoch endpoint without touching
other processes or modifying an in-flight training configuration.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import signal
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def proc_identity(pid: int, trainer: Path, config: Path) -> tuple[bool, str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False, "process_absent"
    args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    if str(trainer) not in args or str(config) not in args or "--train" not in args:
        return False, f"pid_command_mismatch:{args!r}"
    return True, "matched"


def csv_epochs(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [int(row["epoch"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target", type=int, default=125)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.config = args.config.resolve()
    args.trainer = args.trainer.resolve()
    if ROOT not in args.run.parents or ROOT not in args.config.parents or ROOT not in args.trainer.parents:
        raise ValueError("Every target must be inside this research directory")
    if args.target != 125 or args.pid <= 1 or not 0.05 <= args.poll_seconds <= 2:
        raise ValueError("Unexpected early-stop arguments")
    names = {
        "A_original": "resnet50_A_native500_s2005_e200",
        "B_global_blur": "resnet50_B_global_blur_native500_s2005_e200",
        "C_context_removed": "resnet50_C_context_removed_native500_s2005_e200",
        "D_gradual_background": "resnet50_D_gradual_background_native500_s2005_e200",
        "E_grayscale": "resnet50_E_grayscale_native500_s2005_e200",
        "F_scharr": "resnet50_F_scharr_native500_s2005_e200",
    }
    if args.run.name not in names.values():
        raise ValueError("Unexpected run directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["epochs"] != 200 or config["seed"] != 2005 or Path(config["run_dir"]).resolve() != args.run:
        raise ValueError("Run/config identity mismatch")
    if config["group"] not in names:
        raise ValueError("Unexpected group")
    if names[config["group"]] != args.run.name:
        raise ValueError("Group/run mismatch")
    expected_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    stop_path = args.run / "revised_e125_stop.json"
    if stop_path.exists():
        raise RuntimeError("Stop record already exists; refusing duplicate watcher")
    matched, reason = proc_identity(args.pid, args.trainer, args.config)
    if not matched:
        raise RuntimeError(f"Trainer not running at watcher launch: {reason}")
    print(json.dumps({"status": "watching", "group": config["group"], "pid": args.pid,
                      "target_epoch": args.target, "utc": utc_now()}), flush=True)
    while True:
        matched, reason = proc_identity(args.pid, args.trainer, args.config)
        if not matched:
            atomic_json(stop_path, {"status": "trainer_disappeared_before_target", "reason": reason,
                                    "target_epoch": args.target, "pid": args.pid, "utc": utc_now()})
            raise RuntimeError(reason)
        state_path = args.run / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(args.poll_seconds)
                continue
            epoch = int(state["epoch"])
            if epoch > args.target:
                atomic_json(stop_path, {"status": "missed_target_no_signal", "observed_epoch": epoch,
                                        "target_epoch": args.target, "pid": args.pid, "utc": utc_now()})
                raise RuntimeError("Trainer already passed target; no signal sent")
            if epoch == args.target:
                if state.get("config_sha256") != expected_hash:
                    raise RuntimeError("State/config hash mismatch; no signal sent")
                epochs = csv_epochs(args.run / "results.csv")
                if epochs != list(range(1, args.target + 1)):
                    raise RuntimeError("CSV not exactly continuous through target; no signal sent")
                if not (args.run / "last.pt").is_file():
                    raise RuntimeError("Target checkpoint absent; no signal sent")
                matched, reason = proc_identity(args.pid, args.trainer, args.config)
                if not matched:
                    raise RuntimeError(f"Trainer identity changed before signal: {reason}")
                os.kill(args.pid, signal.SIGTERM)
                payload = {"status": "sigterm_sent_after_completed_target", "group": config["group"],
                           "target_epoch": args.target, "pid": args.pid,
                           "config_sha256": expected_hash, "observed_csv_rows": len(epochs),
                           "utc": utc_now(), "note": "Checkpoint must pass a separate CPU audit."}
                atomic_json(stop_path, payload)
                print(json.dumps(payload), flush=True)
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
