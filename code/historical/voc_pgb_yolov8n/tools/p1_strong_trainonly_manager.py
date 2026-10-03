"""Run and verify the 20-epoch strong Oracle train-only blur control."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from p1_blackout_manager import now, read_rows, resumable, weights_ok, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "voc_oracle_strong_trainonly_p1_e20.json"
STATE = ROOT / "research" / "p1_strong_trainonly_state.json"
EVENTS = ROOT / "research" / "p1_strong_trainonly_events.jsonl"
LOG = ROOT / "research" / "p1_strong_trainonly.log"


def event(name: str, **details: object) -> None:
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), "event": name, **details}) + "\n")


def audit() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    directory = Path(config["project"]) / config["name"]
    rows = read_rows(directory / "results.csv")
    expected = int(config["epochs"])
    result = {
        "updated": now(),
        "status": "new",
        "reason": None,
        "completed_epochs": len(rows),
        "expected_epochs": expected,
        "run_dir": str(directory),
    }
    if rows:
        try:
            epochs = [int(float(row["epoch"])) for row in rows]
            values_finite = all(
                math.isfinite(float(value))
                for row in rows
                for key, value in row.items()
                if key and value and key != "epoch"
            )
        except (KeyError, TypeError, ValueError) as error:
            result.update(status="invalid", reason=f"bad CSV: {error}")
            return result
        if epochs != list(range(1, len(rows) + 1)) or not values_finite:
            result.update(status="invalid", reason="bad epoch sequence or non-finite result")
        elif len(rows) > expected:
            result.update(status="invalid", reason="more than expected epochs")
        elif len(rows) == expected:
            best_ok, best_reason = weights_ok(directory / "weights" / "best.pt")
            last_ok, last_reason = weights_ok(directory / "weights" / "last.pt")
            if best_ok and last_ok:
                result["status"] = "complete"
            else:
                result.update(status="invalid", reason=f"best={best_reason}; last={last_reason}")
        elif resumable(directory / "weights" / "last.pt"):
            result["status"] = "resumable"
        else:
            result.update(status="blocked", reason="partial run without valid resume checkpoint")
    elif directory.exists():
        result.update(status="blocked", reason="run directory exists without result row")
    return result


def main() -> None:
    global CONFIG, STATE, EVENTS, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--tag", default="p1_strong_trainonly")
    parser.add_argument("--kind", choices=("oracle_train_only", "baseline"), default="oracle_train_only")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.tag):
        raise ValueError("tag must use lowercase letters, digits and underscore")
    CONFIG = args.config.resolve()
    if CONFIG.parent != (ROOT / "configs").resolve():
        raise ValueError("config must be in this project's configs directory")
    STATE = ROOT / "research" / f"{args.tag}_state.json"
    EVENTS = ROOT / "research" / f"{args.tag}_events.jsonl"
    LOG = ROOT / "research" / f"{args.tag}.log"
    state = audit()
    write_json(STATE, state)
    if state["status"] in {"blocked", "invalid"}:
        event("blocked_before_start", reason=state["reason"])
        raise RuntimeError(state["reason"])
    if state["status"] != "complete":
        command = [
            sys.executable,
            "-u",
            str(ROOT / "tools" / "run_experiment.py"),
            args.kind,
            str(CONFIG),
        ]
        if state["status"] == "resumable":
            command += ["--resume", str(Path(state["run_dir"]) / "weights" / "last.pt")]
        event("started", mode=state["status"], command=command)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{now()}] command: {' '.join(command)}\n")
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            state = audit()
            state.update(status="failed", reason=f"exit code {completed.returncode}")
            write_json(STATE, state)
            event("failed", returncode=completed.returncode)
            raise SystemExit(completed.returncode)
    state = audit()
    if state["status"] != "complete":
        write_json(STATE, state)
        event("post_run_audit_failed", state=state)
        raise RuntimeError(f"run not verified: {state}")
    state["status"] = "verified_complete"
    write_json(STATE, state)
    event("verified_complete", epochs=state["completed_epochs"])
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
