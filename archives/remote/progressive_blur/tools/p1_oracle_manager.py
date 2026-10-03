"""Idempotent manager and audit for the P1 ground-truth oracle control."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import load_checkpoint, torch_safe_load


CONFIG = ROOT / "configs" / "voc_oracle_p1_e20.json"
BASELINE_CONFIG = ROOT / "configs" / "voc_a0_baseline_p1_e20.json"
STATE = ROOT / "research" / "p1_oracle_state.json"
EVENTS = ROOT / "research" / "p1_oracle_events.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def event(event_type: str, **details) -> None:
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), "event": event_type, **details}) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_dir(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return Path(config["project"]) / config["name"]


def readable_weights(path: Path) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing"
    try:
        model, _ = load_checkpoint(path, device="cpu", fuse=False)
        finite = all(
            value.isfinite().all()
            for value in model.state_dict().values()
            if value.is_floating_point()
        )
        return (True, None) if finite else (False, "non-finite tensor")
    except Exception as error:
        return False, repr(error)


def resumable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint, _ = torch_safe_load(path)
        return checkpoint.get("optimizer") is not None and checkpoint.get("epoch", -1) >= 0
    except Exception:
        return False


def audit() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    directory = run_dir(CONFIG)
    rows = read_rows(directory / "results.csv")
    expected = int(config["epochs"])
    status, reason = "new", None
    if rows:
        numeric_ok = all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key and value and key != "epoch"
        )
        if not numeric_ok:
            status, reason = "invalid", "non-finite results.csv value"
        elif len(rows) == expected:
            best_ok, best_reason = readable_weights(directory / "weights" / "best.pt")
            last_ok, last_reason = readable_weights(directory / "weights" / "last.pt")
            if best_ok and last_ok:
                status = "complete"
            else:
                status, reason = "invalid", f"best={best_reason}; last={last_reason}"
        elif len(rows) < expected and resumable(directory / "weights" / "last.pt"):
            status = "resumable"
        else:
            status, reason = "blocked", f"{len(rows)}/{expected} rows without resumable checkpoint"
    elif directory.exists():
        status, reason = "blocked", "run directory exists without results"
    return {
        "updated": now(),
        "status": status,
        "reason": reason,
        "completed_epochs": len(rows),
        "expected_epochs": expected,
        "run_dir": str(directory),
    }


def best_metrics(config_path: Path) -> dict[str, float | int]:
    rows = read_rows(run_dir(config_path) / "results.csv")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "best_epoch": int(float(best["epoch"])),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best["metrics/mAP50-95(B)"]),
    }


def write_comparison() -> None:
    baseline = best_metrics(BASELINE_CONFIG)
    oracle = best_metrics(CONFIG)
    delta = {
        key: oracle[key] - baseline[key]
        for key in ("precision", "recall", "mAP50", "mAP50_95")
    }
    write_json(
        ROOT / "research" / "p1_oracle_comparison.json",
        {
            "status": "diagnostic_oracle_not_deployable",
            "generated": now(),
            "selection": "maximum development mAP50-95",
            "baseline": baseline,
            "oracle": oracle,
            "oracle_minus_baseline": delta,
        },
    )


def main() -> None:
    event("oracle_manager_started")
    state = audit()
    write_json(STATE, state)
    if state["status"] in {"invalid", "blocked"}:
        event("oracle_manager_blocked", reason=state["reason"])
        raise RuntimeError(state["reason"])
    if state["status"] != "complete":
        command = [
            sys.executable,
            "-u",
            str(ROOT / "tools" / "run_experiment.py"),
            "oracle",
            str(CONFIG),
        ]
        if state["status"] == "resumable":
            command.extend(["--resume", str(run_dir(CONFIG) / "weights" / "last.pt")])
        event("oracle_started", mode=state["status"], command=command)
        with (ROOT / "research" / "p1_oracle.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{now()}] command: {' '.join(command)}\n")
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            state = audit()
            state["status"] = "failed"
            state["reason"] = f"exit code {completed.returncode}"
            write_json(STATE, state)
            event("oracle_failed", returncode=completed.returncode)
            raise SystemExit(completed.returncode)
    state = audit()
    if state["status"] != "complete":
        event("oracle_post_audit_failed", audit=state)
        raise RuntimeError(f"oracle post-run audit failed: {state}")
    state["status"] = "verified_complete"
    write_json(STATE, state)
    write_comparison()
    event("oracle_verified_complete", epochs=state["completed_epochs"])
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

