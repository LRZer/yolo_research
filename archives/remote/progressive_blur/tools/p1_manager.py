"""Idempotent sequential manager for the paired P1 VOC experiments."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Running this file as ``python tools/p1_manager.py`` otherwise puts only the
# tools directory at sys.path[0].  Checkpoint unpickling must resolve the
# task-local ``pgb`` package from the research root, never a same-named package
# from site-packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import load_checkpoint, torch_safe_load


STATE = ROOT / "research" / "p1_queue_state.json"
EVENTS = ROOT / "research" / "p1_queue_events.jsonl"
TASKS = (
    ("baseline", ROOT / "configs" / "voc_a0_baseline_p1_e20.json"),
    ("pgb", ROOT / "configs" / "voc_d_pgb_p1_e20.json"),
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def event(event_type: str, **details) -> None:
    record = {"time": now(), "event": event_type, **details}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_rows(results: Path) -> list[dict[str, str]]:
    if not results.is_file():
        return []
    with results.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weights_readable(path: Path) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing"
    try:
        model, _ = load_checkpoint(path, device="cpu", fuse=False)
        if not all(
            tensor.isfinite().all()
            for tensor in model.state_dict().values()
            if tensor.is_floating_point()
        ):
            return False, "non-finite tensor"
        return True, None
    except Exception as error:  # preserve exact reason in state
        return False, repr(error)


def resumable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint, _ = torch_safe_load(path)
        return checkpoint.get("optimizer") is not None and checkpoint.get("epoch", -1) >= 0
    except Exception:
        return False


def run_dir(config: dict) -> Path:
    return Path(config["project"]) / config["name"]


def audit_task(kind: str, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    directory = run_dir(config)
    rows = read_rows(directory / "results.csv")
    expected = int(config["epochs"])
    status = "new"
    reason = None
    if rows:
        numeric_ok = True
        for row in rows:
            for key, value in row.items():
                if key and value and key != "epoch":
                    try:
                        numeric_ok &= math.isfinite(float(value))
                    except ValueError:
                        pass
        if not numeric_ok:
            status, reason = "invalid", "non-finite results.csv value"
        elif len(rows) == expected:
            best_ok, best_reason = weights_readable(directory / "weights" / "best.pt")
            last_ok, last_reason = weights_readable(directory / "weights" / "last.pt")
            if best_ok and last_ok:
                status = "complete"
            else:
                status, reason = "invalid", f"best={best_reason}; last={last_reason}"
        elif len(rows) < expected and resumable(directory / "weights" / "last.pt"):
            status = "resumable"
        else:
            status, reason = "blocked", f"{len(rows)}/{expected} rows and no resumable checkpoint"
    elif directory.exists():
        status, reason = "blocked", "run directory exists without results"
    return {
        "kind": kind,
        "config": str(config_path),
        "run_dir": str(directory),
        "expected_epochs": expected,
        "completed_epochs": len(rows),
        "status": status,
        "reason": reason,
    }


def summarize(tasks: list[dict]) -> dict:
    summary = {"updated": now(), "tasks": tasks}
    if all(task["status"] == "complete" for task in tasks):
        summary["status"] = "verified_complete"
    elif any(task["status"] in {"invalid", "blocked", "failed"} for task in tasks):
        summary["status"] = "blocked"
    else:
        summary["status"] = "running"
    return summary


def final_comparison(tasks: list[dict]) -> None:
    comparison = {"status": "preliminary_p1", "generated": now(), "experiments": {}}
    for task in tasks:
        rows = read_rows(Path(task["run_dir"]) / "results.csv")
        best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        comparison["experiments"][task["kind"]] = {
            # Ultralytics writes one-based epoch numbers to results.csv.
            "best_epoch": int(float(best["epoch"])),
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50_95": float(best["metrics/mAP50-95(B)"]),
            "final_epoch": rows[-1],
        }
    baseline = comparison["experiments"]["baseline"]
    pgb = comparison["experiments"]["pgb"]
    comparison["pgb_minus_baseline"] = {
        "precision": pgb["precision"] - baseline["precision"],
        "recall": pgb["recall"] - baseline["recall"],
        "mAP50": pgb["mAP50"] - baseline["mAP50"],
        "mAP50_95": pgb["mAP50_95"] - baseline["mAP50_95"],
    }
    write_json(ROOT / "research" / "p1_preliminary_comparison.json", comparison)


def main() -> None:
    event("manager_started", pid=None)
    for kind, config_path in TASKS:
        tasks = [audit_task(item_kind, item_config) for item_kind, item_config in TASKS]
        current = next(task for task in tasks if task["kind"] == kind)
        write_json(STATE, summarize(tasks))
        if current["status"] == "complete":
            event("task_skipped_complete", kind=kind)
            continue
        if current["status"] in {"blocked", "invalid"}:
            event("manager_blocked", kind=kind, reason=current["reason"])
            raise RuntimeError(current["reason"])

        config = json.loads(config_path.read_text(encoding="utf-8"))
        directory = run_dir(config)
        log = ROOT / "research" / f"p1_{kind}.log"
        command = [sys.executable, "-u", str(ROOT / "tools" / "run_experiment.py"), kind, str(config_path)]
        if current["status"] == "resumable":
            command.extend(["--resume", str(directory / "weights" / "last.pt")])
        event("task_started", kind=kind, mode=current["status"], command=command)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{now()}] command: {' '.join(command)}\n")
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            current["status"] = "failed"
            current["reason"] = f"exit code {completed.returncode}"
            write_json(STATE, summarize(tasks))
            event("task_failed", kind=kind, returncode=completed.returncode)
            raise SystemExit(completed.returncode)
        verified = audit_task(kind, config_path)
        if verified["status"] != "complete":
            event("task_post_audit_failed", kind=kind, audit=verified)
            raise RuntimeError(f"post-run audit failed: {verified}")
        event("task_verified_complete", kind=kind, epochs=verified["completed_epochs"])

    tasks = [audit_task(kind, config) for kind, config in TASKS]
    state = summarize(tasks)
    write_json(STATE, state)
    if state["status"] == "verified_complete":
        final_comparison(tasks)
        event("queue_verified_complete")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
