#!/usr/bin/env python3
"""Durable sequential queue for 22 YOLOv8n scratch baselines.

The queue is intentionally task-local. It never installs a system service or
touches another process. A heartbeat may relaunch this manager after a server
reboot; completed jobs are detected from their immutable run artifacts and
partial jobs resume from the newest readable checkpoint.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test")
DATA_ROOT = ROOT / "processed_yolo"
MODEL = ROOT / "models" / "yolov8n_scratch.yaml"
LAUNCHER = ROOT / "_management" / "train_yolov8n_scratch.sh"
QUEUE_ROOT = ROOT / "_management" / "yolov8n_scratch_queue"
STATE_PATH = QUEUE_ROOT / "queue_state.json"
LOCK_PATH = QUEUE_ROOT / "queue.lock"
MANAGER_LOG = QUEUE_ROOT / "manager_events.jsonl"
RUN_PROJECT = ROOT / "runs" / "yolov8n_scratch_baselines"
YOLO = Path("/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/yolo")
EXPECTED_EPOCHS = 125
EXPECTED_BATCH = 64
EXPECTED_IMGSZ = 640
SEEDS = (2027, 2005)
DATASETS = (
    "AP-10K", "DUO", "ExDark", "HIT-UAV", "HRSID-JPG",
    "LLVIP-infrared", "LLVIP-visible", "ORCA-class-level-seen",
    "SeaClear", "VisDrone2019-DET", "ZeroWaste-f",
)
MAX_FAILED_ATTEMPTS = 5
POLL_SECONDS = 120


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def log_event(event: str, **fields):
    rec = {"time": now_iso(), "event": event, **fields}
    with MANAGER_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def job_id(dataset: str, seed: int):
    return f"{dataset}__seed{seed}"


def run_name(dataset: str, seed: int):
    return f"yolov8n_scratch_{dataset}_e{EXPECTED_EPOCHS}_b{EXPECTED_BATCH}_i{EXPECTED_IMGSZ}_s{seed}"


def count_epochs(run_dir: Path):
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def last_metrics(run_dir: Path):
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return {k.strip(): v.strip() for k, v in rows[-1].items()}
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def is_complete(run_dir: Path):
    return (
        count_epochs(run_dir) >= EXPECTED_EPOCHS
        and (run_dir / "weights" / "last.pt").is_file()
        and (run_dir / "weights" / "best.pt").is_file()
    )


def pid_matches(pid: int):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return False
    return any(token in cmd for token in (
        "train_yolov8n_scratch.sh",
        "yolov8n_scratch_queue",
        "/bin/yolo detect train",
        "yolov8n_scratch_baselines",
    ))


def newest_valid_checkpoint(run_dir: Path):
    weights = run_dir / "weights"
    candidates = []
    last = weights / "last.pt"
    if last.is_file():
        candidates.append(last)
    periodic = sorted(
        weights.glob("epoch*.pt"),
        key=lambda p: int(re.search(r"epoch(\d+)", p.stem).group(1)) if re.search(r"epoch(\d+)", p.stem) else -1,
        reverse=True,
    )
    candidates.extend(periodic)
    best = weights / "best.pt"
    if best.is_file():
        candidates.append(best)
    if not candidates:
        return None
    import torch
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
                del ckpt
                return path
        except BaseException as exc:
            log_event("checkpoint_rejected", path=str(path), error=f"{type(exc).__name__}: {exc}")
    return None


def gpu_uuids_4_7():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    result = set()
    for line in out.splitlines():
        idx, uuid = [v.strip() for v in line.split(",", 1)]
        if int(idx) in (4, 5, 6, 7):
            result.add(uuid)
    return result


def gpu_processes_on_4_7():
    uuids = gpu_uuids_4_7()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    procs = []
    for line in out.splitlines():
        parts = [v.strip() for v in line.split(",", 2)]
        if len(parts) >= 2 and parts[0] in uuids:
            procs.append(parts)
    return procs


def load_or_create_state():
    jobs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            jobs.append({
                "id": job_id(dataset, seed), "dataset": dataset, "seed": seed,
                "status": "pending", "attempts": 0, "epochs_completed": 0,
                "run_dir": str(RUN_PROJECT / run_name(dataset, seed)),
            })
    if STATE_PATH.is_file():
        old = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        old_by_id = {j["id"]: j for j in old.get("jobs", [])}
        for job in jobs:
            if job["id"] in old_by_id:
                job.update(old_by_id[job["id"]])
        state = old
        state["jobs"] = jobs
    else:
        state = {
            "status": "initialized", "created": now_iso(), "updated": now_iso(),
            "configuration": {
                "epochs": EXPECTED_EPOCHS, "batch": EXPECTED_BATCH, "imgsz": EXPECTED_IMGSZ,
                "seeds": list(SEEDS), "datasets": list(DATASETS), "gpus_physical": [4, 5, 6, 7],
                "numa_node": 1, "workers_per_rank": 4, "pretrained": False,
                "model_sha256": sha256(MODEL),
                "dataset_yaml_sha256": {d: sha256(DATA_ROOT / d / "dataset.yaml") for d in DATASETS},
            },
            "jobs": jobs, "active_child_pid": None, "manager_pid": os.getpid(),
        }
    state["manager_pid"] = os.getpid()
    state["updated"] = now_iso()
    return state


def save_state(state, status=None):
    if status is not None:
        state["status"] = status
    state["updated"] = now_iso()
    state["completed_jobs"] = sum(j["status"] == "completed" for j in state["jobs"])
    state["total_jobs"] = len(state["jobs"])
    atomic_json(STATE_PATH, state)


def reconcile(state):
    for job in state["jobs"]:
        run_dir = Path(job["run_dir"])
        epochs = count_epochs(run_dir)
        job["epochs_completed"] = epochs
        if is_complete(run_dir):
            if job["status"] != "completed":
                log_event("job_reconciled_complete", job=job["id"], epochs=epochs)
            job["status"] = "completed"
            job["completed_at"] = job.get("completed_at", now_iso())
            job["last_metrics"] = last_metrics(run_dir)
    pid = state.get("active_child_pid")
    if pid and not pid_matches(int(pid)):
        state["active_child_pid"] = None
    save_state(state, "running")


def archive_unresumable(run_dir: Path):
    if not run_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = run_dir.with_name(run_dir.name + f".incomplete_no_valid_checkpoint.{stamp}")
    os.rename(run_dir, dest)
    log_event("unresumable_run_archived", source=str(run_dir), destination=str(dest))
    return dest


def command_for(job):
    run_dir = Path(job["run_dir"])
    checkpoint = newest_valid_checkpoint(run_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    env["PYTHONPATH"] = "/home/ubuntu/rz/yolo_study/ultralytics"
    env["YOLO_NUMA_BOUND"] = "1"
    prefix = ["numactl", "--cpunodebind=1", "--membind=1"]
    if checkpoint is not None and count_epochs(run_dir) < EXPECTED_EPOCHS:
        cmd = prefix + [
            str(YOLO), "detect", "train", f"model={checkpoint}", "resume=True",
            "device=0,1,2,3", "workers=4",
        ]
        return cmd, env, "resume", checkpoint
    if run_dir.exists() and count_epochs(run_dir) < EXPECTED_EPOCHS:
        archive_unresumable(run_dir)
    cmd = prefix + [
        str(LAUNCHER), job["dataset"], str(EXPECTED_EPOCHS), str(EXPECTED_BATCH),
        str(EXPECTED_IMGSZ), str(job["seed"]),
    ]
    return cmd, env, "fresh", None


def wait_for_existing_child(state):
    pid = state.get("active_child_pid")
    if not pid or not pid_matches(int(pid)):
        state["active_child_pid"] = None
        return
    log_event("reattached_to_existing_child", pid=pid, job=state.get("active_job"))
    while pid_matches(int(pid)):
        time.sleep(POLL_SECONDS)
    state["active_child_pid"] = None
    save_state(state, "running")


def run_job(state, job):
    while gpu_processes_on_4_7():
        procs = gpu_processes_on_4_7()
        state["status"] = "waiting_for_gpus_4_7"
        state["gpu_processes"] = procs
        save_state(state)
        log_event("waiting_for_gpus", processes=procs)
        time.sleep(300)
    cmd, env, mode, checkpoint = command_for(job)
    job["attempts"] = int(job.get("attempts", 0)) + 1
    job["status"] = "running" if mode == "fresh" else "resuming"
    job["started_at"] = job.get("started_at", now_iso())
    job["last_attempt_at"] = now_iso()
    state["active_job"] = job["id"]
    log_path = QUEUE_ROOT / f"{job['id']}.log"
    err_path = QUEUE_ROOT / f"{job['id']}.err.log"
    log_event("job_launch", job=job["id"], mode=mode, checkpoint=str(checkpoint) if checkpoint else None,
              attempt=job["attempts"], command=cmd)
    with log_path.open("ab", buffering=0) as stdout, err_path.open("ab", buffering=0) as stderr:
        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        state["active_child_pid"] = proc.pid
        save_state(state, "running")
        return_code = proc.wait()
    state["active_child_pid"] = None
    state["active_job"] = None
    epochs = count_epochs(Path(job["run_dir"]))
    job["epochs_completed"] = epochs
    job["last_return_code"] = return_code
    if is_complete(Path(job["run_dir"])):
        job["status"] = "completed"
        job["completed_at"] = now_iso()
        job["last_metrics"] = last_metrics(Path(job["run_dir"]))
        log_event("job_complete", job=job["id"], epochs=epochs, return_code=return_code,
                  metrics=job["last_metrics"])
        save_state(state, "running")
        return
    job["status"] = "retry_wait"
    log_event("job_incomplete", job=job["id"], epochs=epochs, return_code=return_code,
              attempt=job["attempts"])
    if job["attempts"] >= MAX_FAILED_ATTEMPTS:
        job["status"] = "blocked"
        state["blocking_job"] = job["id"]
        save_state(state, "blocked_repeated_failure")
        raise RuntimeError(f"{job['id']} failed {job['attempts']} times")
    save_state(state, "retry_wait")
    time.sleep(120)


def main():
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_PROJECT.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("queue manager already running", flush=True)
        return 0
    for required in (MODEL, LAUNCHER, YOLO):
        if not required.is_file():
            raise FileNotFoundError(required)
    state = load_or_create_state()
    log_event("manager_started", pid=os.getpid(), model_sha256=sha256(MODEL))
    reconcile(state)
    wait_for_existing_child(state)
    while True:
        reconcile(state)
        pending = [j for j in state["jobs"] if j["status"] != "completed"]
        if not pending:
            state["finished"] = now_iso()
            save_state(state, "all_22_jobs_completed")
            log_event("queue_complete", jobs=len(state["jobs"]))
            return 0
        blocked = [j for j in pending if j["status"] == "blocked"]
        if blocked:
            save_state(state, "blocked_repeated_failure")
            return 2
        run_job(state, pending[0])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        try:
            log_event("manager_fatal", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
