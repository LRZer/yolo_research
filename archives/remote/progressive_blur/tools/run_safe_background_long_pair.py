"""Run the preregistered 125-epoch VOC pair sequentially and conservatively.

Run this task-scoped manager detached on gpu8-team. It will not overwrite an
existing run, resume a partial checkpoint, use GPU4-7, or kill a process.
After a failure/reboot a human or the thread monitor must inspect the state
and checkpoint before attempting a separate, explicit resume.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


MANAGEMENT = ROOT / "_management" / "sbg_long_e125_s2027"
RUNS = ROOT / "runs" / "sbg_long_e125"
DATA = ROOT / "data_splits" / "voc_strict_seed2027.yaml"
EXPECTED_UE = (0, 52)
EXPECTED_DEV_IMAGES = 1655
MIN_TRAIN_IMAGES = 14000
MIN_NODE0_FREE_MB = 25000
MIN_DISK_FREE_GB = 100
JOBS = (
    {
        "key": "safe_background",
        "kind": "safe_background",
        "config": "safe_background_voc_long_e125_s2027.json",
        "sha256": "4dd94bdf466e45d0b0df2372ea83e15257d02336c87eeaf3d2402e4529c38c8b",
        "name": "sbg_joint_voc_train90_e125_s2027_g0123_numa0",
    },
    {
        "key": "baseline",
        "kind": "baseline",
        "config": "baseline_voc_long_e125_s2027.json",
        "sha256": "b946cd73fd9186a3b0bf81178e75f8d0ae547bfb28a4ec17eddb4ce1d9802d64",
        "name": "baseline_voc_train90_e125_s2027_g0123_numa0",
    },
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def record(kind: str, **details: object) -> None:
    event = {"time_utc": now(), "event": kind, **details}
    with (MANAGEMENT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def state(status: str, **details: object) -> None:
    payload = {"time_utc": now(), "status": status, "manager_pid": os.getpid(), **details}
    target = MANAGEMENT / "state.json"
    temp = MANAGEMENT / "state.json.tmp"
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(target)


def require_configs() -> None:
    for job in JOBS:
        path = ROOT / "configs" / str(job["config"])
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != job["sha256"]:
            raise RuntimeError(f"preregistered config hash changed: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        if (
            config["name"] != job["name"]
            or config["epochs"] != 125
            or config["batch"] != 64
            or config["workers"] != 4
            or config["device"] != "0,1,2,3"
            or Path(config["project"]) != RUNS
            or Path(config["data"]) != DATA
        ):
            raise RuntimeError(f"unexpected long-run settings: {path}")
    if len([line for line in (ROOT / "data_splits" / "voc_dev10_seed2027.txt").read_text().splitlines() if line]) != EXPECTED_DEV_IMAGES:
        raise RuntimeError("VOC dev list changed")
    train_count = len([line for line in (ROOT / "data_splits" / "voc_train90_seed2027.txt").read_text().splitlines() if line])
    if train_count < MIN_TRAIN_IMAGES:
        raise RuntimeError(f"VOC train list unexpectedly short: {train_count}")


def current_ue() -> tuple[int, int]:
    return tuple(
        int((Path("/sys/devices/system/edac/mc") / f"mc{index}" / "ue_count").read_text().strip())
        for index in (0, 1)
    )


def gpu_usage() -> tuple[dict[int, tuple[str, int]], list[tuple[int, str]]]:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    mapping: dict[int, tuple[str, int]] = {}
    for line in result.splitlines():
        index, uuid, memory = (part.strip() for part in line.split(","))
        mapping[int(index)] = (uuid, int(memory))
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    uuid_to_index = {uuid: index for index, (uuid, _) in mapping.items()}
    active: list[tuple[int, str]] = []
    for line in processes.splitlines():
        uuid, pid = (part.strip() for part in line.split(","))
        if uuid in uuid_to_index:
            active.append((uuid_to_index[uuid], pid))
    return mapping, active


def preflight() -> dict[str, object]:
    require_configs()
    ue = current_ue()
    if ue != EXPECTED_UE:
        raise RuntimeError(f"EDAC counts changed from preflight {EXPECTED_UE} to {ue}; review hardware before a new job")
    gpu, active = gpu_usage()
    occupied = [(index, pid) for index, pid in active if index in (0, 1, 2, 3)]
    if occupied:
        raise RuntimeError(f"physical GPU0-3 are occupied: {occupied}")
    if any(gpu[index][1] > 1000 for index in (0, 1, 2, 3)):
        raise RuntimeError(f"physical GPU0-3 have substantial untracked memory use: {gpu}")
    numa = subprocess.check_output(["numactl", "--hardware"], text=True, timeout=15)
    match = re.search(r"node 0 free:\s*(\d+) MB", numa)
    if not match or int(match.group(1)) < MIN_NODE0_FREE_MB:
        raise RuntimeError(f"NUMA0 free memory is below {MIN_NODE0_FREE_MB} MB or unavailable")
    disk_gb = shutil.disk_usage(ROOT).free / (1024**3)
    if disk_gb < MIN_DISK_FREE_GB:
        raise RuntimeError(f"project disk has only {disk_gb:.1f} GiB free")
    return {
        "edac_ue": ue,
        "numa0_free_mb": int(match.group(1)),
        "disk_free_gib": round(disk_gb, 1),
        "gpu_0_3_memory_mib": [gpu[index][1] for index in (0, 1, 2, 3)],
        "other_active_gpu_processes": active,
    }


def completed_checkpoint(run: Path) -> None:
    result_path = run / "results.csv"
    with result_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, 126)):
        raise RuntimeError(f"{run.name}: expected consecutive epochs 1..125, got {epochs[-5:]}")
    for filename in ("best.pt", "last.pt"):
        model, _ = load_checkpoint(run / "weights" / filename, device="cpu", fuse=False)
        if getattr(model, "nc", None) != 20:
            raise RuntimeError(f"{run.name}/{filename}: unexpected number of classes")


def wait_for_gpu_release() -> None:
    for _ in range(12):
        gpu, active = gpu_usage()
        if not any(index in (0, 1, 2, 3) for index, _ in active) and all(gpu[index][1] <= 1000 for index in (0, 1, 2, 3)):
            return
        time.sleep(5)
    raise RuntimeError("GPU0-3 did not become idle after the previous task")


def run_job(job: dict[str, object]) -> None:
    run = RUNS / str(job["name"])
    if run.exists():
        completed_checkpoint(run)
        record("already_verified_complete", job=job["key"], run=str(run))
        return
    checks = preflight()
    record("preflight_passed", job=job["key"], **checks)
    state("launching", job=job["key"], run=str(run), **checks)
    command = [
        "numactl", "--cpunodebind=0", "--membind=0",
        sys.executable, str(ROOT / "tools" / "run_experiment.py"),
        str(job["kind"]), str(ROOT / "configs" / str(job["config"])),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONUNBUFFERED"] = "1"
    with (MANAGEMENT / f"{job['key']}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        record("job_started", job=job["key"], pid=child.pid, command=command)
        state("running", job=job["key"], pid=child.pid, run=str(run), **checks)
        exit_code = child.wait()
    record("job_exited", job=job["key"], pid=child.pid, exit_code=exit_code)
    state("auditing" if exit_code == 0 else "failed", job=job["key"], exit_code=exit_code, run=str(run))
    if exit_code:
        raise RuntimeError(f"{job['key']} exited with {exit_code}; inspect its task log and do not auto-resume")
    completed_checkpoint(run)
    record("job_verified_complete", job=job["key"], run=str(run))
    state("job_verified_complete", job=job["key"], run=str(run))
    wait_for_gpu_release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="read-only preflight; do not start training")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(preflight(), indent=2, ensure_ascii=False))
        return
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    with (MANAGEMENT / "manager.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another long-pair manager already holds the lock") from exc
        try:
            record("manager_started", pid=os.getpid())
            for job in JOBS:
                run_job(job)
            record("training_pair_verified_complete")
            state("verified_complete", jobs=[job["key"] for job in JOBS])
        except Exception as exc:
            record("manager_blocked", error=str(exc))
            state("blocked", error=str(exc))
            raise


if __name__ == "__main__":
    main()
