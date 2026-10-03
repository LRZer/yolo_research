"""Conservative task-owned manager for the preregistered HBS-only pair.

No automatic checkpoint resume, no overwrite, no interaction with other jobs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.set_hbs_integration import HBSOnlyDetectionModel  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402

RUNS = ROOT / "runs" / "set_hbs_visdrone_960_e125"
MANAGEMENT = ROOT / "_management" / "set_hbs_visdrone_960_e125_s2027"
EXPECTED_UE = (0, 52)
JOBS = (
    ("hbs", "set_hbs_only_visdrone_960_s2027_e125.json",
     "8b6fcdb44508ed15dd08cdb294ad1abe789bda1c56a2f0e095e6474dc0857d4c",
     "hbs_only_s2027_g0123_numa0"),
    ("identity", "set_hbs_identity_visdrone_960_s2027_e125.json",
     "15cdc6e7579064e425dfa4700044233a23cd96039c3a604ae907d2fce7706609",
     "identity_aux_s2027_g0123_numa0"),
)


def record(event: str, **extra: object) -> None:
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    with (MANAGEMENT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                 "event": event, **extra}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def state(status: str, **extra: object) -> None:
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    payload = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": status,
               "manager_pid": os.getpid(), **extra}
    temp = MANAGEMENT / "state.json.tmp"
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(MANAGEMENT / "state.json")


def configs() -> dict[str, dict[str, object]]:
    found = {}
    for key, filename, expected_hash, name in JOBS:
        path = ROOT / "configs" / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"frozen configuration hash changed: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        expected = {"epochs": 125, "fraction": 1.0, "batch": 64, "imgsz": 960,
                    "workers": 4, "seed": 2027, "device": "0,1,2,3", "name": name,
                    "project": str(RUNS), "hbs_mode": key, "hbs_aux_weight": 1.0}
        if key == "hbs":
            expected["hbs_mode"] = "hbs"
        for field, value in expected.items():
            if config.get(field) != value:
                raise RuntimeError(f"{filename}: unexpected {field}={config.get(field)!r}")
        found[key] = config
    left, right = found["hbs"], found["identity"]
    differing = [key for key in set(left) | set(right)
                 if left.get(key) != right.get(key) and key not in {"name", "hbs_mode"}]
    if differing:
        raise RuntimeError(f"paired configurations differ unexpectedly: {sorted(differing)}")
    if not Path(str(left["data"])).is_file() or not Path(str(left["model"])).is_file():
        raise FileNotFoundError("data YAML or scratch model YAML missing")
    return found


def gpu_state() -> tuple[dict[int, tuple[str, int]], list[tuple[int, str]]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    gpus = {}
    for line in output.splitlines():
        index, uuid, used = (part.strip() for part in line.split(","))
        gpus[int(index)] = (uuid, int(used))
    process_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True, timeout=15)
    by_uuid = {uuid: index for index, (uuid, _) in gpus.items()}
    active = []
    for line in process_output.splitlines():
        uuid, pid = (part.strip() for part in line.split(","))
        if uuid in by_uuid:
            active.append((by_uuid[uuid], pid))
    return gpus, active


def preflight() -> dict[str, object]:
    configs()
    ue = tuple(int((Path("/sys/devices/system/edac/mc") / f"mc{i}" / "ue_count").read_text().strip())
               for i in (0, 1))
    if ue != EXPECTED_UE:
        raise RuntimeError(f"EDAC changed from {EXPECTED_UE} to {ue}; human review required")
    recent = subprocess.check_output(["journalctl", "-k", "--since", "10 minutes ago", "--no-pager"],
                                     text=True, timeout=15)
    if re.search(r"Uncorrected hardware memory error|Sending SIGBUS|Memory failure", recent, re.I):
        raise RuntimeError("new kernel MCE/SIGBUS/memory-failure record in last ten minutes")
    gpu, active = gpu_state()
    if any(index in (0, 1, 2, 3) for index, _ in active):
        raise RuntimeError(f"GPU0-3 compute process present: {active}")
    if any(gpu[index][1] > 1000 for index in (0, 1, 2, 3)):
        raise RuntimeError("GPU0-3 substantial untracked memory use")
    numa = subprocess.check_output(["numactl", "--hardware"], text=True, timeout=15)
    free_match = re.search(r"node 0 free:\s*(\d+) MB", numa)
    if not free_match or int(free_match.group(1)) < 25000:
        raise RuntimeError("NUMA0 local memory below 25 GB")
    disk_free = shutil.disk_usage(ROOT).free / (1024**3)
    if disk_free < 100:
        raise RuntimeError("project filesystem less than 100 GiB free")
    return {"edac_ue": ue, "numa0_free_mb": int(free_match.group(1)),
            "disk_free_gib": round(disk_free, 1), "other_gpu_processes": active}


def audit_run(run: Path, mode: str) -> None:
    with (run / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 125:
        raise RuntimeError(f"{run}: expected 125 rows, got {len(rows)}")
    for index, row in enumerate(rows, start=1):
        if int(float(row["epoch"])) != index:
            raise RuntimeError(f"{run}: discontinuous CSV at {index}")
        if any(value is None or not math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError(f"{run}: nonfinite CSV at {index}")
    for filename in ("best.pt", "last.pt"):
        model, checkpoint = load_checkpoint(run / "weights" / filename, device="cpu", fuse=False)
        if not isinstance(model, HBSOnlyDetectionModel) or model.nc != 10 or model.hbs_mode != mode:
            raise RuntimeError(f"{run}/{filename}: wrong model type/nc/HBS mode")
        if any(checkpoint["train_args"].get(k) != v for k, v in
               {"epochs": 125, "batch": 64, "imgsz": 960, "seed": 2027}.items()):
            raise RuntimeError(f"{run}/{filename}: wrong checkpoint training parameters")


def wait_for_gpu_release() -> None:
    for _ in range(12):
        gpu, active = gpu_state()
        if not any(index in (0, 1, 2, 3) for index, _ in active) and all(
            gpu[index][1] <= 1000 for index in (0, 1, 2, 3)):
            return
        time.sleep(5)
    raise RuntimeError("GPU0-3 did not become idle after previous task")


def run_job(key: str, filename: str, name: str) -> None:
    run = RUNS / name
    if run.exists():
        audit_run(run, key)
        record("already_verified_complete", job=key, run=str(run))
        return
    checks = preflight()
    record("preflight_passed", job=key, **checks)
    state("launching", job=key, run=str(run), **checks)
    command = ["numactl", "--cpunodebind=0", "--membind=0", sys.executable,
               str(ROOT / "tools" / "run_experiment.py"), "hbs_only",
               str(ROOT / "configs" / filename)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONUNBUFFERED"] = "1"
    with (MANAGEMENT / f"{key}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        record("job_started", job=key, pid=child.pid, command=command)
        state("running", job=key, pid=child.pid, run=str(run), **checks)
        return_code = child.wait()
    record("job_exited", job=key, pid=child.pid, return_code=return_code)
    state("auditing" if return_code == 0 else "failed", job=key,
          return_code=return_code, run=str(run))
    if return_code != 0:
        raise RuntimeError(f"{key} failed with exit {return_code}; no automatic resume")
    audit_run(run, key)
    record("job_verified_complete", job=key, run=str(run))
    state("job_verified_complete", job=key, run=str(run))
    wait_for_gpu_release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(preflight(), indent=2))
        return
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    with (MANAGEMENT / "manager.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another HBS pair manager is active") from error
        try:
            record("manager_started", pid=os.getpid())
            for key, filename, _hash, name in JOBS:
                run_job(key, filename, name)
            record("pair_verified_complete")
            state("verified_complete", jobs=[job[0] for job in JOBS])
        except Exception as error:
            record("manager_blocked", error=str(error))
            state("blocked", error=str(error))
            raise


if __name__ == "__main__":
    main()
