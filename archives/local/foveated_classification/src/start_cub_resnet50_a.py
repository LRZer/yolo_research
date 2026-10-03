"""Preflight and detach the task-specific single-GPU CUB baseline."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
RUN = ROOT / "runs/resnet50_A_native500_s2005_e200"
CONFIG = ROOT / "configs/cub_resnet50_A_native500_s2005_e200_gpu0.json"
TRAINER = ROOT / "src/train_cub_resnet50_a.py"
PYTHON = Path("/home/ubuntu/miniconda3/envs/LZCyolo/bin/python")
LOG = RUN / "train.log"


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, timeout=12).strip()


def active_trainers() -> list[int]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            fields = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if any(str(TRAINER).encode() == field for field in fields) and b"--train" in fields:
            matches.append(int(proc.name))
    return matches


def main() -> None:
    if active_trainers():
        raise RuntimeError(f"A task trainer is already running: {active_trainers()}")
    gpu_rows = run("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits").splitlines()
    gpu0 = [int(part.strip()) for part in gpu_rows[0].split(",")]
    if gpu0[0] != 0 or gpu0[1] > 256 or gpu0[2] > 5:
        raise RuntimeError(f"Physical GPU0 is not idle: {gpu0}")
    ue_count = int(Path("/sys/devices/system/edac/mc/mc0/ue_count").read_text().strip())
    if ue_count != 0:
        raise RuntimeError(f"NUMA0 memory controller has uncorrected errors this boot: {ue_count}")
    node_text = run("numactl", "--hardware")
    free_line = next(line for line in node_text.splitlines() if line.startswith("node 0 free:"))
    free_mb = int(free_line.split()[3])
    if free_mb < 16000:
        raise RuntimeError(f"NUMA0 has too little free memory: {free_mb} MB")
    if not CONFIG.is_file() or not TRAINER.is_file() or not PYTHON.is_file():
        raise FileNotFoundError("Missing task config, trainer, or Python runtime")
    RUN.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with LOG.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            ["numactl", "--cpunodebind=0", "--membind=0", str(PYTHON), "-u",
             str(TRAINER), "--config", str(CONFIG), "--train"],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f"Trainer exited immediately with code {process.returncode}; inspect {LOG}")
    launch = {
        "status": "launched", "pid": process.pid, "physical_gpu": 0, "numa_node": 0,
        "gpu0_memory_mib_before": gpu0[1], "numa0_free_mb_before": free_mb,
        "mc0_uncorrected_this_boot_before": ue_count,
        "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": ["numactl", "--cpunodebind=0", "--membind=0", str(PYTHON),
                    "-u", str(TRAINER), "--config", str(CONFIG), "--train"],
        "log": str(LOG),
    }
    (RUN / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
