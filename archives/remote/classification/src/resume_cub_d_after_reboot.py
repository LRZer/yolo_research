"""One-shot, task-scoped D resume after a verified server reboot."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
CONFIG = ROOT / "configs/cub_resnet50_D_gradual_background_native500_s2005_e200_gpu0.json"
RUN = ROOT / "runs/resnet50_D_gradual_background_native500_s2005_e200"
TRAINER = ROOT / "src/train_cub_resnet50_oracle.py"
AUDITOR = ROOT / "src/audit_cub_resume.py"
PYTHON = Path("/home/ubuntu/miniconda3/envs/LZCyolo/bin/python")
RECORD = RUN / "resume_launch_epoch15.json"
TRAINER_NAMES = {
    "train_cub_resnet50_a.py", "train_cub_resnet50_b.py",
    "train_cub_resnet50_oracle.py", "train_cub_resnet50_e.py",
}


def active_task_trainers() -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [value.decode("utf-8", "replace") for value in
                    (proc / "cmdline").read_bytes().split(b"\0") if value]
        except (OSError, PermissionError):
            continue
        if "--train" in args and any(str(ROOT / "src") in arg and Path(arg).name in TRAINER_NAMES for arg in args):
            found.append(int(proc.name))
    return found


def main() -> None:
    if RECORD.exists():
        raise RuntimeError("This resume has already been launched")
    if (RUN / "revised_e125_stop.json").exists():
        raise RuntimeError("D already has a stop record")
    active = active_task_trainers()
    if active:
        raise RuntimeError(f"CUB trainer is active: {active}")
    original = json.loads((RUN / "launch.json").read_text(encoding="utf-8"))
    if original["pid"] != 1189729 or (Path("/proc") / str(original["pid"])).exists():
        raise RuntimeError("Original D process identity is unexpected or still present")
    audit = json.loads(subprocess.check_output(
        [str(PYTHON), str(AUDITOR), "--config", str(CONFIG), "--target-epochs", "125"],
        text=True, timeout=30))
    if audit["status"] != "verified_resumable_checkpoint" or audit["epoch"] != 15:
        raise RuntimeError("D checkpoint is not the verified epoch-15 checkpoint")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=12).splitlines()
    gpu = [int(field.strip()) for field in rows[0].split(",")]
    if gpu[0] != 0 or gpu[1] > 256 or gpu[2] > 5:
        raise RuntimeError(f"GPU0 is not idle: {gpu}")
    node = subprocess.check_output(["numactl", "--hardware"], text=True, timeout=12)
    free_line = next(line for line in node.splitlines() if line.startswith("node 0 free:"))
    free_mb = int(free_line.split()[3])
    if free_mb < 16000:
        raise RuntimeError(f"NUMA0 has insufficient free memory: {free_mb} MB")
    ue = int(Path("/sys/devices/system/edac/mc/mc0/ue_count").read_text().strip())
    if ue != 0:
        raise RuntimeError(f"NUMA0 has new uncorrected memory errors: {ue}")
    kernel = subprocess.check_output(["journalctl", "-k", "-b", "--no-pager"], text=True, timeout=12).lower()
    if "sigbus" in kernel or "uncorrectable" in kernel or "uncorrected" in kernel:
        raise RuntimeError("Current boot has SIGBUS or explicit uncorrectable hardware error")
    command = ["numactl", "--cpunodebind=0", "--membind=0", str(PYTHON), "-u",
               str(TRAINER), "--config", str(CONFIG), "--train"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with (RUN / "train.log").open("ab", buffering=0) as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f"D resume exited immediately with code {process.returncode}")
    record = {"status": "launched_resume", "pid": process.pid, "resume_from_epoch": audit["epoch"],
              "checkpoint_sha256": audit["last_checkpoint_sha256"],
              "config_sha256": audit["config_sha256"], "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "gpu0_before": gpu, "numa0_free_mb_before": free_mb,
              "mc0_ue_count_before": ue, "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command": command}
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
