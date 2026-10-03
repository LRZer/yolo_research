"""Preflight, smoke and launch only the authorized Scharr F on GPU6/NUMA0."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/rz/foveated_classification_research")
CONFIG = ROOT / "configs/cub_resnet50_F_scharr_native500_s2005_e200_gpu6.json"
RUN = ROOT / "runs/resnet50_F_scharr_native500_s2005_e200"
TRAINER = ROOT / "src/train_cub_resnet50_f.py"
WATCHER = ROOT / "src/stop_cub_run_at_epoch.py"
PYTHON = Path("/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python")
REPORT = ROOT / "reports/CUB_RESNET50_F_SCHARR_PREPARATION_2026-09-30.json"
SMOKE = ROOT / "reports/CUB_RESNET50_F_SCHARR_SMOKE_2026-09-30.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def active_trainers():
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [part.decode("utf-8", "replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if "--train" in args and any(str(ROOT / "src/train_cub_resnet50_") in arg for arg in args):
            matches.append({"pid": int(proc.name), "args": args})
    return matches


def preflight(allow_shared=False):
    config = json.loads(CONFIG.read_bytes())
    if config["device_physical"] != 6 or config["numa_node"] != 0 or config["group"] != "F_scharr":
        raise RuntimeError("Not the explicitly authorized GPU6/NUMA0 F run")
    report = json.loads(REPORT.read_bytes())
    if report["status"] != "F_cpu_verified_preview_exact" or report["config_sha256"] != sha(CONFIG):
        raise RuntimeError("CPU/preview audit is absent or stale")
    if report["trainer_sha256"] != sha(TRAINER) or report["baseline_engine_sha256"] != sha(ROOT / "src/train_cub_resnet50_a.py"):
        raise RuntimeError("Trainer/engine changed after CPU audit")
    if active_trainers():
        raise RuntimeError("Another task-owned CUB trainer is still active")
    gpu_lines = command("nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,memory.total",
                        "--format=csv,noheader,nounits").splitlines()
    fields = next(line.split(",") for line in gpu_lines if int(line.split(",")[0]) == 6)
    gpu = {"index": int(fields[0]), "uuid": fields[1].strip(),
           "memory_mib": int(fields[2]), "utilization_percent": int(fields[3]),
           "total_memory_mib": int(fields[4])}
    if not allow_shared and (gpu["memory_mib"] > 256 or gpu["utilization_percent"] > 5):
        raise RuntimeError(f"GPU6 is occupied; no process touched: {gpu}")
    apps = command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    gpu_apps = [line for line in apps.splitlines() if line.startswith(gpu["uuid"])]
    if not allow_shared and gpu_apps:
        raise RuntimeError("GPU6 has a compute process; refusing launch")
    if allow_shared and gpu["total_memory_mib"] - gpu["memory_mib"] < 11000:
        raise RuntimeError("Authorized shared GPU6 has insufficient VRAM headroom (requires 11000 MiB)")
    numa = command("numactl", "--hardware")
    free_mb = int(next(line for line in numa.splitlines() if line.startswith("node 0 free:")).split()[3])
    node_info = {}
    for line in Path("/sys/devices/system/node/node0/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        node_info[key.split()[-1]] = int(value.split()[0])
    global_info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        global_info[key] = int(value.split()[0])
    pressure = Path("/proc/pressure/memory").read_text()
    full_pressure = float(re.search(r"full avg10=([0-9.]+)", pressure).group(1))
    some_pressure = float(re.search(r"some avg10=([0-9.]+)", pressure).group(1))
    # MemFree alone excludes reclaimable cache. Discount file cache heavily and
    # require system-wide headroom/low PSI instead of forcing shared-cache drops.
    clean_inactive = max(0, node_info["Inactive(file)"] - node_info["Dirty"]
                         - node_info["Writeback"] - node_info["Mapped"] - node_info["Shmem"])
    budget_mb = (node_info["MemFree"] + 0.5 * clean_inactive
                 + 0.25 * node_info["SReclaimable"]) / 1024
    if (budget_mb < 16000 or global_info["MemAvailable"] < 32 * 1024 ** 2
            or full_pressure > 0.1 or some_pressure > 0.5):
        raise RuntimeError(f"Insufficient conservative NUMA0 memory headroom: {budget_mb:.0f} MB; PSI={pressure}")
    edac = {str(path): int(path.read_text().strip()) for path in Path("/sys/devices/system/edac/mc").glob("mc*/ue_count")}
    if not edac or any(edac.values()):
        raise RuntimeError(f"Current boot has uncorrected EDAC errors: {edac}")
    entries = [json.loads(line) for line in command("journalctl", "-k", "-b", "--no-pager", "-o", "json").splitlines()]
    fatal_entries = [entry for entry in entries if re.search(
        r"Hardware Error|Machine check events logged|uncorrected|uncorrectable|SIGBUS|EDAC .*\bUE\b",
        entry.get("MESSAGE", ""), re.I)]
    # Use journal record timestamps, not wall-clock-minus-uptime: firmware and
    # early-boot wall clocks can differ by seconds. BERT explicitly labels replay.
    first_kernel_wall_time = int(entries[0]["__REALTIME_TIMESTAMP"]) / 1_000_000
    messages = [entry.get("MESSAGE", "") for entry in entries]
    mce_times = [tuple(map(int, match.groups())) for message in messages
                 if (match := re.search(r"PROCESSOR .* TIME (\d+) SOCKET (\d+)", message))]
    prior_bert = any("BERT: Error records from previous boot:" in message for message in messages)
    historical_replay_verified = (prior_bert and bool(mce_times)
        and all(timestamp < first_kernel_wall_time and socket == 1 for timestamp, socket in mce_times)
        and all(int(entry["__MONOTONIC_TIMESTAMP"]) < 120_000_000 for entry in fatal_entries))
    if fatal_entries and not historical_replay_verified:
        raise RuntimeError(f"New or unlocalized hardware errors: {[entry.get('MESSAGE') for entry in fatal_entries][-10:]}")
    fatal = [] if historical_replay_verified else [entry.get("MESSAGE") for entry in fatal_entries]
    if not PYTHON.is_file() or not WATCHER.is_file():
        raise FileNotFoundError("Missing existing runtime or stopping watcher")
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "gpu": gpu, "numa0_free_mb": free_mb, "ue_counts": edac,
            "shared_gpu_user_authorized": allow_shared,
            "existing_gpu6_compute_apps": gpu_apps,
            "memory_gate": {"conservative_node0_reclaim_budget_mb": budget_mb,
                            "system_memavailable_mb": global_info["MemAvailable"] / 1024,
                            "memory_psi": pressure, "node0_meminfo_kb": node_info},
            "historical_previous_boot_socket1_replay_verified": historical_replay_verified,
            "historical_hardware_records": [entry.get("MESSAGE") for entry in fatal_entries] if historical_replay_verified else [],
            "new_boot_fatal_records": fatal,
            "hardware_limit": "Zero current-boot counters do not prove historical memory faults were repaired"}


def environment():
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "6"
    return env


def trainer_command(mode):
    return ["numactl", "--cpunodebind=0", "--membind=0", str(PYTHON), "-u", str(TRAINER),
            "--config", str(CONFIG), mode]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--allow-shared", action="store_true", help="Only after explicit human approval to share GPU6")
    args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.launch)) != 1:
        parser.error("Choose one operation")
    evidence = preflight(args.allow_shared)
    if args.preflight:
        print(json.dumps({"status": "F_preflight_passed", **evidence}, indent=2))
        return
    if args.smoke:
        output = subprocess.check_output(trainer_command("--smoke"), text=True, stderr=subprocess.STDOUT,
                                         env=environment(), timeout=180)
        lines = [line for line in output.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1])
        if result.get("status") != "smoke_passed" or result.get("images") != 32 or result.get("updates") != 1:
            raise RuntimeError(output)
        payload = {"config_sha256": sha(CONFIG), "trainer_sha256": sha(TRAINER),
                   "baseline_engine_sha256": sha(ROOT / "src/train_cub_resnet50_a.py"),
                   "physical_gpu": 6, "numa_node": 0, "preflight": evidence, "result": result,
                   "raw_output": output}
        write_json(SMOKE, payload)
        print(json.dumps(payload, indent=2))
        return
    smoke = json.loads(SMOKE.read_bytes())
    if smoke["config_sha256"] != sha(CONFIG) or smoke["trainer_sha256"] != sha(TRAINER):
        raise RuntimeError("GPU smoke result is stale")
    if smoke["baseline_engine_sha256"] != sha(ROOT / "src/train_cub_resnet50_a.py"):
        raise RuntimeError("Engine changed after smoke")
    if any((RUN / name).exists() for name in ("launch.json", "results.csv", "last.pt", "launch.lock")):
        raise RuntimeError("F run has artifacts; refusing duplicate launch")
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "launch.lock").open("x") as lock:
        lock.write(f"launcher_pid={os.getpid()}\n")
    with (RUN / "train.log").open("ab", buffering=0) as output:
        trainer = subprocess.Popen(trainer_command("--train"), stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=environment(), start_new_session=True)
    time.sleep(3)
    if trainer.poll() is not None:
        raise RuntimeError(f"Trainer exited at launch: {trainer.returncode}")
    watcher_command = ["numactl", "--cpunodebind=0", "--membind=0", str(PYTHON), "-u", str(WATCHER),
                       "--run", str(RUN), "--config", str(CONFIG), "--trainer", str(TRAINER),
                       "--pid", str(trainer.pid), "--target", "125"]
    watcher_env = os.environ.copy()
    watcher_env["CUDA_VISIBLE_DEVICES"] = ""
    with (RUN / "revised_e125_watcher.log").open("ab", buffering=0) as output:
        watcher = subprocess.Popen(watcher_command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=watcher_env, start_new_session=True)
    time.sleep(1)
    if watcher.poll() is not None or '"status": "watching"' not in (RUN / "revised_e125_watcher.log").read_text():
        os.kill(trainer.pid, signal.SIGTERM)
        raise RuntimeError("Stopping watcher did not initialize; stopped only the new F trainer")
    launch = {"status": "launched", "pid": trainer.pid, "watcher_pid": watcher.pid,
              "physical_gpu": 6, "numa_node": 0, "actual_target_epochs": 125,
              "learning_rate_schedule_total_epochs": 200, "config_sha256": sha(CONFIG),
              "command": trainer_command("--train"), "watcher_command": watcher_command,
              "runtime_change": "A-E recorded LZCyolo runtime is absent; F uses existing yolo26, versions in CPU report",
              "preflight": evidence, "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(RUN / "launch.json", launch)
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
