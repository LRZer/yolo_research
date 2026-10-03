"""Task-owned sequential supervisor, sharing ONLY explicitly authorized GPUs2,3.

No process killing, environment changes, dataset writes, or automatic recovery.
GPU preflight and CPU checkpoint audit gate each NEXT launch. Healthy running
training is not terminated merely because a later hardware check would fail.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/ubuntu/rz/foveated_classification_research')
PYTHON = ROOT.parent / 'yolo_study/.conda/yolo26/bin/python'
TRAINER = ROOT / 'src/train_cub_global_local_ddp.py'
ENGINE = ROOT / 'src/train_cub_resnet50_a.py'
MANAGEMENT = ROOT / '_management/cub_global_local_ddp23_s2005'
CONFIGS = [ROOT / 'configs/cub_resnet50_G4_global_crop_s2005_e125_ddp23.json',
           ROOT / 'configs/cub_resnet50_G5_blank_crop_s2005_e125_ddp23.json']
UUIDS = {2: 'GPU-66f2712d-a00e-2fdb-e6da-be24056fa1d4',
         3: 'GPU-a47e8163-6e6f-6686-c5ee-c98142f6cb27'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def command(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def active_task_trainers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = [p.decode('utf-8', 'replace') for p in (proc / 'cmdline').read_bytes().split(b'\0') if p]
        except OSError:
            continue
        if any(p.startswith(str(ROOT / 'src/train_cub')) for p in parts) and ('--train' in parts or '--smoke' in parts):
            found.append(int(proc.name))
    return found


def preflight():
    if active_task_trainers():
        raise RuntimeError(f'Another task-owned CUB trainer active: {active_task_trainers()}')
    lines = command('nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits').splitlines()
    gpus = []
    for line in lines:
        fields = [x.strip() for x in line.split(',')]
        index = int(fields[0])
        if index not in UUIDS:
            continue
        if fields[1] != UUIDS[index] or int(fields[2]) < 12000:
            raise RuntimeError(f'GPU{index} mapping or free VRAM gate failed: {fields}')
        gpus.append(dict(index=index, uuid=fields[1], free_mib=int(fields[2]), utilization=int(fields[3])))
    if len(gpus) != 2:
        raise RuntimeError('Both authorized GPUs must be visible')
    edac = {str(p): int(p.read_text().strip()) for p in Path('/sys/devices/system/edac/mc').glob('mc*/ue_count')}
    if not edac or any(edac.values()):
        raise RuntimeError(f'Current EDAC UE prevents a NEW launch: {edac}')
    entries = [json.loads(line) for line in command('journalctl', '-k', '-b', '--no-pager', '-o', 'json').splitlines()]
    if not entries:
        raise RuntimeError('Kernel hardware log unavailable')
    fatal = [e for e in entries if re.search(r'Hardware Error|Machine check events logged|uncorrected|uncorrectable|SIGBUS|EDAC .*\bUE\b', e.get('MESSAGE', ''), re.I)]
    messages = [e.get('MESSAGE', '') for e in entries]
    times = [tuple(map(int, match.groups())) for message in messages
             if (match := re.search(r'PROCESSOR .* TIME (\d+) SOCKET (\d+)', message))]
    prior = (any('BERT: Error records from previous boot:' in m for m in messages) and bool(times)
             and all(t < int(entries[0]['__REALTIME_TIMESTAMP']) / 1e6 and socket == 1 for t, socket in times)
             and all(int(e['__MONOTONIC_TIMESTAMP']) < 120_000_000 for e in fatal))
    if fatal and not prior:
        raise RuntimeError('New or unlocalized MCE/SIGBUS; no new job launched')
    node = {}
    for line in Path('/sys/devices/system/node/node0/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        node[key.split()[-1]] = int(value.split()[0])
    mem = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
           for line in Path('/proc/meminfo').read_text().splitlines()}
    clean = max(0, node['Inactive(file)'] - node['Dirty'] - node['Writeback'] - node['Mapped'] - node['Shmem'])
    budget = (node['MemFree'] + .5 * clean + .25 * node['SReclaimable']) / 1024
    psi = Path('/proc/pressure/memory').read_text()
    if (budget < 16000 or mem['MemAvailable'] < 32 * 1024**2 or
        float(re.search(r'full avg10=([0-9.]+)', psi).group(1)) > .1 or
        float(re.search(r'some avg10=([0-9.]+)', psi).group(1)) > .5):
        raise RuntimeError('NUMA0 memory/pressure headroom gate failed')
    return dict(checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), gpus=gpus,
                numa_node=0, ue_counts=edac, memory_budget_mb=budget, psi=psi,
                historical_socket1_bert_replay=prior, hardware_repaired_not_claimed=True,
                shared_gpus_user_authorized=True)


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='2,3', CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return env


def job_command(config, mode, out=None):
    args = ['numactl', '--cpunodebind=0', '--membind=0', str(PYTHON), '-u', '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', '--nproc-per-node=2', str(TRAINER), '--config', str(config), mode]
    return args + (['--out', str(out)] if out else [])


def verified_cpu(config):
    group = json.loads(config.read_text())['group']
    report = json.loads((ROOT / 'reports' / f'{group}_DDP23_CPU_2026-10-01.json').read_text())
    if (report['status'] != 'verified_cpu_pair_pipeline' or report['config_sha256'] != sha(config) or
        report['source_sha256']['trainer'] != sha(TRAINER) or report['source_sha256']['historical_engine'] != sha(ENGINE)):
        raise RuntimeError(f'CPU/preview verification stale for {group}')
    return report


def smoke(config):
    cpu = verified_cpu(config)
    evidence = preflight()
    group = json.loads(config.read_text())['group']
    out = ROOT / 'reports' / f'{group}_DDP23_SMOKE_2026-10-01.json'
    log = ROOT / 'reports' / f'{group}_DDP23_SMOKE_2026-10-01.log'
    with log.open('wb') as f:
        result = subprocess.run(job_command(config, '--smoke', out), env=environment(),
                                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=600)
    if result.returncode:
        raise RuntimeError(f'DDP smoke failed: inspect {log}; no formal job launched')
    report = json.loads(out.read_text())
    if (report['status'] != 'verified_ddp_smoke' or report['config_sha256'] != sha(config)
        or report['source_sha256'] != cpu['source_sha256'] or report['row']['train_images'] != 170
        or report['row']['test_images'] != 34 or report['row']['update_groups'] != 3
        or report['row']['optimizer_steps'] != 3 or report['row']['amp_skipped_steps'] != 0):
        raise RuntimeError(f'Smoke count/gradient audit failed for {group}')
    if any(rank['peak_reserved_gib'] * 1024 + 3000 > gpu['free_mib']
           for rank, gpu in zip(report['rank_reports'], evidence['gpus'], strict=True)):
        raise RuntimeError('Measured task VRAM plus margin exceeds available VRAM')
    report['preflight'] = evidence
    json_write(out, report)
    return report


def audit_command(config):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    output = subprocess.check_output(['numactl', '--cpunodebind=0', '--membind=0', str(PYTHON), '-u',
                                     str(TRAINER), '--config', str(config), '--audit'], env=env, text=True, timeout=300)
    result = json.loads(output.splitlines()[-1])
    if result['status'] != 'verified_native_e125_ddp_pair':
        raise RuntimeError('E125 strict audit did not pass')
    return result


def fresh_smoke_or_run(config):
    """Reuse only a just-completed, hash-identical same-boot smoke, not old trials."""
    cpu = verified_cpu(config)
    evidence = preflight()
    group = json.loads(config.read_text())['group']
    path = ROOT / 'reports' / f'{group}_DDP23_SMOKE_2026-10-01.json'
    if path.is_file():
        report = json.loads(path.read_text())
        checked = report.get('preflight', {}).get('checked_at_utc')
        age = ((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(checked)).total_seconds()
               if checked else float('inf'))
        if (0 <= age < 900 and report.get('status') == 'verified_ddp_smoke'
            and report.get('config_sha256') == sha(config) and report.get('source_sha256') == cpu['source_sha256']
            and report['preflight']['boot_id'] == evidence['boot_id']
            and report['row']['optimizer_steps'] == 3 and report['row']['amp_skipped_steps'] == 0
            and all(rank['peak_reserved_gib'] * 1024 + 3000 <= gpu['free_mib']
                    for rank, gpu in zip(report['rank_reports'], evidence['gpus'], strict=True))):
            return report
    return smoke(config)


def event(stage, **details):
    item = dict(stage=stage, manager_pid=os.getpid(), at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **details)
    json_write(MANAGEMENT / 'state.json', item)
    with (MANAGEMENT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(item) + '\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(item), flush=True)


def supervise():
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    with (MANAGEMENT / 'manager.lock').open('x') as f:
        f.write(str(os.getpid()))
    # Freeze the pair contrast: only these three configuration fields may differ.
    first, second = [json.loads(p.read_text()) for p in CONFIGS]
    for key in set(first) | set(second):
        if key not in ('group', 'global_view', 'run_dir') and first.get(key) != second.get(key):
            raise RuntimeError(f'Paired control mismatch: {key}')
    if any(Path(c['run_dir']).exists() for c in (first, second)):
        raise RuntimeError('Existing run artifacts prevent duplicate launch')
    hashes = {str(p): sha(p) for p in CONFIGS + [TRAINER, ENGINE, Path(__file__)]}
    json_write(MANAGEMENT / 'launch.json', dict(manager_pid=os.getpid(), hashes=hashes,
        devices_physical=[2, 3], numa_node=0, actual_epochs_each=125, lr_horizon=200,
        control='sequential G4 full-global+crop then G5 constant-global+identical-crop',
        allowed_external_action='share existing GPUs without changing other jobs'))
    completed = []
    try:
        for config in CONFIGS:
            group = json.loads(config.read_text())['group']
            if any(sha(Path(p)) != digest for p, digest in hashes.items()):
                raise RuntimeError('Sources or configuration changed after launch')
            event('smoke_preflight', group=group)
            report = fresh_smoke_or_run(config)
            evidence = preflight()
            run = Path(json.loads(config.read_text())['run_dir'])
            log = MANAGEMENT / f'{group}.log'
            with log.open('wb', buffering=0) as f:
                process = subprocess.Popen(job_command(config, '--train'), env=environment(),
                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            event('training', group=group, torchrun_pid=process.pid, command=job_command(config, '--train'),
                  preflight=evidence, smoke_report=str(ROOT / 'reports' / f'{group}_DDP23_SMOKE_2026-10-01.json'))
            code = process.wait()
            if code:
                raise RuntimeError(f'{group} exited {code}; no unsafe automatic resume; inspect {log}')
            if active_task_trainers():
                raise RuntimeError('Task-owned rank/worker still active; hold next group for identity audit')
            event('cpu_auditing', group=group)
            result = audit_command(config)
            completed.append(result)
            json_write(ROOT / 'reports' / f'{group}_DDP23_E125_RESULT_2026-10-01.json', result)
            event('group_verified', group=group, result=result)
        a, b = completed
        comparison = dict(status='both_native_e125_verified', results=completed,
            g4_minus_g5_top1_pp=float(a['final_row']['test_top1']) - float(b['final_row']['test_top1']),
            g4_minus_g5_top5_pp=float(a['final_row']['test_top5']) - float(b['final_row']['test_top5']),
            limits=['GT boxes in train and official test: Oracle, not deployable',
                    'Monitored official test, single seed2005', 'Global contrast is not background-only causality',
                    'DDP SyncBN and newer runtime differ from historical single-card A-F',
                    'Shared GPUs: speed is not a controlled exclusive-compute benchmark'])
        json_write(ROOT / 'reports/CUB_GLOBAL_LOCAL_DDP23_FINAL_2026-10-01.json', comparison)
        event('all_verified', comparison=comparison)
    except Exception as error:
        event('blocked_or_failed', reason=str(error), completed_groups=[r['group'] for r in completed])
        raise


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--supervise', action='store_true')
    p.add_argument('--group', choices=['G4', 'G5'], default='G4')
    args = p.parse_args()
    if args.preflight:
        print(json.dumps(preflight()), flush=True)
    elif args.smoke:
        print(json.dumps(smoke(CONFIGS[0 if args.group == 'G4' else 1])), flush=True)
    else:
        supervise()


if __name__ == '__main__':
    main()
