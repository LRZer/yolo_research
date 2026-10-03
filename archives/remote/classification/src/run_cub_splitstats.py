"""Task-owned G6 supervisor. No historical writes, killing, or unsafe resume."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import run_cub_global_local_pair as safety

ROOT, PYTHON = safety.ROOT, safety.PYTHON
TRAINER = ROOT / 'src/train_cub_splitstats_ddp.py'
CONFIG = ROOT / 'configs/cub_resnet50_G6_splitstats_global_crop_s2005_e125_ddp23.json'
CONTROL = ROOT / 'configs/cub_resnet50_G4_global_crop_s2005_e125_ddp23.json'
MANAGEMENT = ROOT / '_management/cub_splitstats_ddp23_s2005'
PREFIX = ROOT / 'reports/G6_splitstats_DDP23_2026-10-02'
CPU = PREFIX.with_name(PREFIX.name + '_CPU.json')
CONTROL_REPORT = PREFIX.with_name(PREFIX.name + '_CONTROL.json')
NORM_SMOKE = PREFIX.with_name(PREFIX.name + '_NORM_SMOKE.json')
SMOKE = PREFIX.with_name(PREFIX.name + '_SMOKE.json')


def job_command(mode, out=None):
    args = ['numactl', '--cpunodebind=0', '--membind=0', str(PYTHON), '-u', '-m',
            'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2',
            str(TRAINER), '--config', str(CONFIG), mode]
    return args + (['--out', str(out)] if out else [])


def files():
    return [CONFIG, CONTROL, TRAINER, Path(__file__), safety.TRAINER, safety.ENGINE, Path(safety.__file__)]


def frozen_hashes():
    return {str(path): safety.sha(path) for path in files()}


def cpu_verified():
    cpu = json.loads(CPU.read_text())
    control = json.loads(CONTROL_REPORT.read_text())
    if (cpu['status'] != 'verified_cpu_splitstats_pipeline' or cpu['config_sha256'] != safety.sha(CONFIG)
        or cpu['source_sha256']['trainer'] != safety.sha(TRAINER)
        or cpu['source_sha256']['paired_trainer'] != safety.sha(safety.TRAINER)
        or cpu['source_sha256']['historical_engine'] != safety.sha(safety.ENGINE)
        or control['status'] != 'verified_readonly_g4_control'
        or control['config_sha256'] != safety.sha(CONTROL)
        or not cpu['normalization_checks']['all_initial_parameters_identical_to_g4']):
        raise RuntimeError('CPU/G4 comparison verification is missing or stale')
    control_run = Path(json.loads(CONTROL.read_text())['run_dir'])
    if safety.sha(control_run / 'results.csv') != control['csv_sha256']:
        raise RuntimeError('Historical G4 CSV changed')
    for name, digest in control['checkpoint_sha256'].items():
        if safety.sha(control_run / name) != digest:
            raise RuntimeError('Historical G4 checkpoint changed')
    return cpu, control


def smoke():
    cpu, _ = cpu_verified()
    evidence = safety.preflight()
    reports = []
    for mode, out in (('--norm-smoke', NORM_SMOKE), ('--smoke', SMOKE)):
        log = out.with_suffix('.log')
        with log.open('wb') as handle:
            result = subprocess.run(job_command(mode, out), env=safety.environment(), stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError(f'{mode} failed; inspect {log}; no formal training launched')
        report = json.loads(out.read_text())
        if report['source_sha256'] != cpu['source_sha256']:
            raise RuntimeError('Smoke sources differ from the approved CPU test')
        if mode == '--norm-smoke':
            if report['status'] != 'verified_splitstats_syncbn_ddp' or not report['shared_affine_ddp_gradients']:
                raise RuntimeError('Separate SyncBN numerical/gradient check failed')
        elif (report['status'] != 'verified_splitstats_ddp_smoke' or report['config_sha256'] != safety.sha(CONFIG)
              or report['parameter_count'] != 24_327_432 or report['row']['train_images'] != 170
              or report['row']['test_images'] != 34 or report['row']['update_groups'] != 3
              or report['row']['optimizer_steps'] != 3 or report['row']['amp_skipped_steps'] != 0
              or not report['normalization_checks']['all_model_parameters_and_buffers_identical_across_ranks']):
            raise RuntimeError('Full-model DDP smoke count/gradient/state check failed')
        report['preflight'] = evidence
        safety.json_write(out, report)
        reports.append(report)
    if any(rank['peak_reserved_gib'] * 1024 + 3000 > gpu['free_mib']
           for rank, gpu in zip(reports[-1]['rank_reports'], evidence['gpus'], strict=True)):
        raise RuntimeError('Measured peak VRAM plus 3000MiB margin does not fit')
    return reports[-1]


def verified_recent_smoke(evidence):
    cpu, _ = cpu_verified()
    reports = [json.loads(p.read_text()) for p in (NORM_SMOKE, SMOKE)]
    for report in reports:
        recorded = report['preflight']
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(recorded['checked_at_utc'])).total_seconds()
        if not (0 <= age < 900 and recorded['boot_id'] == evidence['boot_id']
                and report['source_sha256'] == cpu['source_sha256']):
            raise RuntimeError('Smoke must be hash-identical, same boot, and less than 15 minutes old')
    if (reports[0]['status'] != 'verified_splitstats_syncbn_ddp'
        or reports[1]['status'] != 'verified_splitstats_ddp_smoke'
        or reports[1]['config_sha256'] != safety.sha(CONFIG)
        or reports[1]['row']['optimizer_steps'] != 3 or reports[1]['row']['amp_skipped_steps'] != 0):
        raise RuntimeError('Fresh smoke did not pass')
    if any(rank['peak_reserved_gib'] * 1024 + 3000 > gpu['free_mib']
           for rank, gpu in zip(reports[1]['rank_reports'], evidence['gpus'], strict=True)):
        raise RuntimeError('Fresh free VRAM no longer meets measured peak plus margin')
    return reports[1]


def event(stage, **fields):
    item = dict(stage=stage, manager_pid=os.getpid(), at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)
    safety.json_write(MANAGEMENT / 'state.json', item)
    with (MANAGEMENT / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(item) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(item), flush=True)


def audit():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    output = subprocess.check_output(['numactl', '--cpunodebind=0', '--membind=0', str(PYTHON), '-u',
                                      str(TRAINER), '--config', str(CONFIG), '--audit'],
                                     env=env, text=True, timeout=300)
    result = json.loads(output.splitlines()[-1])
    if result['status'] != 'verified_native_e125_splitstats':
        raise RuntimeError('Strict native125 audit did not pass')
    return result


def supervise():
    with (MANAGEMENT / 'manager.lock').open('x') as handle:
        handle.write(str(os.getpid()))
    reservation = json.loads((MANAGEMENT / 'reservation.json').read_text())
    try:
        if reservation['hashes'] != frozen_hashes():
            raise RuntimeError('Source/config changed after launch reservation')
        _, control = cpu_verified()
        c = json.loads(CONFIG.read_text())
        if Path(c['run_dir']).exists():
            raise RuntimeError('Existing G6 artifacts prohibit duplicate launch')
        evidence = safety.preflight()
        verified_recent_smoke(evidence)
        command = job_command('--train')
        with (MANAGEMENT / 'train.log').open('xb', buffering=0) as handle:
            process = subprocess.Popen(command, env=safety.environment(), stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        event('training', torchrun_pid=process.pid, command=command, preflight=evidence,
              hashes=reservation['hashes'], control_config_sha256=control['config_sha256'])
        code = process.wait()
        if code:
            raise RuntimeError(f'Training exited {code}; no unsafe automatic resume')
        if safety.active_task_trainers():
            raise RuntimeError('Task-owned trainer/worker still active; hold audit for identity inspection')
        if reservation['hashes'] != frozen_hashes():
            raise RuntimeError('Sources/config changed during training')
        event('cpu_auditing')
        result = audit()
        _, control = cpu_verified()
        reference = control['historical_audit']
        comparison = dict(status='g6_and_frozen_g4_verified', experiment=result, control=reference,
                          g6_minus_g4_top1_pp=float(result['final_row']['test_top1']) - float(reference['final_row']['test_top1']),
                          g6_minus_g4_top5_pp=float(result['final_row']['test_top5']) - float(reference['final_row']['test_top5']),
                          g6_minus_g4_test_loss=float(result['final_row']['test_loss']) - float(reference['final_row']['test_loss']),
                          limits=['Single seed2005 and monitored official test, not blind selection',
                                  'GT boxes in train/test: Oracle, not deployable',
                                  'Separate statistics include 16 views per route instead of 32 mixed views',
                                  'Affine parameters tied; trainable capacity and initialization unchanged',
                                  'Frozen completed G4 reused, not concurrent rerun; shared resource timing uncontrolled',
                                  'A better G6 would support this normalization adaptation, not prove all G4 failures were BN'])
        safety.json_write(PREFIX.with_name(PREFIX.name + '_FINAL.json'), comparison)
        event('all_verified', comparison=comparison)
    except Exception as error:
        event('blocked_or_failed', reason=str(error))
        raise


def launch():
    cpu_verified()
    evidence = safety.preflight()
    verified_recent_smoke(evidence)
    if Path(json.loads(CONFIG.read_text())['run_dir']).exists():
        raise RuntimeError('Existing G6 run blocks duplicate launch')
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    reservation = dict(hashes=frozen_hashes(), created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       devices_physical=[2, 3], numa_node=0, actual_epochs=125, lr_horizon=200,
                       sole_intervention='per-view BN statistics; shared affine and identical initialization')
    with (MANAGEMENT / 'reservation.json').open('x') as handle:
        handle.write(json.dumps(reservation, indent=2) + '\n')
    with (MANAGEMENT / 'manager.log').open('xb', buffering=0) as handle:
        process = subprocess.Popen(['numactl', '--cpunodebind=0', '--membind=0', str(PYTHON), '-u',
                                    str(Path(__file__)), '--supervise'], env=safety.environment(),
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    result = dict(status='manager_started', manager_pid=process.pid, management=str(MANAGEMENT),
                  run_dir=json.loads(CONFIG.read_text())['run_dir'], preflight=evidence)
    safety.json_write(MANAGEMENT / 'launch.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('preflight', 'smoke', 'launch', 'supervise'):
        mode.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(safety.preflight()), flush=True)
    elif args.smoke:
        print(json.dumps(smoke()), flush=True)
    elif args.launch:
        launch()
    else:
        supervise()


if __name__ == '__main__':
    main()
