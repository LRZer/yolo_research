#!/usr/bin/env python3
"""Strict final audit for the 22-run YOLOv8n scratch baseline queue."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test")
MODEL = ROOT / "models" / "yolov8n_scratch.yaml"
DATA_ROOT = ROOT / "processed_yolo"
PROJECT = ROOT / "runs" / "yolov8n_scratch_baselines"
QUEUE_ROOT = ROOT / "_management" / "yolov8n_scratch_queue"
STATE_PATH = QUEUE_ROOT / "queue_state.json"
EXPECTED_MODEL_SHA256 = "44c9efc76c9dad388ad1ef6c4efb37e47ec885fe2ad12f110c76f6cdb750889d"
DATASETS = (
    "AP-10K", "DUO", "ExDark", "HIT-UAV", "HRSID-JPG",
    "LLVIP-infrared", "LLVIP-visible", "ORCA-class-level-seen",
    "SeaClear", "VisDrone2019-DET", "ZeroWaste-f",
)
SEEDS = (2027, 2005)
EPOCHS = 125
BATCH = 64
IMGSZ = 640
METRICS = (
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_name(dataset: str, seed: int) -> str:
    return f"yolov8n_scratch_{dataset}_e{EPOCHS}_b{BATCH}_i{IMGSZ}_s{seed}"


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def as_float(value, label, errors):
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("not finite")
        return number
    except Exception as exc:
        errors.append(f"{label}: invalid number {value!r} ({exc})")
        return math.nan


def checkpoint_audit(path: Path, dataset: str, seed: int, errors):
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except BaseException as exc:
        errors.append(f"{path}: unreadable checkpoint ({type(exc).__name__}: {exc})")
        return None
    if not isinstance(checkpoint, dict) or not ("model" in checkpoint or "ema" in checkpoint):
        errors.append(f"{path}: missing model/ema payload")
        return None
    args = checkpoint.get("train_args")
    if not isinstance(args, dict):
        errors.append(f"{path}: missing train_args dictionary")
        return None
    expected = {
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "workers": 4,
        "seed": seed,
        "device": "0,1,2,3",
        "pretrained": False,
        "deterministic": True,
        "optimizer": "SGD",
        "model": str(MODEL),
        "data": str(DATA_ROOT / dataset / "dataset.yaml"),
    }
    for key, wanted in expected.items():
        if args.get(key) != wanted:
            errors.append(f"{path}: train_args[{key!r}]={args.get(key)!r}, expected {wanted!r}")
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "version": checkpoint.get("version"),
    }
    del checkpoint
    gc.collect()
    return result


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    audited_at = datetime.now().astimezone().isoformat(timespec="seconds")
    errors = []
    incomplete = []
    run_records = []
    checkpoint_records = []

    if not STATE_PATH.is_file():
        raise FileNotFoundError(STATE_PATH)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    configuration = state.get("configuration", {})
    expected_configuration = {
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "seeds": list(SEEDS),
        "datasets": list(DATASETS),
        "gpus_physical": [4, 5, 6, 7],
        "numa_node": 1,
        "workers_per_rank": 4,
        "pretrained": False,
        "model_sha256": EXPECTED_MODEL_SHA256,
    }
    for key, wanted in expected_configuration.items():
        if configuration.get(key) != wanted:
            errors.append(f"queue configuration {key}={configuration.get(key)!r}, expected {wanted!r}")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        errors.append("model YAML SHA256 changed")

    expected_ids = {f"{dataset}__seed{seed}" for dataset in DATASETS for seed in SEEDS}
    state_ids = {job.get("id") for job in state.get("jobs", [])}
    if state_ids != expected_ids:
        errors.append(f"queue job IDs mismatch: missing={sorted(expected_ids-state_ids)}, extra={sorted(state_ids-expected_ids)}")

    for dataset in DATASETS:
        for seed in SEEDS:
            jid = f"{dataset}__seed{seed}"
            run_dir = PROJECT / run_name(dataset, seed)
            results = run_dir / "results.csv"
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            if not results.is_file():
                incomplete.append({"job": jid, "reason": "missing results.csv"})
                continue
            rows = read_rows(results)
            if len(rows) != EPOCHS:
                incomplete.append({"job": jid, "reason": f"results rows={len(rows)}, expected {EPOCHS}"})
                continue
            epochs = [int(float(row.get("epoch", "nan"))) for row in rows]
            if epochs != list(range(1, EPOCHS + 1)):
                errors.append(f"{jid}: epoch sequence is not exactly 1..{EPOCHS}")
            if not best.is_file() or not last.is_file():
                incomplete.append({"job": jid, "reason": f"weights missing: best={best.is_file()} last={last.is_file()}"})
                continue

            values_by_metric = {
                metric: [as_float(row.get(metric), f"{jid}/{metric}", errors) for row in rows]
                for metric in METRICS
            }
            map_metric = "metrics/mAP50-95(B)"
            best_index = max(range(len(rows)), key=lambda i: values_by_metric[map_metric][i])
            last_row = rows[-1]
            record = {
                "dataset": dataset,
                "seed": seed,
                "epochs": len(rows),
                "best_epoch_by_map50_95": epochs[best_index],
            }
            for metric in METRICS:
                short = metric.removeprefix("metrics/").removesuffix("(B)").strip()
                record[f"last_{short}"] = as_float(last_row.get(metric), f"{jid}/last/{metric}", errors)
                record[f"best_epoch_{short}"] = values_by_metric[metric][best_index]
            p = record["last_precision"]
            r = record["last_recall"]
            record["last_F1"] = 2 * p * r / (p + r) if p + r else 0.0
            bp = record["best_epoch_precision"]
            br = record["best_epoch_recall"]
            record["best_epoch_F1"] = 2 * bp * br / (bp + br) if bp + br else 0.0
            run_records.append(record)
            for weight in (best, last):
                result = checkpoint_audit(weight, dataset, seed, errors)
                if result:
                    checkpoint_records.append({"job": jid, "kind": weight.stem, **result})

    summary_records = []
    numeric_fields = [
        "last_precision", "last_recall", "last_mAP50", "last_mAP50-95", "last_F1",
        "best_epoch_precision", "best_epoch_recall", "best_epoch_mAP50",
        "best_epoch_mAP50-95", "best_epoch_F1",
    ]
    for dataset in DATASETS:
        records = [r for r in run_records if r["dataset"] == dataset]
        if len(records) != len(SEEDS):
            continue
        summary = {"dataset": dataset, "seeds": ",".join(str(r["seed"]) for r in records)}
        for field in numeric_fields:
            vals = [r[field] for r in records]
            summary[f"{field}_mean"] = statistics.mean(vals)
            summary[f"{field}_std"] = statistics.stdev(vals)
        summary_records.append(summary)

    complete = not errors and not incomplete and len(run_records) == len(expected_ids)
    audit = {
        "audited_at": audited_at,
        "status": "verified_complete" if complete else ("invalid" if errors else "incomplete"),
        "requirements": {
            "datasets": list(DATASETS), "seeds": list(SEEDS), "runs": len(expected_ids),
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ,
            "workers_per_rank": 4, "physical_gpus": [4, 5, 6, 7],
            "numa_node": 1, "pretrained": False,
        },
        "queue_state_status": state.get("status"),
        "verified_runs": len(run_records),
        "verified_checkpoints": len(checkpoint_records),
        "errors": errors,
        "incomplete": incomplete,
        "runs": run_records,
        "checkpoints": checkpoint_records,
        "two_seed_summary": summary_records,
    }
    (QUEUE_ROOT / "audit_latest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if run_records:
        fields = list(run_records[0].keys())
        write_csv(QUEUE_ROOT / "audit_per_run_metrics.csv", run_records, fields)
    if summary_records:
        fields = list(summary_records[0].keys())
        write_csv(QUEUE_ROOT / "audit_two_seed_summary.csv", summary_records, fields)

    lines = [
        "# YOLOv8n scratch baseline final audit",
        "",
        f"- Audited at: `{audited_at}`",
        f"- Status: **{audit['status']}**",
        f"- Verified runs: **{len(run_records)}/{len(expected_ids)}**",
        f"- Verified readable checkpoints: **{len(checkpoint_records)}/{2*len(expected_ids)}**",
        f"- Errors: **{len(errors)}**",
        f"- Incomplete: **{len(incomplete)}**",
        "",
    ]
    if summary_records:
        lines.extend([
            "## Two-seed summary",
            "",
            "| Dataset | Last mAP50-95 mean±std | Best-epoch mAP50-95 mean±std | Last F1 mean±std |",
            "|---|---:|---:|---:|",
        ])
        for row in summary_records:
            lines.append(
                f"| {row['dataset']} | {row['last_mAP50-95_mean']:.6f}±{row['last_mAP50-95_std']:.6f} "
                f"| {row['best_epoch_mAP50-95_mean']:.6f}±{row['best_epoch_mAP50-95_std']:.6f} "
                f"| {row['last_F1_mean']:.6f}±{row['last_F1_std']:.6f} |"
            )
        lines.append("")
    if incomplete:
        lines.extend(["## Incomplete runs", ""])
        lines.extend(f"- `{item['job']}`: {item['reason']}" for item in incomplete)
        lines.append("")
    if errors:
        lines.extend(["## Audit errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    (QUEUE_ROOT / "AUDIT_LATEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"], "verified_runs": len(run_records),
        "verified_checkpoints": len(checkpoint_records),
        "errors": len(errors), "incomplete": len(incomplete),
    }, ensure_ascii=False))
    return 0 if complete else (2 if errors else 3)


if __name__ == "__main__":
    raise SystemExit(main())
