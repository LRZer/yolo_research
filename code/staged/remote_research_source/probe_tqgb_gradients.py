"""Calibrate TQGB classification-route gradients on fixed training checkpoints.

The probe uses deterministic, non-augmented training batches, creates no optimizer,
and never updates weights.  It measures m2o, o2m, and bidirectional projected routes
against the official base objective on the exact same predictions and assignments.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--m2o-gain", type=float, default=1.0)
    parser.add_argument("--o2m-gain", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="tqgb_gradient_probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_batches <= 0 or args.batch <= 0 or args.clip_norm <= 0:
        parser.error("max-batches, batch, and clip-norm must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.m2o_gain, args.o2m_gain)):
        parser.error("directional gains must be finite and non-negative")
    return args


def resolve_checkpoints(patterns):
    resolved = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(matches)
    unique, seen = [], set()
    for path in resolved:
        absolute = str(Path(path).resolve())
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    if not unique:
        raise FileNotFoundError(f"No checkpoints matched: {patterns}")
    return unique


ARGS = parse_args()
CHECKPOINTS = resolve_checkpoints(ARGS.checkpoints)
if ARGS.dry_run:
    print(json.dumps({"checkpoints": CHECKPOINTS, **vars(ARGS)}, indent=2, default=str))
    raise SystemExit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

import torch
from torch import nn

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer


MODES = ("m2o", "o2m", "bidirectional")
GROUPS = ("shared", "o2m_box", "o2m_cls", "o2o_box", "o2o_cls")


def checkpoint_epoch(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1
    train_args = checkpoint.get("train_args") or {}
    configured = train_args.get("epochs") if isinstance(train_args, dict) else None
    if isinstance(configured, int) and configured > 0 and Path(path).stem in {"last", "best"}:
        return configured
    name = Path(path).stem
    return int(name[5:]) + 1 if name.startswith("epoch") and name[5:].isdigit() else None


def checkpoint_total_epochs(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("train_args") or {}
    configured = train_args.get("epochs") if isinstance(train_args, dict) else None
    return configured if isinstance(configured, int) and configured > 0 else None


def make_trainer(checkpoint):
    total_epochs = checkpoint_total_epochs(checkpoint) or 50
    overrides = {
        "model": checkpoint,
        "data": ARGS.data,
        "imgsz": ARGS.imgsz,
        "batch": ARGS.batch,
        "workers": ARGS.workers,
        "epochs": total_epochs,
        "device": "0",
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "o2o_kd_gain": 0.0,
        "tqgb_cls_m2o_gain": ARGS.m2o_gain,
        "tqgb_cls_o2m_gain": ARGS.o2m_gain,
        "tqgb_min_score": 0.25,
        "tqgb_cls_margin": 0.05,
        "tqgb_direction": "bidirectional",
        "tqgb_projection": True,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.model.train()
    for module in trainer.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    trainer.model.criterion = trainer.model.init_criterion()
    criterion = trainer.model.criterion
    epoch = checkpoint_epoch(checkpoint)
    if epoch is not None:
        criterion.updates = max(epoch - 1, 0)
        criterion.o2m = criterion.decay(criterion.updates)
        criterion.o2o = max(criterion.total - criterion.o2m, 0.0)
    return trainer, epoch


def make_fixed_loader(trainer):
    """Build the unaugmented train panel without validation-mode worker doubling."""
    dataset = build_yolo_dataset(
        trainer.args,
        trainer.data["train"],
        ARGS.batch,
        trainer.data,
        mode="val",
        rect=False,
        stride=trainer.stride,
    )
    loader = build_dataloader(
        dataset,
        batch=ARGS.batch,
        workers=ARGS.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=True,
        device=trainer.device,
    )
    actual_workers = int(getattr(loader, "num_workers", -1))
    if actual_workers != ARGS.workers:
        raise RuntimeError(f"Requested workers={ARGS.workers}, loader constructed workers={actual_workers}")
    if not bool(getattr(loader, "pin_memory", False)):
        raise RuntimeError("Pinned memory is required by the frozen probe protocol")
    return loader


def parameter_groups(model):
    head = model.model[-1]
    group_ids = {
        "o2m_box": {id(p) for p in head.cv2.parameters()},
        "o2m_cls": {id(p) for p in head.cv3.parameters()},
        "o2o_box": {id(p) for p in head.one2one_cv2.parameters()},
        "o2o_cls": {id(p) for p in head.one2one_cv3.parameters()},
    }
    parameters = [p for p in model.parameters() if p.requires_grad]
    labels = [next((name for name, ids in group_ids.items() if id(p) in ids), "shared") for p in parameters]
    return parameters, labels


def gradients(loss, parameters, retain_graph):
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def grad_norm(grads, labels=None, group=None):
    total = 0.0
    for index, grad in enumerate(grads):
        if grad is not None and (group is None or labels[index] == group):
            total += float(grad.detach().float().square().sum())
    return math.sqrt(total)


def combined_norm(base, route, labels=None, group=None):
    total = 0.0
    for index, (left, right) in enumerate(zip(base, route)):
        if group is not None and labels[index] != group:
            continue
        if left is None and right is None:
            continue
        value = right if left is None else left if right is None else left + right
        total += float(value.detach().float().square().sum())
    return math.sqrt(total)


def cosine(left, right, labels, group):
    dot = left_sq = right_sq = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if labels[index] != group or a is None or b is None:
            continue
        a, b = a.detach().float(), b.detach().float()
        dot += float((a * b).sum())
        left_sq += float(a.square().sum())
        right_sq += float(b.square().sum())
    denominator = math.sqrt(left_sq * right_sq)
    return dot / denominator if denominator else 0.0


def percentile(values, probability):
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def record(samples, mode, base_grads, route_grads, labels, route_loss, stats):
    base_total = grad_norm(base_grads)
    route_total = grad_norm(route_grads)
    combined_total = combined_norm(base_grads, route_grads)
    alpha_base = min(1.0, ARGS.clip_norm / max(base_total, 1e-12))
    alpha_combined = min(1.0, ARGS.clip_norm / max(combined_total, 1e-12))
    values = {
        "route_loss": float(route_loss.detach()),
        "base_grad_norm": base_total,
        "route_grad_norm": route_total,
        "route_to_base_ratio": route_total / max(base_total, 1e-12),
        "combined_grad_norm": combined_total,
        "base_clip_hit": float(base_total > ARGS.clip_norm),
        "combined_clip_hit": float(combined_total > ARGS.clip_norm),
        "shared_scale_ratio": alpha_combined / max(alpha_base, 1e-12),
        "m2o_rate": float(stats["tqgb_cls_m2o_rate"]),
        "o2m_rate": float(stats["tqgb_cls_o2m_rate"]),
        "m2o_projection_keep": float(stats["tqgb_cls_m2o_projection_keep"]),
        "o2m_projection_keep": float(stats["tqgb_cls_o2m_projection_keep"]),
        "cosine_o2o_cls": cosine(base_grads, route_grads, labels, "o2o_cls"),
        "cosine_o2m_cls": cosine(base_grads, route_grads, labels, "o2m_cls"),
        "cosine_shared": cosine(base_grads, route_grads, labels, "shared"),
    }
    for group in GROUPS:
        base_norm = grad_norm(base_grads, labels, group)
        route_norm = grad_norm(route_grads, labels, group)
        values[f"base_{group}_norm"] = base_norm
        values[f"route_{group}_norm"] = route_norm
        values[f"route_to_base_{group}_ratio"] = route_norm / max(base_norm, 1e-12)
    for key, value in values.items():
        samples[(mode, key)].append(value)


def probe_checkpoint(checkpoint):
    trainer, epoch = make_trainer(checkpoint)
    loader = make_fixed_loader(trainer)
    criterion = trainer.model.criterion
    parameters, labels = parameter_groups(trainer.model)
    samples = defaultdict(list)
    images = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= ARGS.max_batches:
            break
        batch = trainer.preprocess_batch(batch)
        images += int(batch["img"].shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parsed = criterion.one2many.parse_output(trainer.model.predict(batch["img"]))
            one2many, one2one = parsed["one2many"], parsed["one2one"]
            assign_o2m, raw_o2m, _ = criterion.one2many.get_assigned_targets_and_loss(
                one2many, batch, return_details=True
            )
            assign_o2o, raw_o2o, _ = criterion.one2one.get_assigned_targets_and_loss(
                one2one, batch, return_details=True
            )
            batch_size = int(batch["img"].shape[0])
            base_loss = (raw_o2m * criterion.o2m + raw_o2o * criterion.o2o).sum() * batch_size
            route_losses, mode_stats = {}, {}
            for mode in MODES:
                criterion.tqgb_direction = mode
                loss, stats = criterion._tqgb_classification_loss(one2many, one2one, assign_o2m, assign_o2o)
                route_losses[mode] = loss * batch_size
                mode_stats[mode] = stats

        base_grads = gradients(base_loss, parameters, retain_graph=True)
        for index, mode in enumerate(MODES):
            route_grads = gradients(route_losses[mode], parameters, retain_graph=index < len(MODES) - 1)
            record(samples, mode, base_grads, route_grads, labels, route_losses[mode], mode_stats[mode])

    rows = []
    for mode in MODES:
        mode_keys = [key for stored_mode, key in samples if stored_mode == mode]
        count = len(samples[(mode, mode_keys[0])]) if mode_keys else 0
        row = {
            "checkpoint": checkpoint,
            "checkpoint_name": Path(checkpoint).name,
            "epoch": epoch,
            "mode": mode,
            "images": images,
            "batches": int(count),
            "m2o_gain": ARGS.m2o_gain,
            "o2m_gain": ARGS.o2m_gain,
            "clip_norm": ARGS.clip_norm,
            "prog_o2m": criterion.o2m,
            "prog_o2o": criterion.o2o,
        }
        for key in mode_keys:
            values = samples[(mode, key)]
            row[f"mean_{key}"] = sum(values) / max(len(values), 1)
            row[f"p95_{key}"] = percentile(values, 0.95)
            row[f"max_{key}"] = max(values)
            row[f"min_{key}"] = min(values)
        rows.append(row)

    if hasattr(loader, "close"):
        loader.close()
    del trainer, criterion, loader
    torch.cuda.empty_cache()
    return rows


def write_outputs(rows):
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / f"{ARGS.name}.csv",
        "json": output_dir / f"{ARGS.name}.json",
        "markdown": output_dir / f"{ARGS.name}.md",
    }
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TQGB 分类路由梯度校准",
        "",
        f"> 固定 checkpoint、训练划分、无增强、无 optimizer、无权重更新；m2o/o2m gains="
        f"{ARGS.m2o_gain:g}/{ARGS.o2m_gain:g}。",
        "",
        "| epoch | mode | route rate m2o/o2m | keep m2o/o2m | route/base mean/P95/max | o2o-cls mean/P95/max | o2m-cls mean/P95/max | shared mean/P95/max | min clip scale |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('epoch') or '?'} | {row['mode']} | "
            f"{row['mean_m2o_rate']:.3f}/{row['mean_o2m_rate']:.3f} | "
            f"{row['mean_m2o_projection_keep']:.3f}/{row['mean_o2m_projection_keep']:.3f} | "
            f"{row['mean_route_to_base_ratio']:.4f}/{row['p95_route_to_base_ratio']:.4f}/"
            f"{row['max_route_to_base_ratio']:.4f} | "
            f"{row['mean_route_to_base_o2o_cls_ratio']:.4f}/"
            f"{row['p95_route_to_base_o2o_cls_ratio']:.4f}/"
            f"{row['max_route_to_base_o2o_cls_ratio']:.4f} | "
            f"{row['mean_route_to_base_o2m_cls_ratio']:.4f}/"
            f"{row['p95_route_to_base_o2m_cls_ratio']:.4f}/"
            f"{row['max_route_to_base_o2m_cls_ratio']:.4f} | "
            f"{row['mean_route_to_base_shared_ratio']:.4f}/"
            f"{row['p95_route_to_base_shared_ratio']:.4f}/"
            f"{row['max_route_to_base_shared_ratio']:.4f} | "
            f"{row['min_shared_scale_ratio']:.4f} |"
        )
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main():
    rows = []
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = probe_checkpoint(checkpoint)
        rows.extend(checkpoint_rows)
        for row in checkpoint_rows:
            print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(write_outputs(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
