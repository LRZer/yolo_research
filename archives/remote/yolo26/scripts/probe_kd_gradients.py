"""Measure KD strength and global-clipping coupling on fixed YOLO26 checkpoints.

The probe performs forward/backward differentiation on deterministic, non-augmented
training batches but never creates an optimizer and never updates model weights.  It
compares the official base objective, ungated matched-anchor KD, and quality-gated KD
on exactly the same weights and images.
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
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="kd_gradient_probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    if args.batch <= 0 or args.gain < 0 or args.clip_norm <= 0:
        parser.error("batch and clip-norm must be positive; gain must be non-negative")
    return args


def resolve_checkpoints(patterns):
    resolved = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(matches)
    unique = []
    seen = set()
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

from ultralytics.models.yolo.detect import DetectionTrainer


MODES = ("ungated", "quality")
GROUPS = ("shared", "o2m_head", "o2o_box", "o2o_cls")


def checkpoint_epoch(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1
    train_args = checkpoint.get("train_args") or {}
    configured_epochs = train_args.get("epochs") if isinstance(train_args, dict) else None
    if isinstance(configured_epochs, int) and configured_epochs > 0 and Path(path).stem in {"last", "best"}:
        return configured_epochs
    name = Path(path).stem
    return int(name[5:]) + 1 if name.startswith("epoch") and name[5:].isdigit() else None


def make_trainer(checkpoint):
    overrides = {
        "model": checkpoint,
        "data": ARGS.data,
        "imgsz": ARGS.imgsz,
        "batch": ARGS.batch,
        "workers": ARGS.workers,
        "device": "0",
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "o2o_kd_gain": ARGS.gain,
        "o2o_kd_warmup": 0.0,
        "o2o_kd_ramp": 0.0,
        "o2o_kd_fade": 0.0,
        "o2o_kd_supervision_gate": "none",
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
    criterion.kd_gain = ARGS.gain
    criterion.kd_warmup = criterion.kd_ramp = criterion.kd_fade = 0.0
    return trainer


def parameter_groups(model):
    head = model.model[-1]
    group_ids = {
        "o2m_head": {id(p) for module in (head.cv2, head.cv3) for p in module.parameters()},
        "o2o_box": {id(p) for p in head.one2one_cv2.parameters()},
        "o2o_cls": {id(p) for p in head.one2one_cv3.parameters()},
    }
    params = [p for p in model.parameters() if p.requires_grad]
    labels = []
    for parameter in params:
        label = next((name for name, ids in group_ids.items() if id(parameter) in ids), "shared")
        labels.append(label)
    return params, labels


def grad_norm(grads, labels=None, group=None):
    total = 0.0
    for index, grad in enumerate(grads):
        if grad is not None and (group is None or labels[index] == group):
            total += float(grad.detach().float().square().sum())
    return math.sqrt(total)


def combined_norm(base_grads, kd_grads, labels=None, group=None):
    total = 0.0
    for index, (base, kd) in enumerate(zip(base_grads, kd_grads)):
        if group is not None and labels[index] != group:
            continue
        if base is None and kd is None:
            continue
        value = kd if base is None else base if kd is None else base + kd
        total += float(value.detach().float().square().sum())
    return math.sqrt(total)


def grad_cosine(left, right, labels, group="o2o_cls"):
    dot = left_sq = right_sq = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if labels[index] != group or a is None or b is None:
            continue
        af, bf = a.detach().float(), b.detach().float()
        dot += float((af * bf).sum())
        left_sq += float(af.square().sum())
        right_sq += float(bf.square().sum())
    denominator = math.sqrt(left_sq * right_sq)
    return dot / denominator if denominator else 0.0


def gradients(loss, parameters, retain_graph):
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def add_batch(totals, mode, base_grads, kd_grads, labels, kd_loss, stats):
    base_total = grad_norm(base_grads)
    kd_total = grad_norm(kd_grads)
    combined_total = combined_norm(base_grads, kd_grads)
    alpha_base = min(1.0, ARGS.clip_norm / max(base_total, 1e-12))
    alpha_combined = min(1.0, ARGS.clip_norm / max(combined_total, 1e-12))
    values = {
        "base_grad_norm": base_total,
        "kd_grad_norm": kd_total,
        "combined_grad_norm": combined_total,
        "base_clip_hit": float(base_total > ARGS.clip_norm),
        "combined_clip_hit": float(combined_total > ARGS.clip_norm),
        "base_clip_alpha": alpha_base,
        "combined_clip_alpha": alpha_combined,
        "shared_scale_ratio": alpha_combined / max(alpha_base, 1e-12),
        "base_kd_cosine_o2o_cls": grad_cosine(base_grads, kd_grads, labels),
        "kd_loss": float(kd_loss.detach()),
        "candidate_rate": float(stats["kd_candidate_rate"]),
        "teacher_better": float(stats["teacher_better"]),
        "quality_gap": float(stats["quality_gap"]),
    }
    for group in GROUPS:
        values[f"base_{group}_norm"] = grad_norm(base_grads, labels, group)
        values[f"kd_{group}_norm"] = grad_norm(kd_grads, labels, group)
        values[f"combined_{group}_norm"] = combined_norm(base_grads, kd_grads, labels, group)
    for key, value in values.items():
        totals[(mode, key)] += value
    totals[(mode, "batches")] += 1.0


def probe_checkpoint(checkpoint):
    trainer = make_trainer(checkpoint)
    dataset_path = trainer.data["train"]
    loader = trainer.get_dataloader(dataset_path, batch_size=ARGS.batch, rank=-1, mode="val")
    criterion = trainer.model.criterion
    parameters, labels = parameter_groups(trainer.model)
    totals = defaultdict(float)
    images = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= ARGS.max_batches:
            break
        batch = trainer.preprocess_batch(batch)
        images += int(batch["img"].shape[0])
        trainer.model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = trainer.model.predict(batch["img"])
            criterion.kd_gate = False
            ungated_losses, ungated_stats = criterion(preds, batch)
            criterion.kd_gate = True
            quality_losses, quality_stats = criterion(preds, batch)
            base_loss = quality_losses[:3].sum()
            kd_losses = {
                "ungated": ungated_losses[3],
                "quality": quality_losses[3],
            }
            mode_stats = {"ungated": ungated_stats, "quality": quality_stats}

        base_grads = gradients(base_loss, parameters, retain_graph=True)
        for mode_index, mode in enumerate(MODES):
            kd_grads = gradients(kd_losses[mode], parameters, retain_graph=mode_index < len(MODES) - 1)
            add_batch(totals, mode, base_grads, kd_grads, labels, kd_losses[mode], mode_stats[mode])

    rows = []
    for mode in MODES:
        count = totals[(mode, "batches")]
        row = {
            "checkpoint": checkpoint,
            "checkpoint_name": Path(checkpoint).name,
            "epoch": checkpoint_epoch(checkpoint),
            "mode": mode,
            "images": images,
            "batches": int(count),
            "gain": ARGS.gain,
            "clip_norm": ARGS.clip_norm,
        }
        for (stored_mode, key), value in totals.items():
            if stored_mode == mode and key != "batches":
                row[f"mean_{key}"] = value / max(count, 1.0)
        rows.append(row)

    if hasattr(loader, "close"):
        loader.close()
    del criterion, trainer, loader
    torch.cuda.empty_cache()
    return rows


def write_outputs(rows):
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{ARGS.name}.csv"
    json_path = output_dir / f"{ARGS.name}.json"
    md_path = output_dir / f"{ARGS.name}.md"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# KD 梯度与 global clipping 探针",
        "",
        "> 固定 checkpoint、无增强、无 optimizer、无参数更新；同一权重/图像比较基础目标与两种 KD。",
        "",
        "| epoch | mode | KD loss | base norm | KD norm | combined norm | base clip | combined clip | shared scale | cosine(cls) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {mode} | {kd:.5f} | {base:.3f} | {kdn:.3f} | {combined:.3f} | "
            "{base_hit:.1%} | {combined_hit:.1%} | {scale:.4f} | {cosine:.4f} |".format(
                epoch=row.get("epoch") or "?",
                mode=row["mode"],
                kd=row["mean_kd_loss"],
                base=row["mean_base_grad_norm"],
                kdn=row["mean_kd_grad_norm"],
                combined=row["mean_combined_grad_norm"],
                base_hit=row["mean_base_clip_hit"],
                combined_hit=row["mean_combined_clip_hit"],
                scale=row["mean_shared_scale_ratio"],
                cosine=row["mean_base_kd_cosine_o2o_cls"],
            )
        )
    lines.extend(
        [
            "",
            "- `shared scale` 是加入 KD 后相对基础目标的 global-clip 缩放比；小于 1 表示 KD 会间接缩小共享参数更新。",
            "- `cosine(cls)` 只在 O2O 分类头上比较基础监督与 KD 梯度方向。",
            "- 单卡 batch 等于正式四卡 global batch，用于近似 DDP all-reduce 后的裁剪前范数。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


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
