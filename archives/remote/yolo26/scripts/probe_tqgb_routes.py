"""Probe TQGB route coverage and projection survival on fixed YOLO26 checkpoints.

This script is deliberately read-only with respect to model weights: it uses a deterministic,
non-augmented data loader, performs no backward pass, and never updates an optimizer.  It measures
whether the proposed task-decoupled bidirectional transfer has enough usable support before any
training budget is spent on it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths or glob patterns.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=("train", "val", "test"))
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means the complete split.")
    parser.add_argument("--cls-min-quality", type=float, default=0.25)
    parser.add_argument("--cls-margin", type=float, default=0.05)
    parser.add_argument("--box-min-quality", type=float, default=0.25)
    parser.add_argument("--box-margin", type=float, default=0.05)
    parser.add_argument("--projection-eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="tqgb_route_probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("cls_min_quality", "box_min_quality"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("cls_margin", "box_margin", "projection_eps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")
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
    print(
        json.dumps(
            {
                "checkpoints": CHECKPOINTS,
                "splits": ARGS.splits,
                "physical_device": ARGS.physical_device,
                "max_batches": ARGS.max_batches,
            },
            indent=2,
        )
    )
    raise SystemExit(0)

# Restrict visibility before importing torch/Ultralytics. Logical cuda:0 is the selected allowed physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

import torch
from torch import nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.metrics import bbox_iou


ROUTES = ("cls_m2o", "cls_o2m", "box_m2o", "box_o2m")


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
    if name.startswith("epoch") and name[5:].isdigit():
        return int(name[5:]) + 1
    return None


def make_trainer(checkpoint):
    overrides = {
        "model": checkpoint,
        "data": ARGS.data,
        "imgsz": ARGS.imgsz,
        "batch": ARGS.batch,
        "workers": ARGS.workers,
        "device": "0",
        # BaseTrainer otherwise rebuilds from YAML and silently discards the checkpoint weights.
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "o2o_kd_gain": 0.0,
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
    return trainer


def assigned_class(values, class_idx):
    """Gather each foreground assignment's discrete GT class without inferring it from a soft score."""
    return values.gather(-1, class_idx).squeeze(-1)


def xyxy_residual_to_gt(boxes, target_boxes):
    """Return [dx/w_gt, dy/h_gt, log(w/w_gt), log(h/h_gt)] residuals."""
    boxes = boxes.float()
    target_boxes = target_boxes.float()
    pred_wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp_min(1e-6)
    target_wh = (target_boxes[..., 2:4] - target_boxes[..., 0:2]).clamp_min(1e-6)
    pred_center = (boxes[..., 0:2] + boxes[..., 2:4]) * 0.5
    target_center = (target_boxes[..., 0:2] + target_boxes[..., 2:4]) * 0.5
    delta_center = (pred_center - target_center) / target_wh
    delta_scale = (pred_wh / target_wh).clamp_min(1e-6).log()
    return torch.cat((delta_center, delta_scale), dim=-1)


def segment_project(teacher, student, target):
    """Project a detached teacher onto the closed segment from student to its supervision target."""
    lower = torch.minimum(student, target)
    upper = torch.maximum(student, target)
    return torch.minimum(torch.maximum(teacher, lower), upper)


def add_scalar_projection_stats(totals, prefix, route, teacher, student, target):
    """Accumulate mutually interpretable projection outcomes for a scalar classification target."""
    eps = ARGS.projection_eps
    projected = segment_project(teacher, student, target)
    changed = (projected - teacher).abs() > eps
    nonzero = (projected - student).abs() > eps
    clipped_to_target = changed & nonzero & ((projected - target).abs() <= eps)
    clipped_to_student = changed & ~nonzero
    retained = ~changed

    totals[f"{prefix}_route_count"] += route.sum().item()
    totals[f"{prefix}_projected_nonzero_count"] += (route & nonzero).sum().item()
    totals[f"{prefix}_teacher_retained_count"] += (route & retained).sum().item()
    totals[f"{prefix}_clipped_to_target_count"] += (route & clipped_to_target).sum().item()
    totals[f"{prefix}_clipped_to_student_count"] += (route & clipped_to_student).sum().item()
    totals[f"{prefix}_raw_step_sum"] += ((teacher - student).abs() * route).sum().item()
    totals[f"{prefix}_projected_step_sum"] += ((projected - student).abs() * route).sum().item()


def add_box_projection_stats(totals, prefix, route, teacher_residual, student_residual):
    """Accumulate anchor- and coordinate-level outcomes for box-residual projection toward zero (GT)."""
    eps = ARGS.projection_eps
    target = torch.zeros_like(student_residual)
    projected = segment_project(teacher_residual, student_residual, target)
    changed = (projected - teacher_residual).abs() > eps
    nonzero_coord = (projected - student_residual).abs() > eps
    route_coord = route.unsqueeze(-1).expand_as(nonzero_coord)
    nonzero_anchor = nonzero_coord.any(-1)
    clipped_to_target = changed & nonzero_coord & (projected.abs() <= eps)
    clipped_to_student = changed & ~nonzero_coord
    retained = ~changed

    totals[f"{prefix}_route_count"] += route.sum().item()
    totals[f"{prefix}_projected_nonzero_count"] += (route & nonzero_anchor).sum().item()
    totals[f"{prefix}_coordinate_count"] += route_coord.sum().item()
    totals[f"{prefix}_coordinate_nonzero_count"] += (route_coord & nonzero_coord).sum().item()
    totals[f"{prefix}_teacher_retained_coord_count"] += (route_coord & retained).sum().item()
    totals[f"{prefix}_clipped_to_target_coord_count"] += (route_coord & clipped_to_target).sum().item()
    totals[f"{prefix}_clipped_to_student_coord_count"] += (route_coord & clipped_to_student).sum().item()
    raw_step = (teacher_residual - student_residual).abs().mean(-1)
    projected_step = (projected - student_residual).abs().mean(-1)
    totals[f"{prefix}_raw_step_sum"] += (raw_step * route).sum().item()
    totals[f"{prefix}_projected_step_sum"] += (projected_step * route).sum().item()


def batch_route_stats(one2many, one2one, assign_o2m, assign_o2o):
    totals = defaultdict(float)
    logits_o2m = one2many["scores"].permute(0, 2, 1).float()
    logits_o2o = one2one["scores"].permute(0, 2, 1).float()
    labels_o2m = assign_o2m["target_labels"].long()
    labels_o2o = assign_o2o["target_labels"].long()
    if labels_o2m.ndim == logits_o2m.ndim:
        labels_o2m = labels_o2m.squeeze(-1)
        labels_o2o = labels_o2o.squeeze(-1)
    label_agree = labels_o2m == labels_o2o
    class_idx = labels_o2o.clamp(0, logits_o2o.shape[-1] - 1).unsqueeze(-1)

    same_gt = assign_o2m["target_gt_idx"] == assign_o2o["target_gt_idx"]
    common = assign_o2m["fg_mask"] & assign_o2o["fg_mask"] & same_gt & label_agree
    p_o2m = assigned_class(logits_o2m, class_idx).sigmoid()
    p_o2o = assigned_class(logits_o2o, class_idx).sigmoid()
    y_o2m = assigned_class(assign_o2m["target_scores"].float(), class_idx).clamp_(0, 1)
    y_o2o = assigned_class(assign_o2o["target_scores"].float(), class_idx).clamp_(0, 1)

    target_boxes = assign_o2o["target_bboxes"].float()
    boxes_o2m = assign_o2m["pred_bboxes"].float()
    boxes_o2o = assign_o2o["pred_bboxes"].float()
    iou_o2m = bbox_iou(boxes_o2m, target_boxes, xywh=False).squeeze(-1).clamp_(0, 1)
    iou_o2o = bbox_iou(boxes_o2o, target_boxes, xywh=False).squeeze(-1).clamp_(0, 1)

    cls_gap = p_o2m - p_o2o
    box_gap = iou_o2m - iou_o2o
    cls_m2o = common & (p_o2m >= ARGS.cls_min_quality) & (cls_gap >= ARGS.cls_margin)
    cls_o2m = common & (p_o2o >= ARGS.cls_min_quality) & (-cls_gap >= ARGS.cls_margin)
    box_m2o = common & (iou_o2m >= ARGS.box_min_quality) & (box_gap >= ARGS.box_margin)
    box_o2m = common & (iou_o2o >= ARGS.box_min_quality) & (-box_gap >= ARGS.box_margin)

    common_count = common.sum().item()
    totals["common_count"] += common_count
    totals["label_disagree_common_gt_count"] += (
        assign_o2m["fg_mask"] & assign_o2o["fg_mask"] & same_gt & ~label_agree
    ).sum().item()
    totals["cls_raw_winner_disagree_box_count"] += (common & (cls_gap * box_gap < 0)).sum().item()
    cls_any = cls_m2o | cls_o2m
    box_any = box_m2o | box_o2m
    both_routed = cls_any & box_any
    direction_disagree = (cls_m2o & box_o2m) | (cls_o2m & box_m2o)
    totals["both_task_routed_count"] += both_routed.sum().item()
    totals["task_direction_disagree_count"] += direction_disagree.sum().item()
    totals["cls_any_route_count"] += cls_any.sum().item()
    totals["box_any_route_count"] += box_any.sum().item()

    totals["p_o2m_common_sum"] += (p_o2m * common).sum().item()
    totals["p_o2o_common_sum"] += (p_o2o * common).sum().item()
    totals["iou_o2m_common_sum"] += (iou_o2m * common).sum().item()
    totals["iou_o2o_common_sum"] += (iou_o2o * common).sum().item()
    totals["abs_cls_gap_common_sum"] += (cls_gap.abs() * common).sum().item()
    totals["abs_box_gap_common_sum"] += (box_gap.abs() * common).sum().item()

    add_scalar_projection_stats(totals, "cls_m2o", cls_m2o, p_o2m, p_o2o, y_o2o)
    add_scalar_projection_stats(totals, "cls_o2m", cls_o2m, p_o2o, p_o2m, y_o2m)

    residual_o2m = xyxy_residual_to_gt(boxes_o2m, target_boxes)
    residual_o2o = xyxy_residual_to_gt(boxes_o2o, target_boxes)
    add_box_projection_stats(totals, "box_m2o", box_m2o, residual_o2m, residual_o2o)
    add_box_projection_stats(totals, "box_o2m", box_o2m, residual_o2o, residual_o2m)
    return totals


def finalize_rates(row):
    common = row["common_count"]
    row["cls_any_route_rate_common"] = row["cls_any_route_count"] / common if common else 0.0
    row["box_any_route_rate_common"] = row["box_any_route_count"] / common if common else 0.0
    row["raw_task_winner_disagree_rate"] = row["cls_raw_winner_disagree_box_count"] / common if common else 0.0
    both = row["both_task_routed_count"]
    row["task_direction_disagree_rate"] = row["task_direction_disagree_count"] / both if both else 0.0
    for key in (
        "p_o2m_common",
        "p_o2o_common",
        "iou_o2m_common",
        "iou_o2o_common",
        "abs_cls_gap_common",
        "abs_box_gap_common",
    ):
        row[f"{key}_mean"] = row.pop(f"{key}_sum") / common if common else 0.0

    for prefix in ROUTES:
        count = row[f"{prefix}_route_count"]
        row[f"{prefix}_rate_common"] = count / common if common else 0.0
        row[f"{prefix}_projection_nonzero_rate"] = (
            row[f"{prefix}_projected_nonzero_count"] / count if count else 0.0
        )
        row[f"{prefix}_raw_step_mean"] = row.pop(f"{prefix}_raw_step_sum") / count if count else 0.0
        row[f"{prefix}_projected_step_mean"] = (
            row.pop(f"{prefix}_projected_step_sum") / count if count else 0.0
        )
        if prefix.startswith("cls_"):
            for outcome in ("teacher_retained", "clipped_to_target", "clipped_to_student"):
                row[f"{prefix}_{outcome}_rate"] = row[f"{prefix}_{outcome}_count"] / count if count else 0.0
        else:
            coordinates = row[f"{prefix}_coordinate_count"]
            row[f"{prefix}_coordinate_nonzero_rate"] = (
                row[f"{prefix}_coordinate_nonzero_count"] / coordinates if coordinates else 0.0
            )
            for outcome in ("teacher_retained", "clipped_to_target", "clipped_to_student"):
                row[f"{prefix}_{outcome}_coord_rate"] = (
                    row[f"{prefix}_{outcome}_coord_count"] / coordinates if coordinates else 0.0
                )
    return row


def probe_split(checkpoint, split):
    trainer = make_trainer(checkpoint)
    dataset_path = trainer.data.get(split)
    if not dataset_path:
        raise KeyError(f"Dataset YAML has no non-empty '{split}' entry")
    # mode='val' intentionally disables training augmentation: this is a fixed-checkpoint diagnostic.
    loader = trainer.get_dataloader(dataset_path, batch_size=ARGS.batch, rank=-1, mode="val")
    criterion = trainer.model.criterion
    totals = defaultdict(float)
    images = 0
    batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if ARGS.max_batches and batch_index >= ARGS.max_batches:
                break
            batch = trainer.preprocess_batch(batch)
            images += int(batch["img"].shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = trainer.model.predict(batch["img"])
                preds = criterion.one2many.parse_output(preds)
                one2many, one2one = preds["one2many"], preds["one2one"]
                assign_o2m, _, _ = criterion.one2many.get_assigned_targets_and_loss(
                    one2many, batch, return_details=True
                )
                assign_o2o, _, _ = criterion.one2one.get_assigned_targets_and_loss(
                    one2one, batch, return_details=True
                )
                batch_totals = batch_route_stats(one2many, one2one, assign_o2m, assign_o2o)
            for key, value in batch_totals.items():
                totals[key] += float(value)
            batches += 1

    row = {
        "checkpoint": checkpoint,
        "checkpoint_name": Path(checkpoint).name,
        "epoch": checkpoint_epoch(checkpoint),
        "split": split,
        "images": images,
        "batches": batches,
        "complete_split": ARGS.max_batches == 0,
        "cls_min_quality": ARGS.cls_min_quality,
        "cls_margin": ARGS.cls_margin,
        "box_min_quality": ARGS.box_min_quality,
        "box_margin": ARGS.box_margin,
        "projection_eps": ARGS.projection_eps,
        **totals,
    }
    # Ensure zero-count route fields exist so all rows share a stable schema.
    for prefix in ROUTES:
        for suffix in ("route_count", "projected_nonzero_count", "raw_step_sum", "projected_step_sum"):
            row.setdefault(f"{prefix}_{suffix}", 0.0)
        if prefix.startswith("cls_"):
            for suffix in ("teacher_retained_count", "clipped_to_target_count", "clipped_to_student_count"):
                row.setdefault(f"{prefix}_{suffix}", 0.0)
        else:
            for suffix in (
                "coordinate_count",
                "coordinate_nonzero_count",
                "teacher_retained_coord_count",
                "clipped_to_target_coord_count",
                "clipped_to_student_coord_count",
            ):
                row.setdefault(f"{prefix}_{suffix}", 0.0)
    for key in (
        "common_count",
        "cls_any_route_count",
        "box_any_route_count",
        "both_task_routed_count",
        "task_direction_disagree_count",
        "cls_raw_winner_disagree_box_count",
        "p_o2m_common_sum",
        "p_o2o_common_sum",
        "iou_o2m_common_sum",
        "iou_o2o_common_sum",
        "abs_cls_gap_common_sum",
        "abs_box_gap_common_sum",
    ):
        row.setdefault(key, 0.0)
    finalize_rates(row)

    if hasattr(loader, "close"):
        loader.close()
    del criterion, trainer, loader
    torch.cuda.empty_cache()
    return row


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
        "# TQGB 双向路由可行性探针",
        "",
        "> 固定 checkpoint、无训练增强、无反向传播、无权重更新。比例均由跨 batch 原始计数汇总。",
        "",
        "| epoch | split | common | cls M→O | cls O→M | box M→O | box O→M | raw task disagree | routed disagree | cls O→M survives | box O→M survives |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("epoch") or 10**9, item["split"])):
        lines.append(
            "| {epoch} | {split} | {common:.0f} | {cm2o:.2%} | {co2m:.2%} | {bm2o:.2%} | {bo2m:.2%} | "
            "{raw_disagree:.2%} | {route_disagree:.2%} | {cls_survive:.2%} | {box_survive:.2%} |".format(
                epoch=row.get("epoch") or "?",
                split=row["split"],
                common=row["common_count"],
                cm2o=row["cls_m2o_rate_common"],
                co2m=row["cls_o2m_rate_common"],
                bm2o=row["box_m2o_rate_common"],
                bo2m=row["box_o2m_rate_common"],
                raw_disagree=row["raw_task_winner_disagree_rate"],
                route_disagree=row["task_direction_disagree_rate"],
                cls_survive=row["cls_o2m_projection_nonzero_rate"],
                box_survive=row["box_o2m_projection_nonzero_rate"],
            )
        )
    lines.extend(
        [
            "",
            "- `M→O`：O2M 教 O2O；`O→M`：部署 O2O 教 O2M，并可能把梯度送入共享 trunk。",
            "- 分类质量使用 GT 类概率，定位质量使用预测框对同一 GT 的 IoU；两个任务独立路由。",
            "- `raw task disagree` 是不加阈值时分类与定位优胜方向相反的共同位置比例。",
            "- `routed disagree` 只在两个任务都越过阈值时统计方向相反比例。",
            "- `survives` 是监督线段投影后仍产生非零 student 更新的 route 比例。",
            "- 预注册裁决：若中后期 O→M 有效覆盖长期低于约 1%，停止双向方案，回到状态感知单向调度。",
            "- 本报告只判断机制覆盖，不构成精度或创新性结论。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def main():
    rows = []
    for checkpoint in CHECKPOINTS:
        for split in ARGS.splits:
            row = probe_split(checkpoint, split)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(write_outputs(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
