"""Probe YOLO26 O2M-to-O2O supervision compatibility on fixed checkpoints without training.

The probe runs the detection loss path on a non-augmented dataset split, aggregates true
candidate/common counts across all batches, and reports whether quality-gated O2M teachers
agree with or are closer to the O2O TAL soft target. It never updates model weights.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths or glob patterns.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--splits", nargs="+", default=["val"], choices=("train", "val", "test"))
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kd-gain", type=float, default=0.0)
    parser.add_argument("--kd-gate", choices=("true", "false"), default="true")
    parser.add_argument("--kd-min-quality", type=float, default=0.25)
    parser.add_argument("--kd-margin", type=float, default=0.05)
    parser.add_argument("--kd-supervision-gate", choices=("none", "direction", "closer"), default="none")
    parser.add_argument("--kd-warmup", type=float, default=5.0)
    parser.add_argument("--kd-ramp", type=float, default=10.0)
    parser.add_argument("--kd-fade", type=float, default=10.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="supervision_conflict_trajectory")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    print(json.dumps({"checkpoints": CHECKPOINTS, "splits": ARGS.splits}, indent=2))
    raise SystemExit(0)

# Set visibility before importing torch/Ultralytics so logical cuda:0 is one allowed physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

import torch
from torch import nn

from ultralytics.models.yolo.detect import DetectionTrainer


COUNT_KEYS = ("kd_common_count", "kd_candidate_count", "kd_effective_count")
COMMON_MEAN_KEYS = ("teacher_better", "teacher_cls_better", "quality_gap", "head_box_iou")
CANDIDATE_MEAN_KEYS = (
    "kd_p_teacher_mean",
    "kd_p_student_mean",
    "kd_y_mean",
    "kd_student_below_y",
    "kd_grad_direction_agree",
    "kd_teacher_closer",
    "kd_teacher_overshoot",
)


def checkpoint_epoch(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1  # checkpoint metadata is zero-based, report human epoch.
    name = Path(path).stem
    if name.startswith("epoch") and name[5:].isdigit():
        return int(name[5:]) + 1
    return 150 if name in {"last", "best"} else None


def make_trainer(checkpoint):
    overrides = {
        "model": checkpoint,
        "data": ARGS.data,
        "imgsz": ARGS.imgsz,
        "batch": ARGS.batch,
        "workers": ARGS.workers,
        "device": "0",
        # Keep checkpoint weights. In BaseTrainer.setup_model(), pretrained=False deliberately discards even a .pt
        # model's weights and rebuilds from YAML, which would make the copied O2M/O2O heads artificially identical.
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "o2o_kd_gain": ARGS.kd_gain,
        "o2o_kd_gate": ARGS.kd_gate == "true",
        "o2o_kd_min_quality": ARGS.kd_min_quality,
        "o2o_kd_margin": ARGS.kd_margin,
        "o2o_kd_supervision_gate": ARGS.kd_supervision_gate,
        "o2o_kd_warmup": ARGS.kd_warmup,
        "o2o_kd_ramp": ARGS.kd_ramp,
        "o2o_kd_fade": ARGS.kd_fade,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.model.criterion = None
    trainer.model.train()
    # Use the head's training output while freezing all normalization statistics.
    for module in trainer.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return trainer


def probe_split(checkpoint, split):
    trainer = make_trainer(checkpoint)
    dataset_path = trainer.data.get(split)
    if not dataset_path:
        raise KeyError(f"Dataset YAML has no non-empty '{split}' entry")
    loader = trainer.get_dataloader(dataset_path, batch_size=ARGS.batch, rank=-1, mode="val")
    totals = {key: 0.0 for key in COUNT_KEYS}
    common_sums = {key: 0.0 for key in COMMON_MEAN_KEYS}
    candidate_sums = {key: 0.0 for key in CANDIDATE_MEAN_KEYS}
    loss_sums = {}
    images = 0
    batches = 0

    with torch.inference_mode():
        for batch in loader:
            batch = trainer.preprocess_batch(batch)
            batch_images = int(batch["img"].shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, items = trainer.model.loss(batch)
            values = {key: float(value.detach().float().cpu()) for key, value in items.items()}
            common_count = values["kd_common_count"]
            candidate_count = values["kd_candidate_count"]
            for key in COUNT_KEYS:
                totals[key] += values[key]
            for key in COMMON_MEAN_KEYS:
                common_sums[key] += values[key] * common_count
            for key in CANDIDATE_MEAN_KEYS:
                candidate_sums[key] += values[key] * candidate_count
            for key, value in values.items():
                if key.startswith(("o2o_", "o2m_")) and key.endswith("_loss"):
                    loss_sums[key] = loss_sums.get(key, 0.0) + value * batch_images
            images += batch_images
            batches += 1

    common = totals["kd_common_count"]
    candidate = totals["kd_candidate_count"]
    effective = totals["kd_effective_count"]
    row = {
        "checkpoint": checkpoint,
        "checkpoint_name": Path(checkpoint).name,
        "epoch": checkpoint_epoch(checkpoint),
        "split": split,
        "images": images,
        "batches": batches,
        "kd_gain": ARGS.kd_gain,
        "kd_gate": ARGS.kd_gate == "true",
        "kd_min_quality": ARGS.kd_min_quality,
        "kd_margin": ARGS.kd_margin,
        "kd_supervision_gate": ARGS.kd_supervision_gate,
        **totals,
        "kd_candidate_rate_global": candidate / common if common else 0.0,
        "kd_effective_rate_global": effective / common if common else 0.0,
    }
    row.update({key: value / common if common else 0.0 for key, value in common_sums.items()})
    row.update({key: value / candidate if candidate else 0.0 for key, value in candidate_sums.items()})
    row.update({key: value / images if images else 0.0 for key, value in loss_sums.items()})
    if hasattr(loader, "close"):
        loader.close()
    del trainer, loader
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
        "# YOLO26 O2M→O2O 监督冲突轨迹",
        "",
        "> 固定 checkpoint、无权重更新、无训练增强；所有比例由跨 batch 原始计数汇总。",
        "",
        "| epoch | split | common | quality candidates | candidate rate | grad agree | teacher closer | overshoot | p_t | p_s | y |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("epoch") or 10**9, item["split"])):
        lines.append(
            "| {epoch} | {split} | {common:.0f} | {candidate:.0f} | {rate:.2%} | {agree:.2%} | "
            "{closer:.2%} | {overshoot:.2%} | {pt:.4f} | {ps:.4f} | {y:.4f} |".format(
                epoch=row.get("epoch") or "?",
                split=row["split"],
                common=row["kd_common_count"],
                candidate=row["kd_candidate_count"],
                rate=row["kd_candidate_rate_global"],
                agree=row["kd_grad_direction_agree"],
                closer=row["kd_teacher_closer"],
                overshoot=row["kd_teacher_overshoot"],
                pt=row["kd_p_teacher_mean"],
                ps=row["kd_p_student_mean"],
                y=row["kd_y_mean"],
            )
        )
    lines.extend(
        [
            "",
            "- `grad agree`：KD 与 O2O soft-target BCE 对类别 logit 的更新方向一致。",
            "- `teacher closer`：教师概率比学生概率更接近 O2O TAL soft target；这是更严格的安全条件。",
            "- `overshoot`：教师概率高于 O2O soft target；它不必然有害，但提示原始教师概率可能过强。",
            "- 本报告只诊断机制，不包含精度改进结论。",
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
