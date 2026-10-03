"""Read-only SARC assignment-scale probe for fixed YOLO26 checkpoints.

The probe answers one narrow question before SARC is implemented: do the P3/P4/P5
levels receive different, temporally stable distributions of assigned object scale
relative to stride?

It deliberately uses the one-to-many TAL branch because that is the branch whose
features are not detached from the shared backbone/neck in YOLO26.  No optimizer is
created, no backward pass is performed, and model weights are never changed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths or glob patterns.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML; its train split is probed.")
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=8, help="Fixed panel size; 8x64=512 images by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-strides", type=float, nargs=3, default=(8.0, 16.0, 32.0))
    parser.add_argument("--expected-epochs", type=int, nargs="+", default=(11, 21, 31, 41, 50))
    parser.add_argument("--hist-min", type=float, default=-4.0)
    parser.add_argument("--hist-max", type=float, default=8.0)
    parser.add_argument("--hist-step", type=float, default=0.5)
    parser.add_argument("--similarity-js", type=float, default=0.02)
    parser.add_argument("--instability-js", type=float, default=0.30)
    parser.add_argument(
        "--min-pooled-count",
        type=int,
        default=100,
        help="Below this per-level count the verdict is INCONCLUSIVE, not GO/NO-GO.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="sarc_assignment_histogram_probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch <= 0 or args.workers < 0 or args.max_batches <= 0:
        parser.error("batch and max-batches must be positive; workers must be non-negative")
    if not args.hist_min < args.hist_max or args.hist_step <= 0:
        parser.error("hist-min < hist-max and hist-step > 0 are required")
    span = (args.hist_max - args.hist_min) / args.hist_step
    if not math.isclose(span, round(span), rel_tol=0.0, abs_tol=1e-9):
        parser.error("(hist-max - hist-min) must be an integer multiple of hist-step")
    if args.similarity_js < 0 or args.instability_js < 0:
        parser.error("JS thresholds must be non-negative")
    if args.min_pooled_count <= 0:
        parser.error("min-pooled-count must be positive")
    if any(right <= left for left, right in zip(args.expected_strides, args.expected_strides[1:])):
        parser.error("expected-strides must be strictly ordered from fine to coarse")
    if any(right <= left for left, right in zip(args.expected_epochs, args.expected_epochs[1:])):
        parser.error("expected-epochs must be strictly increasing")
    return args


def resolve_checkpoints(patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(matches)
    unique: list[str] = []
    seen: set[str] = set()
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
    print(json.dumps({"checkpoints": CHECKPOINTS, **vars(ARGS)}, ensure_ascii=False, indent=2))
    raise SystemExit(0)

# Restrict CUDA visibility before importing torch/Ultralytics.  Logical cuda:0 is one allowed physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

import numpy as np
import torch
from torch import nn

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.tal import make_anchors


WEIGHTINGS = ("positive_anchor", "unique_gt_per_image_level")
LEVEL_NAMES = ("P3", "P4", "P5")
HIST_EDGES = np.linspace(
    ARGS.hist_min,
    ARGS.hist_max,
    int(round((ARGS.hist_max - ARGS.hist_min) / ARGS.hist_step)) + 1,
    dtype=np.float64,
)


def seed_everything(seed: int) -> None:
    """Reset all relevant RNGs before constructing each checkpoint's identical panel."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path: str) -> int | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1
    train_args = checkpoint.get("train_args") or {}
    configured = train_args.get("epochs") if isinstance(train_args, dict) else None
    if isinstance(configured, int) and configured > 0 and Path(path).stem in {"last", "best"}:
        return configured
    stem = Path(path).stem
    if stem.startswith("epoch") and stem[5:].isdigit():
        return int(stem[5:]) + 1
    return None


def checkpoint_total_epochs(path: str) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("train_args") or {}
    configured = train_args.get("epochs") if isinstance(train_args, dict) else None
    return configured if isinstance(configured, int) and configured > 0 else 50


def make_trainer(checkpoint: str) -> DetectionTrainer:
    overrides = {
        "model": checkpoint,
        "data": ARGS.data,
        "imgsz": ARGS.imgsz,
        "batch": ARGS.batch,
        "workers": ARGS.workers,
        "epochs": checkpoint_total_epochs(checkpoint),
        "device": "0",
        # Required: otherwise BaseTrainer may rebuild from YAML and discard checkpoint weights.
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "seed": ARGS.seed,
        "deterministic": True,
        "o2o_kd_gain": 0.0,
        "tqgb_cls_m2o_gain": 0.0,
        "tqgb_cls_o2m_gain": 0.0,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.model.train()  # Detect must return raw per-head dictionaries.
    for module in trainer.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # fixed-checkpoint inference, not BN-stat mutation
    trainer.model.criterion = trainer.model.init_criterion()
    return trainer


def make_fixed_loader(trainer: DetectionTrainer):
    """Build an unaugmented, non-rectangular train-split panel with exactly the requested workers."""
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
        raise RuntimeError("Pinned memory is required by the frozen protocol but the loader disabled it")
    return loader


def update_panel_digest(digest: Any, batch: dict[str, Any]) -> None:
    """Hash paths and labels/boxes before device transfer to prove the panel is identical."""
    files = batch.get("im_file")
    if files is None:
        files = []
    digest.update(json.dumps([str(x) for x in files], ensure_ascii=False).encode("utf-8"))
    for key in ("batch_idx", "cls", "bboxes"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes())


def build_level_map(feats: list[torch.Tensor], strides: torch.Tensor, anchor_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map concatenated head anchors to P3/P4/P5, with two independent consistency checks.

    Detect.forward_head concatenates each feature level after flattening HxW, and
    make_anchors iterates over that same feats list and concatenates the same HxW
    blocks.  We reconstruct those blocks, then require make_anchors' per-anchor
    stride tensor to equal the head stride expected for every element.
    """
    if len(feats) != 3:
        raise RuntimeError(f"SARC probe expects exactly 3 detection levels, got {len(feats)}")
    model_strides = strides.detach().float().cpu()
    expected = torch.tensor(ARGS.expected_strides, dtype=torch.float32)
    if model_strides.numel() != 3 or not torch.allclose(model_strides, expected, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"Expected YOLO26 P3/P4/P5 strides {expected.tolist()}, got {model_strides.tolist()}")

    lengths = [int(feature.shape[-2] * feature.shape[-1]) for feature in feats]
    if sum(lengths) != anchor_count:
        raise RuntimeError(f"Feature HxW lengths {lengths} sum to {sum(lengths)}, assignment has {anchor_count}")
    device = feats[0].device
    level_ids = torch.cat(
        [torch.full((length,), level, dtype=torch.long, device=device) for level, length in enumerate(lengths)]
    )
    _, stride_tensor = make_anchors(feats, strides, 0.5)
    expected_per_anchor = strides.to(device=device, dtype=stride_tensor.dtype)[level_ids]
    if stride_tensor.shape != (anchor_count, 1):
        raise RuntimeError(f"Unexpected stride_tensor shape {tuple(stride_tensor.shape)}")
    if not torch.equal(stride_tensor[:, 0], expected_per_anchor):
        mismatch = int((stride_tensor[:, 0] != expected_per_anchor).sum().item())
        raise RuntimeError(f"Anchor-level P3/P4/P5 mapping disagrees with make_anchors at {mismatch} positions")
    return level_ids, stride_tensor[:, 0]


def append_batch_values(
    storage: dict[str, dict[str, list[float]]],
    batch_counts: dict[str, dict[str, list[int]]],
    assignment: dict[str, torch.Tensor],
    feats: list[torch.Tensor],
    strides: torch.Tensor,
) -> None:
    fg_mask = assignment["fg_mask"].bool()
    target_gt_idx = assignment["target_gt_idx"].long()
    target_boxes = assignment["target_bboxes"].float()
    if fg_mask.ndim != 2 or target_boxes.shape[:2] != fg_mask.shape or target_boxes.shape[-1] != 4:
        raise RuntimeError(
            f"Unexpected assignment shapes fg={tuple(fg_mask.shape)}, boxes={tuple(target_boxes.shape)}"
        )
    level_ids, stride_per_anchor = build_level_map(feats, strides, fg_mask.shape[1])
    wh = (target_boxes[..., 2:4] - target_boxes[..., 0:2]).clamp_min(0.0)
    relative_scale = torch.log2(wh.prod(-1).sqrt().clamp_min(1e-6) / stride_per_anchor.unsqueeze(0))
    batch_indices = torch.arange(fg_mask.shape[0], device=fg_mask.device).unsqueeze(1).expand_as(fg_mask)

    for level, level_name in enumerate(LEVEL_NAMES):
        selected = fg_mask & (level_ids.unsqueeze(0) == level)
        values = relative_scale[selected].detach().float().cpu().tolist()
        storage["positive_anchor"][level_name].extend(values)
        batch_counts["positive_anchor"][level_name].append(len(values))

        # Sensitivity view: TAL may assign top-k anchors to one GT.  Count each
        # (image, local-GT-index, level) only once so multiplicity cannot create separation.
        selected_batch = batch_indices[selected].detach().cpu().tolist()
        selected_gt = target_gt_idx[selected].detach().cpu().tolist()
        selected_values = relative_scale[selected].detach().float().cpu().tolist()
        first_by_gt: dict[tuple[int, int], float] = {}
        for image_index, gt_index, value in zip(selected_batch, selected_gt, selected_values):
            key = (int(image_index), int(gt_index))
            previous = first_by_gt.setdefault(key, float(value))
            if not math.isclose(previous, float(value), rel_tol=0.0, abs_tol=1e-5):
                raise RuntimeError(f"Assigned GT {key} has inconsistent relative scale within {level_name}")
        unique_values = list(first_by_gt.values())
        storage["unique_gt_per_image_level"][level_name].extend(unique_values)
        batch_counts["unique_gt_per_image_level"][level_name].append(len(unique_values))


def histogram(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "underflow": 0,
            "overflow": 0,
            "histogram_counts": [0] * (len(HIST_EDGES) - 1),
            "histogram_probabilities": [0.0] * (len(HIST_EDGES) - 1),
        }
    underflow = int(np.count_nonzero(array < HIST_EDGES[0]))
    overflow = int(np.count_nonzero(array > HIST_EDGES[-1]))
    clipped = np.clip(array, HIST_EDGES[0], np.nextafter(HIST_EDGES[-1], HIST_EDGES[0]))
    counts, _ = np.histogram(clipped, bins=HIST_EDGES)
    probabilities = counts / counts.sum()
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "max": float(array.max()),
        "underflow": underflow,
        "overflow": overflow,
        "histogram_counts": counts.astype(int).tolist(),
        "histogram_probabilities": probabilities.tolist(),
    }


def summarize_checkpoint(
    checkpoint: str,
    epoch: int,
    storage: dict[str, dict[str, list[float]]],
    batch_counts: dict[str, dict[str, list[int]]],
    images: int,
    batches: int,
    panel_sha256: str,
) -> dict[str, Any]:
    levels: dict[str, dict[str, dict[str, Any]]] = {}
    for weighting in WEIGHTINGS:
        levels[weighting] = {}
        total = sum(len(storage[weighting][name]) for name in LEVEL_NAMES)
        for name in LEVEL_NAMES:
            row = histogram(storage[weighting][name])
            counts = np.asarray(batch_counts[weighting][name], dtype=np.float64)
            row.update(
                {
                    "assignment_share": row["count"] / total if total else 0.0,
                    "batch_count_mean": float(counts.mean()) if counts.size else 0.0,
                    "batch_count_std": float(counts.std(ddof=0)) if counts.size else 0.0,
                    "batch_count_cv": (
                        float(counts.std(ddof=0) / counts.mean()) if counts.size and counts.mean() > 0 else 0.0
                    ),
                }
            )
            levels[weighting][name] = row
    return {
        "checkpoint": checkpoint,
        "checkpoint_name": Path(checkpoint).name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "epoch": epoch,
        "images": images,
        "batches": batches,
        "panel_sha256": panel_sha256,
        "levels": levels,
    }


def probe_checkpoint(checkpoint: str) -> dict[str, Any]:
    epoch = checkpoint_epoch(checkpoint)
    if epoch is None:
        raise RuntimeError(f"Cannot determine checkpoint epoch: {checkpoint}")
    seed_everything(ARGS.seed)
    trainer = make_trainer(checkpoint)
    # Validation transforms intentionally disable augmentation, while rect=False
    # avoids aspect-ratio sorting. Inputs still come exclusively from the train split.
    loader = make_fixed_loader(trainer)
    criterion = trainer.model.criterion
    storage = {weight: {name: [] for name in LEVEL_NAMES} for weight in WEIGHTINGS}
    batch_counts = {weight: {name: [] for name in LEVEL_NAMES} for weight in WEIGHTINGS}
    panel_digest = hashlib.sha256()
    images = batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= ARGS.max_batches:
                break
            update_panel_digest(panel_digest, batch)
            batch = trainer.preprocess_batch(batch)
            images += int(batch["img"].shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                parsed = criterion.one2many.parse_output(trainer.model.predict(batch["img"]))
                one2many = parsed["one2many"]
                assignment, _, _ = criterion.one2many.get_assigned_targets_and_loss(
                    one2many, batch, return_details=True
                )
            append_batch_values(storage, batch_counts, assignment, one2many["feats"], criterion.one2many.stride)
            batches += 1

    if batches != ARGS.max_batches:
        raise RuntimeError(f"Requested {ARGS.max_batches} fixed batches but loader yielded only {batches}")
    result = summarize_checkpoint(
        checkpoint, epoch, storage, batch_counts, images, batches, panel_digest.hexdigest()
    )
    if hasattr(loader, "close"):
        loader.close()
    del criterion, trainer, loader
    torch.cuda.empty_cache()
    return result


def js_divergence(left: list[float], right: list[float]) -> float | None:
    """Jensen-Shannon divergence in base-2 units, bounded by [0, 1]."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError(f"Histogram shape mismatch: {p.shape} vs {q.shape}")
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(term: np.ndarray) -> float:
        mask = term > 0
        return float(np.sum(term[mask] * np.log2(term[mask] / midpoint[mask])))

    return 0.5 * (kl(p) + kl(q))


def hist_counts(result: dict[str, Any], weighting: str, level: str) -> list[int]:
    return result["levels"][weighting][level]["histogram_counts"]


def build_js_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(results, key=lambda item: item["epoch"])
    for weighting in WEIGHTINGS:
        # Between levels at each checkpoint.
        for result in ordered:
            for left_index, left in enumerate(LEVEL_NAMES):
                for right in LEVEL_NAMES[left_index + 1 :]:
                    rows.append(
                        {
                            "comparison": "between_levels",
                            "weighting": weighting,
                            "epoch_left": result["epoch"],
                            "epoch_right": result["epoch"],
                            "level_left": left,
                            "level_right": right,
                            "consecutive": False,
                            "js_base2": js_divergence(
                                hist_counts(result, weighting, left), hist_counts(result, weighting, right)
                            ),
                        }
                    )
        # Same level across every checkpoint pair; consecutive pairs drive the instability gate.
        for level in LEVEL_NAMES:
            for left_index, left_result in enumerate(ordered):
                for right_index in range(left_index + 1, len(ordered)):
                    right_result = ordered[right_index]
                    rows.append(
                        {
                            "comparison": "across_epochs",
                            "weighting": weighting,
                            "epoch_left": left_result["epoch"],
                            "epoch_right": right_result["epoch"],
                            "level_left": level,
                            "level_right": level,
                            "consecutive": right_index == left_index + 1,
                            "js_base2": js_divergence(
                                hist_counts(left_result, weighting, level),
                                hist_counts(right_result, weighting, level),
                            ),
                        }
                    )
    return rows


def pooled_level_counts(results: list[dict[str, Any]], weighting: str, level: str) -> np.ndarray:
    arrays = [np.asarray(hist_counts(result, weighting, level), dtype=np.int64) for result in results]
    return np.sum(arrays, axis=0)


def make_decision(results: list[dict[str, Any]], js_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = "positive_anchor"
    panel_hashes = sorted({result["panel_sha256"] for result in results})
    epochs = [result["epoch"] for result in sorted(results, key=lambda item: item["epoch"])]
    if len(epochs) != len(set(epochs)):
        raise RuntimeError(f"Duplicate checkpoint epochs are not allowed: {epochs}")
    if epochs != list(ARGS.expected_epochs):
        raise RuntimeError(f"Expected checkpoint epochs {list(ARGS.expected_epochs)}, got {epochs}")
    if len(panel_hashes) != 1:
        raise RuntimeError(f"Fixed panel mismatch across checkpoints: {panel_hashes}")

    pooled_js: dict[str, dict[str, float | None]] = {}
    pooled_counts: dict[str, dict[str, int]] = {}
    for weighting in WEIGHTINGS:
        pooled_counts[weighting] = {
            level: int(pooled_level_counts(results, weighting, level).sum()) for level in LEVEL_NAMES
        }
        pooled_js[weighting] = {}
        for left_index, left in enumerate(LEVEL_NAMES):
            for right in LEVEL_NAMES[left_index + 1 :]:
                key = f"{left}_vs_{right}"
                pooled_js[weighting][key] = js_divergence(
                    pooled_level_counts(results, weighting, left).tolist(),
                    pooled_level_counts(results, weighting, right).tolist(),
                )

    primary_pooled = [value for value in pooled_js[primary].values() if value is not None]
    similarity_no_go = bool(primary_pooled) and max(primary_pooled) < ARGS.similarity_js
    consecutive = [
        row["js_base2"]
        for row in js_rows
        if row["weighting"] == primary
        and row["comparison"] == "across_epochs"
        and row["consecutive"]
        and row["js_base2"] is not None
    ]
    instability_no_go = bool(consecutive) and max(consecutive) > ARGS.instability_js
    inadequate_levels = [
        level for level, count in pooled_counts[primary].items() if count < ARGS.min_pooled_count
    ]
    unique_values = [value for value in pooled_js["unique_gt_per_image_level"].values() if value is not None]
    unique_similarity_no_go = bool(unique_values) and max(unique_values) < ARGS.similarity_js

    if inadequate_levels:
        verdict = "INCONCLUSIVE"
        reason = f"insufficient pooled assignments for {', '.join(inadequate_levels)}"
    elif similarity_no_go:
        verdict = "NO-GO"
        reason = "all pooled P3/P4/P5 distributions are below the pre-registered separation threshold"
    elif instability_no_go:
        verdict = "NO-GO"
        reason = "at least one consecutive-checkpoint JS exceeds the pre-registered instability threshold"
    else:
        verdict = "GO"
        reason = "level separation exists and consecutive-checkpoint distributions remain within the stability bound"

    warnings: list[str] = []
    if verdict == "GO" and unique_similarity_no_go:
        warnings.append(
            "The de-duplicated GT sensitivity view is below the separation threshold; apparent structure may be "
            "driven by TAL positive-anchor multiplicity rather than object scale alone."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "primary_weighting": primary,
        "panel_sha256": panel_hashes[0],
        "epochs": epochs,
        "pooled_counts": pooled_counts,
        "pooled_between_level_js_base2": pooled_js,
        "max_primary_pooled_between_level_js": max(primary_pooled) if primary_pooled else None,
        "max_primary_consecutive_epoch_js": max(consecutive) if consecutive else None,
        "similarity_no_go": similarity_no_go,
        "instability_no_go": instability_no_go,
        "unique_gt_similarity_no_go": unique_similarity_no_go,
        "thresholds": {
            "similarity_js_strictly_below": ARGS.similarity_js,
            "instability_js_strictly_above": ARGS.instability_js,
            "min_pooled_count": ARGS.min_pooled_count,
        },
        "warnings": warnings,
    }


def flatten_stat_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item["epoch"]):
        for weighting in WEIGHTINGS:
            for level in LEVEL_NAMES:
                stats = result["levels"][weighting][level]
                rows.append(
                    {
                        "checkpoint": result["checkpoint"],
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "epoch": result["epoch"],
                        "panel_sha256": result["panel_sha256"],
                        "images": result["images"],
                        "batches": result["batches"],
                        "weighting": weighting,
                        "level": level,
                        "stride": ARGS.expected_strides[LEVEL_NAMES.index(level)],
                        **{key: value for key, value in stats.items() if not isinstance(value, list)},
                        "histogram_counts": json.dumps(stats["histogram_counts"]),
                        "histogram_probabilities": json.dumps(stats["histogram_probabilities"]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def write_outputs(results: list[dict[str, Any]], js_rows: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{ARGS.name}.json",
        "stats_csv": output_dir / f"{ARGS.name}_level_stats.csv",
        "js_csv": output_dir / f"{ARGS.name}_js.csv",
        "decision_csv": output_dir / f"{ARGS.name}_decision.csv",
        "markdown": output_dir / f"{ARGS.name}.md",
    }
    payload = {
        "schema_version": 1,
        "method": {
            "branch": "one2many",
            "split": "train",
            "loader_mode": "val (deterministic/no augmentation)",
            "rect": False,
            "workers": ARGS.workers,
            "pinned_memory_required": True,
            "scale": "u=log2(sqrt(target_width*target_height)/level_stride)",
            "js_log_base": 2,
            "histogram_edges": HIST_EDGES.tolist(),
            "args": vars(ARGS),
        },
        "checkpoints": results,
        "js_comparisons": js_rows,
        "decision": decision,
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(paths["stats_csv"], flatten_stat_rows(results))
    write_csv(paths["js_csv"], js_rows)
    write_csv(
        paths["decision_csv"],
        [
            {
                "verdict": decision["verdict"],
                "reason": decision["reason"],
                "max_primary_pooled_between_level_js": decision["max_primary_pooled_between_level_js"],
                "max_primary_consecutive_epoch_js": decision["max_primary_consecutive_epoch_js"],
                "similarity_no_go": decision["similarity_no_go"],
                "instability_no_go": decision["instability_no_go"],
                "unique_gt_similarity_no_go": decision["unique_gt_similarity_no_go"],
                "panel_sha256": decision["panel_sha256"],
            }
        ],
    )

    primary_rows = flatten_stat_rows(results)
    primary_rows = [row for row in primary_rows if row["weighting"] == "positive_anchor"]
    lines = [
        "# SARC assignment histogram probe",
        "",
        f"**Decision: {decision['verdict']}** — {decision['reason']}.",
        "",
        "> Read-only fixed-checkpoint diagnostic: train split, deterministic non-augmented panel, no optimizer, "
        "no backward pass, no weight or BN-stat updates. JS uses log base 2 and is bounded by [0,1].",
        "",
        "## Reliable P3/P4/P5 mapping",
        "",
        "`Detect.forward_head` flattens each feature map and concatenates levels in `feats` order. "
        "`make_anchors` iterates over the identical list and concatenates the identical HxW blocks. The probe "
        "reconstructs these block indices and then requires every anchor's generated stride to match the "
        "corresponding model stride (8/16/32); mismatch aborts instead of silently assigning a level.",
        "",
        "Only O2M/TAL assignments are primary because YOLO26 detaches O2O inputs from the shared "
        "backbone/neck. `positive_anchor` is the SARC training-signal view; `unique_gt_per_image_level` "
        "de-duplicates TAL top-k repetitions as a sensitivity check.",
        "",
        f"Panel: {results[0]['images']} images in {results[0]['batches']} batches; fingerprint "
        f"`{decision['panel_sha256']}`.",
        "",
        "## Positive-anchor distributions",
        "",
        "| epoch | level | count | share | mean u | p10 | p50 | p90 | per-batch CV |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['epoch']} | {row['level']} | {row['count']} | {row['assignment_share']:.2%} | "
            f"{fmt(row.get('mean'), 3)} | {fmt(row.get('p10'), 3)} | {fmt(row.get('p50'), 3)} | "
            f"{fmt(row.get('p90'), 3)} | {row['batch_count_cv']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Pooled between-level JS",
            "",
            "| weighting | P3 vs P4 | P3 vs P5 | P4 vs P5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for weighting in WEIGHTINGS:
        pooled = decision["pooled_between_level_js_base2"][weighting]
        lines.append(
            f"| {weighting} | {fmt(pooled['P3_vs_P4'])} | {fmt(pooled['P3_vs_P5'])} | "
            f"{fmt(pooled['P4_vs_P5'])} |"
        )

    lines.extend(
        [
            "",
            "## Consecutive-checkpoint JS (positive anchors)",
            "",
            "| epochs | level | JS |",
            "|---|---|---:|",
        ]
    )
    for row in js_rows:
        if (
            row["weighting"] == "positive_anchor"
            and row["comparison"] == "across_epochs"
            and row["consecutive"]
        ):
            lines.append(
                f"| {row['epoch_left']}→{row['epoch_right']} | {row['level_left']} | "
                f"{fmt(row['js_base2'])} |"
            )

    thresholds = decision["thresholds"]
    lines.extend(
        [
            "",
            "## Pre-registered gate",
            "",
            f"- Similarity NO-GO: maximum pooled positive-anchor between-level JS is strictly below "
            f"`{thresholds['similarity_js_strictly_below']}`. Observed: "
            f"`{fmt(decision['max_primary_pooled_between_level_js'])}`.",
            f"- Instability NO-GO: any consecutive-checkpoint positive-anchor JS is strictly above "
            f"`{thresholds['instability_js_strictly_above']}`. Maximum observed: "
            f"`{fmt(decision['max_primary_consecutive_epoch_js'])}`.",
            f"- Sample sufficiency: at least `{thresholds['min_pooled_count']}` pooled positive assignments "
            "per level; otherwise the result is INCONCLUSIVE.",
        ]
    )
    if decision["warnings"]:
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in decision["warnings"]])
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    results: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        result = probe_checkpoint(checkpoint)
        results.append(result)
        compact = {
            "checkpoint": result["checkpoint"],
            "epoch": result["epoch"],
            "panel_sha256": result["panel_sha256"],
            "positive_counts": {
                level: result["levels"]["positive_anchor"][level]["count"] for level in LEVEL_NAMES
            },
        }
        print(json.dumps(compact, ensure_ascii=False))
    results.sort(key=lambda item: item["epoch"])
    js_rows = build_js_rows(results)
    decision = make_decision(results, js_rows)
    paths = write_outputs(results, js_rows, decision)
    print(json.dumps({"decision": decision, "outputs": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
