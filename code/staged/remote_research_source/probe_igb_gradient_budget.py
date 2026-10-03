#!/usr/bin/env python3
"""IGB-P0: exact zero-update per-GT O2M gradient ledger for YOLO26.

The executable intentionally contains no training path.  It reuses the strict,
content-pinned B50 provenance verifier, reads 24 deterministic train batches,
and differentiates each GT's native positive O2M cls+box loss with respect to
the three tensors entering Detect.  See work/yolo26_research/IGB_P0_SPEC.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


SCHEMA = "IGB-P0-v1"
LEVELS = ("P3", "P4", "P5")
FROZEN_PROTOCOL: dict[str, Any] = {
    "batch": 4,
    "imgsz": 640,
    "workers": 4,
    "batches": 24,
    "max_gt": 128,
    "min_matched_gt": 96,
    "min_assignment_coverage": 0.75,
    "min_gini_batches": 12,
    "seed": 42,
    "strides": [8.0, 16.0, 32.0],
    "autocast": "cuda_bfloat16",
    "checkpoint_files": ["epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "last.pt"],
    "completed_epochs": [11, 21, 31, 41, 50],
}
# Pinned to the exact B50/TQGB source snapshot, not the separately staged BRC
# control source.  A compatible-looking but modified core is rejected until
# these pins and the implementation audit are deliberately regenerated.
EXPECTED_RUNTIME_SHA256 = {
    "strict_provenance_helper": "b0d764fdb00c54a926e27a0a5b8ffd8f016c626998c75321fc1697f4ed444bf9",
    "loss": "6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9",
    "tal": "9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861",
    "detect_head": "1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8",
    "model_tasks": "c1da24859fd56b9869759bd4bfb753456c9074bfc3238b51a08ea3c6cbeda1ae",
    "data_build": "1c3131adc60dc9763b9edcde853d3bc461ec908c2ed9d1965b2610bfd8c27847",
    "detection_trainer": "76d6b181259edc09b03f7cea1f9d59bd4eda2f7964146bbb2e76d4f7201dcf5f",
}
GATES: dict[str, Any] = {
    "median_batch_gini_min": 0.35,
    "median_checkpoint_top20_min": 0.45,
    "median_checkpoint_rho_mass_min": 0.40,
    "positive_rho_checkpoint_min": 3,
}


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for an audit signature."""

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(entry) for entry in item]
        if isinstance(item, Path):
            return str(item.resolve())
        if isinstance(item, torch.Tensor):
            return normalize(item.detach().cpu().tolist())
        if isinstance(item, float):
            if not math.isfinite(item):
                return None
            return item
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        return repr(item)

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def gini(values: Iterable[float]) -> float | None:
    """Population Gini for non-negative values; zero mass has Gini 0."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(value < 0 or not math.isfinite(value) for value in ordered):
        raise ValueError("Gini values must be finite and non-negative")
    total = sum(ordered)
    if total == 0:
        return 0.0
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def top_fraction_share(values: Iterable[float], fraction: float = 0.20) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    if not 0 < fraction <= 1 or any(value < 0 or not math.isfinite(value) for value in numbers):
        raise ValueError("top-share inputs are invalid")
    total = sum(numbers)
    if total == 0:
        return 0.0
    count = max(1, int(math.ceil(fraction * len(numbers))))
    return float(sum(sorted(numbers, reverse=True)[:count]) / total)


def effective_number(values: Iterable[float]) -> tuple[float | None, float | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return None, None
    if any(value < 0 or not math.isfinite(value) for value in numbers):
        raise ValueError("effective-number inputs are invalid")
    denominator = sum(value * value for value in numbers)
    effective = 0.0 if denominator == 0 else sum(numbers) ** 2 / denominator
    return float(effective), float(effective / len(numbers))


def average_ranks(values: Iterable[float]) -> list[float]:
    """One-based average ranks with exact tie handling."""
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("ranks require finite values")
    order = sorted(range(len(numbers)), key=numbers.__getitem__)
    ranks = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and numbers[order[stop]] == numbers[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = [float(v) for v in left], [float(v) for v in right]
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    left_ss = sum((a - mx) ** 2 for a in rx)
    right_ss = sum((b - my) ** 2 for b in ry)
    if left_ss == 0 or right_ss == 0:
        return None
    return float(numerator / math.sqrt(left_ss * right_ss))


def aligned_iou_xyxy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Element-aligned IoU without importing Ultralytics at module import time."""
    intersection_wh = (
        torch.minimum(left[..., 2:], right[..., 2:]) - torch.maximum(left[..., :2], right[..., :2])
    ).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    left_area = (left[..., 2] - left[..., 0]).clamp_min(0) * (left[..., 3] - left[..., 1]).clamp_min(0)
    right_area = (right[..., 2] - right[..., 0]).clamp_min(0) * (right[..., 3] - right[..., 1]).clamp_min(0)
    return intersection / (left_area + right_area - intersection).clamp_min(1e-9)


def mask_for_gt(assignment: dict[str, torch.Tensor], image_index: int, gt_index: int) -> torch.Tensor:
    foreground = assignment["fg_mask"].bool()
    mask = torch.zeros_like(foreground)
    mask[image_index] = foreground[image_index] & (assignment["target_gt_idx"][image_index].long() == gt_index)
    return mask


def native_positive_terms_for_mask(
    criterion: Any,
    tensors: dict[str, torch.Tensor],
    assignment: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reproduce the additive native positive cls/box terms for an arbitrary foreground mask.

    The denominator remains the full native batch target-score sum.  This seam is
    also used by the synthetic test with a small fake native bbox criterion.
    """
    if mask.shape != assignment["fg_mask"].shape:
        raise ValueError("foreground mask shape mismatch")
    target_scores = assignment["target_scores"]
    denominator = target_scores.sum().clamp_min(1)
    labels = assignment["target_labels"].long()
    if labels.ndim == tensors["pred_scores"].ndim:
        labels = labels.squeeze(-1)
    labels = labels.clamp(0, tensors["pred_scores"].shape[-1] - 1)
    selected_logits = tensors["pred_scores"].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    selected_targets = target_scores.gather(-1, labels.unsqueeze(-1)).squeeze(-1).to(selected_logits.dtype)
    cls_points = criterion.bce(selected_logits, selected_targets)
    class_weights = getattr(criterion, "class_weights", None)
    if class_weights is not None:
        cls_points = cls_points * class_weights.reshape(-1).to(cls_points)[labels]
    cls = cls_points[mask].sum() / denominator

    zero = tensors["pred_scores"].sum() * 0.0
    if mask.any():
        box, distribution = criterion.bbox_loss(
            tensors["pred_distri"],
            tensors["pred_bboxes"],
            tensors["anchor_points"],
            assignment["target_bboxes"] / tensors["stride_tensor"],
            target_scores,
            denominator,
            mask,
            tensors["imgsz"],
            tensors["stride_tensor"],
        )
    else:
        box, distribution = zero, zero
    box = box * float(getattr(criterion.hyp, "box"))
    cls = cls * float(getattr(criterion.hyp, "cls"))
    distribution = distribution * float(getattr(criterion.hyp, "dfl"))
    return {"box": box, "cls": cls, "dfl_or_l1": distribution, "total": box + cls + distribution}


def feature_gradient_norms(
    loss: torch.Tensor, features: list[torch.Tensor], retain_graph: bool = True
) -> tuple[list[float], float]:
    """Exact VJP norms for one scalar GT loss and three feature leaves."""
    gradients = torch.autograd.grad(
        loss,
        features,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    squared = []
    norms = []
    for gradient in gradients:
        value = gradient.detach().double().square().sum().item()
        squared.append(value)
        norms.append(math.sqrt(value))
    return norms, math.sqrt(sum(squared))


class DetectBoundaryCapture:
    """Cut the backbone graph at Detect while preserving exact local boundary VJPs."""

    def __init__(self, head: torch.nn.Module):
        self.features: list[torch.Tensor] | None = None
        self.handle = head.register_forward_pre_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not isinstance(inputs[0], (list, tuple)) or len(inputs[0]) != 3:
            raise RuntimeError("IGB-P0 requires Detect to receive exactly P3/P4/P5")
        leaves = [feature.detach().requires_grad_(True) for feature in inputs[0]]
        self.features = leaves
        return (leaves, *inputs[1:])

    def close(self) -> None:
        self.handle.remove()


def level_ids_from_features(features: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    sizes = [int(feature.shape[-2] * feature.shape[-1]) for feature in features]
    level_ids = torch.cat(
        [torch.full((size,), index, device=features[0].device, dtype=torch.long) for index, size in enumerate(sizes)]
    )
    return level_ids, sizes


def gt_diagnostics(
    tensors: dict[str, torch.Tensor],
    assignment: dict[str, torch.Tensor],
    mask: torch.Tensor,
    level_ids: torch.Tensor,
) -> dict[str, Any]:
    """Detached assignment descriptors for one matched GT."""
    with torch.no_grad():
        labels = assignment["target_labels"].long()
        if labels.ndim == tensors["pred_scores"].ndim:
            labels = labels.squeeze(-1)
        labels = labels.clamp(0, tensors["pred_scores"].shape[-1] - 1)
        q = assignment["target_scores"].gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
        logits = tensors["pred_scores"].gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
        pred_pixels = tensors["pred_bboxes"] * tensors["stride_tensor"]
        target_pixels = assignment["target_bboxes"].float()
        iou = aligned_iou_xyxy(pred_pixels.float()[mask], target_pixels[mask]).clamp(0, 1)
        quality = (logits[mask].sigmoid() * iou).clamp_min(0).sqrt()
        selected_q = q[mask]
        flat_mask = mask.reshape(-1)
        repeated_levels = level_ids.repeat(mask.shape[0])
        selected_levels = repeated_levels[flat_mask]
        counts = [int((selected_levels == index).sum().item()) for index in range(3)]
        masses = [float(selected_q[selected_levels == index].sum().item()) for index in range(3)]
        box = target_pixels[mask][0]
        image_area = float((tensors["imgsz"][0] * tensors["imgsz"][1]).item())
        area = float(((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)).item() / image_area)
        return {
            "positive_count": int(mask.sum().item()),
            "target_mass": float(selected_q.sum().item()),
            "target_quality_mean": float(selected_q.mean().item()),
            "matched_quality_mean": float(quality.mean().item()),
            "area": area,
            **{f"{LEVELS[i].lower()}_positive_count": counts[i] for i in range(3)},
            **{f"{LEVELS[i].lower()}_target_mass": masses[i] for i in range(3)},
        }


def summarize_checkpoint(rows: list[dict[str, Any]], completed_epoch: int) -> dict[str, Any]:
    by_batch: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_batch[int(row["batch_index"])].append(float(row["grad_joint"]))
    batch_ginis = [
        {"batch_index": batch, "matched_gt": len(values), "gini": gini(values)}
        for batch, values in sorted(by_batch.items())
        if len(values) >= 2
    ]
    norms = [float(row["grad_joint"]) for row in rows]
    effective, effective_fraction = effective_number(norms)
    correlations = {}
    for name in ("target_mass", "positive_count", "area", "matched_quality_mean", "target_quality_mean"):
        correlations[name] = spearman(norms, [float(row[name]) for row in rows])
    return {
        "completed_epoch": int(completed_epoch),
        "matched_gt": len(rows),
        "batches_with_two_matched_gt": len(batch_ginis),
        "batch_ginis": batch_ginis,
        "median_batch_gini": median(entry["gini"] for entry in batch_ginis),
        "top20_share": top_fraction_share(norms),
        "effective_number": effective,
        "effective_fraction": effective_fraction,
        "spearman_gradient_vs": correlations,
    }


def make_decision(checkpoints: list[dict[str, Any]], gates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the preregistered conjunctive IGB-P0 gates with no fallback inference."""
    gates = GATES if gates is None else gates
    all_batch_ginis = [
        float(row["gini"])
        for checkpoint in checkpoints
        for row in checkpoint.get("batch_ginis", [])
        if row.get("gini") is not None
    ]
    checkpoint_top20 = [checkpoint.get("top20_share") for checkpoint in checkpoints]
    checkpoint_rho = [checkpoint.get("spearman_gradient_vs", {}).get("target_mass") for checkpoint in checkpoints]
    finite_rho = [float(value) for value in checkpoint_rho if value is not None and math.isfinite(float(value))]
    evidence = {
        "median_batch_gini": median(all_batch_ginis),
        "median_checkpoint_top20_share": median(checkpoint_top20),
        "median_checkpoint_rho_mass": median(finite_rho),
        "positive_rho_checkpoints": sum(value > 0 for value in finite_rho),
        "finite_rho_checkpoints": len(finite_rho),
    }
    protocol_checks = {
        "five_checkpoints": len(checkpoints) == 5,
        "all_24_batches_processed": all(
            int(checkpoint.get("processed_batches", -1)) == FROZEN_PROTOCOL["batches"]
            for checkpoint in checkpoints
        ),
        "all_128_gt_selected": all(
            int(checkpoint.get("selected_gt", -1)) == FROZEN_PROTOCOL["max_gt"] for checkpoint in checkpoints
        ),
        "matched_gt_sufficient": all(
            int(checkpoint.get("matched_selected_gt", -1)) >= FROZEN_PROTOCOL["min_matched_gt"]
            for checkpoint in checkpoints
        ),
        "assignment_coverage_sufficient": all(
            float(checkpoint.get("assignment_coverage", -1)) >= FROZEN_PROTOCOL["min_assignment_coverage"]
            for checkpoint in checkpoints
        ),
        "gini_batches_sufficient": all(
            int(checkpoint.get("batches_with_two_matched_gt", -1)) >= FROZEN_PROTOCOL["min_gini_batches"]
            for checkpoint in checkpoints
        ),
        "rho_finite_at_all_checkpoints": len(finite_rho) == len(checkpoints) == 5,
    }
    checks = {
        "median_batch_gini": evidence["median_batch_gini"] is not None
        and evidence["median_batch_gini"] >= gates["median_batch_gini_min"],
        "median_checkpoint_top20_share": evidence["median_checkpoint_top20_share"] is not None
        and evidence["median_checkpoint_top20_share"] >= gates["median_checkpoint_top20_min"],
        "median_checkpoint_rho_mass": evidence["median_checkpoint_rho_mass"] is not None
        and evidence["median_checkpoint_rho_mass"] >= gates["median_checkpoint_rho_mass_min"],
        "positive_rho_checkpoint_count": evidence["positive_rho_checkpoints"]
        >= gates["positive_rho_checkpoint_min"],
    }
    verdict = "GO" if all(protocol_checks.values()) and all(checks.values()) else "NO-GO"
    return {
        "verdict": verdict,
        "reason": (
            "all four preregistered gradient-domination gates passed"
            if verdict == "GO"
            else "one or more preregistered gradient-domination gates failed or lacked finite evidence"
        ),
        "evidence": evidence,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "thresholds": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Exact frozen B50 checkpoint paths/globs.")
    parser.add_argument("--data", required=True, help="Exact B50 dataset YAML; train split only.")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="igb_p0_gradient_budget")
    parser.add_argument("--dry-run", action="store_true", help="Print the frozen protocol without loading CUDA/data.")
    return parser.parse_args()


def initialize_exact_runtime(source_root: Path, physical_device: int):
    """Bind imports to the requested source snapshot before any Ultralytics import."""
    root = source_root.resolve()
    package = root / "ultralytics"
    helper_path = root / "scripts" / "probe_asac_assignment_stability.py"
    if not package.is_dir() or not helper_path.is_file():
        raise RuntimeError(f"source root lacks ultralytics/ or strict provenance helper: {root}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    provenance = importlib.import_module("probe_asac_assignment_stability")
    provenance.initialize_runtime(physical_device)
    imported_root = Path(provenance.ultralytics.__file__).resolve().parents[1]
    if imported_root != root:
        raise RuntimeError(f"Ultralytics import escaped requested source: {imported_root} != {root}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("IGB-P0 requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("IGB-P0 frozen protocol requires CUDA BF16 support")
    return provenance


def trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data=str(Path(args.data).resolve()),
        imgsz=FROZEN_PROTOCOL["imgsz"],
        batch=FROZEN_PROTOCOL["batch"],
        workers=FROZEN_PROTOCOL["workers"],
        seed=FROZEN_PROTOCOL["seed"],
    )


def canonical_batch(batch: dict[str, Any], batch_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = [str(Path(path).resolve()) for path in batch.get("im_file", [])]
    batch_ids = batch["batch_idx"].detach().cpu().long()
    classes = batch["cls"].detach().cpu().view(-1)
    boxes = batch["bboxes"].detach().cpu()
    local_counts: dict[int, int] = defaultdict(int)
    objects = []
    for row_index, image_index_tensor in enumerate(batch_ids):
        image_index = int(image_index_tensor.item())
        local_index = local_counts[image_index]
        local_counts[image_index] += 1
        objects.append(
            {
                "batch_index": batch_index,
                "image_index": image_index,
                "image_path": paths[image_index],
                "local_gt_index": local_index,
                "class": int(classes[row_index].item()),
                "bbox_xywhn": [float(value) for value in boxes[row_index].tolist()],
            }
        )
    payload = {
        "batch_index": batch_index,
        "paths": paths,
        "batch_idx": batch_ids.tolist(),
        "classes": [float(value) for value in classes.tolist()],
        "boxes": [[float(value) for value in row] for row in boxes.tolist()],
        "image_tensor_sha256": tensor_sha256(batch["img"]),
    }
    payload["sha256"] = sha256_json(payload)
    return payload, objects


def scan_panel(loader: Any) -> dict[str, Any]:
    batches, objects = [], []
    for batch_index, batch in enumerate(loader):
        if batch_index >= FROZEN_PROTOCOL["batches"]:
            break
        canonical, batch_objects = canonical_batch(batch, batch_index)
        batches.append(canonical)
        objects.extend(batch_objects)
    if len(batches) != FROZEN_PROTOCOL["batches"]:
        raise RuntimeError(f"dataset yielded only {len(batches)} of 24 frozen batches")
    if len(objects) < 2:
        raise RuntimeError("fixed panel has fewer than two ground-truth objects")
    for row in objects:
        selection_payload = {
            "seed": FROZEN_PROTOCOL["seed"],
            "image_path": row["image_path"],
            "local_gt_index": row["local_gt_index"],
            "class": row["class"],
            "bbox_xywhn": row["bbox_xywhn"],
        }
        row["selection_sha256"] = sha256_json(selection_payload)
    selected = sorted(objects, key=lambda row: (row["selection_sha256"], row["batch_index"], row["image_index"]))[
        : FROZEN_PROTOCOL["max_gt"]
    ]
    for row in selected:
        row["id"] = f"b{row['batch_index']}:i{row['image_index']}:g{row['local_gt_index']}"
    panel_public = {
        "batch_sha256": [row["sha256"] for row in batches],
        "total_raw_gt": len(objects),
        "selected_gt": len(selected),
        "selected": selected,
    }
    return {
        **panel_public,
        "panel_sha256": sha256_json(panel_public),
        "selected_sha256": sha256_json(selected),
    }


def selected_by_batch(panel: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in panel["selected"]:
        result[int(row["batch_index"])].append(row)
    return result


def verify_batch(batch: dict[str, Any], batch_index: int, panel: dict[str, Any]) -> None:
    canonical, _ = canonical_batch(batch, batch_index)
    expected = panel["batch_sha256"][batch_index]
    if canonical["sha256"] != expected:
        raise RuntimeError(f"fixed train batch {batch_index} changed: {canonical['sha256']} != {expected}")


def extract_o2m(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, tuple):
        raw = raw[1]
    if not isinstance(raw, dict) or "one2many" not in raw:
        raise RuntimeError("live Detect output lacks native one2many dictionary")
    branch = raw["one2many"]
    if not isinstance(branch, dict) or not {"boxes", "scores", "feats"}.issubset(branch):
        raise RuntimeError("native one2many output contract changed")
    return branch


def native_tensors(criterion: Any, predictions: dict[str, torch.Tensor], make_anchors: Any) -> dict[str, torch.Tensor]:
    pred_distri = predictions["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = predictions["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(predictions["feats"], criterion.stride, 0.5)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    imgsz = torch.tensor(
        predictions["feats"][0].shape[2:], device=pred_scores.device, dtype=pred_scores.dtype
    ) * criterion.stride[0]
    return {
        "pred_distri": pred_distri,
        "pred_scores": pred_scores,
        "pred_bboxes": pred_bboxes,
        "anchor_points": anchor_points,
        "stride_tensor": stride_tensor,
        "imgsz": imgsz,
    }


def detach_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in assignment.items()}


def float_close(left: float, right: float, atol: float = 2e-3, rtol: float = 8e-3) -> bool:
    return abs(left - right) <= atol + rtol * max(abs(left), abs(right))


def no_parameter_gradients(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def full_state_sha256(model: torch.nn.Module, provenance: Any) -> tuple[str, int]:
    """Hash every parameter and persistent buffer, not only BatchNorm state."""
    return provenance.named_tensor_sha256(model.state_dict().items())


def validate_live_contract(trainer: Any, provenance: Any) -> dict[str, Any]:
    contract = provenance.validate_criterion_contract(trainer, FROZEN_PROTOCOL["strides"])
    criterion = trainer.model.criterion.one2many
    head = trainer.model.model[-1]
    if int(getattr(head, "nl", -1)) != 3:
        raise RuntimeError("IGB-P0 supports exactly three Detect levels")
    if int(getattr(criterion.assigner, "topk", -1)) != 10:
        raise RuntimeError("IGB-P0 must use native O2M TAL topk=10")
    if not callable(getattr(criterion, "get_assigned_targets_and_loss", None)):
        raise RuntimeError("native O2M assignment-details interface is missing")
    return contract


def runtime_sources(trainer: Any, provenance: Any, trajectory: dict[str, Any]) -> dict[str, Any]:
    criterion = trainer.model.criterion.one2many
    head = trainer.model.model[-1]
    paths = {
        "igb_probe": Path(__file__).resolve(),
        "strict_provenance_helper": Path(provenance.__file__).resolve(),
        "loss": Path(inspect.getsourcefile(criterion.__class__)).resolve(),
        "tal": Path(inspect.getsourcefile(criterion.assigner.__class__)).resolve(),
        "detect_head": Path(inspect.getsourcefile(head.__class__)).resolve(),
        "model_tasks": Path(inspect.getsourcefile(trainer.model.__class__)).resolve(),
        "data_build": Path(inspect.getsourcefile(provenance.build_yolo_dataset)).resolve(),
        "detection_trainer": Path(inspect.getsourcefile(provenance.DetectionTrainer)).resolve(),
    }
    for key, row in trajectory["run_evidence"]["config_files"].items():
        paths[f"config_{key}"] = Path(row["path"]).resolve()
    rows = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected, "actual": rows.get(name, {}).get("sha256")}
        for name, expected in EXPECTED_RUNTIME_SHA256.items()
        if rows.get(name, {}).get("sha256") != expected
    }
    if mismatches:
        raise RuntimeError(f"runtime source differs from the pinned exact research snapshot: {mismatches}")
    return {
        "files": rows,
        "bundle_sha256": sha256_json(rows),
        "pinned_core_sha256": EXPECTED_RUNTIME_SHA256,
    }


def probe_checkpoint(
    record: dict[str, Any],
    args: argparse.Namespace,
    provenance: Any,
    panel: dict[str, Any],
    trajectory: dict[str, Any],
    trainer: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    trainer = provenance.make_trainer(record["path"], trainer_args(args)) if trainer is None else trainer
    contract = validate_live_contract(trainer, provenance)
    source_meta = runtime_sources(trainer, provenance, trajectory)  # content pins fail before any panel forward
    loader, loader_meta = provenance.make_fixed_loader(trainer, trainer_args(args))
    criterion = trainer.model.criterion.one2many
    model = trainer.model
    head = model.model[-1]
    if not no_parameter_gradients(model):
        raise RuntimeError("checkpoint model already has parameter gradients before IGB-P0")
    before = provenance.integrity_snapshot(trainer)
    before["parameter_grads_all_none"] = True
    before["full_state_sha256"], before["full_state_tensors"] = full_state_sha256(model, provenance)
    capture = DetectBoundaryCapture(head)
    selected = selected_by_batch(panel)
    rows: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    decomposition_checks = []
    autocast_evidence = None
    processed_batches = 0
    try:
        for batch_index, host_batch in enumerate(loader):
            if batch_index >= FROZEN_PROTOCOL["batches"]:
                break
            processed_batches += 1
            verify_batch(host_batch, batch_index, panel)
            batch = trainer.preprocess_batch(host_batch)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = model(batch["img"])
                predictions = extract_o2m(raw)
                features = capture.features
                if features is None or len(predictions["feats"]) != len(features) or any(
                    left is not right for left, right in zip(predictions["feats"], features)
                ):
                    raise RuntimeError("captured P3/P4/P5 are not the live O2M shared feature tensors")
                if autocast_evidence is None:
                    autocast_evidence = {
                        "enabled": bool(torch.is_autocast_enabled("cuda")),
                        "dtype": str(torch.get_autocast_dtype("cuda")),
                        "feature_dtypes": [str(feature.dtype) for feature in features],
                        "score_dtype": str(predictions["scores"].dtype),
                    }
                with torch.no_grad():
                    assignment, _, _ = criterion.get_assigned_targets_and_loss(
                        predictions, batch, return_details=True
                    )
                assignment = detach_assignment(assignment)
                tensors = native_tensors(criterion, predictions, provenance.make_anchors)
                level_ids, level_sizes = level_ids_from_features(features)
                union = torch.zeros_like(assignment["fg_mask"], dtype=torch.bool)
                component_sums = {"box": 0.0, "cls": 0.0, "dfl_or_l1": 0.0, "total": 0.0}
                for selected_gt in selected.get(batch_index, []):
                    image_index = int(selected_gt["image_index"])
                    gt_index = int(selected_gt["local_gt_index"])
                    gt_mask = mask_for_gt(assignment, image_index, gt_index)
                    if not gt_mask.any():
                        continue
                    terms = native_positive_terms_for_mask(criterion, tensors, assignment, gt_mask)
                    level_norms, joint_norm = feature_gradient_norms(terms["total"], features, retain_graph=True)
                    descriptors = gt_diagnostics(tensors, assignment, gt_mask, level_ids)
                    row = {
                        "checkpoint": record["filename"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "completed_epoch": int(record["epoch"]),
                        "batch_index": batch_index,
                        "image_index": image_index,
                        "image_path_sha256": hashlib.sha256(
                            selected_gt["image_path"].encode("utf-8")
                        ).hexdigest(),
                        "local_gt_index": gt_index,
                        "class": int(selected_gt["class"]),
                        "panel_gt_id": selected_gt["id"],
                        **descriptors,
                        "positive_cls_loss": float(terms["cls"].detach().float().item()),
                        "box_iou_loss": float(terms["box"].detach().float().item()),
                        "dfl_or_l1_loss": float(terms["dfl_or_l1"].detach().float().item()),
                        "positive_total_loss": float(terms["total"].detach().float().item()),
                        **{f"grad_{LEVELS[index].lower()}": level_norms[index] for index in range(3)},
                        "grad_joint": joint_norm,
                    }
                    rows.append(row)
                    matched_ids.add(selected_gt["id"])
                    union |= gt_mask
                    for name in component_sums:
                        component_sums[name] += float(terms[name].detach().float().item())
                if union.any():
                    with torch.no_grad():
                        union_terms = native_positive_terms_for_mask(criterion, tensors, assignment, union)
                    union_values = {name: float(value.float().item()) for name, value in union_terms.items()}
                    errors = {
                        name: abs(component_sums[name] - union_values[name]) for name in component_sums
                    }
                    if any(
                        not float_close(component_sums[name], union_values[name]) for name in component_sums
                    ):
                        raise RuntimeError(
                            f"per-GT decomposition is not additive at batch {batch_index}: "
                            f"sum={component_sums}, union={union_values}"
                        )
                    decomposition_checks.append(
                        {"batch_index": batch_index, "component_abs_error": errors, "level_sizes": level_sizes}
                    )
            del raw, predictions, batch
            if not no_parameter_gradients(model):
                raise RuntimeError(f"autograd.grad populated parameter .grad at batch {batch_index}")
    finally:
        capture.close()
    if processed_batches != FROZEN_PROTOCOL["batches"]:
        raise RuntimeError(f"processed {processed_batches} batches; frozen protocol requires 24")
    if autocast_evidence is None or not autocast_evidence["enabled"] or autocast_evidence["dtype"] != "torch.bfloat16":
        raise RuntimeError(f"BF16 autocast contract failed: {autocast_evidence}")
    after = provenance.integrity_snapshot(trainer)
    after["parameter_grads_all_none"] = no_parameter_gradients(model)
    after["full_state_sha256"], after["full_state_tensors"] = full_state_sha256(model, provenance)
    provenance.assert_integrity_unchanged(before, after, record["path"])
    if (before["full_state_sha256"], before["full_state_tensors"]) != (
        after["full_state_sha256"],
        after["full_state_tensors"],
    ):
        raise RuntimeError("model parameter/buffer state changed during zero-update probe")
    if not after["parameter_grads_all_none"]:
        raise RuntimeError("zero-update probe left parameter gradients")
    summary = summarize_checkpoint(rows, int(record["epoch"]))
    summary.update(
        {
            "checkpoint": record["filename"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "selected_gt": panel["selected_gt"],
            "processed_batches": processed_batches,
            "matched_selected_gt": len(matched_ids),
            "assignment_coverage": len(matched_ids) / max(panel["selected_gt"], 1),
            "unmatched_selected_gt": panel["selected_gt"] - len(matched_ids),
            "loader": loader_meta,
            "autocast": autocast_evidence,
            "decomposition_checks": decomposition_checks,
            "integrity_before": before,
            "integrity_after": after,
            "criterion_contract": contract,
        }
    )
    del trainer
    torch.cuda.empty_cache()
    return summary, rows, source_meta


def write_outputs(result: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}_per_gt.csv"
    markdown_path = output_dir / f"{name}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    decision = result["decision"]
    evidence = decision["evidence"]
    lines = [
        "# IGB-P0 zero-update gradient-budget result",
        "",
        f"Verdict: **{decision['verdict']}** — {decision['reason']}.",
        "",
        "> Train-only fixed panel; native O2M assignment; exact per-GT boundary VJPs; eval BN; no optimizer, update, checkpoint write, validation AP, or test data.",
        "",
        f"Panel SHA256: `{result['panel']['panel_sha256']}`; selected GT: `{result['panel']['selected_gt']}`; source bundle: `{result['runtime']['source_bundle_sha256']}`.",
        "",
        "## Gate evidence",
        "",
        f"- Median batch Gini: `{evidence['median_batch_gini']}` (requires >= `{GATES['median_batch_gini_min']}`).",
        f"- Median checkpoint top-20 share: `{evidence['median_checkpoint_top20_share']}` (requires >= `{GATES['median_checkpoint_top20_min']}`).",
        f"- Median checkpoint rho(gradient,mass): `{evidence['median_checkpoint_rho_mass']}` (requires >= `{GATES['median_checkpoint_rho_mass_min']}`).",
        f"- Positive rho checkpoints: `{evidence['positive_rho_checkpoints']}` (requires >= `{GATES['positive_rho_checkpoint_min']}`).",
        "",
        "## Per checkpoint",
        "",
        "| checkpoint | epoch | matched/selected | median Gini | top20 | N_eff | rho mass | rho count | rho area | rho quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in result["checkpoints"]:
        rho = checkpoint["spearman_gradient_vs"]
        lines.append(
            f"| {checkpoint['checkpoint']} | {checkpoint['completed_epoch']} | "
            f"{checkpoint['matched_selected_gt']}/{checkpoint['selected_gt']} | "
            f"{checkpoint['median_batch_gini']} | {checkpoint['top20_share']} | "
            f"{checkpoint['effective_number']} | {rho['target_mass']} | {rho['positive_count']} | "
            f"{rho['area']} | {rho['matched_quality_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Semantics warning",
            "",
            "The ledger excludes background/non-assigned-class BCE because it has no unique GT owner. It also omits the checkpoint-common E2E `batch_size * prog_o2m_w` scalar. The boundary detach preserves dL/d(P3,P4,P5) exactly but cannot be used to infer upstream parameter gradients.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "protocol": FROZEN_PROTOCOL, "gates": GATES}, indent=2))
        return
    source_root = Path(args.source_root).resolve()
    provenance = initialize_exact_runtime(source_root, args.physical_device)
    random.seed(FROZEN_PROTOCOL["seed"])
    torch.manual_seed(FROZEN_PROTOCOL["seed"])
    torch.cuda.manual_seed_all(FROZEN_PROTOCOL["seed"])
    resolved = provenance.resolve_checkpoints(args.checkpoints)
    records, trajectory = provenance.inspect_checkpoints(resolved, str(Path(args.data).resolve()))
    if [row["filename"] for row in records] != FROZEN_PROTOCOL["checkpoint_files"]:
        raise RuntimeError("strict verifier returned a checkpoint panel different from frozen IGB-P0")
    if [int(row["epoch"]) for row in records] != FROZEN_PROTOCOL["completed_epochs"]:
        raise RuntimeError("strict verifier returned completed epochs different from frozen IGB-P0")

    first_trainer = provenance.make_trainer(records[0]["path"], trainer_args(args))
    first_loader, panel_loader_meta = provenance.make_fixed_loader(first_trainer, trainer_args(args))
    panel = scan_panel(first_loader)
    panel["loader"] = panel_loader_meta
    checkpoint_summaries, all_rows, source_meta = [], [], None
    for index, record in enumerate(records):
        summary, rows, live_sources = probe_checkpoint(
            record,
            args,
            provenance,
            panel,
            trajectory,
            trainer=first_trainer if index == 0 else None,
        )
        first_trainer = None
        checkpoint_summaries.append(summary)
        all_rows.extend(rows)
        if source_meta is None:
            source_meta = live_sources
        elif source_meta["bundle_sha256"] != live_sources["bundle_sha256"]:
            raise RuntimeError("live source bundle changed between checkpoints")

    for record in records:
        if file_sha256(record["path"]) != record["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint content changed during zero-update probe: {record['path']}")
    decision = make_decision(checkpoint_summaries)
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_count": torch.cuda.device_count(),
        "source_root": str(source_root),
        "sources": source_meta["files"],
        "source_bundle_sha256": source_meta["bundle_sha256"],
    }
    result = {
        "schema": SCHEMA,
        "protocol": FROZEN_PROTOCOL,
        "gates": GATES,
        "decision": decision,
        "panel": panel,
        "trajectory": trajectory,
        "checkpoint_provenance": records,
        "checkpoints": checkpoint_summaries,
        "runtime": runtime,
        "semantics": {
            "assignment": "live native O2M TAL topk=10",
            "classification": "assigned-class BCE on foreground anchors only",
            "localization": "live CIoU plus DFL/L1 bbox_loss",
            "normalization": "full-batch native target_scores_sum",
            "feature_boundary": "equal-valued detached Detect inputs; exact local VJP",
            "omitted_common_scalar": "E2E batch_size * progressive O2M weight",
            "optimizer_or_update": False,
        },
    }
    write_outputs(result, all_rows, Path(args.output_dir).resolve(), args.name)
    print(json.dumps({"decision": decision, "panel_sha256": panel["panel_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
