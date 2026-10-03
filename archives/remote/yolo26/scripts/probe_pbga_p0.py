"""PBGA-P0: zero-update pyramid-boundary gradient conflict probe.

This script is deliberately isolated from the training implementation. It stops the
model immediately before Detect, evaluates the native O2M and O2O heads on two
independent leaf copies of the same P3/P4/P5 tensors, and measures the gradients of
their native losses with respect to those leaves. No optimizer is constructed and
``Tensor.backward`` is never called.

For the total loss only, it also simulates the full-model gradient norm under four
shared-feature routes (official detach, naive reattach, norm-cap, PBGA) so that the
effect of the training-time global max-norm clip is not confused with arbitration.
The simulated routes preserve native gradients to both heads and inject only the
chosen shared gradient through a first-order surrogate at the pyramid boundary.
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
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCALES = ("P3", "P4", "P5")
COMPONENTS = ("cls", "box", "total")
ROUTES = ("official", "naive", "cap", "pbga")
DEFAULT_EXPECTED_EPOCHS = (11, 21, 31, 41, 50)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure native O2M/O2O feature-gradient conflict without updating a YOLO26 checkpoint."
    )
    parser.add_argument("--checkpoints", nargs="+", help="Checkpoint paths or glob patterns.")
    parser.add_argument("--data", help="Dataset YAML. Its train split is loaded with deterministic val transforms.")
    parser.add_argument("--output-dir", help="Directory for JSON, CSV and Markdown outputs.")
    parser.add_argument("--name", default="pbga_p0_probe")
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-epochs", nargs="*", type=int, default=list(DEFAULT_EXPECTED_EPOCHS))
    parser.add_argument("--allow-epoch-mismatch", action="store_true")
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only projection/topology tests and exit.")
    args = parser.parse_args()
    if args.self_test:
        return args
    missing = [flag for flag, value in (("--checkpoints", args.checkpoints), ("--data", args.data), ("--output-dir", args.output_dir)) if not value]
    if missing:
        parser.error(f"required unless --self-test: {', '.join(missing)}")
    if args.batch <= 0 or args.imgsz <= 0 or args.max_batches <= 0 or args.clip_norm <= 0:
        parser.error("batch, imgsz, max-batches and clip-norm must be positive")
    if args.workers != 4:
        parser.error("PBGA-P0 protocol is frozen at workers=4")
    return args


def resolve_checkpoints(patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(str(Path(path).resolve()) for path in matches)
    unique = list(dict.fromkeys(resolved))
    if not unique:
        raise FileNotFoundError(f"No checkpoints matched: {patterns}")
    return unique


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def tensor_dot(left, right) -> float:
    return float((left.detach().float() * right.detach().float()).sum())


def tensor_norm(value) -> float:
    return math.sqrt(float(value.detach().float().square().sum()))


def projection_statistics(g_m, g_o) -> tuple[Any, dict[str, float]]:
    """Project only the O2O component that conflicts with O2M and return diagnostics."""
    gm = g_m.detach().float()
    go = g_o.detach().float()
    dot = float((gm * go).sum())
    gm_sq = float(gm.square().sum())
    gm_norm = math.sqrt(gm_sq)
    go_norm = math.sqrt(float(go.square().sum()))
    valid = gm_norm > EPS and go_norm > EPS
    cosine = dot / max(gm_norm * go_norm, EPS) if valid else 0.0
    projected = go
    if valid and dot < 0:
        projected = go - (dot / max(gm_sq, EPS)) * gm
    removed = go - projected
    projected_norm = tensor_norm(projected)
    removed_norm = tensor_norm(removed)
    cap = min(1.0, gm_norm / max(projected_norm, EPS)) if valid else 0.0
    capped = projected * cap
    post_dot = float((gm * capped).sum())
    return capped.to(dtype=g_o.dtype), {
        "dot": dot,
        "cosine": cosine,
        "o2m_norm": gm_norm,
        "o2o_norm": go_norm,
        "norm_ratio_o2o_to_o2m": go_norm / max(gm_norm, EPS),
        "negative_dot": float(valid and dot < 0),
        "valid_pair": float(valid),
        "projection_removed_norm_fraction": removed_norm / max(go_norm, EPS),
        "projection_removed_energy_fraction": (removed_norm / max(go_norm, EPS)) ** 2,
        "projected_o2o_norm": projected_norm,
        "cap": cap,
        "capped_o2o_norm": tensor_norm(capped),
        "post_projection_dot": post_dot,
    }


def norm_cap(g_m, g_o):
    gm_norm, go_norm = tensor_norm(g_m), tensor_norm(g_o)
    cap = min(1.0, gm_norm / max(go_norm, EPS)) if gm_norm > EPS and go_norm > EPS else 0.0
    return g_o.detach() * cap


def combined_parameter_norm(grads) -> float:
    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))


def run_self_test() -> None:
    """CPU-only math and autograd-topology checks; imports no Ultralytics code."""
    import torch
    from torch import nn

    gm = torch.tensor([1.0, 0.0])
    go = torch.tensor([-1.0, 1.0])
    projected, stats = projection_statistics(gm, go)
    assert stats["negative_dot"] == 1.0
    assert abs(float((gm * projected).sum())) < 1e-6
    assert tensor_norm(projected) <= tensor_norm(gm) + 1e-6
    positive = torch.tensor([1.0, 2.0])
    kept, kept_stats = projection_statistics(gm, positive)
    expected = norm_cap(gm, positive)
    assert torch.allclose(kept, expected)
    assert kept_stats["projection_removed_norm_fraction"] == 0.0

    torch.manual_seed(7)
    backbone = nn.Linear(3, 2, bias=False)
    o2m_head = nn.Linear(2, 1, bias=False)
    o2o_head = nn.Linear(2, 1, bias=False)
    x = torch.randn(4, 3)
    features = backbone(x)
    fm = features.detach().requires_grad_(True)
    fo = features.detach().requires_grad_(True)
    lm = o2m_head(fm).square().mean()
    lo = o2o_head(fo).square().mean()
    grad_m = torch.autograd.grad(lm, fm, retain_graph=True)[0]
    grad_o = torch.autograd.grad(lo, fo, retain_graph=True)[0]
    official = lm + lo + (features * grad_m.detach()).sum()
    params = tuple(backbone.parameters()) + tuple(o2m_head.parameters()) + tuple(o2o_head.parameters())
    grads = torch.autograd.grad(official, params)
    assert tensor_norm(grads[0]) > 0
    assert tensor_norm(grads[1]) > 0 and tensor_norm(grads[2]) > 0
    direct_o2o_backbone = torch.autograd.grad(lo, tuple(backbone.parameters()), allow_unused=True)
    assert all(value is None for value in direct_o2o_backbone)
    assert all(parameter.grad is None for parameter in params)
    print(json.dumps({"status": "PASS", "tests": ["projection", "norm_cap", "isolated_head_surrogate_topology"]}))


ARGS = parse_args()
if ARGS.self_test:
    run_self_test()
    raise SystemExit(0)

CHECKPOINTS = resolve_checkpoints(ARGS.checkpoints)
if ARGS.dry_run:
    print(json.dumps({"checkpoints": CHECKPOINTS, **vars(ARGS)}, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0)

# This must be set before importing torch. Inside the process, the selected physical
# card is always CUDA device 0; the allowed physical-card range is enforced by argparse.
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_device)

import numpy as np
import torch

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_metadata(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored_epoch = checkpoint.get("epoch")
    epoch = stored_epoch + 1 if isinstance(stored_epoch, int) and stored_epoch >= 0 else None
    train_args = checkpoint.get("train_args") or {}
    total_epochs = train_args.get("epochs") if isinstance(train_args, dict) else None
    if epoch is None and Path(path).stem in {"last", "best"} and isinstance(total_epochs, int):
        epoch = total_epochs
    if epoch is None and Path(path).stem.startswith("epoch") and Path(path).stem[5:].isdigit():
        epoch = int(Path(path).stem[5:]) + 1
    return {
        "path": str(Path(path).resolve()),
        "name": Path(path).name,
        "epoch": epoch,
        "total_epochs": total_epochs if isinstance(total_epochs, int) and total_epochs > 0 else 50,
        "sha256": sha256_file(path),
    }


def make_trainer(checkpoint: str, total_epochs: int):
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
        "tqgb_cls_m2o_gain": 0.0,
        "tqgb_cls_o2m_gain": 0.0,
        "tqgb_direction": "none",
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.model.train()
    trainer.model.criterion = trainer.model.init_criterion()
    return trainer


def make_fixed_loader(trainer: DetectionTrainer):
    """Build an unaugmented train-split panel with exactly four workers and pinned memory."""
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
        raise RuntimeError("Probe dataloader is not pinned; protocol requires pinned memory")
    return loader


def recover_prog_weights(criterion, epoch: int | None) -> None:
    if epoch is None:
        raise ValueError("Cannot recover ProgLoss weights without a checkpoint epoch")
    criterion.updates = max(epoch - 1, 0)
    criterion.o2m = criterion.decay(criterion.updates)
    criterion.o2o = max(criterion.total - criterion.o2m, 0.0)


def forward_to_detect_inputs(model, image) -> list[torch.Tensor]:
    """Mirror BaseModel._predict_once only until the final Detect input list."""
    x, saved = image, []
    final = model.model[-1]
    for module in model.model:
        if module.f != -1:
            x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
        if module is final:
            if not isinstance(x, list) or len(x) != final.nl:
                raise RuntimeError(f"Expected {final.nl} Detect inputs, received {type(x).__name__}")
            return x
        x = module(x)
        saved.append(x if module.i in model.save else None)
    raise RuntimeError("Final Detect module was not reached")


def snapshot_batchnorm(model) -> list[tuple[Any, Any, Any, Any]]:
    snapshots = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            snapshots.append(
                (
                    module,
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone(),
                )
            )
    return snapshots


def restore_batchnorm(snapshots) -> None:
    with torch.no_grad():
        for module, mean, var, count in snapshots:
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(count)


def verify_batchnorm_restored(snapshots) -> None:
    for module, mean, var, count in snapshots:
        if not torch.equal(module.running_mean, mean) or not torch.equal(module.running_var, var):
            raise RuntimeError("BatchNorm running statistics changed during the zero-update probe")
        if not torch.equal(module.num_batches_tracked, count):
            raise RuntimeError("BatchNorm batch counter changed during the zero-update probe")


def component_loss(raw: torch.Tensor, component: str, weight: float, batch_size: int) -> torch.Tensor:
    # YOLO26n has reg_max=1. Its two localization terms are box_loss and l1_loss;
    # PBGA-P0 reports their sum as `box` and keeps classification separate.
    if component == "cls":
        value = raw[1]
    elif component == "box":
        value = raw[0] + raw[2]
    elif component == "total":
        value = raw.sum()
    else:
        raise KeyError(component)
    return value * weight * batch_size


def feature_grads(loss: torch.Tensor, features: list[torch.Tensor], retain_graph: bool = True) -> tuple[torch.Tensor, ...]:
    grads = torch.autograd.grad(loss, features, retain_graph=retain_graph, allow_unused=False)
    if any(grad is None for grad in grads):
        raise RuntimeError("A native branch loss was disconnected from a pyramid feature")
    return tuple(grads)


def batch_identity(batch: dict[str, Any]) -> tuple[str, list[str]]:
    files = batch.get("im_file")
    if files:
        normalized = [str(path) for path in files]
        payload = "\n".join(normalized).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), normalized
    image = batch["img"].detach().cpu().contiguous()
    digest = hashlib.sha256(memoryview(image.numpy())).hexdigest()
    return digest, [f"tensor-sha256:{digest}"]


def route_gradients(gm_total, go_total) -> dict[str, list[torch.Tensor]]:
    result = {route: [] for route in ROUTES}
    for gm, go in zip(gm_total, go_total):
        pbga_o, _ = projection_statistics(gm, go)
        capped_o = norm_cap(gm, go)
        result["official"].append(gm.detach())
        result["naive"].append(gm.detach() + go.detach())
        result["cap"].append(gm.detach() + capped_o)
        result["pbga"].append(gm.detach() + pbga_o.detach())
    return result


def simulate_global_clip(
    model,
    original_features: list[torch.Tensor],
    head_loss: torch.Tensor,
    gm_total: tuple[torch.Tensor, ...],
    go_total: tuple[torch.Tensor, ...],
) -> dict[str, dict[str, float]]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    routed = route_gradients(gm_total, go_total)
    output = {}
    for index, route in enumerate(ROUTES):
        surrogate = sum((feature * gradient).sum() for feature, gradient in zip(original_features, routed[route]))
        grads = torch.autograd.grad(
            head_loss + surrogate,
            parameters,
            retain_graph=index < len(ROUTES) - 1,
            allow_unused=True,
        )
        norm = combined_parameter_norm(grads)
        scale = min(1.0, ARGS.clip_norm / max(norm, EPS))
        output[route] = {"global_grad_norm": norm, "global_clip_scale": scale}
    base_scale = output["official"]["global_clip_scale"]
    for route in ROUTES:
        output[route]["clip_scale_relative_to_official"] = output[route]["global_clip_scale"] / max(base_scale, EPS)
    return output


def probe_checkpoint(metadata: dict[str, Any], expected_fingerprints: list[str] | None):
    set_deterministic_seed(ARGS.seed)
    trainer = make_trainer(metadata["path"], metadata["total_epochs"])
    criterion = trainer.model.criterion
    recover_prog_weights(criterion, metadata["epoch"])
    head = trainer.model.model[-1]
    if head.nl != 3 or not hasattr(head, "one2one_cv2"):
        raise RuntimeError("PBGA-P0 currently requires a three-scale O2M/O2O Detect head")
    if int(head.reg_max) != 1:
        raise RuntimeError(f"Expected YOLO26n direct regression (reg_max=1), received {head.reg_max}")

    loader = make_fixed_loader(trainer)
    pinned = bool(getattr(loader, "pin_memory", False))
    if not pinned and not ARGS.allow_unpinned:
        raise RuntimeError("Probe dataloader is not pinned; protocol requires pinned memory")

    initial_bn = snapshot_batchnorm(trainer.model)
    samples: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    global_samples: list[dict[str, dict[str, float]]] = []
    fingerprints, file_panels = [], []
    images = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= ARGS.max_batches:
            break
        batch = trainer.preprocess_batch(batch)
        fingerprint, files = batch_identity(batch)
        fingerprints.append(fingerprint)
        file_panels.append(files)
        images += int(batch["img"].shape[0])
        bn_before = snapshot_batchnorm(trainer.model)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                original_features = forward_to_detect_inputs(trainer.model, batch["img"])
                o2m_features = [feature.detach().requires_grad_(True) for feature in original_features]
                o2o_features = [feature.detach().requires_grad_(True) for feature in original_features]
                one2many = head.forward_head(o2m_features, **head.one2many)
                one2one = head.forward_head(o2o_features, **head.one2one)
                _, raw_o2m, _ = criterion.one2many.get_assigned_targets_and_loss(
                    one2many, batch, return_details=True
                )
                _, raw_o2o, _ = criterion.one2one.get_assigned_targets_and_loss(
                    one2one, batch, return_details=True
                )
                batch_size = int(batch["img"].shape[0])
                losses_m = {
                    component: component_loss(raw_o2m, component, criterion.o2m, batch_size)
                    for component in COMPONENTS
                }
                losses_o = {
                    component: component_loss(raw_o2o, component, criterion.o2o, batch_size)
                    for component in COMPONENTS
                }
                grads_m = {
                    component: feature_grads(losses_m[component], o2m_features) for component in COMPONENTS
                }
                grads_o = {
                    component: feature_grads(losses_o[component], o2o_features) for component in COMPONENTS
                }

            for scale_index, scale in enumerate(SCALES):
                for component in COMPONENTS:
                    _, metrics = projection_statistics(
                        grads_m[component][scale_index], grads_o[component][scale_index]
                    )
                    metrics.update({"batch_index": batch_index, "batch_fingerprint": fingerprint})
                    samples[(scale, component)].append(metrics)

            head_loss = losses_m["total"] + losses_o["total"]
            global_samples.append(
                simulate_global_clip(
                    trainer.model,
                    original_features,
                    head_loss,
                    grads_m["total"],
                    grads_o["total"],
                )
            )
        finally:
            restore_batchnorm(bn_before)

    if len(fingerprints) != ARGS.max_batches:
        raise RuntimeError(f"Expected {ARGS.max_batches} batches, observed {len(fingerprints)}")
    if expected_fingerprints is not None and fingerprints != expected_fingerprints:
        raise RuntimeError("Fixed train batch fingerprints changed between checkpoints")
    restore_batchnorm(initial_bn)
    verify_batchnorm_restored(initial_bn)
    if any(parameter.grad is not None for parameter in trainer.model.parameters()):
        raise RuntimeError("Probe populated .grad despite using autograd.grad")

    feature_rows = []
    for (scale, component), records in samples.items():
        valid = [record for record in records if record["valid_pair"] > 0]
        row = {
            "row_type": "feature",
            "checkpoint": metadata["path"],
            "checkpoint_name": metadata["name"],
            "checkpoint_sha256": metadata["sha256"],
            "epoch": metadata["epoch"],
            "scale": scale,
            "component": component,
            "batches": len(records),
            "valid_pairs": len(valid),
            "prog_o2m": criterion.o2m,
            "prog_o2o": criterion.o2o,
        }
        for metric in (
            "cosine",
            "norm_ratio_o2o_to_o2m",
            "projection_removed_norm_fraction",
            "projection_removed_energy_fraction",
            "cap",
            "post_projection_dot",
        ):
            for key, value in summary([record[metric] for record in valid]).items():
                row[f"{metric}_{key}"] = value
        row["negative_dot_count"] = sum(record["negative_dot"] for record in valid)
        row["negative_dot_rate"] = row["negative_dot_count"] / max(len(valid), 1)
        feature_rows.append(row)

    global_rows = []
    for route in ROUTES:
        row = {
            "row_type": "global_clip",
            "checkpoint": metadata["path"],
            "checkpoint_name": metadata["name"],
            "checkpoint_sha256": metadata["sha256"],
            "epoch": metadata["epoch"],
            "scale": "ALL",
            "component": "total",
            "route": route,
            "batches": len(global_samples),
            "clip_norm": ARGS.clip_norm,
        }
        for metric in ("global_grad_norm", "global_clip_scale", "clip_scale_relative_to_official"):
            for key, value in summary([record[route][metric] for record in global_samples]).items():
                row[f"{metric}_{key}"] = value
        global_rows.append(row)

    result = {
        "metadata": metadata,
        "images": images,
        "batches": len(fingerprints),
        "pinned_memory": pinned,
        "prog_o2m": criterion.o2m,
        "prog_o2o": criterion.o2o,
        "batch_fingerprints": fingerprints,
        "batch_files": file_panels,
        "feature_summaries": feature_rows,
        "global_clip_summaries": global_rows,
        "feature_batch_records": {
            f"{scale}/{component}": records for (scale, component), records in samples.items()
        },
        "global_clip_batch_records": global_samples,
        "zero_update_checks": {"batchnorm_restored": True, "parameter_grad_buffers_empty": True},
    }
    if hasattr(loader, "close"):
        loader.close()
    del loader, criterion, trainer
    torch.cuda.empty_cache()
    return result, fingerprints


def decide(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = [row for row in feature_rows if row["component"] == "total"]
    qualifying = {}
    for scale in SCALES:
        rows = [row for row in total_rows if row["scale"] == scale]
        epochs = [
            row["epoch"]
            for row in rows
            if row["negative_dot_rate"] >= 0.10
            and (row.get("norm_ratio_o2o_to_o2m_median") or 0.0) >= 0.05
        ]
        qualifying[scale] = epochs
    go_scales = [scale for scale, epochs in qualifying.items() if len(epochs) >= 2]
    all_conflict_below_five = all(row["negative_dot_rate"] < 0.05 for row in total_rows)
    all_norm_degenerate = all((row.get("norm_ratio_o2o_to_o2m_median") or 0.0) < 0.05 for row in total_rows)
    decision = "GO" if go_scales else "NO_GO"
    return {
        "decision": decision,
        "rule": (
            "GO iff at least one scale has total-gradient negative-dot rate >=0.10 and median "
            "||g_o2o||/||g_o2m|| >=0.05 at two or more checkpoints."
        ),
        "qualifying_epochs_by_scale": qualifying,
        "qualifying_scales": go_scales,
        "preregistered_no_go_conflict_condition": all_conflict_below_five,
        "preregistered_no_go_norm_condition": all_norm_degenerate,
        "note": "With eight batches, a nonzero conflict rate has 0.125 granularity.",
    }


def source_manifest() -> dict[str, dict[str, str]]:
    import inspect
    import ultralytics.nn.modules.head as head_module
    import ultralytics.nn.tasks as tasks_module
    import ultralytics.utils.loss as loss_module

    manifest = {}
    for name, module in (("head", head_module), ("tasks", tasks_module), ("loss", loss_module)):
        path = Path(inspect.getfile(module)).resolve()
        manifest[name] = {"path": str(path), "sha256": sha256_file(path)}
    return manifest


def format_float(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def write_outputs(results: list[dict[str, Any]]) -> dict[str, str]:
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = [row for result in results for row in result["feature_summaries"]]
    global_rows = [row for result in results for row in result["global_clip_summaries"]]
    verdict = decide(feature_rows)
    payload = {
        "protocol": {
            "name": "PBGA-P0",
            "zero_update": True,
            "split": "train loaded with deterministic val transforms",
            "batch": ARGS.batch,
            "imgsz": ARGS.imgsz,
            "workers": ARGS.workers,
            "pinned_memory_required": not ARGS.allow_unpinned,
            "max_batches": ARGS.max_batches,
            "physical_device": ARGS.physical_device,
            "amp": "bfloat16",
            "clip_norm": ARGS.clip_norm,
            "seed": ARGS.seed,
            "box_definition": "native box_loss + native l1_loss (YOLO26n reg_max=1)",
            "bn_policy": "train-mode batch statistics; running buffers restored exactly after each batch",
        },
        "source_manifest": source_manifest(),
        "verdict": verdict,
        "checkpoints": results,
    }
    paths = {
        "json": output_dir / f"{ARGS.name}.json",
        "csv": output_dir / f"{ARGS.name}.csv",
        "markdown": output_dir / f"{ARGS.name}.md",
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = feature_rows + global_rows
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PBGA-P0 零更新金字塔边界梯度探针",
        "",
        f"**自动裁决：{verdict['decision']}**",
        "",
        "> 固定 train batches（val transforms）、原生 O2M/O2O 监督、无 optimizer、无 backward、无权重更新。"
        " box 表示 YOLO26n 的 box_loss+l1_loss；所有结果使用相应 epoch 的 ProgLoss 原生权重。",
        "",
        "## 总梯度冲突",
        "",
        "| epoch | scale | cosine median | O2O/O2M norm median | negative-dot | projected norm removed | cap median |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in feature_rows:
        if row["component"] != "total":
            continue
        lines.append(
            f"| {row['epoch']} | {row['scale']} | {format_float(row['cosine_median'])} | "
            f"{format_float(row['norm_ratio_o2o_to_o2m_median'])} | {row['negative_dot_rate']:.3f} "
            f"({int(row['negative_dot_count'])}/{row['valid_pairs']}) | "
            f"{format_float(row['projection_removed_norm_fraction_mean'])} | {format_float(row['cap_median'])} |"
        )
    lines.extend(
        [
            "",
            "## 分类与定位分解",
            "",
            "| epoch | scale | component | cosine median | norm ratio median | negative-dot | removed mean |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in feature_rows:
        if row["component"] == "total":
            continue
        lines.append(
            f"| {row['epoch']} | {row['scale']} | {row['component']} | {format_float(row['cosine_median'])} | "
            f"{format_float(row['norm_ratio_o2o_to_o2m_median'])} | {row['negative_dot_rate']:.3f} | "
            f"{format_float(row['projection_removed_norm_fraction_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 模拟全局梯度裁剪",
            "",
            "| epoch | route | grad norm mean/P95 | clip scale mean/min | relative scale mean/min |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in global_rows:
        lines.append(
            f"| {row['epoch']} | {row['route']} | {format_float(row['global_grad_norm_mean'])}/"
            f"{format_float(row['global_grad_norm_p95'])} | {format_float(row['global_clip_scale_mean'])}/"
            f"{format_float(row['global_clip_scale_min'])} | "
            f"{format_float(row['clip_scale_relative_to_official_mean'])}/"
            f"{format_float(row['clip_scale_relative_to_official_min'])} |"
        )
    lines.extend(
        [
            "",
            "## 预注册裁决",
            "",
            verdict["rule"],
            "",
            f"- qualifying epochs by scale: `{json.dumps(verdict['qualifying_epochs_by_scale'])}`",
            f"- qualifying scales: `{verdict['qualifying_scales']}`",
            f"- all conflict rates <5%: `{verdict['preregistered_no_go_conflict_condition']}`",
            f"- all median norm ratios <0.05: `{verdict['preregistered_no_go_norm_condition']}`",
            f"- {verdict['note']}",
            "",
            "## 审计边界",
            "",
            "- 全局裁剪统计是同一计算图上的完整模型参数梯度模拟，不是仅用 feature norm 代替。",
            "- O2O/O2M head 参数始终接收各自原生损失；四条 route 只改变进入 Backbone/Neck 的边界梯度。",
            "- BatchNorm 使用训练批统计，但每个 batch 后精确恢复 running buffers；`.grad` 始终为空。",
            "- 本探针回答是否存在可利用的边界梯度冲突，不构成 AP 改善证据。",
        ]
    )
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def validate_epoch_set(metadata: list[dict[str, Any]]) -> None:
    if ARGS.allow_epoch_mismatch or not ARGS.expected_epochs:
        return
    observed = [item["epoch"] for item in metadata]
    if observed != ARGS.expected_epochs:
        raise RuntimeError(f"Expected checkpoint epochs {ARGS.expected_epochs}, observed {observed}")


def main() -> None:
    metadata = [checkpoint_metadata(path) for path in CHECKPOINTS]
    metadata.sort(key=lambda item: (-1 if item["epoch"] is None else item["epoch"]))
    validate_epoch_set(metadata)
    results, fixed_fingerprints = [], None
    for item in metadata:
        result, fingerprints = probe_checkpoint(item, fixed_fingerprints)
        fixed_fingerprints = fingerprints
        results.append(result)
        print(
            json.dumps(
                {
                    "checkpoint": item["name"],
                    "epoch": item["epoch"],
                    "batches": result["batches"],
                    "pinned_memory": result["pinned_memory"],
                },
                ensure_ascii=False,
            )
        )
    print(json.dumps(write_outputs(results), ensure_ascii=False))


if __name__ == "__main__":
    main()
