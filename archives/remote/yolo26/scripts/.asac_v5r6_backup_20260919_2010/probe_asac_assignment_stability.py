"""Read-only ASAC assignment-stability probe for fixed YOLO26 checkpoints.

This probe measures whether deterministic scale/translation perturbations produce
an intensity-sensitive, quality-relevant assignment-stability signal.  It uses a
fixed train-only panel, performs no backward pass or optimizer update, freezes BN
statistics, and never reads validation/test AP.

For every retained ground-truth object, positive anchors from both native O2M and
O2O assigners are inverse-mapped into the clean image and expressed in GT-relative
coordinates.  A smooth (level, dx, dy) occupancy distribution makes comparison
well-defined even for the one-positive O2O branch.  Stability is exp(-JS), where
JS is base-2 Jensen-Shannon divergence.  Assignment quality agreement is computed
separately from predicted class probability and IoU and is not used to construct
the occupancy distribution.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HEADS = ("O2M", "O2O")
LEVEL_NAMES = ("P3", "P4", "P5")
SCHEMA_VERSION = 2
PROVENANCE_REVISION = 5

# ASAC-P0 v2 is frozen here rather than being defined by argparse defaults.  CLI options remain available for
# sensitivity diagnostics, but any deviation forces the final verdict to NON_PREREGISTERED and can never yield
# GO/NO-GO.  Execution-integrity constraints (256 images, workers=4, non-augmented sequential loader) remain hard.
FROZEN_PROTOCOL: dict[str, Any] = {
    "batch": 64,
    "imgsz": 640,
    "workers": 4,
    "max_batches": 4,
    "panel_images": 256,
    "seed": 42,
    "strengths": [0.25, 0.50, 0.75],
    "max_scale_delta": 0.50,
    "max_translate": 0.10,
    "min_visibility": 0.70,
    "min_box_pixels": 2.0,
    "grid_bins": 13,
    "grid_extent": 1.5,
    "kernel_sigma": 0.25,
    "expected_strides": [8.0, 16.0, 32.0],
    "expected_epochs": [11, 21, 31, 41, 50],
    "min_object_pairs": 100,
    "min_assignment_coverage": 0.50,
    "monotonic_tolerance": 0.01,
    "min_weak_strong_drop": 0.02,
    "min_monotonic_checkpoints": 3,
    "min_spearman": 0.20,
    "min_correlated_checkpoints": 2,
}

EXPECTED_CHECKPOINT_FILES = {
    "epoch10.pt": 11,
    "epoch20.pt": 21,
    "epoch30.pt": 31,
    "epoch40.pt": 41,
    "last.pt": 50,
}

# Independently recorded by the earlier PBGA and SARC read-only diagnostics.
# Pinning file content makes path/metadata spoofing insufficient to pass C6.
EXPECTED_CHECKPOINT_SHA256 = {
    "epoch10.pt": "ac6833319b8b39509ac4b02cedcdd90994220620fc2559b6d0549eddc94e21d6",
    "epoch20.pt": "ea4b17ae9f09c8774cf17970fc0071e95c1cc8abf48b7bdeac4315c79da4e9de",
    "epoch30.pt": "cdbb6e3d72b587aba9754a3fd0450a64a50170ea1cdb13e318a046ddd16e6aa9",
    "epoch40.pt": "a2f334e6c260790f5ba03702e26746f856dcde20cb31a36229be25f9d62b0765",
    "last.pt": "49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb",
}

EXPECTED_CONFIG_SHA256 = {
    "model": "afc4600876cc3a996aa65f5894d2a3005f25baa29b574bf46eff373779451b26",
    "data": "5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40",
}

# This is the complete 124-key args.yaml saved by the real B50 run, plus the
# three TQGB fields that did not yet exist when B50 was trained.  The latter
# have explicit semantic defaults; every other field must be present on both
# sides of the provenance proof.  Freezing the complete manifest avoids the
# old failure mode where five mutually consistent checkpoints from a different
# model/augmentation/KD trajectory could pass merely because they agreed with
# one another.
FROZEN_B50_ARGS: dict[str, Any] = {
    "task": "detect",
    "mode": "train",
    "model": "/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/models/yolo26n_baseline.yaml",
    "data": "/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml",
    "epochs": 50,
    "time": None,
    "patience": 50,
    "batch": 64,
    "imgsz": 640,
    "save": True,
    "save_period": 10,
    "cache": False,
    "device": "4,5,6,7",
    "workers": 4,
    "project": "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs",
    "name": "b50_no_kd_e50_g4567_s42",
    "exist_ok": False,
    "pretrained": False,
    "cls_remap": True,
    "optimizer": "auto",
    "verbose": True,
    "seed": 42,
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "resume": False,
    "amp": "bf16",
    "fraction": 1.0,
    "profile": False,
    "freeze": None,
    "multi_scale": 0.0,
    "compile": False,
    "channels_last": None,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.0,
    "val": True,
    "split": "val",
    "save_json": False,
    "conf": None,
    "iou": 0.7,
    "max_det": 300,
    "quantize": None,
    "dnn": False,
    "plots": False,
    "nms": False,
    "source": None,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "embed": None,
    "show": False,
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "line_width": None,
    "format": "torchscript",
    "keras": False,
    "optimize": False,
    "dynamic": False,
    "simplify": True,
    "opset": None,
    "workspace": None,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "distill_model": None,
    "dis": 6.0,
    "o2o_kd_gain": 0.0,
    "o2o_kd_temp": 1.0,
    "o2o_kd_min_quality": 0.25,
    "o2o_kd_margin": 0.05,
    "o2o_kd_warmup": 5.0,
    "o2o_kd_ramp": 10.0,
    "o2o_kd_fade": 10.0,
    "o2o_kd_gate": True,
    "o2o_kd_supervision_gate": "none",
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "rle": 1.0,
    "angle": 1.0,
    "dlog": 1.0,
    "dgrad": 0.5,
    "dlam": 1.0,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "cfg": None,
    "tracker": "tracktrack.yaml",
    "save_dir": "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42",
    "tqgb_cls_m2o_gain": 0.0,
    "tqgb_cls_o2m_gain": 0.0,
    "tqgb_direction": "none",
}

# `args.yaml` is written before `build_optimizer()` mutates the live trainer
# arguments.  For optimizer=auto, Ultralytics deterministically sets
# warmup_bias_lr from the launch value 0.1 to 0.0 inside build_optimizer()
# after choosing the concrete optimizer from the iteration count.  Checkpoints therefore contain the latter
# while args.yaml correctly retains the former.  Keep both manifests explicit:
# this is a one-field, source-proven transformation rather than a general
# mismatch allowance.
RUNTIME_MANIFEST_TRANSFORMS: dict[str, dict[str, Any]] = {
    "warmup_bias_lr": {
        "launch_value": 0.1,
        "checkpoint_value": 0.0,
        "precondition": {"optimizer": "auto"},
        "reason": "Ultralytics build_optimizer(name='auto') sets self.args.warmup_bias_lr=0.0",
    }
}


def checkpoint_runtime_manifest(launch_manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact checkpoint manifest from the frozen launch manifest."""
    runtime = dict(launch_manifest)
    for key, transform in RUNTIME_MANIFEST_TRANSFORMS.items():
        for guard_key, guard_value in transform["precondition"].items():
            if runtime.get(guard_key) != guard_value:
                raise RuntimeError(
                    f"cannot apply frozen runtime transform for {key!r}: "
                    f"{guard_key}={runtime.get(guard_key)!r}, expected {guard_value!r}"
                )
        if runtime.get(key) != transform["launch_value"]:
            raise RuntimeError(
                f"cannot apply frozen runtime transform for {key!r}: "
                f"launch={runtime.get(key)!r}, expected {transform['launch_value']!r}"
            )
        runtime[key] = transform["checkpoint_value"]
    return runtime


FROZEN_B50_CHECKPOINT_ARGS: dict[str, Any] = checkpoint_runtime_manifest(FROZEN_B50_ARGS)

TRAJECTORY_FIELDS = tuple(FROZEN_B50_ARGS)
SEMANTIC_DEFAULTS = {
    "tqgb_cls_m2o_gain": 0.0,
    "tqgb_cls_o2m_gain": 0.0,
    "tqgb_direction": "none",
}
# Ultralytics final_eval() deterministically runs strip_optimizer(last.pt).
# strip_optimizer rebuilds train_args from DEFAULT_CFG_KEYS, which removes the
# runtime-only save_dir key and sets epoch=-1.  This is the sole non-semantic
# checkpoint-field recovery introduced in provenance revision 4 and retained
# unchanged in revision 5.  The recovered
# value is not taken from a mutable default: it is derived from the exact
# content-pinned .../<run>/weights/last.pt path and is later required to agree
# with project/name and the independently validated run args.yaml.
TERMINAL_STRIPPED_RECOVERABLE_FIELDS = frozenset({"save_dir"})
REQUIRED_TRAJECTORY_FIELDS = frozenset(TRAJECTORY_FIELDS) - frozenset(SEMANTIC_DEFAULTS)
PATH_MANIFEST_FIELDS = frozenset({"model", "data", "project", "save_dir"})

EXPECTED_B50_MODEL_STRUCTURE = {
    "model_class": "ultralytics.nn.tasks.DetectionModel",
    "head_class": "ultralytics.nn.modules.head.Detect",
    "nc": 20,
    "stride": [8.0, 16.0, 32.0],
    "parameter_count": 2_511_600,
    "names": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths or glob patterns.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML; only its train split is used.")
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    parser.add_argument("--batch", type=int, default=FROZEN_PROTOCOL["batch"])
    parser.add_argument("--imgsz", type=int, default=FROZEN_PROTOCOL["imgsz"])
    parser.add_argument("--workers", type=int, default=FROZEN_PROTOCOL["workers"])
    parser.add_argument(
        "--max-batches", type=int, default=FROZEN_PROTOCOL["max_batches"], help="Frozen formal panel is 4x64=256."
    )
    parser.add_argument("--seed", type=int, default=FROZEN_PROTOCOL["seed"])
    parser.add_argument("--strengths", type=float, nargs="+", default=FROZEN_PROTOCOL["strengths"])
    parser.add_argument("--max-scale-delta", type=float, default=FROZEN_PROTOCOL["max_scale_delta"])
    parser.add_argument("--max-translate", type=float, default=FROZEN_PROTOCOL["max_translate"])
    parser.add_argument("--min-visibility", type=float, default=FROZEN_PROTOCOL["min_visibility"])
    parser.add_argument("--min-box-pixels", type=float, default=FROZEN_PROTOCOL["min_box_pixels"])
    parser.add_argument("--grid-bins", type=int, default=FROZEN_PROTOCOL["grid_bins"])
    parser.add_argument("--grid-extent", type=float, default=FROZEN_PROTOCOL["grid_extent"])
    parser.add_argument("--kernel-sigma", type=float, default=FROZEN_PROTOCOL["kernel_sigma"])
    parser.add_argument("--expected-strides", type=float, nargs=3, default=FROZEN_PROTOCOL["expected_strides"])
    parser.add_argument("--expected-epochs", type=int, nargs="+", default=FROZEN_PROTOCOL["expected_epochs"])
    parser.add_argument("--min-object-pairs", type=int, default=FROZEN_PROTOCOL["min_object_pairs"])
    parser.add_argument(
        "--min-assignment-coverage", type=float, default=FROZEN_PROTOCOL["min_assignment_coverage"]
    )
    parser.add_argument("--monotonic-tolerance", type=float, default=FROZEN_PROTOCOL["monotonic_tolerance"])
    parser.add_argument("--min-weak-strong-drop", type=float, default=FROZEN_PROTOCOL["min_weak_strong_drop"])
    parser.add_argument(
        "--min-monotonic-checkpoints", type=int, default=FROZEN_PROTOCOL["min_monotonic_checkpoints"]
    )
    parser.add_argument("--min-spearman", type=float, default=FROZEN_PROTOCOL["min_spearman"])
    parser.add_argument(
        "--min-correlated-checkpoints", type=int, default=FROZEN_PROTOCOL["min_correlated_checkpoints"]
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="asac_assignment_stability_probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch <= 0 or args.max_batches <= 0:
        parser.error("batch and max-batches must be positive")
    if args.workers < 0:
        parser.error("workers must be non-negative")
    if len(args.strengths) != 3 or any(not 0 < value <= 1 for value in args.strengths):
        parser.error("exactly three strengths in (0,1] are required")
    if any(right <= left for left, right in zip(args.strengths, args.strengths[1:])):
        parser.error("strengths must be strictly increasing")
    if args.max_scale_delta <= 0 or args.max_scale_delta >= 1:
        parser.error("max-scale-delta must be in (0,1)")
    if not 0 <= args.max_translate < 0.5:
        parser.error("max-translate must be in [0,0.5)")
    if not 0 < args.min_visibility <= 1:
        parser.error("min-visibility must be in (0,1]")
    if args.grid_bins < 5 or args.grid_bins % 2 == 0:
        parser.error("grid-bins must be an odd integer >=5")
    if args.grid_extent <= 0 or args.kernel_sigma <= 0:
        parser.error("grid-extent and kernel-sigma must be positive")
    if args.min_object_pairs <= 0 or not 0 <= args.min_assignment_coverage <= 1:
        parser.error("invalid sample sufficiency thresholds")
    if args.min_monotonic_checkpoints <= 0 or args.min_correlated_checkpoints <= 0:
        parser.error("checkpoint-count thresholds must be positive")
    if any(right <= left for left, right in zip(args.expected_epochs, args.expected_epochs[1:])):
        parser.error("expected-epochs must be strictly increasing")
    return args


def canonical_json(value: Any) -> str:
    """Stable JSON for hashes/signatures; Path/tuple/numpy scalars are normalized."""

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, Path):
            return str(item.resolve())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            values_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def protocol_status(args: argparse.Namespace) -> dict[str, Any]:
    """Return every CLI deviation from the frozen v2 protocol."""
    deviations = {}
    for key, expected in FROZEN_PROTOCOL.items():
        if key == "panel_images":
            continue
        actual = getattr(args, key)
        if not values_equal(actual, expected):
            deviations[key] = {"expected": expected, "actual": actual}
    return {
        "schema": "ASAC-P0-v2",
        "preregistered": not deviations,
        "deviations": deviations,
        "frozen": FROZEN_PROTOCOL,
    }


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


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verified_checkpoint_sha256(
    path: str | Path, expected_hashes: dict[str, str] | None = None
) -> str:
    """Hash checkpoint bytes before any pickle deserialization and require the pinned digest."""
    checkpoint_path = Path(path)
    expected_hashes = EXPECTED_CHECKPOINT_SHA256 if expected_hashes is None else expected_hashes
    filename = checkpoint_path.name.lower()
    if filename not in expected_hashes:
        raise RuntimeError(f"checkpoint is not in the frozen SHA256 pin set: {checkpoint_path}")
    actual = file_sha256(str(checkpoint_path))
    expected = expected_hashes[filename]
    if actual != expected:
        raise RuntimeError(
            f"checkpoint content SHA256 does not match before deserialization: "
            f"path={checkpoint_path}, expected={expected}, actual={actual}"
        )
    return actual


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_manifest_value(key: str, value: Any, expected: Any, source: str) -> Any:
    """Normalize one B50 manifest value while retaining exact type/value semantics."""
    if key in PATH_MANIFEST_FIELDS:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise RuntimeError(f"{source} field {key!r} must be a non-empty absolute path, got {value!r}")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"{source} field {key!r} must be absolute, got {value!r}")
        return str(path.resolve())
    if expected is None:
        if value is not None:
            raise RuntimeError(f"{source} field {key!r} must be null, got {value!r}")
        return None
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            raise RuntimeError(f"{source} field {key!r} must be bool, got {type(value).__name__}")
        return value
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(f"{source} field {key!r} must be int, got {value!r}")
        return int(value)
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise RuntimeError(f"{source} field {key!r} must be numeric, got {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"{source} field {key!r} must be finite, got {value!r}")
        return numeric
    if isinstance(expected, str):
        if not isinstance(value, str):
            raise RuntimeError(f"{source} field {key!r} must be string, got {value!r}")
        return value
    if isinstance(expected, (list, tuple)):
        if not isinstance(value, (list, tuple)) or len(value) != len(expected):
            raise RuntimeError(f"{source} field {key!r} has invalid sequence value {value!r}")
        return [
            _normalize_manifest_value(f"{key}[{index}]", actual, wanted, source)
            for index, (actual, wanted) in enumerate(zip(value, expected))
        ]
    raise RuntimeError(f"unsupported frozen-manifest type for {key!r}: {type(expected).__name__}")


def normalize_b50_manifest(
    raw: dict[str, Any], source: str, expected_manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the complete typed/path-normalized B50 manifest or fail on missing evidence.

    The three TQGB keys were absent from the historical B50 artifacts and are
    normalized to their no-op semantics.  No other absent field is accepted.
    """
    if not isinstance(raw, dict):
        raise RuntimeError(f"{source} manifest must be a mapping")
    expected_manifest = FROZEN_B50_ARGS if expected_manifest is None else expected_manifest
    missing = sorted(set(expected_manifest) - set(raw) - set(SEMANTIC_DEFAULTS))
    if missing:
        raise RuntimeError(f"{source} lacks frozen B50 manifest fields {missing}")
    unexpected = sorted(set(raw) - set(expected_manifest))
    if unexpected:
        raise RuntimeError(f"{source} has fields outside the frozen B50 manifest {unexpected}")
    normalized: dict[str, Any] = {}
    for key, expected in expected_manifest.items():
        value = raw[key] if key in raw else SEMANTIC_DEFAULTS[key]
        normalized[key] = _normalize_manifest_value(key, value, expected, source)
    return normalized


def manifest_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compare normalized provenance manifests exactly, without metric tolerances."""
    return {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected.get(key)
    }


FROZEN_B50_MANIFEST_SHA256 = sha256_json(FROZEN_B50_ARGS)
FROZEN_B50_CHECKPOINT_MANIFEST_SHA256 = sha256_json(FROZEN_B50_CHECKPOINT_ARGS)


def train_results_epoch_count(train_results: Any) -> tuple[int, float | None]:
    """Return row count and final logged epoch from an embedded train_results dictionary."""
    if not isinstance(train_results, dict) or not train_results:
        return 0, None
    list_lengths = {len(value) for value in train_results.values() if isinstance(value, (list, tuple))}
    if len(list_lengths) != 1:
        return 0, None
    count = next(iter(list_lengths), 0)
    epoch_values = None
    for key, value in train_results.items():
        if str(key).strip().lower() == "epoch" and isinstance(value, (list, tuple)):
            epoch_values = value
            break
    if count <= 0 or epoch_values is None or len(epoch_values) != count:
        return count, None
    try:
        return count, float(epoch_values[-1])
    except (TypeError, ValueError):
        return count, None


def checkpoint_record_from_payload(path: str | Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Strictly identify one frozen B50 checkpoint; no filename-only epoch inference is allowed."""
    checkpoint_path = Path(path).resolve()
    filename = checkpoint_path.name.lower()
    if filename == "best.pt":
        raise RuntimeError("best.pt is forbidden: best epoch is not the frozen final epoch")
    if filename not in EXPECTED_CHECKPOINT_FILES:
        raise RuntimeError(
            f"ASAC-P0 requires exactly {sorted(EXPECTED_CHECKPOINT_FILES)}, got {checkpoint_path.name}"
        )
    expected_epoch = EXPECTED_CHECKPOINT_FILES[filename]
    internal_epoch = checkpoint.get("epoch")
    if not isinstance(internal_epoch, int):
        raise RuntimeError(f"checkpoint epoch metadata is missing/non-integer: {checkpoint_path}")
    if filename.startswith("epoch"):
        if internal_epoch < 0:
            raise RuntimeError(f"stripped numbered checkpoint is forbidden: {checkpoint_path}")
        if internal_epoch + 1 != expected_epoch:
            raise RuntimeError(
                f"filename/internal epoch mismatch for {checkpoint_path}: expected {expected_epoch}, "
                f"metadata gives {internal_epoch + 1}"
            )
    elif internal_epoch not in {-1, expected_epoch - 1}:
        raise RuntimeError(
            f"last.pt epoch metadata must be -1 (stripped) or {expected_epoch - 1}, got {internal_epoch}"
        )

    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise RuntimeError(f"checkpoint train_args metadata is missing: {checkpoint_path}")
    manifest_input = dict(train_args)
    required_missing = (
        set(FROZEN_B50_CHECKPOINT_ARGS) - set(manifest_input) - set(SEMANTIC_DEFAULTS)
    )
    terminal_recovery: dict[str, Any] | None = None
    if (
        filename == "last.pt"
        and internal_epoch == -1
        and required_missing == set(TERMINAL_STRIPPED_RECOVERABLE_FIELDS)
    ):
        derived_save_dir = str(checkpoint_path.parent.parent.resolve())
        manifest_input["save_dir"] = derived_save_dir
        terminal_recovery = {
            "provenance_revision": PROVENANCE_REVISION,
            "reason": "Ultralytics terminal strip_optimizer(last.pt)",
            "recovered_fields": ["save_dir"],
            "save_dir_source": "checkpoint_path.parent.parent",
            "derived_save_dir": derived_save_dir,
        }
    trajectory_args = normalize_b50_manifest(
        manifest_input,
        f"checkpoint {checkpoint_path} train_args",
        FROZEN_B50_CHECKPOINT_ARGS,
    )
    result_count, final_logged_epoch = train_results_epoch_count(checkpoint.get("train_results"))
    if result_count != expected_epoch or final_logged_epoch is None or not math.isclose(
        final_logged_epoch, float(expected_epoch), rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"checkpoint train_results do not prove completion through epoch {expected_epoch}: "
            f"rows={result_count}, final={final_logged_epoch}, path={checkpoint_path}"
        )
    if checkpoint.get("ema") is None and checkpoint.get("model") is None:
        raise RuntimeError(f"checkpoint has no model/EMA payload: {checkpoint_path}")

    return {
        "path": str(checkpoint_path),
        "filename": checkpoint_path.name,
        "epoch": expected_epoch,
        "internal_epoch": internal_epoch,
        "stripped": internal_epoch == -1,
        "train_results_rows": result_count,
        "train_results_final_epoch": final_logged_epoch,
        "trajectory_args": trajectory_args,
        "trajectory_args_sha256": sha256_json(trajectory_args),
        "terminal_stripped_recovery": terminal_recovery,
        "git": checkpoint.get("git") if isinstance(checkpoint.get("git"), dict) else {},
        "checkpoint_version": checkpoint.get("version"),
    }


def model_structure_record(model: Any) -> dict[str, Any]:
    head = model.model[-1] if hasattr(model, "model") and len(model.model) else None
    record = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "yaml": getattr(model, "yaml", None),
        "names": getattr(model, "names", None),
        "head_class": (
            f"{head.__class__.__module__}.{head.__class__.__qualname__}" if head is not None else None
        ),
        "nc": getattr(head, "nc", None),
        "stride": (
            [float(value) for value in getattr(head, "stride").detach().cpu().tolist()]
            if head is not None and hasattr(getattr(head, "stride", None), "detach")
            else None
        ),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
    }
    record["sha256"] = sha256_json(record)
    return record


def csv_run_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required run evidence is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise RuntimeError(f"results.csv must contain exactly 50 epochs, got {len(rows)}: {path}")
    epoch_key = next((key for key in rows[-1] if key.strip().lower() == "epoch"), None)
    if epoch_key is None or not math.isclose(float(rows[-1][epoch_key]), 50.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"results.csv does not end at epoch 50: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(str(path)), "rows": 50, "final_epoch": 50.0}


def load_yaml_mapping_strict(path: Path) -> dict[str, Any]:
    """Load one provenance YAML mapping while rejecting duplicate keys."""
    import yaml

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise RuntimeError(f"unhashable YAML mapping key in {path}: {key!r}") from error
            if duplicate:
                raise RuntimeError(f"duplicate YAML mapping key in {path}: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
    )
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise RuntimeError(f"invalid YAML provenance file {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise RuntimeError(f"run args.yaml is not a mapping: {path}")
    return loaded


def canonical_model_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            keyed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"model names have non-integer keys: {names!r}") from error
        if sorted(keyed) != list(range(len(keyed))):
            raise RuntimeError(f"model names are not contiguous from class 0: {sorted(keyed)}")
        return [keyed[index] for index in range(len(keyed))]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    raise RuntimeError(f"model names metadata is not a mapping/list: {names!r}")


def validate_b50_model_structure(
    structure: dict[str, Any], expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Require the checkpoint model itself to be the expected YOLO26n/VOC20 graph."""
    expected = EXPECTED_B50_MODEL_STRUCTURE if expected is None else expected
    if not isinstance(structure, dict):
        raise RuntimeError("checkpoint model structure record is missing")
    claimed_sha = structure.get("sha256")
    payload = {key: value for key, value in structure.items() if key != "sha256"}
    actual_sha = sha256_json(payload)
    if claimed_sha != actual_sha:
        raise RuntimeError(f"model structure record hash is invalid: expected {actual_sha}, got {claimed_sha}")
    names = canonical_model_names(structure.get("names"))
    mismatches: dict[str, dict[str, Any]] = {}
    for key in ("model_class", "head_class", "nc", "stride", "parameter_count"):
        if not values_equal(structure.get(key), expected[key]):
            mismatches[key] = {"expected": expected[key], "actual": structure.get(key)}
    if names != expected["names"]:
        mismatches["names"] = {"expected": expected["names"], "actual": names}
    yaml_metadata = structure.get("yaml")
    if not isinstance(yaml_metadata, dict) or not values_equal(yaml_metadata.get("nc"), expected["nc"]):
        mismatches["yaml.nc"] = {
            "expected": expected["nc"],
            "actual": yaml_metadata.get("nc") if isinstance(yaml_metadata, dict) else None,
        }
    if isinstance(yaml_metadata, dict) and yaml_metadata.get("scale") not in {None, "n"}:
        mismatches["yaml.scale"] = {"expected": "n or absent", "actual": yaml_metadata.get("scale")}
    if not isinstance(yaml_metadata, dict) or yaml_metadata.get("end2end") is not True:
        mismatches["yaml.end2end"] = {
            "expected": True,
            "actual": yaml_metadata.get("end2end") if isinstance(yaml_metadata, dict) else None,
        }
    if mismatches:
        raise RuntimeError(f"checkpoint is not the frozen YOLO26n/VOC20 model structure: {mismatches}")
    return {
        "sha256": actual_sha,
        "model_class": structure["model_class"],
        "head_class": structure["head_class"],
        "nc": int(structure["nc"]),
        "stride": [float(value) for value in structure["stride"]],
        "parameter_count": int(structure["parameter_count"]),
        "names": names,
        "yaml_nc": int(yaml_metadata["nc"]),
        "yaml_scale": yaml_metadata.get("scale"),
        "yaml_end2end": bool(yaml_metadata.get("end2end")),
    }


def validate_b50_trajectory(
    records: list[dict[str, Any]],
    data_path: str,
    frozen_manifest: dict[str, Any] | None = None,
    expected_model_structure: dict[str, Any] | None = None,
    expected_checkpoint_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Prove all five checkpoints are the exact frozen B50 run, not merely mutually consistent.

    The optional manifests are an internal CPU-test seam and are not exposed by
    the CLI.  Production calls always use the immutable real B50 constants.
    """
    frozen_raw = FROZEN_B50_ARGS if frozen_manifest is None else frozen_manifest
    expected_hashes = (
        EXPECTED_CHECKPOINT_SHA256 if expected_checkpoint_sha256 is None else expected_checkpoint_sha256
    )
    frozen = normalize_b50_manifest(frozen_raw, "frozen B50 launch manifest", frozen_raw)
    frozen_checkpoint_raw = checkpoint_runtime_manifest(frozen_raw)
    frozen_checkpoint = normalize_b50_manifest(
        frozen_checkpoint_raw,
        "frozen B50 checkpoint manifest",
        frozen_checkpoint_raw,
    )
    frozen_sha256 = sha256_json(frozen)
    frozen_checkpoint_sha256 = sha256_json(frozen_checkpoint)
    expected_epochs = FROZEN_PROTOCOL["expected_epochs"]
    ordered = []
    for input_row in sorted(records, key=lambda row: row["epoch"]):
        row = dict(input_row)
        trajectory_args = normalize_b50_manifest(
            row.get("trajectory_args"),
            f"checkpoint {row.get('path')} train_args",
            frozen_checkpoint_raw,
        )
        actual_args_sha = sha256_json(trajectory_args)
        if row.get("trajectory_args_sha256") != actual_args_sha:
            raise RuntimeError(
                f"checkpoint trajectory manifest hash is invalid for {row.get('path')}: "
                f"expected {actual_args_sha}, got {row.get('trajectory_args_sha256')}"
            )
        row["trajectory_args"] = trajectory_args
        row["trajectory_args_sha256"] = actual_args_sha
        row["validated_model_structure"] = validate_b50_model_structure(
            row.get("model_structure"), expected_model_structure
        )
        ordered.append(row)
    if [row["epoch"] for row in ordered] != expected_epochs:
        raise RuntimeError(f"strict checkpoint epochs must be {expected_epochs}, got {[row['epoch'] for row in ordered]}")
    actual_files = {Path(row["path"]).name.lower(): row["epoch"] for row in ordered}
    if actual_files != EXPECTED_CHECKPOINT_FILES:
        raise RuntimeError(f"strict checkpoint filenames/epochs must be {EXPECTED_CHECKPOINT_FILES}, got {actual_files}")
    if set(expected_hashes) != set(EXPECTED_CHECKPOINT_FILES):
        raise RuntimeError(f"checkpoint SHA256 pin set is incomplete: {sorted(expected_hashes)}")
    hash_mismatches = {
        Path(row["path"]).name.lower(): {
            "expected": expected_hashes[Path(row["path"]).name.lower()],
            "actual": row.get("checkpoint_sha256"),
        }
        for row in ordered
        if row.get("checkpoint_sha256") != expected_hashes[Path(row["path"]).name.lower()]
    }
    if hash_mismatches:
        raise RuntimeError(f"checkpoint content SHA256 does not match the pinned B50 artifacts: {hash_mismatches}")
    parents = {str(Path(row["path"]).parent.resolve()) for row in ordered}
    if len(parents) != 1 or Path(next(iter(parents))).name != "weights":
        raise RuntimeError(f"all checkpoints must share one weights/ directory, got {sorted(parents)}")
    signatures = {row["trajectory_args_sha256"] for row in ordered}
    if len(signatures) != 1:
        raise RuntimeError("checkpoint train_args differ; mixed trajectories are forbidden")
    git_signatures = {sha256_json(row["git"]) for row in ordered}
    if len(git_signatures) != 1:
        raise RuntimeError("checkpoint source/git metadata differ; mixed trajectories are forbidden")
    structure_signatures = {row["model_structure"]["sha256"] for row in ordered}
    if len(structure_signatures) != 1:
        raise RuntimeError("checkpoint model structures differ; mixed trajectories are forbidden")

    train_args = ordered[0]["trajectory_args"]
    for row in ordered:
        mismatches = manifest_mismatches(row["trajectory_args"], frozen_checkpoint)
        if mismatches:
            raise RuntimeError(
                f"checkpoint is not the frozen B50 manifest ({row['path']}): {mismatches}"
            )
    checkpoint_data = Path(str(train_args["data"])).resolve()
    if checkpoint_data != Path(data_path).resolve():
        raise RuntimeError(f"--data differs from B50 checkpoint metadata: {data_path} vs {checkpoint_data}")

    weights_dir = Path(next(iter(parents)))
    run_dir = weights_dir.parent
    expected_run_dir = (Path(str(train_args["project"])) / str(train_args["name"])).resolve()
    if run_dir.resolve() != expected_run_dir:
        raise RuntimeError(
            f"run directory must equal project/name: run={run_dir.resolve()}, expected={expected_run_dir}"
        )
    if Path(str(train_args["save_dir"])).resolve() != expected_run_dir:
        raise RuntimeError(
            f"save_dir must equal project/name: {train_args['save_dir']} vs {expected_run_dir}"
        )
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"required run args.yaml is missing: {args_path}")
    run_args = load_yaml_mapping_strict(args_path)
    normalized_run_args = normalize_b50_manifest(run_args, f"run args.yaml {args_path}", frozen_raw)
    frozen_run_mismatches = manifest_mismatches(normalized_run_args, frozen)
    if frozen_run_mismatches:
        raise RuntimeError(f"run args.yaml is not the frozen B50 manifest: {frozen_run_mismatches}")
    expected_runtime_from_run = checkpoint_runtime_manifest(normalized_run_args)
    run_mismatches = manifest_mismatches(expected_runtime_from_run, train_args)
    if run_mismatches:
        raise RuntimeError(
            "run args.yaml disagrees with checkpoint metadata after the frozen runtime transform: "
            f"{run_mismatches}"
        )
    config_evidence = {}
    for key in ("model", "data"):
        config_path = Path(str(train_args[key]))
        if not config_path.is_file():
            raise RuntimeError(f"frozen B50 {key} file is missing: {config_path}")
        config_sha = file_sha256(str(config_path))
        if frozen_manifest is None and config_sha != EXPECTED_CONFIG_SHA256[key]:
            raise RuntimeError(
                f"frozen B50 {key} config SHA256 changed: "
                f"expected={EXPECTED_CONFIG_SHA256[key]}, actual={config_sha}, path={config_path}"
            )
        config_evidence[key] = {
            "path": str(config_path.resolve()),
            "sha256": config_sha,
            "expected_sha256": EXPECTED_CONFIG_SHA256[key] if frozen_manifest is None else None,
            "content_pinned": frozen_manifest is None,
        }
    results_evidence = csv_run_evidence(run_dir / "results.csv")
    run_evidence = {
        "run_dir": str(run_dir),
        "weights_dir": str(weights_dir),
        "project_name_relation_verified": True,
        "config_files": config_evidence,
        "args_yaml": {
            "path": str(args_path.resolve()),
            "sha256": file_sha256(str(args_path)),
            "normalized_manifest_sha256": sha256_json(normalized_run_args),
            "checkpoint_runtime_manifest_sha256": sha256_json(expected_runtime_from_run),
            "runtime_transforms": RUNTIME_MANIFEST_TRANSFORMS,
        },
        "results_csv": results_evidence,
    }
    signature_payload = {
        "trajectory_args": train_args,
        "frozen_manifest_sha256": frozen_sha256,
        "frozen_checkpoint_manifest_sha256": frozen_checkpoint_sha256,
        "model_structure": ordered[0]["model_structure"],
        "git": ordered[0]["git"],
        "run_evidence": run_evidence,
    }
    return {
        "signature_sha256": sha256_json(signature_payload),
        "trajectory_args": train_args,
        "frozen_manifest_sha256": frozen_sha256,
        "frozen_checkpoint_manifest_sha256": frozen_checkpoint_sha256,
        "production_frozen_manifest_sha256": FROZEN_B50_MANIFEST_SHA256,
        "production_frozen_checkpoint_manifest_sha256": FROZEN_B50_CHECKPOINT_MANIFEST_SHA256,
        "runtime_manifest_transforms": RUNTIME_MANIFEST_TRANSFORMS,
        "model_structure": ordered[0]["model_structure"],
        "validated_model_structure": ordered[0]["validated_model_structure"],
        "git": ordered[0]["git"],
        "run_evidence": run_evidence,
    }


def deterministic_unit_triplet(path: str, seed: int) -> tuple[float, float, float]:
    """Return reproducible signed values in [-1,1], independent of Python hash randomization."""
    digest = hashlib.sha256(f"asac-v1|{seed}|{Path(path).as_posix()}".encode("utf-8")).digest()
    values = []
    for offset in (0, 8, 16):
        integer = int.from_bytes(digest[offset : offset + 8], byteorder="big", signed=False)
        values.append(2.0 * integer / float(2**64 - 1) - 1.0)
    return tuple(values)  # type: ignore[return-value]


def affine_parameters(
    paths: list[str], strength: float, seed: int, max_scale_delta: float, max_translate: float
) -> list[dict[str, float]]:
    """Nested perturbations: each image keeps the same signed direction as strength increases."""
    result = []
    for path in paths:
        scale_unit, tx_unit, ty_unit = deterministic_unit_triplet(path, seed)
        result.append(
            {
                "scale": 1.0 + strength * max_scale_delta * scale_unit,
                "tx": strength * max_translate * tx_unit,
                "ty": strength * max_translate * ty_unit,
            }
        )
    return result


def js_divergence(left: np.ndarray | list[float], right: np.ndarray | list[float]) -> float | None:
    """Base-2 Jensen-Shannon divergence, bounded by [0,1]."""
    p = np.asarray(left, dtype=np.float64).reshape(-1)
    q = np.asarray(right, dtype=np.float64).reshape(-1)
    if p.shape != q.shape:
        raise ValueError(f"distribution shape mismatch: {p.shape} vs {q.shape}")
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(term: np.ndarray) -> float:
        mask = term > 0
        return float(np.sum(term[mask] * np.log2(term[mask] / midpoint[mask])))

    return min(max(0.5 * (kl(p) + kl(q)), 0.0), 1.0)


def smooth_occupancy(
    points: list[tuple[int, float, float]], bins: int, extent: float, sigma: float
) -> np.ndarray | None:
    """Build an unweighted smooth occupancy over (P3/P4/P5, GT-relative dx,dy)."""
    if not points:
        return None
    centers = np.linspace(-extent, extent, bins, dtype=np.float64)
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    occupancy = np.zeros((len(LEVEL_NAMES), bins, bins), dtype=np.float64)
    for level, dx, dy in points:
        if not 0 <= level < len(LEVEL_NAMES):
            raise ValueError(f"invalid level id {level}")
        clipped_x = float(np.clip(dx, -extent, extent))
        clipped_y = float(np.clip(dy, -extent, extent))
        occupancy[level] += np.exp(-((xx - clipped_x) ** 2 + (yy - clipped_y) ** 2) / (2 * sigma**2))
    total = occupancy.sum()
    return occupancy.reshape(-1) / total if total > 0 else None


def rankdata(values: list[float]) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x, y = rankdata(left), rankdata(right)
    if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compare_object_records(
    clean: dict[str, Any] | None,
    view: dict[str, Any] | None,
    bins: int,
    extent: float,
    sigma: float,
) -> dict[str, Any]:
    """Compare one canonical GT across two views without using quality as occupancy weight."""
    clean_points = [] if clean is None else clean.get("points", [])
    view_points = [] if view is None else view.get("points", [])
    assigned_both = bool(clean_points and view_points)
    clean_quality = max(clean.get("qualities", [0.0])) if clean_points else 0.0
    view_quality = max(view.get("qualities", [0.0])) if view_points else 0.0
    clean_target_score = max(clean.get("target_scores", [0.0])) if clean_points else 0.0
    view_target_score = max(view.get("target_scores", [0.0])) if view_points else 0.0
    if assigned_both:
        clean_distribution = smooth_occupancy(clean_points, bins, extent, sigma)
        view_distribution = smooth_occupancy(view_points, bins, extent, sigma)
        js = js_divergence(clean_distribution, view_distribution)
        if js is None:
            raise RuntimeError("non-empty assignment yielded an empty occupancy distribution")
        stability = math.exp(-js)
        clean_centroid = np.asarray([(p[1], p[2]) for p in clean_points], dtype=np.float64).mean(0)
        view_centroid = np.asarray([(p[1], p[2]) for p in view_points], dtype=np.float64).mean(0)
        centroid_drift = float(np.linalg.norm(clean_centroid - view_centroid))
        clean_level = int(np.bincount([point[0] for point in clean_points], minlength=3).argmax())
        view_level = int(np.bincount([point[0] for point in view_points], minlength=3).argmax())
        level_flip = clean_level != view_level
    else:
        # Missing foreground is qualitatively worse than the maximum finite JS (whose exp(-JS) is still 0.368).
        js, stability, centroid_drift, level_flip = 1.0, 0.0, None, None
    quality_max = max(clean_quality, view_quality)
    quality_agreement = min(clean_quality, view_quality) / quality_max if quality_max > 0 else 0.0
    target_max = max(clean_target_score, view_target_score)
    target_score_agreement = min(clean_target_score, view_target_score) / target_max if target_max > 0 else 0.0
    return {
        "assigned_clean": bool(clean_points),
        "assigned_view": bool(view_points),
        "assigned_both": assigned_both,
        "clean_positive_count": len(clean_points),
        "view_positive_count": len(view_points),
        "js_base2": js,
        "stability": stability,
        "centroid_drift_gt_units": centroid_drift,
        "dominant_level_flip": level_flip,
        "clean_quality": clean_quality,
        "view_quality": view_quality,
        "quality_agreement": quality_agreement,
        "clean_target_score": clean_target_score,
        "view_target_score": view_target_score,
        "target_score_agreement": target_score_agreement,
    }


def summarize_object_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["epoch"]), str(row["head"]), float(row["strength"]))].append(row)
    summaries = []
    for (epoch, head, strength), group in sorted(groups.items()):
        matched = [row for row in group if row["assigned_both"]]
        stability = [float(row["stability"]) for row in group]
        matched_stability = [float(row["stability"]) for row in matched]
        matched_quality = [float(row["quality_agreement"]) for row in matched]
        level_flips = [bool(row["dominant_level_flip"]) for row in matched]
        summaries.append(
            {
                "epoch": epoch,
                "head": head,
                "strength": strength,
                "object_pairs": len(group),
                "assigned_both": len(matched),
                "assignment_coverage": len(matched) / len(group) if group else 0.0,
                "mean_stability": float(np.mean(stability)) if stability else None,
                "mean_stability_assigned": float(np.mean(matched_stability)) if matched_stability else None,
                "mean_js_assigned": (
                    float(np.mean([row["js_base2"] for row in matched])) if matched else None
                ),
                "mean_quality_agreement_assigned": float(np.mean(matched_quality)) if matched else None,
                "mean_clean_quality_assigned": (
                    float(np.mean([row["clean_quality"] for row in matched])) if matched else None
                ),
                "mean_view_quality_assigned": (
                    float(np.mean([row["view_quality"] for row in matched])) if matched else None
                ),
                "mean_target_score_agreement_assigned": (
                    float(np.mean([row["target_score_agreement"] for row in matched])) if matched else None
                ),
                "spearman_stability_quality_assigned": spearman(matched_stability, matched_quality),
                "dominant_level_flip_rate": float(np.mean(level_flips)) if level_flips else None,
                "mean_centroid_drift_gt_units": (
                    float(np.mean([row["centroid_drift_gt_units"] for row in matched])) if matched else None
                ),
            }
        )
    return summaries


def make_decision(
    checkpoint_results: list[dict[str, Any]], summaries: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    panel_hashes = sorted({row["panel_sha256"] for row in checkpoint_results})
    if len(panel_hashes) != 1:
        raise RuntimeError(f"Fixed panel mismatch across checkpoints: {panel_hashes}")
    geometry_hashes = sorted({row.get("geometry_sha256", "synthetic-test") for row in checkpoint_results})
    if len(geometry_hashes) != 1:
        raise RuntimeError(f"GT geometry panel mismatch across checkpoints: {geometry_hashes}")
    epochs = sorted(int(row["epoch"]) for row in checkpoint_results)
    expected_epochs = list(FROZEN_PROTOCOL["expected_epochs"])
    if epochs != expected_epochs or len(epochs) != len(set(epochs)):
        raise RuntimeError(f"Expected unique checkpoint epochs {expected_epochs}, got {epochs}")
    strengths = list(protocol["strengths"])
    lookup = {(row["epoch"], row["head"], row["strength"]): row for row in summaries}
    missing = [
        (epoch, head, strength)
        for epoch in epochs
        for head in HEADS
        for strength in strengths
        if (epoch, head, strength) not in lookup
    ]
    if missing:
        raise RuntimeError(f"Missing summary cells: {missing}")

    insufficiency = []
    for key, row in lookup.items():
        if row["object_pairs"] < protocol["min_object_pairs"]:
            insufficiency.append(f"{key}: objects={row['object_pairs']}")
        if row["assigned_both"] < protocol["min_object_pairs"]:
            insufficiency.append(f"{key}: matched={row['assigned_both']}")
        if row["assignment_coverage"] < protocol["min_assignment_coverage"]:
            insufficiency.append(f"{key}: coverage={row['assignment_coverage']:.3f}")

    head_evidence: dict[str, Any] = {}
    for head in HEADS:
        monotonic_epochs = []
        drops = []
        correlated_epochs = []
        for epoch in epochs:
            curve = [lookup[(epoch, head, strength)]["mean_stability"] for strength in strengths]
            if any(value is None for value in curve):
                continue
            monotonic = all(
                float(left) + protocol["monotonic_tolerance"] >= float(right)
                for left, right in zip(curve, curve[1:])
            )
            if monotonic:
                monotonic_epochs.append(epoch)
            drops.append(float(curve[0]) - float(curve[-1]))
            rho = lookup[(epoch, head, strengths[-1])]["spearman_stability_quality_assigned"]
            if rho is not None and rho >= protocol["min_spearman"]:
                correlated_epochs.append(epoch)
        head_evidence[head] = {
            "monotonic_epochs": monotonic_epochs,
            "monotonic_checkpoint_count": len(monotonic_epochs),
            "mean_weak_strong_drop": float(np.mean(drops)) if drops else None,
            "per_checkpoint_weak_strong_drop": drops,
            "correlated_epochs_at_strongest_view": correlated_epochs,
            "correlated_checkpoint_count": len(correlated_epochs),
        }

    preregistered = bool(protocol.get("preregistered", True))
    deviations = protocol.get("deviations", {})
    if not preregistered:
        verdict = "NON_PREREGISTERED"
        reason = "CLI parameters deviate from frozen ASAC-P0-v2; diagnostic metrics are not an automatic decision"
    elif insufficiency:
        verdict = "INCONCLUSIVE"
        reason = "insufficient paired or assigned objects for a stable automatic decision"
    elif any(
        head_evidence[head]["monotonic_checkpoint_count"] < protocol["min_monotonic_checkpoints"]
        for head in HEADS
    ):
        verdict = "NO-GO"
        reason = "O2M or O2O stability is not consistently intensity-sensitive across checkpoints"
    elif any(
        head_evidence[head]["mean_weak_strong_drop"] < protocol["min_weak_strong_drop"] for head in HEADS
    ):
        verdict = "NO-GO"
        reason = "weak-to-strong stability change is too small to support feedback control"
    elif any(
        head_evidence[head]["correlated_checkpoint_count"] < protocol["min_correlated_checkpoints"]
        for head in HEADS
    ):
        verdict = "NO-GO"
        reason = "object-level stability lacks repeatable positive association with supervision-quality retention"
    else:
        verdict = "GO"
        reason = "both native heads show intensity-sensitive and quality-relevant affine assignment stability"

    warnings = []
    if verdict == "GO":
        warnings.append("GO authorizes only a scale/translate ASAC prototype; Mosaic correspondence was not tested.")
    weakest_coverage = min(float(row["assignment_coverage"]) for row in summaries)
    if weakest_coverage < 0.80:
        warnings.append("Some cells retain fewer than 80% paired assignments; controller design must log coverage.")
    return {
        "verdict": verdict,
        "reason": reason,
        "panel_sha256": panel_hashes[0],
        "geometry_sha256": geometry_hashes[0],
        "epochs": epochs,
        "strengths": strengths,
        "head_evidence": head_evidence,
        "sample_insufficiency": insufficiency,
        "weakest_assignment_coverage": weakest_coverage,
        "thresholds": protocol,
        "preregistered": preregistered,
        "protocol_deviations": deviations,
        "warnings": warnings,
    }


# Runtime-only helpers below rely on torch/Ultralytics globals installed by initialize_runtime().
torch = None
F = None
DetectionTrainer = None
build_dataloader = None
build_yolo_dataset = None
make_anchors = None
ultralytics = None


def initialize_runtime(physical_device: int) -> None:
    global torch, F, DetectionTrainer, build_dataloader, build_yolo_dataset, make_anchors, ultralytics
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch as torch_module
    import torch.nn.functional as functional
    import ultralytics as ultralytics_module
    from ultralytics.data.build import build_dataloader as build_loader
    from ultralytics.data.build import build_yolo_dataset as build_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer as Trainer
    from ultralytics.utils.tal import make_anchors as anchors

    torch = torch_module
    F = functional
    DetectionTrainer = Trainer
    build_dataloader = build_loader
    build_yolo_dataset = build_dataset
    make_anchors = anchors
    ultralytics = ultralytics_module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_checkpoints(checkpoints: list[str], data_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for path in checkpoints:
        verified_sha = verified_checkpoint_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint payload is not a dictionary: {path}")
        record = checkpoint_record_from_payload(path, payload)
        model = payload.get("ema") if payload.get("ema") is not None else payload.get("model")
        record["model_structure"] = model_structure_record(model)
        post_load_sha = verified_checkpoint_sha256(path)
        if post_load_sha != verified_sha:
            raise RuntimeError(f"checkpoint bytes changed during inspection: {path}")
        record["checkpoint_sha256"] = verified_sha
        records.append(record)
        del model, payload
    trajectory = validate_b50_trajectory(records, data_path)
    return sorted(records, key=lambda row: row["epoch"]), trajectory


def class_path(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def tensor_or_sequence_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def validate_criterion_contract(trainer, expected_strides: list[float]) -> dict[str, Any]:
    """C1: prove the live loss is current YOLO26 E2ELoss with exact native assigners."""
    criterion = getattr(trainer.model, "criterion", None)
    if criterion is None or class_path(criterion) != "ultralytics.utils.loss.E2ELoss":
        raise RuntimeError(f"expected ultralytics.utils.loss.E2ELoss, got {class_path(criterion) if criterion else None}")
    expected = {"one2many": (10, 10), "one2one": (7, 1)}
    branches = {}
    for branch_name, (topk, topk2) in expected.items():
        branch = getattr(criterion, branch_name, None)
        assigner = getattr(branch, "assigner", None)
        if branch is None or assigner is None:
            raise RuntimeError(f"criterion lacks {branch_name}.assigner")
        actual = (getattr(assigner, "topk", None), getattr(assigner, "topk2", None))
        strides = tensor_or_sequence_floats(getattr(branch, "stride", []))
        if actual != (topk, topk2):
            raise RuntimeError(f"{branch_name} assigner must be topk/topk2={(topk, topk2)}, got {actual}")
        if not values_equal(strides, expected_strides):
            raise RuntimeError(f"{branch_name} strides must be {expected_strides}, got {strides}")
        if not callable(getattr(branch, "get_assigned_targets_and_loss", None)):
            raise RuntimeError(f"{branch_name} is not the native assignment/loss path")
        branches[branch_name] = {
            "loss_class": class_path(branch),
            "assigner_class": class_path(assigner),
            "topk": topk,
            "topk2": topk2,
            "stride": strides,
            "assignment_method": "get_assigned_targets_and_loss",
        }
    head = trainer.model.model[-1]
    dual_head_modules_present = all(
        getattr(head, name, None) is not None for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
    )
    if class_path(head) != "ultralytics.nn.modules.head.Detect" or not dual_head_modules_present:
        raise RuntimeError("model head lacks the native O2M/O2O YOLO26 module pair")
    return {
        "criterion_class": class_path(criterion),
        "branches": branches,
        "head_class": class_path(head),
        "head_runtime_end2end": bool(head.end2end),
        "dual_head_modules_present": dual_head_modules_present,
        "one2one_head_present": True,
    }


def named_tensor_sha256(named_tensors) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() or value.is_complex():
            value = value.float()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        count += 1
    return digest.hexdigest(), count


def integrity_snapshot(trainer) -> dict[str, Any]:
    model = trainer.model
    parameter_hash, parameter_tensors = named_tensor_sha256(model.named_parameters())
    bn_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    bn_buffers = []
    for name, module in bn_modules:
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                bn_buffers.append((f"{name}.{buffer_name}", buffer))
    bn_hash, bn_buffer_tensors = named_tensor_sha256(bn_buffers)
    return {
        "parameter_sha256": parameter_hash,
        "parameter_tensors": parameter_tensors,
        "bn_buffer_sha256": bn_hash,
        "bn_buffer_tensors": bn_buffer_tensors,
        "bn_total": len(bn_modules),
        "bn_eval": sum(not module.training for _, module in bn_modules),
        "bn_all_eval": all(not module.training for _, module in bn_modules),
        "model_training": bool(model.training),
        "optimizer_created": getattr(trainer, "optimizer", None) is not None,
        "grad_enabled_at_snapshot": bool(torch.is_grad_enabled()),
    }


def assert_integrity_unchanged(before: dict[str, Any], after: dict[str, Any], checkpoint: str) -> None:
    for key in ("parameter_sha256", "bn_buffer_sha256", "parameter_tensors", "bn_buffer_tensors", "bn_total"):
        if before[key] != after[key]:
            raise RuntimeError(f"fixed-checkpoint state changed ({key}) while probing {checkpoint}")
    if not before["bn_all_eval"] or not after["bn_all_eval"]:
        raise RuntimeError(f"all BatchNorm modules must remain in eval mode while probing {checkpoint}")
    if before["optimizer_created"] or after["optimizer_created"]:
        raise RuntimeError(f"zero-update probe unexpectedly created an optimizer for {checkpoint}")


def source_runtime_metadata(trainer, criterion_contract: dict[str, Any]) -> dict[str, Any]:
    import inspect

    head = trainer.model.model[-1]
    sources = {
        "probe": Path(__file__).resolve(),
        "loss": Path(inspect.getsourcefile(trainer.model.criterion.__class__)).resolve(),
        "tal": Path(inspect.getsourcefile(trainer.model.criterion.one2many.assigner.__class__)).resolve(),
        "data_build": Path(inspect.getsourcefile(build_yolo_dataset)).resolve(),
        "detection_trainer": Path(inspect.getsourcefile(DetectionTrainer)).resolve(),
        "detect_head": Path(inspect.getsourcefile(head.__class__)).resolve(),
    }
    source_hashes = {
        name: {"path": str(path), "sha256": file_sha256(str(path))} for name, path in sources.items()
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": getattr(ultralytics, "__version__", None),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "autocast_device": "cuda",
        "autocast_dtype": "torch.bfloat16",
        "criterion_contract": criterion_contract,
        "sources": source_hashes,
        "source_bundle_sha256": sha256_json(source_hashes),
    }


def make_trainer(checkpoint: str, args: argparse.Namespace):
    verified_before = verified_checkpoint_sha256(checkpoint)
    overrides = {
        "model": checkpoint,
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "epochs": 50,
        "device": "0",
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "seed": args.seed,
        "deterministic": True,
        "rect": False,
        "multi_scale": 0.0,
        "fraction": 1.0,
        "cache": False,
        "o2o_kd_gain": 0.0,
        "tqgb_cls_m2o_gain": 0.0,
        "tqgb_cls_o2m_gain": 0.0,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    verified_after = verified_checkpoint_sha256(checkpoint)
    if verified_after != verified_before:
        raise RuntimeError(f"checkpoint bytes changed while trainer loaded the model: {checkpoint}")
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.model.train()  # raw O2M/O2O dictionaries are required
    for module in trainer.model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    trainer.model.criterion = trainer.model.init_criterion()
    actual_args = {
        "rect": bool(trainer.args.rect),
        "multi_scale": float(trainer.args.multi_scale),
        "fraction": float(trainer.args.fraction),
        "cache": trainer.args.cache,
    }
    expected_args = {"rect": False, "multi_scale": 0.0, "fraction": 1.0, "cache": False}
    if any(not values_equal(actual_args[key], expected) for key, expected in expected_args.items()):
        raise RuntimeError(f"trainer violated fixed loader args: actual={actual_args}, expected={expected_args}")
    return trainer


def make_fixed_loader(trainer, args: argparse.Namespace):
    dataset = build_yolo_dataset(
        trainer.args,
        trainer.data["train"],
        args.batch,
        trainer.data,
        mode="val",
        rect=False,
        stride=trainer.stride,
    )
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=FROZEN_PROTOCOL["workers"],
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=True,
        device=trainer.device,
    )
    if bool(getattr(dataset, "rect", True)):
        raise RuntimeError("fixed ASAC panel requires dataset.rect=False")
    if bool(getattr(dataset, "augment", True)):
        raise RuntimeError("fixed ASAC panel requires dataset.augment=False")
    if len(dataset) < FROZEN_PROTOCOL["panel_images"]:
        raise RuntimeError(
            f"dataset has only {len(dataset)} images; fixed panel requires {FROZEN_PROTOCOL['panel_images']}"
        )
    if int(getattr(loader, "num_workers", -1)) != FROZEN_PROTOCOL["workers"]:
        raise RuntimeError(f"ASAC protocol requires actual workers=4, got {getattr(loader, 'num_workers', None)}")
    if not bool(getattr(loader, "pin_memory", False)):
        raise RuntimeError("ASAC protocol requires pinned memory, but the loader disabled it")
    sampler_classes = []
    for candidate in (
        getattr(loader, "sampler", None),
        getattr(getattr(loader, "batch_sampler", None), "sampler", None),
    ):
        if candidate is not None:
            sampler_classes.append(class_path(candidate))
    if any(name.endswith("RandomSampler") for name in sampler_classes) or not any(
        name.endswith("SequentialSampler") for name in sampler_classes
    ):
        raise RuntimeError(f"fixed panel requires a sequential sampler, got {sampler_classes}")
    metadata = {
        "dataset_length": len(dataset),
        "dataset_rect": bool(dataset.rect),
        "dataset_augment": bool(dataset.augment),
        "loader_workers": int(loader.num_workers),
        "loader_pin_memory": bool(loader.pin_memory),
        "sampler_classes": sampler_classes,
        "batch_size": int(args.batch),
        "shuffle": False,
        "fraction": float(trainer.args.fraction),
        "multi_scale": float(trainer.args.multi_scale),
        "cache": trainer.args.cache,
    }
    return loader, metadata


def update_panel_digest(digest: Any, batch: dict[str, Any], args: argparse.Namespace) -> None:
    paths = [str(path) for path in batch.get("im_file", [])]
    digest.update(json.dumps(paths, ensure_ascii=False).encode("utf-8"))
    for key in ("batch_idx", "cls", "bboxes"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    protocol = {
        "seed": args.seed,
        "strengths": list(args.strengths),
        "max_scale_delta": args.max_scale_delta,
        "max_translate": args.max_translate,
        "min_visibility": args.min_visibility,
        "min_box_pixels": args.min_box_pixels,
    }
    digest.update(json.dumps(protocol, sort_keys=True).encode("ascii"))


def base_local_maps(batch: dict[str, Any], batch_size: int) -> list[list[int]]:
    maps = []
    batch_idx = batch["batch_idx"].long()
    for image_index in range(batch_size):
        count = int((batch_idx == image_index).sum().item())
        maps.append(list(range(count)))
    return maps


def canonical_boxes(batch: dict[str, Any], batch_size: int) -> dict[tuple[int, int], list[float]]:
    result = {}
    batch_idx = batch["batch_idx"].long()
    for image_index in range(batch_size):
        rows = batch["bboxes"][batch_idx == image_index]
        for original_index, box in enumerate(rows.detach().float().cpu().tolist()):
            result[(image_index, original_index)] = box
    return result


def geometry_panel_rows(
    canonical: dict[tuple[int, int], list[float]],
    view_maps: dict[float, list[list[int]]],
    strengths: list[float],
    paths: list[str],
    image_offset: int,
) -> list[dict[str, Any]]:
    retained = {
        strength: {
            (image_index, original_index)
            for image_index, mapping in enumerate(view_maps[strength])
            for original_index in mapping
        }
        for strength in strengths
    }
    rows = []
    for (image_index, original_index), (cx, cy, width, height) in sorted(canonical.items()):
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        flags = {f"retained_q{strength:.2f}": (image_index, original_index) in retained[strength] for strength in strengths}
        rows.append(
            {
                "panel_image_index": image_offset + image_index,
                "image_path": paths[image_index],
                "original_gt_index": original_index,
                "width_fraction": width,
                "height_fraction": height,
                "area_fraction": width * height,
                "sqrt_area_fraction": math.sqrt(max(width * height, 0.0)),
                "box_boundary_margin": min(x1, y1, 1.0 - x2, 1.0 - y2),
                "center_edge_distance": min(cx, cy, 1.0 - cx, 1.0 - cy),
                **flags,
                "retained_all": all(flags.values()),
            }
        )
    return rows


def quantile_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def summarize_geometry(rows: list[dict[str, Any]], strengths: list[float]) -> dict[str, Any]:
    total = len(rows)
    retained_all = [row for row in rows if row["retained_all"]]
    dropped = [row for row in rows if not row["retained_all"]]

    def distributions(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "area_fraction": quantile_summary([float(row["area_fraction"]) for row in group]),
            "sqrt_area_fraction": quantile_summary([float(row["sqrt_area_fraction"]) for row in group]),
            "box_boundary_margin": quantile_summary([float(row["box_boundary_margin"]) for row in group]),
            "center_edge_distance": quantile_summary([float(row["center_edge_distance"]) for row in group]),
        }

    per_strength = {}
    for strength in strengths:
        key = f"retained_q{strength:.2f}"
        count = sum(bool(row[key]) for row in rows)
        per_strength[f"q{strength:.2f}"] = {"count": count, "rate": count / total if total else 0.0}
    return {
        "clean_gt_total": total,
        "per_strength": per_strength,
        "intersection_count": len(retained_all),
        "intersection_rate": len(retained_all) / total if total else 0.0,
        "retained_distributions": distributions(retained_all),
        "dropped_distributions": distributions(dropped),
    }


def transform_batch(
    batch: dict[str, Any], parameters: list[dict[str, float]], args: argparse.Namespace
) -> tuple[dict[str, Any], list[list[int]]]:
    images = batch["img"]
    batch_size, _, height, width = images.shape
    if len(parameters) != batch_size:
        raise RuntimeError(f"parameter/image mismatch: {len(parameters)} vs {batch_size}")
    theta = images.new_zeros((batch_size, 2, 3), dtype=torch.float32)
    for image_index, parameter in enumerate(parameters):
        scale, tx, ty = parameter["scale"], parameter["tx"], parameter["ty"]
        theta[image_index, 0, 0] = 1.0 / scale
        theta[image_index, 1, 1] = 1.0 / scale
        theta[image_index, 0, 2] = -2.0 * tx / scale
        theta[image_index, 1, 2] = -2.0 * ty / scale
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    fill = 114.0 / 255.0
    transformed_images = F.grid_sample(images.float() - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    transformed_images = (transformed_images + fill).clamp_(0.0, 1.0)

    original_batch_idx = batch["batch_idx"].long()
    output_batch_idx, output_cls, output_boxes = [], [], []
    local_maps: list[list[int]] = []
    for image_index, parameter in enumerate(parameters):
        selected = torch.where(original_batch_idx == image_index)[0]
        local_maps.append([])
        if selected.numel() == 0:
            continue
        boxes = batch["bboxes"][selected].float()
        classes = batch["cls"][selected]
        xyxy = torch.stack(
            (
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
            ),
            dim=1,
        )
        scale, tx, ty = parameter["scale"], parameter["tx"], parameter["ty"]
        transformed = xyxy.clone()
        transformed[:, (0, 2)] = 0.5 + scale * (xyxy[:, (0, 2)] - 0.5) + tx
        transformed[:, (1, 3)] = 0.5 + scale * (xyxy[:, (1, 3)] - 0.5) + ty
        raw_wh = (transformed[:, 2:4] - transformed[:, 0:2]).clamp_min(0)
        raw_area = raw_wh.prod(1)
        clipped = transformed.clamp(0.0, 1.0)
        clipped_wh = (clipped[:, 2:4] - clipped[:, 0:2]).clamp_min(0)
        clipped_area = clipped_wh.prod(1)
        visibility = clipped_area / raw_area.clamp_min(1e-12)
        valid = (
            (visibility >= args.min_visibility)
            & (clipped_wh[:, 0] * width >= args.min_box_pixels)
            & (clipped_wh[:, 1] * height >= args.min_box_pixels)
        )
        valid_indices = torch.where(valid)[0]
        if valid_indices.numel() == 0:
            continue
        kept = clipped[valid_indices]
        kept_xywh = torch.stack(
            (
                (kept[:, 0] + kept[:, 2]) / 2,
                (kept[:, 1] + kept[:, 3]) / 2,
                kept[:, 2] - kept[:, 0],
                kept[:, 3] - kept[:, 1],
            ),
            dim=1,
        )
        # Preserve the label tensor's device/dtype contract so the criterion can concatenate batch_idx, cls, and
        # bboxes without an accidental CPU/CUDA mix (this also keeps the helper portable across trainer versions).
        output_batch_idx.append(
            torch.full(
                (len(valid_indices),),
                image_index,
                device=batch["batch_idx"].device,
                dtype=batch["batch_idx"].dtype,
            )
        )
        output_cls.append(classes[valid_indices])
        output_boxes.append(kept_xywh.to(batch["bboxes"].dtype))
        local_maps[-1].extend(int(value) for value in valid_indices.detach().cpu().tolist())

    def concatenate(parts: list[Any], template, shape_tail: tuple[int, ...]):
        if parts:
            return torch.cat(parts, dim=0)
        return template.new_empty((0, *shape_tail))

    transformed_batch = {
        "img": transformed_images,
        "batch_idx": concatenate(output_batch_idx, batch["batch_idx"], ()),
        "cls": concatenate(output_cls, batch["cls"], tuple(batch["cls"].shape[1:])),
        "bboxes": concatenate(output_boxes, batch["bboxes"], (4,)),
    }
    return transformed_batch, local_maps


def anchor_geometry(prediction: dict[str, Any], strides, args: argparse.Namespace):
    feats = prediction["feats"]
    if len(feats) != 3:
        raise RuntimeError(f"expected three detection levels, got {len(feats)}")
    actual = strides.detach().float().cpu()
    expected = torch.tensor(args.expected_strides, dtype=torch.float32)
    if actual.numel() != 3 or not torch.allclose(actual, expected, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"expected strides {expected.tolist()}, got {actual.tolist()}")
    anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
    lengths = [int(feature.shape[-2] * feature.shape[-1]) for feature in feats]
    level_ids = torch.cat(
        [torch.full((length,), level, dtype=torch.long, device=feats[0].device) for level, length in enumerate(lengths)]
    )
    if len(level_ids) != len(anchor_points):
        raise RuntimeError("feature-level block lengths disagree with generated anchors")
    expected_stride = strides.to(stride_tensor)[level_ids]
    if not torch.equal(expected_stride, stride_tensor[:, 0]):
        raise RuntimeError("P3/P4/P5 level mapping disagrees with make_anchors")
    return anchor_points * stride_tensor, level_ids


def aligned_iou(left, right):
    intersection_lt = torch.maximum(left[..., :2], right[..., :2])
    intersection_rb = torch.minimum(left[..., 2:], right[..., 2:])
    intersection = (intersection_rb - intersection_lt).clamp_min(0).prod(-1)
    left_area = (left[..., 2:] - left[..., :2]).clamp_min(0).prod(-1)
    right_area = (right[..., 2:] - right[..., :2]).clamp_min(0).prod(-1)
    return intersection / (left_area + right_area - intersection).clamp_min(1e-9)


def extract_object_records(
    assignment: dict[str, Any],
    prediction: dict[str, Any],
    local_maps: list[list[int]],
    canonical: dict[tuple[int, int], list[float]],
    inverse_parameters: list[dict[str, float]],
    image_shape: tuple[int, int],
    strides,
    args: argparse.Namespace,
) -> dict[tuple[int, int], dict[str, Any]]:
    height, width = image_shape
    fg_mask = assignment["fg_mask"].bool()
    target_gt_idx = assignment["target_gt_idx"].long()
    target_labels = assignment["target_labels"].long()
    if target_labels.ndim == 3:
        target_labels = target_labels.squeeze(-1)
    logits = prediction["scores"].permute(0, 2, 1).float()
    class_index = target_labels.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
    class_probability = logits.gather(-1, class_index).squeeze(-1).sigmoid()
    target_score = assignment["target_scores"].float().gather(-1, class_index).squeeze(-1)
    iou = aligned_iou(assignment["pred_bboxes"].float(), assignment["target_bboxes"].float())
    quality = (class_probability * iou.clamp(0, 1)).clamp_min(0).sqrt()
    anchor_pixels, level_ids = anchor_geometry(prediction, strides, args)
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for image_index in range(fg_mask.shape[0]):
        positive_anchors = torch.where(fg_mask[image_index])[0]
        parameter = inverse_parameters[image_index]
        scale, tx, ty = parameter["scale"], parameter["tx"], parameter["ty"]
        for anchor_index in positive_anchors.detach().cpu().tolist():
            local_gt = int(target_gt_idx[image_index, anchor_index].item())
            if local_gt >= len(local_maps[image_index]):
                raise RuntimeError(f"assigner local GT index {local_gt} exceeds map for image {image_index}")
            original_gt = local_maps[image_index][local_gt]
            key = (image_index, original_gt)
            box = canonical[key]
            center_x, center_y, box_w, box_h = box
            anchor_x = float(anchor_pixels[anchor_index, 0].item()) / width
            anchor_y = float(anchor_pixels[anchor_index, 1].item()) / height
            original_x = (anchor_x - 0.5 - tx) / scale + 0.5
            original_y = (anchor_y - 0.5 - ty) / scale + 0.5
            dx = (original_x - center_x) / max(box_w, 1e-9)
            dy = (original_y - center_y) / max(box_h, 1e-9)
            record = groups.setdefault(key, {"points": [], "qualities": [], "target_scores": []})
            record["points"].append((int(level_ids[anchor_index].item()), dx, dy))
            record["qualities"].append(float(quality[image_index, anchor_index].item()))
            record["target_scores"].append(float(target_score[image_index, anchor_index].item()))
    return groups


def forward_assignments(trainer, batch: dict[str, Any]):
    criterion = trainer.model.criterion
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("ASAC-P0 forward must execute under torch.inference_mode()")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise RuntimeError("ASAC-P0 requires active CUDA BF16 autocast")
        parsed = criterion.one2many.parse_output(trainer.model.predict(batch["img"]))
        one2many, one2one = parsed["one2many"], parsed["one2one"]
        assigned_o2m, _, _ = criterion.one2many.get_assigned_targets_and_loss(one2many, batch, return_details=True)
        assigned_o2o, _, _ = criterion.one2one.get_assigned_targets_and_loss(one2one, batch, return_details=True)
        evidence = {
            "inference_mode": bool(torch.is_inference_mode_enabled()),
            "autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "input_dtype": str(batch["img"].dtype),
            "o2m_boxes_dtype": str(one2many["boxes"].dtype),
            "o2m_scores_dtype": str(one2many["scores"].dtype),
            "o2o_boxes_dtype": str(one2one["boxes"].dtype),
            "o2o_scores_dtype": str(one2one["scores"].dtype),
        }
    return {
        "O2M": (assigned_o2m, one2many, criterion.one2many.stride),
        "O2O": (assigned_o2o, one2one, criterion.one2one.stride),
    }, evidence


def probe_checkpoint(
    checkpoint_record: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = checkpoint_record["path"]
    epoch = int(checkpoint_record["epoch"])
    seed_everything(args.seed)
    trainer = make_trainer(checkpoint, args)
    criterion_contract = validate_criterion_contract(trainer, FROZEN_PROTOCOL["expected_strides"])
    loader, loader_metadata = make_fixed_loader(trainer, args)
    before_integrity = integrity_snapshot(trainer)
    if not before_integrity["model_training"] or not before_integrity["bn_all_eval"]:
        raise RuntimeError("model must expose training-format outputs while every BatchNorm remains in eval mode")
    if before_integrity["optimizer_created"]:
        raise RuntimeError("zero-update ASAC-P0 must not create an optimizer")
    runtime_metadata = source_runtime_metadata(trainer, criterion_contract)
    panel_digest = hashlib.sha256()
    object_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []
    autocast_evidence: dict[str, Any] | None = None
    image_offset = images = batches = 0

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            update_panel_digest(panel_digest, raw_batch, args)
            paths = [str(path) for path in raw_batch["im_file"]]
            panel_paths.extend(paths)
            batch = trainer.preprocess_batch(raw_batch)
            batch_size, _, height, width = batch["img"].shape
            base_maps = base_local_maps(batch, batch_size)
            canonical = canonical_boxes(batch, batch_size)
            identity = [{"scale": 1.0, "tx": 0.0, "ty": 0.0} for _ in range(batch_size)]

            transformed: dict[float, tuple[dict[str, Any], list[list[int]], list[dict[str, float]]]] = {}
            view_maps_by_strength: dict[float, list[list[int]]] = {}
            common_ids = {key for key in canonical}
            for strength in args.strengths:
                parameters = affine_parameters(paths, strength, args.seed, args.max_scale_delta, args.max_translate)
                view_batch, view_maps = transform_batch(batch, parameters, args)
                transformed[strength] = (view_batch, view_maps, parameters)
                view_maps_by_strength[strength] = view_maps
                retained = {
                    (image_index, original_index)
                    for image_index, mapping in enumerate(view_maps)
                    for original_index in mapping
                }
                common_ids &= retained

            geometry_rows.extend(
                geometry_panel_rows(canonical, view_maps_by_strength, list(args.strengths), paths, image_offset)
            )

            clean_outputs, clean_evidence = forward_assignments(trainer, batch)
            if autocast_evidence is None:
                autocast_evidence = clean_evidence
            elif clean_evidence != autocast_evidence:
                raise RuntimeError(f"autocast/runtime dtype evidence changed within {checkpoint}")
            clean_records = {}
            for head, (assignment, prediction, strides) in clean_outputs.items():
                clean_records[head] = extract_object_records(
                    assignment, prediction, base_maps, canonical, identity, (height, width), strides, args
                )

            for strength in args.strengths:
                view_batch, view_maps, parameters = transformed[strength]
                view_outputs, view_evidence = forward_assignments(trainer, view_batch)
                if view_evidence != autocast_evidence:
                    raise RuntimeError(f"autocast/runtime dtype evidence changed across views in {checkpoint}")
                for head, (assignment, prediction, strides) in view_outputs.items():
                    view_records = extract_object_records(
                        assignment, prediction, view_maps, canonical, parameters, (height, width), strides, args
                    )
                    for image_index, original_index in sorted(common_ids):
                        comparison = compare_object_records(
                            clean_records[head].get((image_index, original_index)),
                            view_records.get((image_index, original_index)),
                            args.grid_bins,
                            args.grid_extent,
                            args.kernel_sigma,
                        )
                        object_rows.append(
                            {
                                "checkpoint": checkpoint,
                                "epoch": epoch,
                                "head": head,
                                "strength": strength,
                                "panel_image_index": image_offset + image_index,
                                "image_path": paths[image_index],
                                "original_gt_index": original_index,
                                **comparison,
                            }
                        )
            images += batch_size
            image_offset += batch_size
            batches += 1

    if batches != args.max_batches:
        raise RuntimeError(f"requested {args.max_batches} batches but loader yielded only {batches}")
    expected_images = FROZEN_PROTOCOL["panel_images"]
    if args.batch * args.max_batches != expected_images or images != expected_images or len(panel_paths) != expected_images:
        raise RuntimeError(
            f"formal panel must be exactly {expected_images} images and batch*max_batches must match; "
            f"batch={args.batch}, max_batches={args.max_batches}, observed={images}, paths={len(panel_paths)}"
        )
    if len(set(panel_paths)) != expected_images:
        raise RuntimeError(f"fixed panel must contain {expected_images} unique image paths")
    after_integrity = integrity_snapshot(trainer)
    assert_integrity_unchanged(before_integrity, after_integrity, checkpoint)
    geometry_summary = summarize_geometry(geometry_rows, list(args.strengths))
    geometry_sha256 = sha256_json(geometry_rows)
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "epoch": epoch,
        "images": images,
        "batches": batches,
        "unique_image_paths": len(set(panel_paths)),
        "panel_sha256": panel_digest.hexdigest(),
        "geometry_sha256": geometry_sha256,
        "geometry": geometry_summary,
        "object_rows": len(object_rows),
        "loader_actual": loader_metadata,
        "criterion_actual": criterion_contract,
        "autocast_actual": autocast_evidence,
        "integrity_before": before_integrity,
        "integrity_after": after_integrity,
        "runtime": runtime_metadata,
    }
    if hasattr(loader, "close"):
        loader.close()
    del trainer, loader
    torch.cuda.empty_cache()
    return result, object_rows, geometry_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def write_outputs(
    args: argparse.Namespace,
    checkpoints: list[dict[str, Any]],
    checkpoint_provenance: list[dict[str, Any]],
    trajectory: dict[str, Any],
    geometry_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{args.name}.json",
        "objects_csv": output_dir / f"{args.name}_objects.csv",
        "geometry_csv": output_dir / f"{args.name}_geometry.csv",
        "summary_csv": output_dir / f"{args.name}_summary.csv",
        "decision_csv": output_dir / f"{args.name}_decision.csv",
        "markdown": output_dir / f"{args.name}.md",
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "split": "train only",
            "loader_mode": "val (deterministic/no dataset augmentation)",
            "views": "clean reference vs deterministic nested scale+translate",
            "branches": list(HEADS),
            "workers_actual_required": 4,
            "pinned_memory_required": True,
            "occupancy": "unweighted Gaussian occupancy over (P3/P4/P5, GT-relative dx,dy)",
            "stability": "exp(-base2_JS); missing foreground is assigned stability 0",
            "quality": "max per-GT sqrt(class_probability * assigned-box IoU), used only for correlation",
            "args": vars(args),
            "frozen_protocol": FROZEN_PROTOCOL,
        },
        "provenance": {"trajectory": trajectory, "checkpoints": checkpoint_provenance},
        "checkpoints": checkpoints,
        "geometry": checkpoints[0]["geometry"],
        "summaries": summaries,
        "decision": decision,
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(paths["objects_csv"], object_rows)
    write_csv(paths["geometry_csv"], geometry_rows)
    write_csv(paths["summary_csv"], summaries)
    write_csv(
        paths["decision_csv"],
        [
            {
                "verdict": decision["verdict"],
                "reason": decision["reason"],
                "preregistered": decision["preregistered"],
                "protocol_deviations": canonical_json(decision["protocol_deviations"]),
                "panel_sha256": decision["panel_sha256"],
                "geometry_sha256": decision["geometry_sha256"],
                "trajectory_sha256": trajectory["signature_sha256"],
                "source_bundle_sha256": checkpoints[0]["runtime"]["source_bundle_sha256"],
            }
        ],
    )

    lines = [
        "# ASAC assignment-stability probe",
        "",
        f"**Decision: {decision['verdict']}** — {decision['reason']}.",
        "",
        "> Fixed train-only panel; deterministic scale/translate views; inference-only fixed checkpoints; no optimizer, "
        "backward pass, BN-stat update, validation/test AP, or policy tuning.",
        "",
        f"Panel: `{checkpoints[0]['images']}` images, `{checkpoints[0]['batches']}` batches, fingerprint "
        f"`{decision['panel_sha256']}`.",
        "",
        f"Protocol: `{'PREREGISTERED' if decision['preregistered'] else 'NON_PREREGISTERED'}`; trajectory "
        f"`{trajectory['signature_sha256']}`; source bundle "
        f"`{checkpoints[0]['runtime']['source_bundle_sha256']}`.",
        "",
        "## Geometry survival",
        "",
        f"Clean GT: `{checkpoints[0]['geometry']['clean_gt_total']}`; all-view intersection: "
        f"`{checkpoints[0]['geometry']['intersection_count']}` "
        f"(`{checkpoints[0]['geometry']['intersection_rate']:.2%}`).",
        "",
        "| q | retained | rate |",
        "|---:|---:|---:|",
    ]
    for strength in args.strengths:
        geometry = checkpoints[0]["geometry"]["per_strength"][f"q{strength:.2f}"]
        lines.append(f"| {strength:.2f} | {geometry['count']} | {geometry['rate']:.2%} |")
    lines.extend(
        [
        "",
        "Retained/dropped GT area and boundary-distance quantiles are in JSON; per-GT survival flags are in the "
        "geometry CSV.",
        "",
        "## Stability summaries",
        "",
        "| epoch | head | q | objects | coverage | mean S | matched JS | Q clean→view | Q agreement | rho(S,Q-retain) | level flip |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['epoch']} | {row['head']} | {row['strength']:.2f} | {row['object_pairs']} | "
            f"{row['assignment_coverage']:.2%} | {fmt(row['mean_stability'])} | "
            f"{fmt(row['mean_js_assigned'])} | {fmt(row['mean_clean_quality_assigned'])}→"
            f"{fmt(row['mean_view_quality_assigned'])} | {fmt(row['mean_quality_agreement_assigned'])} | "
            f"{fmt(row['spearman_stability_quality_assigned'])} | {fmt(row['dominant_level_flip_rate'])} |"
        )
    if not decision["preregistered"]:
        lines.extend(
            [
                "",
                "## Non-preregistered deviations",
                "",
                f"`{canonical_json(decision['protocol_deviations'])}`",
                "",
                "These metrics are diagnostic only and cannot be cited as an ASAC-P0 GO/NO-GO decision.",
            ]
        )
    lines.extend(["", "## Automatic gate", ""])
    for head in HEADS:
        evidence = decision["head_evidence"][head]
        lines.extend(
            [
                f"- {head}: monotonic checkpoints `{evidence['monotonic_checkpoint_count']}` "
                f"{evidence['monotonic_epochs']}; mean weak→strong drop "
                f"`{fmt(evidence['mean_weak_strong_drop'])}`; correlated checkpoints "
                f"`{evidence['correlated_checkpoint_count']}` {evidence['correlated_epochs_at_strongest_view']}.",
            ]
        )
    thresholds = decision["thresholds"]
    lines.extend(
        [
            "",
            f"GO requires, for both heads: at least `{thresholds['min_monotonic_checkpoints']}` checkpoints with "
            f"non-increasing S (tolerance `{thresholds['monotonic_tolerance']}`), mean weak→strong drop "
            f"`≥{thresholds['min_weak_strong_drop']}`, and strongest-view positive Spearman "
            f"`ρ≥{thresholds['min_spearman']}` at at least `{thresholds['min_correlated_checkpoints']}` checkpoints.",
            f"Every cell also requires `≥{thresholds['min_object_pairs']}` objects, the same number assigned in both "
            f"views, and coverage `≥{thresholds['min_assignment_coverage']}`; otherwise INCONCLUSIVE.",
        ]
    )
    if decision["sample_insufficiency"]:
        lines.extend(["", "## Insufficient cells", ""] + [f"- {item}" for item in decision["sample_insufficiency"]])
    if decision["warnings"]:
        lines.extend(["", "## Warnings", ""] + [f"- {item}" for item in decision["warnings"]])
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    protocol = {
        "expected_epochs": list(args.expected_epochs),
        "strengths": list(args.strengths),
        "min_object_pairs": args.min_object_pairs,
        "min_assignment_coverage": args.min_assignment_coverage,
        "monotonic_tolerance": args.monotonic_tolerance,
        "min_weak_strong_drop": args.min_weak_strong_drop,
        "min_monotonic_checkpoints": args.min_monotonic_checkpoints,
        "min_spearman": args.min_spearman,
        "min_correlated_checkpoints": args.min_correlated_checkpoints,
    }
    protocol.update(protocol_status(args))
    return protocol


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args.checkpoints)
    registration = protocol_status(args)
    if args.dry_run:
        print(
            json.dumps(
                {"checkpoints": checkpoints, "protocol": registration, "args": vars(args)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    initialize_runtime(args.physical_device)
    checkpoint_provenance, trajectory = inspect_checkpoints(checkpoints, args.data)
    checkpoint_results, object_rows = [], []
    canonical_geometry_rows = None
    for checkpoint_record in checkpoint_provenance:
        result, rows, geometry_rows = probe_checkpoint(checkpoint_record, args)
        checkpoint_results.append(result)
        object_rows.extend(rows)
        if canonical_geometry_rows is None:
            canonical_geometry_rows = geometry_rows
        elif sha256_json(geometry_rows) != sha256_json(canonical_geometry_rows):
            raise RuntimeError("per-GT geometry/survival panel changed across checkpoints")
        print(
            json.dumps(
                {
                    "checkpoint": result["checkpoint"],
                    "epoch": result["epoch"],
                    "panel_sha256": result["panel_sha256"],
                    "geometry_sha256": result["geometry_sha256"],
                    "parameter_sha256": result["integrity_after"]["parameter_sha256"],
                    "bn_buffer_sha256": result["integrity_after"]["bn_buffer_sha256"],
                },
                ensure_ascii=False,
            )
        )
    checkpoint_results.sort(key=lambda row: row["epoch"])
    source_bundles = {row["runtime"]["source_bundle_sha256"] for row in checkpoint_results}
    criterion_contracts = {sha256_json(row["criterion_actual"]) for row in checkpoint_results}
    if len(source_bundles) != 1 or len(criterion_contracts) != 1:
        raise RuntimeError("runtime source or criterion contract changed across checkpoints")
    summaries = summarize_object_rows(object_rows)
    decision = make_decision(checkpoint_results, summaries, protocol_from_args(args))
    paths = write_outputs(
        args,
        checkpoint_results,
        checkpoint_provenance,
        trajectory,
        canonical_geometry_rows or [],
        object_rows,
        summaries,
        decision,
    )
    print(json.dumps({"decision": decision, "outputs": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
