"""FCA-P0: preregistered zero-training audit of YOLO26 final GT coverage.

This script never creates an optimizer, never calls backward, never evaluates
VOC2007 test, and never changes the installed TaskAlignedAssigner source.  It
temporarily wraps two methods on the live assigner instance, captures exact
native S0/S1/S2/S3 boundaries, and restores the methods in ``finally``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import probe_asac_assignment_stability as runtime_base
except ModuleNotFoundError:
    remote_helper_dir = Path("/home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts")
    if not (remote_helper_dir / "probe_asac_assignment_stability.py").is_file():
        raise
    sys.path.insert(0, str(remote_helper_dir))
    import probe_asac_assignment_stability as runtime_base
from fca_matching import (
    connected_components,
    lexicographic_quality_matching,
    maximum_matching,
    maximum_matching_preserving_left_set,
    recoverable_gap_excluding_uncovered_left,
)


SCHEMA_VERSION = 2
AUDIT_NAME = "FCA-P0"
HEADS = ("O2M", "O2O")
LEVELS = ("P3", "P4", "P5")
QUALITY_THRESHOLDS = (0.0, 0.25, 0.50, 0.75, 0.90)
EXPECTED_IMAGE_COUNT = 16_551
EXPECTED_GT_COUNT = 40_058
EXPECTED_STRIDES = (8.0, 16.0, 32.0)
EXPECTED_GIT_HEAD = "9d45ee4613e79d7bceef7a39e42c15c278b0d4e8"
EXPECTED_GIT_BRANCH = "research/dual-head-diagnostics"
EXPECTED_SOURCE_SHA256 = {
    "ultralytics/nn/modules/head.py": "1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8",
    "ultralytics/utils/tal.py": "9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861",
    "ultralytics/utils/loss.py": "6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9",
    "ultralytics/data/build.py": "1c3131adc60dc9763b9edcde853d3bc461ec908c2ed9d1965b2610bfd8c27847",
}
EXPECTED_DATA_YAML_SHA256 = "5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40"
EXPECTED_DATA_ROOT = Path("/home/ubuntu/rz/yolo_study/datasets/VOC")
TRAIN_SPLITS = ("train2012", "train2007", "val2012", "val2007")
EXPECTED_CHECKPOINT_SHA256 = {
    "epoch0.pt": "dff866ec3fa3087731de178064e40350b3d9625bb3679d427e9e3a954a79c346",
    "epoch10.pt": "fe96652a133d9f7f13bb586ed06eb517608227f30ea523bc0d717d77762cafbe",
    "epoch20.pt": "17d1b6f0d4089e607899560446ea6102f89427c205906c83bb691d6c9b82acda",
    "epoch30.pt": "cec52cee8bfe0aafb3e10fbb94022ac4e0c98897541b0e440d9557f5846d623a",
    "epoch40.pt": "bfa1467eb89b38124d3a0f7ee98d4b79d7608ac4e513edb608ebe1e672e6965a",
    "last.pt": "a0bab3ef6b42b5cf9b295fbbbe9a322ebd39bc5cf568883aecb9ba4ddf547748",
}
EXPECTED_INTERNAL_EPOCH = {
    "epoch0.pt": 0,
    "epoch10.pt": 10,
    "epoch20.pt": 20,
    "epoch30.pt": 30,
    "epoch40.pt": 40,
    "last.pt": (49, -1),
}
EXPECTED_RUN_NAME = "b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1"
EXPECTED_MODEL_PARAMETER_COUNT = 2_511_600
EXPECTED_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
EXPECTED_IMAGE_MANIFEST_SHA256 = "4a91fffa66e61125f414751d5bef2937457a5aeeb76a4e09fd6497c2be910cc0"
EXPECTED_LABEL_BUNDLE_SHA256 = "39c8623c143706f2caede28dae7b5a248d9d2dd703aca12de043de942508c28a"
EXPECTED_REMOTE_RUNTIME_HELPER_SHA256 = "a0d87624f013d79c69c30ada1b9295e5bd21304b3de9778336030c79b0ea6e74"
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPLICATES = 10_000
GO_RATE = 0.0025
EXPECTED_CPU_TEST_COUNT = 21


OBJECT_FIELDS = (
    "checkpoint", "checkpoint_epoch", "precision", "head", "image_index", "image_path", "local_gt_index",
    "class_id", "class_name", "cx", "cy", "width", "height", "width_px", "height_px", "x1_px", "y1_px",
    "x2_px", "y2_px", "size_bin",
    "image_gt_count", "image_gt_count_bin", "max_other_gt_iou", "max_other_gt_iou_bin",
    "s0_anchor_count", "s1_anchor_count", "s2_anchor_count", "s3_anchor_count",
    "covered_s0", "covered_s1", "covered_s2", "covered_s3", "loss_valid_s0", "loss_s0_s1",
    "loss_s1_s2", "gain_s1_s2", "loss_s2_s3", "loss_s1_s3", "s1_removed_edge_count",
    "s2_added_edge_count", "s2_added_outside_s0_count", "competition_degree", "competition_bin",
    "candidate_p3", "candidate_p4", "candidate_p5", "best_alignment_level", "alignment_max",
    "zero_quality", "native_anchor", "native_edge_in_s1", "native_final_offgraph_edge_count",
    "native_raw_alignment", "native_r", "native_ciou_clipped", "native_iou",
    "native_offgraph_alignment_max", "native_offgraph_ciou_clipped_max", "native_offgraph_iou_max",
    "native_coverage_rematch_feasible", "maximum_anchor", "maximum_r",
    "maximum_ciou_clipped", "maximum_iou", "recoverable_gap_gt", "recoverable_r_ge_050", "lex_anchor", "lex_r",
    "lex_ciou_clipped", "lex_iou", "common_quality_delta", "pair_rewritten",
)

COMPONENT_FIELDS = (
    "checkpoint", "checkpoint_epoch", "precision", "head", "image_index", "image_path", "component_index",
    "gt_count", "anchor_count", "edge_count", "max_gt_degree", "max_anchor_degree", "has_shared_anchor", "native_coverage",
    "matching_coverage_r000", "matching_coverage_r025", "matching_coverage_r050", "matching_coverage_r075",
    "matching_coverage_r090", "signed_gap_r000", "signed_gap_r025", "signed_gap_r050", "signed_gap_r075",
    "signed_gap_r090", "gap_r000", "gap_r025", "gap_r050", "gap_r075", "gap_r090",
    "zero_quality_gt", "matching_seconds", "native_assignment_batch_seconds", "instrumented_criterion_batch_seconds",
)

IMAGE_FIELDS = (
    "checkpoint", "checkpoint_epoch", "precision", "head", "image_index", "image_path", "valid_gt",
    "native_coverage", "matching_coverage_r000", "matching_coverage_r025", "matching_coverage_r050",
    "matching_coverage_r075", "matching_coverage_r090", "signed_gap_r000", "signed_gap_r025", "signed_gap_r050",
    "signed_gap_r075", "signed_gap_r090", "gap_r000", "gap_r025", "gap_r050", "gap_r075", "gap_r090",
    "nonzero_only_gap", "zero_quality_gap_units", "candidate_edges", "s1_removed_edges", "s2_added_edges",
    "s2_added_outside_s0_edges", "native_final_offgraph_edges", "native_coverage_rematch_feasible",
    "component_analysis_valid", "matching_seconds",
    "native_assignment_seconds", "instrumented_criterion_seconds",
)


torch = None
ultralytics = None
RUNTIME_LOG_PATH: Path | None = None


class InvalidRunError(RuntimeError):
    """A completed/partial scan that cannot receive a scientific verdict."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physical-device", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run one checkpoint, one batch, plus native bitwise control.")
    parser.add_argument("--max-batches", type=int, default=None, help="Only valid with --smoke.")
    parser.add_argument("--skip-file-manifest-match", action="store_true", help="Diagnostic only; makes run non-preregistered.")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def command_output(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def execution_affinity_and_gpu() -> dict[str, Any]:
    status = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:")):
            key, value = line.split(":", 1)
            status[key] = value.strip()
    numa_lines = command_output(["numactl", "--show"]).splitlines()
    numa_policy = {}
    for line in numa_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            numa_policy[key.strip()] = value.strip()
    required_numa = {"policy": "bind", "cpubind": "1", "nodebind": "1", "membind": "1"}
    mismatches = {
        key: {"expected": wanted, "actual": numa_policy.get(key)}
        for key, wanted in required_numa.items()
        if numa_policy.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(f"FCA-P0 NUMA policy mismatch: {mismatches}")
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(4)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        processes = [
            {"pid": int(process.pid), "used_gpu_memory": int(process.usedGpuMemory)}
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ]
        foreign = [process for process in processes if process["pid"] != os.getpid()]
        if foreign:
            raise RuntimeError(f"physical GPU 4 has foreign compute processes: {foreign}")
        gpu = {
            "physical_index": 4,
            "memory_used_bytes": int(memory.used),
            "utilization_percent": int(utilization.gpu),
            "compute_processes": processes,
            "foreign_compute_processes": foreign,
        }
    except ImportError as error:
        raise RuntimeError("pynvml is required for trustworthy GPU preflight") from error
    return {"proc_status": status, "numa_policy": numa_policy, "gpu_preflight": gpu}


def log(event: str, **payload: Any) -> None:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **payload}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if RUNTIME_LOG_PATH is not None:
        with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def kernel_fault_events(since_iso: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["journalctl", "-k", "--since", since_iso, "--no-pager"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise InvalidRunError(f"kernel journal health query failed: {completed.stdout.strip()}")
    pattern = re.compile(
        r"Uncorrected hardware memory error|Sending SIGBUS|Memory failure|EDAC.*(?:error|corrected|uncorrected)",
        re.IGNORECASE,
    )
    events = [line for line in completed.stdout.splitlines() if pattern.search(line)]
    return {
        "since": since_iso,
        "checked_at": dt.datetime.now().astimezone().isoformat(),
        "journal_returncode": completed.returncode,
        "fault_pattern": pattern.pattern,
        "events": events,
        "event_count": len(events),
        "valid": not events,
    }


def run_cpu_self_tests(output_dir: Path) -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parent
    test_files = (scripts_dir / "test_fca_matching.py", scripts_dir / "test_probe_final_coverage_audit.py")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", *(path.name for path in test_files)],
        cwd=scripts_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = output_dir / "cpu_tests.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    count_match = re.search(r"Ran (\d+) tests? in", completed.stdout)
    test_count = int(count_match.group(1)) if count_match else 0
    if (
        completed.returncode != 0
        or test_count != EXPECTED_CPU_TEST_COUNT
        or "skipped=" in completed.stdout
        or not completed.stdout.rstrip().endswith("OK")
    ):
        raise RuntimeError(f"FCA-P0 CPU self-tests failed; see {log_path}")
    return {
        "command": [sys.executable, "-m", "unittest", "-v", *(path.name for path in test_files)],
        "returncode": completed.returncode,
        "tests": test_count,
        "result": "OK",
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "test_files": {path.name: sha256_file(path) for path in test_files},
    }


def initialize_runtime(physical_device: int) -> None:
    global torch, ultralytics
    if physical_device != 4:
        raise RuntimeError(f"FCA-P0 is frozen to physical GPU 4, got {physical_device}")
    frozen_source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(frozen_source_root))
    runtime_base.initialize_runtime(physical_device)
    torch = runtime_base.torch
    ultralytics = runtime_base.ultralytics
    imported_root = Path(ultralytics.__file__).resolve().parents[1]
    if imported_root != frozen_source_root:
        raise RuntimeError(f"Ultralytics import escaped frozen source root: {imported_root} != {frozen_source_root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only physical GPU 4")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("FCA-P0 requires exactly one visible CUDA device")


def protocol_record(args: argparse.Namespace) -> dict[str, Any]:
    deviations = {}
    expected = {"physical_device": 4, "batch": 64, "imgsz": 640, "workers": 4, "seed": 0}
    for key, wanted in expected.items():
        if getattr(args, key) != wanted:
            deviations[key] = {"expected": wanted, "actual": getattr(args, key)}
    if args.skip_file_manifest_match:
        deviations["skip_file_manifest_match"] = {"expected": False, "actual": True}
    if not args.smoke and args.max_batches is not None:
        deviations["max_batches"] = {"expected": None, "actual": args.max_batches}
    return {
        "audit": AUDIT_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SMOKE" if args.smoke else "FORMAL",
        "preregistered": not deviations and not args.smoke,
        "protocol_deviations": deviations,
        "data_split": "train only: VOC2007 trainval + VOC2012 trainval",
        "forbidden_split": "VOC2007 test",
        "expected_images": EXPECTED_IMAGE_COUNT,
        "expected_gt": EXPECTED_GT_COUNT,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "pinned_memory": True,
        "shuffle": False,
        "augment": False,
        "rect": False,
        "fraction": 1.0,
        "physical_gpu": args.physical_device,
        "numa_node": 1,
        "primary_precision": "BF16 autocast",
        "sensitivity_precision": "FP32 last.pt",
        "quality_thresholds": list(QUALITY_THRESHOLDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "go_rate": GO_RATE,
        "zero_quality_gap_definition": "max(0, unrestricted_gap - nonzero-GT-only gap)",
        "subgroup_gap_attribution": (
            "native-covered GT identities are first rematched inside E_topk, then that surrogate seed is augmented; "
            "newly covered GTs are additive gap units only when identity rematching is feasible"
        ),
        "protocol_amendment_A1": {
            "frozen_before_full_scan": True,
            "trigger": "formal attempt stopped on batch 2 because native S2 can introduce edges outside S1",
            "unchanged": "dataset, checkpoints, thresholds, E_topk matching graph, GO/NO-GO effect-size gates",
            "stage_semantics": (
                "S2 is raw native post-conflict rewiring, not assumed to be a subset of S1; added and removed edges "
                "are reported explicitly"
            ),
            "offgraph_policy": "raw S2/S3 preserved; off-graph edges never inserted into E_topk",
            "verdict_guard": (
                "any image whose native-covered GT identities cannot all be rematched in E_topk blocks GO and "
                "returns an inconclusive protocol verdict"
            ),
        },
        "checkpoint_files": list(EXPECTED_CHECKPOINT_SHA256),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "approved_by_user": True,
        "approved_request": "执行 FCA-P0：按预注册先做零训练全量审计",
    }


def validate_source_and_git(source_root: Path, data_yaml: Path) -> dict[str, Any]:
    actual_sources = {}
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {relative}: expected {expected}, got {actual}")
        actual_sources[relative] = {"path": str(path.resolve()), "sha256": actual}
    data_hash = sha256_file(data_yaml)
    if data_hash != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError(f"data YAML hash mismatch: expected {EXPECTED_DATA_YAML_SHA256}, got {data_hash}")
    git_head = command_output(["git", "rev-parse", "HEAD"], source_root)
    git_branch = command_output(["git", "branch", "--show-current"], source_root)
    if git_head != EXPECTED_GIT_HEAD or git_branch != EXPECTED_GIT_BRANCH:
        raise RuntimeError(f"git identity mismatch: branch={git_branch}, head={git_head}")
    helper_hash = sha256_file(runtime_base.__file__)
    if Path(runtime_base.__file__).as_posix().startswith("/home/ubuntu/") and helper_hash != EXPECTED_REMOTE_RUNTIME_HELPER_SHA256:
        raise RuntimeError(
            f"runtime helper hash mismatch: expected {EXPECTED_REMOTE_RUNTIME_HELPER_SHA256}, got {helper_hash}"
        )
    return {
        "source_root": str(source_root.resolve()),
        "git_head": git_head,
        "git_branch": git_branch,
        "files": actual_sources,
        "data_yaml": {"path": str(data_yaml.resolve()), "sha256": data_hash},
        "probe": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "matching": {
            "path": str((Path(__file__).parent / "fca_matching.py").resolve()),
            "sha256": sha256_file(Path(__file__).parent / "fca_matching.py"),
        },
        "runtime_helper": {"path": str(Path(runtime_base.__file__).resolve()), "sha256": helper_hash},
    }


def compute_data_file_manifests(data_root: Path, allow_mismatch: bool) -> dict[str, Any]:
    """Recompute the exact shell-equivalent manifests frozen during read-only preflight."""
    if data_root.resolve() != EXPECTED_DATA_ROOT.resolve():
        raise RuntimeError(f"data root must be {EXPECTED_DATA_ROOT}, got {data_root}")
    image_paths = sorted(
        (path for split in TRAIN_SPLITS for path in (data_root / "images" / split).glob("*.*") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    label_paths = sorted(
        (path for split in TRAIN_SPLITS for path in (data_root / "labels" / split).glob("*.txt") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    image_digest = hashlib.sha256()
    for path in image_paths:
        relative = path.relative_to(data_root).as_posix()
        image_digest.update(f"{relative} {path.stat().st_size}\n".encode("utf-8"))
    label_digest = hashlib.sha256()
    gt_count = 0
    for path in label_paths:
        relative = path.relative_to(data_root).as_posix()
        content_hash = sha256_file(path)
        label_digest.update(f"{content_hash}  {relative}\n".encode("utf-8"))
        with path.open("rb") as handle:
            gt_count += sum(1 for line in handle if line.strip())
    record = {
        "data_root": str(data_root.resolve()),
        "train_splits_in_yaml_order": list(TRAIN_SPLITS),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "gt_lines": gt_count,
        "image_relative_path_space_size_sha256": image_digest.hexdigest(),
        "ordered_label_sha256sum_bundle": label_digest.hexdigest(),
        "test2007_touched": False,
    }
    mismatches = {}
    expected = {
        "image_count": EXPECTED_IMAGE_COUNT,
        "label_count": EXPECTED_IMAGE_COUNT,
        "gt_lines": EXPECTED_GT_COUNT,
        "image_relative_path_space_size_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
        "ordered_label_sha256sum_bundle": EXPECTED_LABEL_BUNDLE_SHA256,
    }
    for key, wanted in expected.items():
        if record[key] != wanted:
            mismatches[key] = {"expected": wanted, "actual": record[key]}
    record["mismatches"] = mismatches
    if mismatches and not allow_mismatch:
        raise RuntimeError(f"VOC train file manifest mismatch: {mismatches}")
    return record


def resolve_checkpoints(patterns: list[str], smoke: bool) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for item in patterns]
    if len(set(paths)) != len(paths):
        raise RuntimeError("checkpoint list contains duplicates")
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"checkpoint does not exist: {path}")
        if path.name == "best.pt":
            raise RuntimeError("best.pt is forbidden")
    if smoke:
        if len(paths) != 1:
            raise RuntimeError("smoke must use exactly one checkpoint")
    elif {path.name for path in paths} != set(EXPECTED_CHECKPOINT_SHA256) or len(paths) != 6:
        raise RuntimeError(f"formal FCA-P0 requires exactly {list(EXPECTED_CHECKPOINT_SHA256)}")
    if len({path.parent.resolve() for path in paths}) != 1:
        raise RuntimeError("all checkpoints must share one weights directory")
    if paths[0].parent.parent.name != EXPECTED_RUN_NAME:
        raise RuntimeError(f"checkpoint run directory must be {EXPECTED_RUN_NAME}")
    return sorted(paths, key=lambda path: list(EXPECTED_CHECKPOINT_SHA256).index(path.name))


def inspect_checkpoints(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    trajectory_fingerprints = set()
    structure_fingerprints = set()
    for path in paths:
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_CHECKPOINT_SHA256[path.name]
        if actual_hash != expected_hash:
            raise RuntimeError(f"checkpoint SHA256 mismatch for {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        internal_epoch = payload.get("epoch")
        expected_epoch = EXPECTED_INTERNAL_EPOCH[path.name]
        allowed = expected_epoch if isinstance(expected_epoch, tuple) else (expected_epoch,)
        if internal_epoch not in allowed:
            raise RuntimeError(f"checkpoint epoch mismatch for {path.name}: expected {allowed}, got {internal_epoch}")
        train_args = payload.get("train_args")
        if not isinstance(train_args, dict):
            raise RuntimeError(f"missing train_args in {path.name}")
        frozen_trajectory = {
            key: train_args.get(key)
            for key in (
                "model", "data", "epochs", "batch", "imgsz", "seed", "deterministic", "pretrained", "amp",
                "fraction", "rect", "multi_scale", "cache", "project", "name", "cls_remap", "o2o_kd_gain",
                "tqgb_cls_m2o_gain", "tqgb_cls_o2m_gain",
            )
        }
        trajectory_fingerprints.add(sha256_json(frozen_trajectory))
        if int(train_args.get("seed", -1)) != 0 or int(train_args.get("batch", -1)) != 64:
            raise RuntimeError(f"{path.name} is not the frozen seed-0 batch-64 trajectory")
        model = payload.get("ema") if payload.get("ema") is not None else payload.get("model")
        serialized_head = model.model[-1]
        serialized_end2end = bool(getattr(serialized_head, "end2end", False))
        serialized_o2o_present = (
            getattr(serialized_head, "one2one_cv2", None) is not None
            and getattr(serialized_head, "one2one_cv3", None) is not None
        )
        if not serialized_end2end or not serialized_o2o_present:
            raise RuntimeError(
                f"serialized checkpoint is not the frozen dual-head end-to-end model: {path.name}"
            )
        structure = runtime_base.model_structure_record(model)
        names = runtime_base.canonical_model_names(structure.get("names"))
        if (
            int(structure.get("nc", -1)) != 20
            or tuple(float(value) for value in structure.get("stride", ())) != EXPECTED_STRIDES
            or int(structure.get("parameter_count", -1)) != EXPECTED_MODEL_PARAMETER_COUNT
            or tuple(names) != EXPECTED_CLASS_NAMES
        ):
            raise RuntimeError(f"checkpoint model structure mismatch for {path.name}: {structure}")
        structure_fingerprints.add(structure["sha256"])
        records.append(
            {
                "path": str(path),
                "filename": path.name,
                "sha256": actual_hash,
                "internal_epoch": internal_epoch,
                "completed_epoch": (50 if path.name == "last.pt" else int(path.stem.removeprefix("epoch")) + 1),
                "trajectory": frozen_trajectory,
                "trajectory_sha256": sha256_json(frozen_trajectory),
                "model_structure": structure,
                "serialized_head_end2end": serialized_end2end,
                "serialized_one2one_head_present": serialized_o2o_present,
            }
        )
        del model, payload
    if len(trajectory_fingerprints) != 1 or len(structure_fingerprints) != 1:
        raise RuntimeError("checkpoints do not share one frozen trajectory/model structure")
    return records


def all_tensor_sha256(named_tensors: Iterable[tuple[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        count += 1
    return digest.hexdigest(), count


def integrity_snapshot(trainer: Any) -> dict[str, Any]:
    parameter_hash, parameter_count = all_tensor_sha256(trainer.model.named_parameters())
    buffer_hash, buffer_count = all_tensor_sha256(trainer.model.named_buffers())
    bn_modules = [module for module in trainer.model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    return {
        "parameter_sha256": parameter_hash,
        "parameter_tensor_count": parameter_count,
        "buffer_sha256": buffer_hash,
        "buffer_tensor_count": buffer_count,
        "bn_total": len(bn_modules),
        "bn_eval": sum(not module.training for module in bn_modules),
        "bn_all_eval": all(not module.training for module in bn_modules),
        "model_training": bool(trainer.model.training),
        "optimizer_created": getattr(trainer, "optimizer", None) is not None,
        "parameters_with_grad": sum(parameter.grad is not None for parameter in trainer.model.parameters()),
        "grad_enabled": bool(torch.is_grad_enabled()),
    }


def assert_integrity(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in ("parameter_sha256", "parameter_tensor_count", "buffer_sha256", "buffer_tensor_count", "bn_total"):
        if before[key] != after[key]:
            raise RuntimeError(f"model state changed during zero-training audit: {key}")
    if not before["bn_all_eval"] or not after["bn_all_eval"]:
        raise RuntimeError("all BatchNorm modules must remain in eval mode")
    if before["optimizer_created"] or after["optimizer_created"]:
        raise RuntimeError("zero-training audit unexpectedly created an optimizer")
    if before["parameters_with_grad"] or after["parameters_with_grad"]:
        raise RuntimeError("zero-training audit observed parameter gradients")


def validate_contract(trainer: Any) -> dict[str, Any]:
    contract = runtime_base.validate_criterion_contract(trainer, list(EXPECTED_STRIDES))
    if contract["criterion_class"] != "ultralytics.utils.loss.E2ELoss":
        raise RuntimeError("criterion is not E2ELoss")
    return contract


class BoundaryCapture:
    """Capture exact TaskAlignedAssigner boundaries without changing its returned result."""

    def __init__(self, assigner: Any):
        self.assigner = assigner
        self.had_candidates_instance_attr = "select_candidates_in_gts" in assigner.__dict__
        self.had_highest_instance_attr = "select_highest_overlaps" in assigner.__dict__
        self.previous_candidates_instance_attr = assigner.__dict__.get("select_candidates_in_gts")
        self.previous_highest_instance_attr = assigner.__dict__.get("select_highest_overlaps")
        self.original_candidates = assigner.select_candidates_in_gts
        self.original_highest = assigner.select_highest_overlaps
        self.s0_calls: list[Any] = []
        self.s0_exact: list[Any] = []
        self.highest_calls: list[dict[str, Any]] = []
        source_lines, source_start = inspect.getsourcelines(self.original_highest)
        boundary_offsets = [
            index for index, line in enumerate(source_lines) if line.strip() == "if self.topk2 != self.topk:"
        ]
        if len(boundary_offsets) != 1:
            raise RuntimeError(
                "cannot identify unique native conflict->topk2 boundary in select_highest_overlaps"
            )
        self.s2_boundary_lineno = source_start + boundary_offsets[0]
        self.highest_code = self.original_highest.__func__.__code__

    def __enter__(self) -> "BoundaryCapture":
        capture = self

        def candidates(instance, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
            result = capture.original_candidates(xy_centers, gt_bboxes, mask_gt, eps)
            exact_clone = result.detach().clone()
            capture.s0_calls.append(exact_clone.sum(-1).to(dtype=torch.int16, device="cpu"))
            capture.s0_exact.append(exact_clone)
            return result

        def highest(instance, mask_pos, overlaps, n_max_boxes, align_metric):
            s1 = mask_pos.detach().clone()
            if len(capture.s0_exact) != 1 or capture.s0_exact[0].shape != s1.shape:
                raise RuntimeError("S0/S1 capture shape or call count mismatch")
            if bool((s1.bool() & ~capture.s0_exact[0].bool()).any()):
                raise RuntimeError("stage invariant failed: S1 is not a subset of S0")
            indices = s1.bool().nonzero(as_tuple=False)
            edge_align = align_metric[indices[:, 0], indices[:, 1], indices[:, 2]].detach().float().cpu()
            edge_overlap = overlaps[indices[:, 0], indices[:, 1], indices[:, 2]].detach().float().cpu()
            s2_captures = []
            previous_trace = sys.gettrace()
            if previous_trace is not None:
                raise RuntimeError("FCA-P0 requires no pre-existing Python trace function")

            def trace_native(frame, event, arg):
                if (
                    frame.f_code is capture.highest_code
                    and event == "line"
                    and frame.f_lineno == capture.s2_boundary_lineno
                ):
                    native_mask = frame.f_locals.get("mask_pos")
                    if native_mask is None:
                        raise RuntimeError("native S2 boundary lacks mask_pos local")
                    s2_captures.append(native_mask.detach().clone())
                return trace_native

            sys.settrace(trace_native)
            try:
                target_gt_idx, final_fg, final_mask = capture.original_highest(
                    mask_pos, overlaps, n_max_boxes, align_metric
                )
            finally:
                sys.settrace(previous_trace)
            if len(s2_captures) != 1:
                raise RuntimeError(f"native S2 boundary was captured {len(s2_captures)} times, expected exactly one")
            s2 = s2_captures[0]
            s1_bool = s1.bool()
            s2_bool = s2.bool()
            contested = s1.sum(-2) > 1
            changed_anchors = (s1_bool != s2_bool).any(-2)
            if bool((changed_anchors & ~contested).any()):
                raise RuntimeError("native conflict step changed a non-contested S1 anchor")
            if bool(contested.any()):
                expected_winner = overlaps.argmax(1)
                expected = torch.zeros_like(s2)
                expected.scatter_(1, expected_winner.unsqueeze(1), 1)
                contested_expanded = contested.unsqueeze(1).expand_as(s2_bool)
                if bool((s2_bool[contested_expanded] != expected.bool()[contested_expanded]).any()):
                    raise RuntimeError("native S2 winner does not match all-GT overlaps.argmax")
            s2_added = s2_bool & ~s1_bool
            s1_removed = s1_bool & ~s2_bool
            s2_added_outside_s0 = s2_added & ~capture.s0_exact[0].bool()
            if bool((s2.sum(-2) > 1).any()):
                raise RuntimeError("stage invariant failed: an S2 anchor belongs to multiple GTs")
            s2_indices = s2.bool().nonzero(as_tuple=False)
            s3 = final_mask.detach().clone()
            if bool((s3.bool() & ~s2.bool()).any()):
                raise RuntimeError("stage invariant failed: S3 is not a subset of S2")
            if bool((s3.sum(-2) > 1).any()):
                raise RuntimeError("stage invariant failed: an S3 anchor belongs to multiple GTs")
            if int(instance.topk2) == 1 and bool((s3.sum(-1) > 1).any()):
                raise RuntimeError("stage invariant failed: O2O S3 gives a GT more than one anchor")
            record = {
                "s1_counts": s1.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s2_counts": s2.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s3_counts": s3.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s1_removed_counts": s1_removed.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s2_added_counts": s2_added.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s2_added_outside_s0_counts": s2_added_outside_s0.sum(-1).to(dtype=torch.int16, device="cpu"),
                "s1_indices": indices.to(dtype=torch.int32, device="cpu"),
                "s2_indices": s2_indices.to(dtype=torch.int32, device="cpu"),
                "s3_indices": s3.bool().nonzero(as_tuple=False).to(dtype=torch.int32, device="cpu"),
                "edge_align": edge_align,
                "edge_overlap": edge_overlap,
                "s3_edge_align": align_metric[
                    s3.bool().nonzero(as_tuple=True)
                ].detach().float().cpu(),
                "s3_edge_overlap": overlaps[
                    s3.bool().nonzero(as_tuple=True)
                ].detach().float().cpu(),
                "s2_equals_s3": bool(torch.equal(s2, s3)),
                "returned_fg_matches_s3": bool(torch.equal(final_fg, s3.sum(-2))),
                "returned_gt_matches_s3": bool(torch.equal(target_gt_idx, s3.argmax(-2))),
            }
            capture.highest_calls.append(record)
            capture.s0_exact.clear()
            del s1, s2, s3
            return target_gt_idx, final_fg, final_mask

        object.__setattr__(self.assigner, "select_candidates_in_gts", types.MethodType(candidates, self.assigner))
        object.__setattr__(self.assigner, "select_highest_overlaps", types.MethodType(highest, self.assigner))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.had_candidates_instance_attr:
            object.__setattr__(self.assigner, "select_candidates_in_gts", self.previous_candidates_instance_attr)
        else:
            object.__delattr__(self.assigner, "select_candidates_in_gts")
        if self.had_highest_instance_attr:
            object.__setattr__(self.assigner, "select_highest_overlaps", self.previous_highest_instance_attr)
        else:
            object.__delattr__(self.assigner, "select_highest_overlaps")

    def result(self) -> dict[str, Any]:
        if len(self.s0_calls) != 1 or len(self.highest_calls) != 1:
            raise RuntimeError(
                f"assigner boundary was invoked {len(self.s0_calls)}/{len(self.highest_calls)} times; "
                "OOM fallback or unexpected re-entry makes this batch invalid"
            )
        record = self.highest_calls[0]
        record["s0_counts"] = self.s0_calls[0]
        if not record["returned_fg_matches_s3"] or not record["returned_gt_matches_s3"]:
            raise RuntimeError("captured S3 disagrees with native assigner return tensors")
        return record


def compare_assignments(native: dict[str, Any], instrumented: dict[str, Any], head: str) -> dict[str, Any]:
    keys = ("fg_mask", "target_gt_idx", "target_labels", "target_bboxes", "target_scores", "pred_bboxes", "num_gt")
    evidence = {}
    for key in keys:
        left, right = native[key], instrumented[key]
        equal = bool(torch.equal(left, right)) if isinstance(left, torch.Tensor) else left == right
        evidence[key] = equal
        if not equal:
            raise RuntimeError(f"instrumented/native bitwise mismatch for {head}.{key}")
    return evidence


def autocast_context(precision: str):
    if precision == "BF16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "FP32":
        return contextlib.nullcontext()
    raise ValueError(precision)


def run_forward(trainer: Any, batch: dict[str, Any], precision: str, smoke_compare: bool):
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("forward must run under torch.inference_mode")
    criterion = trainer.model.criterion
    with autocast_context(precision):
        if precision == "BF16":
            if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
                raise RuntimeError("BF16 autocast contract is inactive")
        elif torch.is_autocast_enabled("cuda"):
            raise RuntimeError("FP32 sensitivity pass unexpectedly has autocast enabled")
        parsed = criterion.one2many.parse_output(trainer.model.predict(batch["img"]))
        predictions = {"O2M": parsed["one2many"], "O2O": parsed["one2one"]}
        results = {}
        for head, branch_name in (("O2M", "one2many"), ("O2O", "one2one")):
            branch = getattr(criterion, branch_name)
            native_assignment_times = []

            def native_pre_hook(module, inputs):
                torch.cuda.synchronize()
                native_assignment_times.append(time.perf_counter())

            def native_post_hook(module, inputs, output):
                torch.cuda.synchronize()
                native_assignment_times.append(time.perf_counter())

            pre_handle = branch.assigner.register_forward_pre_hook(native_pre_hook)
            post_handle = branch.assigner.register_forward_hook(native_post_hook)
            try:
                native, _, _ = branch.get_assigned_targets_and_loss(predictions[head], batch, return_details=True)
            finally:
                pre_handle.remove()
                post_handle.remove()
            if len(native_assignment_times) != 2:
                raise RuntimeError(f"native assigner timing hook fired {len(native_assignment_times)} times for {head}")
            native_assignment_seconds = native_assignment_times[1] - native_assignment_times[0]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with BoundaryCapture(branch.assigner) as boundary:
                instrumented, _, _ = branch.get_assigned_targets_and_loss(
                    predictions[head], batch, return_details=True
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            capture = boundary.result()
            if head == "O2M" and not capture["s2_equals_s3"]:
                raise RuntimeError("O2M hard control failed: E_conflict != E_final")
            bitwise = compare_assignments(native, instrumented, head) if smoke_compare else None
            feats = predictions[head]["feats"]
            level_lengths = tuple(int(feature.shape[-2] * feature.shape[-1]) for feature in feats)
            if len(level_lengths) != 3 or sum(level_lengths) != int(predictions[head]["boxes"].shape[-1]):
                raise RuntimeError(f"invalid P3/P4/P5 layout for {head}: {level_lengths}")
            results[head] = {
                "assignment": instrumented,
                "capture": capture,
                "native_assignment_seconds": native_assignment_seconds,
                "instrumented_criterion_seconds": elapsed,
                "level_lengths": level_lengths,
                "prediction_dtypes": {
                    "boxes": str(predictions[head]["boxes"].dtype),
                    "scores": str(predictions[head]["scores"].dtype),
                    "features": [str(feature.dtype) for feature in feats],
                },
                "bitwise_control": bitwise,
            }
    return results


def update_geometry_digest(digest: Any, raw_batch: dict[str, Any]) -> None:
    paths = [str(path) for path in raw_batch["im_file"]]
    digest.update(canonical_json(paths).encode("utf-8"))
    for key in ("batch_idx", "cls", "bboxes"):
        value = raw_batch[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(value.shape).encode("ascii"))
        digest.update(value.tobytes())


def standard_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = np.clip(rb - lt, 0, None).prod(-1)
    areas = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(-1)
    union = areas[:, None] + areas[None, :] - intersection
    matrix = np.divide(intersection, np.maximum(union, 1e-12))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def gt_metadata(batch: dict[str, Any], paths: list[str]) -> list[list[dict[str, Any]]]:
    batch_size, _, height, width = batch["img"].shape
    batch_idx = batch["batch_idx"].detach().cpu().long().numpy()
    classes = batch["cls"].detach().cpu().reshape(-1).long().numpy()
    bboxes = batch["bboxes"].detach().cpu().float().numpy()
    result = []
    for image_index in range(batch_size):
        chosen = np.where(batch_idx == image_index)[0]
        local = bboxes[chosen]
        xyxy = np.zeros((len(local), 4), dtype=np.float64)
        if len(local):
            xyxy[:, 0] = (local[:, 0] - local[:, 2] / 2) * width
            xyxy[:, 1] = (local[:, 1] - local[:, 3] / 2) * height
            xyxy[:, 2] = (local[:, 0] + local[:, 2] / 2) * width
            xyxy[:, 3] = (local[:, 1] + local[:, 3] / 2) * height
        ious = standard_iou_matrix(xyxy)
        entries = []
        for local_index, source_index in enumerate(chosen.tolist()):
            cls = int(classes[source_index])
            entries.append(
                {
                    "local_gt_index": local_index,
                    "class_id": cls,
                    "class_name": EXPECTED_CLASS_NAMES[cls],
                    "cx": float(local[local_index, 0]),
                    "cy": float(local[local_index, 1]),
                    "width": float(local[local_index, 2]),
                    "height": float(local[local_index, 3]),
                    "width_px": float(local[local_index, 2] * width),
                    "height_px": float(local[local_index, 3] * height),
                    "x1_px": float(xyxy[local_index, 0]),
                    "y1_px": float(xyxy[local_index, 1]),
                    "x2_px": float(xyxy[local_index, 2]),
                    "y2_px": float(xyxy[local_index, 3]),
                    "max_other_gt_iou": float(ious[local_index].max()) if len(local) > 1 else 0.0,
                    "image_path": paths[image_index],
                }
            )
        result.append(entries)
    return result


def size_bin(width_px: float, height_px: float) -> str:
    value = min(width_px, height_px)
    if value < 16:
        return "<16"
    if value < 32:
        return "[16,32)"
    if value < 64:
        return "[32,64)"
    return ">=64"


def count_bin(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    return ">=10"


def iou_bin(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.1:
        return "(0,0.1)"
    if value < 0.3:
        return "[0.1,0.3)"
    return ">=0.3"


def competition_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    return ">=4"


def anchor_level(anchor: int, lengths: tuple[int, int, int]) -> str:
    if anchor < lengths[0]:
        return "P3"
    if anchor < lengths[0] + lengths[1]:
        return "P4"
    if anchor < sum(lengths):
        return "P5"
    raise RuntimeError(f"anchor {anchor} exceeds level layout {lengths}")


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def threshold_key(value: float) -> str:
    return f"r{int(round(value * 100)):03d}"


def sparse_by_image(indices: Any, values: Any | None, batch_size: int) -> list[list[Any]]:
    output = [[] for _ in range(batch_size)]
    index_rows = indices.tolist()
    value_rows = values.tolist() if values is not None else None
    for position, (image, gt, anchor) in enumerate(index_rows):
        payload = (int(gt), int(anchor)) if value_rows is None else (int(gt), int(anchor), float(value_rows[position]))
        output[int(image)].append(payload)
    return output


def process_head_batch(
    *,
    checkpoint: str,
    checkpoint_epoch: int,
    precision: str,
    head: str,
    image_offset: int,
    paths: list[str],
    metadata: list[list[dict[str, Any]]],
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    capture = result["capture"]
    batch_size = len(paths)
    s1_rows = sparse_by_image(capture["s1_indices"], None, batch_size)
    s2_rows = sparse_by_image(capture["s2_indices"], None, batch_size)
    s3_rows = sparse_by_image(capture["s3_indices"], None, batch_size)
    edge_align_rows = sparse_by_image(capture["s1_indices"], capture["edge_align"], batch_size)
    edge_overlap_rows = sparse_by_image(capture["s1_indices"], capture["edge_overlap"], batch_size)
    s3_align_rows = sparse_by_image(capture["s3_indices"], capture["s3_edge_align"], batch_size)
    s3_overlap_rows = sparse_by_image(capture["s3_indices"], capture["s3_edge_overlap"], batch_size)
    lengths = result["level_lengths"]
    object_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    quality = defaultdict(list)
    predicted_boxes_batch = result["assignment"]["pred_bboxes"].detach().float().cpu().numpy()

    for local_image in range(batch_size):
        started = time.perf_counter()
        gt_entries = metadata[local_image]
        valid_gt = len(gt_entries)
        for stage_name in (
            "s0_counts", "s1_counts", "s2_counts", "s3_counts", "s1_removed_counts",
            "s2_added_counts", "s2_added_outside_s0_counts",
        ):
            padded = capture[stage_name][local_image, valid_gt:]
            if padded.numel() and bool((padded != 0).any()):
                raise RuntimeError(f"padded GT rows are nonzero at {stage_name}")
        adjacency = {gt: [] for gt in range(valid_gt)}
        align_values: dict[tuple[int, int], float] = {}
        ciou_values: dict[tuple[int, int], float] = {}
        for gt, anchor, value in edge_align_rows[local_image]:
            if gt >= valid_gt:
                raise RuntimeError("padded GT unexpectedly has an S1 edge")
            adjacency[gt].append(anchor)
            align_values[(gt, anchor)] = value
        for gt, anchor, value in edge_overlap_rows[local_image]:
            ciou_values[(gt, anchor)] = value
        adjacency = {gt: sorted(set(anchors)) for gt, anchors in adjacency.items()}
        if sum(map(len, adjacency.values())) != len(s1_rows[local_image]):
            raise RuntimeError("S1 sparse edge de-duplication changed the candidate graph")
        s1_set = set(s1_rows[local_image])
        s2_set = set(s2_rows[local_image])
        s3_set = set(s3_rows[local_image])
        if not s3_set.issubset(s2_set):
            raise RuntimeError("S3 contains an edge outside native post-conflict S2")
        s1_removed_set = s1_set - s2_set
        s2_added_set = s2_set - s1_set
        native_offgraph_set = s3_set - s1_set

        max_align = {gt: max((align_values[(gt, anchor)] for anchor in adjacency[gt]), default=0.0) for gt in adjacency}
        zero_quality = {gt: max_align[gt] <= 0.0 for gt in adjacency}
        weights = {
            (gt, anchor): (align_values[(gt, anchor)] / max_align[gt] if max_align[gt] > 0 else 0.0)
            for gt in adjacency
            for anchor in adjacency[gt]
        }
        native_align_values = {(gt, anchor): value for gt, anchor, value in s3_align_rows[local_image]}
        native_ciou_values = {(gt, anchor): value for gt, anchor, value in s3_overlap_rows[local_image]}
        predicted_boxes = predicted_boxes_batch[local_image]
        iou_values: dict[tuple[int, int], float] = {}
        all_quality_anchors = {
            gt: sorted(set(anchors) | {anchor for native_gt, anchor in s3_set if native_gt == gt})
            for gt, anchors in adjacency.items()
        }
        for gt, anchors in all_quality_anchors.items():
            gt_box = np.asarray(
                [
                    gt_entries[gt]["x1_px"], gt_entries[gt]["y1_px"],
                    gt_entries[gt]["x2_px"], gt_entries[gt]["y2_px"],
                ],
                dtype=np.float64,
            )
            gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            for anchor in anchors:
                pred_box = predicted_boxes[anchor].astype(np.float64, copy=False)
                lt = np.maximum(gt_box[:2], pred_box[:2])
                rb = np.minimum(gt_box[2:], pred_box[2:])
                intersection = float(np.clip(rb - lt, 0, None).prod())
                pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
                iou_values[(gt, anchor)] = intersection / max(gt_area + pred_area - intersection, 1e-12)
        native_edges_by_gt: dict[int, list[int]] = defaultdict(list)
        for gt, anchor in s3_set:
            native_edges_by_gt[gt].append(anchor)
        native_covered = set(native_edges_by_gt)
        if len({anchor for anchors in native_edges_by_gt.values() for anchor in anchors}) != sum(
            len(anchors) for anchors in native_edges_by_gt.values()
        ):
            raise RuntimeError("native final edges do not form an anchor-unique assignment")
        maximum, coverage_seed, native_coverage_rematch_feasible = maximum_matching_preserving_left_set(
            adjacency, native_covered
        )
        unrestricted_cardinality = len(maximum_matching(adjacency))
        if len(maximum) != unrestricted_cardinality:
            raise RuntimeError("coverage-rematched matching is not maximum")
        gap = max(0, unrestricted_cardinality - len(native_covered))
        recoverable = set(maximum) - native_covered if native_coverage_rematch_feasible else set()
        if native_coverage_rematch_feasible and len(recoverable) != gap:
            raise RuntimeError("recoverable GT attribution is not additive to the exact gap")
        lex = lexicographic_quality_matching(adjacency, weights) if unrestricted_cardinality else {}

        threshold_coverages = {}
        threshold_gaps = {}
        for threshold in QUALITY_THRESHOLDS:
            if threshold == 0.0:
                filtered = adjacency
            else:
                filtered = {
                    gt: [anchor for anchor in anchors if not zero_quality[gt] and weights[(gt, anchor)] >= threshold]
                    for gt, anchors in adjacency.items()
                }
            coverage = len(maximum_matching(filtered))
            threshold_coverages[threshold] = coverage
            threshold_gaps[threshold] = max(0, coverage - len(native_covered))
        if threshold_coverages[0.0] != unrestricted_cardinality:
            raise RuntimeError("r=0 matching must equal unrestricted matching")
        if native_coverage_rematch_feasible:
            nonzero_gap = recoverable_gap_excluding_uncovered_left(
                adjacency, coverage_seed, {gt for gt, is_zero in zero_quality.items() if is_zero}
            )
        else:
            nonzero_graph = {
                gt: anchors for gt, anchors in adjacency.items()
                if gt in native_covered or not zero_quality[gt]
            }
            nonzero_gap = max(0, len(maximum_matching(nonzero_graph)) - len(native_covered))
        zero_gap_units = max(0, gap - nonzero_gap)

        anchor_to_gts: dict[int, set[int]] = defaultdict(set)
        for gt, anchors in adjacency.items():
            for anchor in anchors:
                anchor_to_gts[anchor].add(gt)
        competition = {
            gt: len(set().union(*(anchor_to_gts[a] for a in anchors)) - {gt}) if anchors else 0
            for gt, anchors in adjacency.items()
        }

        for gt, entry in enumerate(gt_entries):
            candidates = adjacency[gt]
            level_counts = {level: 0 for level in LEVELS}
            for anchor in candidates:
                level_counts[anchor_level(anchor, lengths)] += 1
            best_anchor = (
                max(candidates, key=lambda anchor: (align_values[(gt, anchor)], -anchor)) if candidates else None
            )
            native_anchor = (
                max(
                    native_edges_by_gt.get(gt, []),
                    key=lambda anchor: (native_align_values[(gt, anchor)], -anchor),
                )
                if gt in native_edges_by_gt else None
            )
            native_offgraph_anchors = [
                anchor for anchor in native_edges_by_gt.get(gt, []) if (gt, anchor) in native_offgraph_set
            ]
            native_offgraph_best = (
                max(native_offgraph_anchors, key=lambda anchor: (native_align_values[(gt, anchor)], -anchor))
                if native_offgraph_anchors else None
            )
            native_edge_in_s1 = native_anchor is not None and (gt, native_anchor) in s1_set
            maximum_anchor = maximum.get(gt)
            lex_anchor = lex.get(gt)
            native_r = weights.get((gt, native_anchor)) if native_edge_in_s1 else None
            maximum_r = weights.get((gt, maximum_anchor)) if maximum_anchor is not None else None
            lex_r = weights.get((gt, lex_anchor)) if lex_anchor is not None else None
            common_delta = lex_r - native_r if native_r is not None and gt in lex else None
            rewritten = None
            if native_edge_in_s1 and gt in lex:
                rewritten = lex_anchor not in native_edges_by_gt.get(gt, [])
            if gt in recoverable:
                quality["new_r"].append(float(maximum_r))
                quality["new_iou"].append(float(iou_values[(gt, maximum_anchor)]))
                quality["new_ciou_clipped"].append(float(ciou_values[(gt, maximum_anchor)]))
            if common_delta is not None:
                quality["common_delta"].append(float(common_delta))
            if rewritten is not None:
                quality["rewritten"].append(float(bool(rewritten)))

            s0_count = int(capture["s0_counts"][local_image, gt].item())
            s1_count = int(capture["s1_counts"][local_image, gt].item())
            s2_count = int(capture["s2_counts"][local_image, gt].item())
            s3_count = int(capture["s3_counts"][local_image, gt].item())
            s1_removed_count = int(capture["s1_removed_counts"][local_image, gt].item())
            s2_added_count = int(capture["s2_added_counts"][local_image, gt].item())
            s2_added_outside_s0_count = int(capture["s2_added_outside_s0_counts"][local_image, gt].item())
            row = {
                "checkpoint": checkpoint,
                "checkpoint_epoch": checkpoint_epoch,
                "precision": precision,
                "head": head,
                "image_index": image_offset + local_image,
                **entry,
                "size_bin": size_bin(entry["width_px"], entry["height_px"]),
                "image_gt_count": valid_gt,
                "image_gt_count_bin": count_bin(valid_gt),
                "max_other_gt_iou_bin": iou_bin(entry["max_other_gt_iou"]),
                "s0_anchor_count": s0_count,
                "s1_anchor_count": s1_count,
                "s2_anchor_count": s2_count,
                "s3_anchor_count": s3_count,
                "covered_s0": int(s0_count > 0),
                "covered_s1": int(s1_count > 0),
                "covered_s2": int(s2_count > 0),
                "covered_s3": int(s3_count > 0),
                "loss_valid_s0": int(s0_count == 0),
                "loss_s0_s1": int(s0_count > 0 and s1_count == 0),
                "loss_s1_s2": int(s1_count > 0 and s2_count == 0),
                "gain_s1_s2": int(s1_count == 0 and s2_count > 0),
                "loss_s2_s3": int(s2_count > 0 and s3_count == 0),
                "loss_s1_s3": int(s1_count > 0 and s3_count == 0),
                "s1_removed_edge_count": s1_removed_count,
                "s2_added_edge_count": s2_added_count,
                "s2_added_outside_s0_count": s2_added_outside_s0_count,
                "competition_degree": competition[gt],
                "competition_bin": competition_bin(competition[gt]),
                "candidate_p3": level_counts["P3"],
                "candidate_p4": level_counts["P4"],
                "candidate_p5": level_counts["P5"],
                "best_alignment_level": anchor_level(best_anchor, lengths) if best_anchor is not None else "none",
                "alignment_max": max_align[gt],
                "zero_quality": int(zero_quality[gt]),
                "native_anchor": native_anchor,
                "native_edge_in_s1": int(native_edge_in_s1),
                "native_final_offgraph_edge_count": sum((gt, anchor) in native_offgraph_set for anchor in native_edges_by_gt.get(gt, [])),
                "native_raw_alignment": native_align_values.get((gt, native_anchor)) if native_anchor is not None else None,
                "native_r": native_r,
                "native_ciou_clipped": native_ciou_values.get((gt, native_anchor)) if native_anchor is not None else None,
                "native_iou": iou_values.get((gt, native_anchor)) if native_anchor is not None else None,
                "native_offgraph_alignment_max": (
                    native_align_values[(gt, native_offgraph_best)] if native_offgraph_best is not None else None
                ),
                "native_offgraph_ciou_clipped_max": (
                    native_ciou_values[(gt, native_offgraph_best)] if native_offgraph_best is not None else None
                ),
                "native_offgraph_iou_max": (
                    iou_values[(gt, native_offgraph_best)] if native_offgraph_best is not None else None
                ),
                "native_coverage_rematch_feasible": int(native_coverage_rematch_feasible),
                "maximum_anchor": maximum_anchor,
                "maximum_r": maximum_r,
                "maximum_ciou_clipped": ciou_values.get((gt, maximum_anchor)) if maximum_anchor is not None else None,
                "maximum_iou": iou_values.get((gt, maximum_anchor)) if maximum_anchor is not None else None,
                "recoverable_gap_gt": int(gt in recoverable),
                "recoverable_r_ge_050": int(gt in recoverable and maximum_r is not None and maximum_r >= 0.50),
                "lex_anchor": lex_anchor,
                "lex_r": lex_r,
                "lex_ciou_clipped": ciou_values.get((gt, lex_anchor)) if lex_anchor is not None else None,
                "lex_iou": iou_values.get((gt, lex_anchor)) if lex_anchor is not None else None,
                "common_quality_delta": common_delta,
                "pair_rewritten": int(rewritten) if rewritten is not None else None,
            }
            object_rows.append(row)

        # A raw native edge outside E_topk can couple the native assignment to a
        # different candidate component.  Component-local signed gaps would then
        # be misattributed, so retain only the valid image-level comparison.
        components = [] if native_offgraph_set else connected_components(adjacency)
        for component_index, (lefts, rights) in enumerate(components):
            anchor_degrees = [sum(anchor in adjacency[gt] for gt in lefts) for anchor in rights]
            component_graph = {gt: [anchor for anchor in adjacency[gt] if anchor in rights] for gt in lefts}
            component_native = sum(gt in native_covered for gt in lefts)
            component_started = time.perf_counter()
            component_coverages = {}
            for threshold in QUALITY_THRESHOLDS:
                filtered = {
                    gt: [
                        anchor for anchor in anchors
                        if threshold == 0.0 or (not zero_quality[gt] and weights[(gt, anchor)] >= threshold)
                    ]
                    for gt, anchors in component_graph.items()
                }
                component_coverages[threshold] = len(maximum_matching(filtered))
            component_elapsed = time.perf_counter() - component_started
            has_shared_anchor = max(anchor_degrees, default=0) > 1
            if not has_shared_anchor and component_coverages[0.0] <= component_native:
                continue
            component_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_epoch": checkpoint_epoch,
                    "precision": precision,
                    "head": head,
                    "image_index": image_offset + local_image,
                    "image_path": paths[local_image],
                    "component_index": component_index,
                    "gt_count": len(lefts),
                    "anchor_count": len(rights),
                    "edge_count": sum(len(component_graph[gt]) for gt in lefts),
                    "max_gt_degree": max(map(len, component_graph.values()), default=0),
                    "max_anchor_degree": max(anchor_degrees, default=0),
                    "has_shared_anchor": int(has_shared_anchor),
                    "native_coverage": component_native,
                    **{f"matching_coverage_{threshold_key(t)}": component_coverages[t] for t in QUALITY_THRESHOLDS},
                    **{f"signed_gap_{threshold_key(t)}": component_coverages[t] - component_native for t in QUALITY_THRESHOLDS},
                    **{f"gap_{threshold_key(t)}": max(0, component_coverages[t] - component_native) for t in QUALITY_THRESHOLDS},
                    "zero_quality_gt": sum(zero_quality[gt] for gt in lefts),
                    "matching_seconds": component_elapsed,
                    "native_assignment_batch_seconds": result["native_assignment_seconds"],
                    "instrumented_criterion_batch_seconds": result["instrumented_criterion_seconds"],
                }
            )
        matching_elapsed = time.perf_counter() - started
        image_rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_epoch": checkpoint_epoch,
                "precision": precision,
                "head": head,
                "image_index": image_offset + local_image,
                "image_path": paths[local_image],
                "valid_gt": valid_gt,
                "native_coverage": len(native_covered),
                **{f"matching_coverage_{threshold_key(t)}": threshold_coverages[t] for t in QUALITY_THRESHOLDS},
                **{f"signed_gap_{threshold_key(t)}": threshold_coverages[t] - len(native_covered) for t in QUALITY_THRESHOLDS},
                **{f"gap_{threshold_key(t)}": threshold_gaps[t] for t in QUALITY_THRESHOLDS},
                "nonzero_only_gap": nonzero_gap,
                "zero_quality_gap_units": zero_gap_units,
                "candidate_edges": sum(map(len, adjacency.values())),
                "s1_removed_edges": len(s1_removed_set),
                "s2_added_edges": len(s2_added_set),
                "s2_added_outside_s0_edges": sum(
                    int(capture["s2_added_outside_s0_counts"][local_image, gt].item())
                    for gt in range(valid_gt)
                ),
                "native_final_offgraph_edges": len(native_offgraph_set),
                "native_coverage_rematch_feasible": int(native_coverage_rematch_feasible),
                "component_analysis_valid": int(not native_offgraph_set),
                "matching_seconds": matching_elapsed,
                "native_assignment_seconds": result["native_assignment_seconds"] / max(batch_size, 1),
                "instrumented_criterion_seconds": result["instrumented_criterion_seconds"] / max(batch_size, 1),
            }
        )
    return object_rows, component_rows, image_rows, dict(quality)


class SummaryAccumulator:
    def __init__(self):
        self.rows: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, row: dict[str, Any]) -> None:
        groups = (
            ("overall", "all"),
            ("size", row["size_bin"]),
            ("competition", row["competition_bin"]),
            ("image_gt_count", row["image_gt_count_bin"]),
            ("max_gt_iou", row["max_other_gt_iou_bin"]),
            ("best_level", row["best_alignment_level"]),
            ("class", f"{row['class_id']:02d}:{row['class_name']}"),
            ("size_x_competition", f"{row['size_bin']}|{row['competition_bin']}"),
        )
        for group_type, group_value in groups:
            key = (row["checkpoint"], row["checkpoint_epoch"], row["precision"], row["head"], group_type, group_value)
            counts = self.rows[key]
            counts["valid_gt"] += 1
            for name in (
                "covered_s0", "covered_s1", "covered_s2", "covered_s3", "loss_valid_s0", "loss_s0_s1",
                "loss_s1_s2", "gain_s1_s2", "loss_s2_s3", "loss_s1_s3", "s1_removed_edge_count",
                "s2_added_edge_count", "s2_added_outside_s0_count", "native_final_offgraph_edge_count",
                "zero_quality", "recoverable_gap_gt", "recoverable_r_ge_050", "candidate_p3", "candidate_p4",
                "candidate_p5",
            ):
                counts[name] += float(row[name])

    def output(self) -> list[dict[str, Any]]:
        output = []
        for key, counts in sorted(self.rows.items(), key=lambda item: tuple(map(str, item[0]))):
            checkpoint, epoch, precision, head, group_type, group_value = key
            valid = counts["valid_gt"]
            if counts["recoverable_r_ge_050"] > counts["recoverable_gap_gt"]:
                raise RuntimeError("quality-retained recoverable attribution is not a subset of recoverable GTs")
            output.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_epoch": epoch,
                    "precision": precision,
                    "head": head,
                    "group_type": group_type,
                    "group_value": group_value,
                    **{name: int(value) for name, value in counts.items()},
                    "s0_coverage_rate": counts["covered_s0"] / valid if valid else None,
                    "s1_coverage_rate": counts["covered_s1"] / valid if valid else None,
                    "s2_coverage_rate": counts["covered_s2"] / valid if valid else None,
                    "s3_coverage_rate": counts["covered_s3"] / valid if valid else None,
                    "recoverable_gap_rate": counts["recoverable_gap_gt"] / valid if valid else None,
                    "recoverable_r_ge_050_share": (
                        counts["recoverable_r_ge_050"] / counts["recoverable_gap_gt"]
                        if counts["recoverable_gap_gt"] else None
                    ),
                }
            )
        return output


def make_loader(trainer: Any, args: argparse.Namespace):
    loader, metadata = runtime_base.make_fixed_loader(trainer, args)
    if int(metadata["dataset_length"]) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"train dataset length must be {EXPECTED_IMAGE_COUNT}, got {metadata['dataset_length']}")
    if metadata["loader_workers"] != 4 or not metadata["loader_pin_memory"]:
        raise RuntimeError("formal loader must have workers=4 and pinned memory")
    return loader, metadata


def scan_checkpoint(
    checkpoint_record: dict[str, Any],
    precision: str,
    args: argparse.Namespace,
    writers: dict[str, Any],
    summary: SummaryAccumulator,
    smoke_compare: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    runtime_base.seed_everything(args.seed)
    trainer = runtime_base.make_trainer(checkpoint_record["path"], args)
    reconstructed_end2end_before = bool(getattr(trainer.model.model[-1], "end2end", False))
    if checkpoint_record["serialized_head_end2end"] and not reconstructed_end2end_before:
        # DetectionTrainer rebuilds the graph and transfers state_dict tensors; _end2end is a non-tensor
        # checkpoint attribute, so restore the already-validated serialized value before any forward.
        trainer.model.end2end = True
    reconstructed_end2end_after = bool(getattr(trainer.model.model[-1], "end2end", False))
    if not reconstructed_end2end_after:
        raise RuntimeError("failed to restore serialized end-to-end head selection")
    contract = validate_contract(trainer)
    loader, loader_metadata = make_loader(trainer, args)
    before = integrity_snapshot(trainer)
    if not before["model_training"] or not before["bn_all_eval"] or before["optimizer_created"]:
        raise RuntimeError("model must expose train-format heads with frozen BN and no optimizer")
    geometry_digest = hashlib.sha256()
    image_rows_all = []
    quality_all = defaultdict(list)
    image_offset = total_gt = batches = 0
    seen_paths: set[str] = set()
    dtype_evidence = None
    bitwise_evidence = None
    max_batches = 1 if args.smoke else None
    if args.smoke and args.max_batches is not None:
        max_batches = args.max_batches
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            update_geometry_digest(geometry_digest, raw_batch)
            paths = [str(path) for path in raw_batch["im_file"]]
            if any("/test2007/" in path.replace("\\", "/") for path in paths):
                raise RuntimeError("forbidden VOC2007 test image entered the train audit")
            duplicates = seen_paths.intersection(paths)
            if duplicates or len(set(paths)) != len(paths):
                raise RuntimeError(f"loader repeated image paths: {sorted(duplicates)[:3]}")
            seen_paths.update(paths)
            batch = trainer.preprocess_batch(raw_batch)
            metadata = gt_metadata(batch, paths)
            total_gt += sum(map(len, metadata))
            forward = run_forward(trainer, batch, precision, smoke_compare and batch_index == 0)
            current_dtypes = {head: result["prediction_dtypes"] for head, result in forward.items()}
            expected_dtype = "torch.bfloat16" if precision == "BF16" else "torch.float32"
            for head, evidence in current_dtypes.items():
                observed = [evidence["boxes"], evidence["scores"], *evidence["features"]]
                if any(value != expected_dtype for value in observed):
                    raise RuntimeError(
                        f"{precision} dtype contract failed for {head}: expected {expected_dtype}, got {observed}"
                    )
            if dtype_evidence is None:
                dtype_evidence = current_dtypes
            elif dtype_evidence != current_dtypes:
                raise RuntimeError("prediction dtype evidence changed within checkpoint")
            current_bitwise = {head: result["bitwise_control"] for head, result in forward.items()}
            if any(value is not None for value in current_bitwise.values()):
                bitwise_evidence = current_bitwise
            for head in HEADS:
                object_rows, component_rows, image_rows, quality = process_head_batch(
                    checkpoint=checkpoint_record["filename"],
                    checkpoint_epoch=checkpoint_record["completed_epoch"],
                    precision=precision,
                    head=head,
                    image_offset=image_offset,
                    paths=paths,
                    metadata=metadata,
                    result=forward[head],
                )
                writers["objects"].writerows(object_rows)
                writers["components"].writerows(component_rows)
                writers["images"].writerows(image_rows)
                for row in object_rows:
                    summary.update(row)
                image_rows_all.extend(image_rows)
                for key, values in quality.items():
                    quality_all[f"{head}:{key}"].extend(values)
            image_offset += len(paths)
            batches += 1
            if batch_index == 0 or (batch_index + 1) % 25 == 0:
                log(
                    "checkpoint_progress",
                    checkpoint=checkpoint_record["filename"],
                    precision=precision,
                    batches=batches,
                    images=image_offset,
                    gt=total_gt,
                )
    after = integrity_snapshot(trainer)
    assert_integrity(before, after)
    elapsed = time.perf_counter() - started
    expected_images = image_offset if args.smoke else EXPECTED_IMAGE_COUNT
    expected_gt = total_gt if args.smoke else EXPECTED_GT_COUNT
    if image_offset != expected_images or total_gt != expected_gt:
        raise RuntimeError(
            f"full scan count mismatch: images={image_offset}/{expected_images}, gt={total_gt}/{expected_gt}"
        )
    if not args.smoke and (batches != math.ceil(EXPECTED_IMAGE_COUNT / args.batch) or len(seen_paths) != EXPECTED_IMAGE_COUNT):
        raise RuntimeError(f"full scan batch/path count mismatch: batches={batches}, unique_paths={len(seen_paths)}")
    if smoke_compare and bitwise_evidence is None:
        raise RuntimeError("smoke did not execute native/instrumented bitwise control")
    result = {
        "checkpoint": checkpoint_record["filename"],
        "checkpoint_epoch": checkpoint_record["completed_epoch"],
        "precision": precision,
        "images": image_offset,
        "gt": total_gt,
        "batches": batches,
        "geometry_sha256": geometry_digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "loader": loader_metadata,
        "criterion": contract,
        "head_state_rehydration": {
            "serialized_end2end": checkpoint_record["serialized_head_end2end"],
            "reconstructed_before": reconstructed_end2end_before,
            "reconstructed_after": reconstructed_end2end_after,
            "tensor_state_changed": False,
        },
        "dtype_evidence": dtype_evidence,
        "inference_mode_enforced": True,
        "bitwise_control": bitwise_evidence,
        "integrity_before": before,
        "integrity_after": after,
    }
    if hasattr(loader, "close"):
        loader.close()
    del trainer, loader
    torch.cuda.empty_cache()
    log("checkpoint_complete", **{key: result[key] for key in ("checkpoint", "precision", "images", "gt", "elapsed_seconds")})
    return result, image_rows_all, dict(quality_all)


def cluster_bootstrap_rates(array: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Paired image-cluster bootstrap used by the formal six-checkpoint pool."""
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError("bootstrap input must be a nonempty N x 2 [gap, denominator] array")
    if (array[:, 1] <= 0).any():
        raise ValueError("every bootstrap cluster denominator must be positive")
    rng = np.random.default_rng(seed)
    rates = np.empty(replicates, dtype=np.float64)
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        sampled = rng.integers(0, len(array), size=(stop - start, len(array)), endpoint=False)
        gaps = array[sampled, 0].sum(axis=1)
        totals = array[sampled, 1].sum(axis=1)
        rates[start:stop] = gaps / totals
    return rates


def bootstrap_pooled(image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in image_rows if row["head"] == "O2O" and row["precision"] == "BF16"]
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in selected:
        values = grouped[int(row["image_index"])]
        values[0] += int(row["gap_r000"])
        values[1] += int(row["valid_gt"])
        values[2] += 1
    if len(grouped) != EXPECTED_IMAGE_COUNT or any(value[2] != 6 for value in grouped.values()):
        raise RuntimeError("cluster bootstrap requires every image to carry all six BF16 checkpoint observations")
    array = np.asarray([grouped[index][:2] for index in sorted(grouped)], dtype=np.int64)
    point = float(array[:, 0].sum() / array[:, 1].sum())
    rates = cluster_bootstrap_rates(array, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "cluster": "image; all six checkpoint observations retained together",
        "images": len(array),
        "gap": int(array[:, 0].sum()),
        "valid_gt": int(array[:, 1].sum()),
        "rate": point,
        "ci95_lower": float(np.quantile(rates, 0.025)),
        "ci95_upper": float(np.quantile(rates, 0.975)),
    }


def checkpoint_metrics(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in image_rows:
        if row["head"] != "O2O":
            continue
        key = (row["checkpoint"], int(row["checkpoint_epoch"]), row["precision"])
        values = groups[key]
        values["valid_gt"] += int(row["valid_gt"])
        values["native_coverage"] += int(row["native_coverage"])
        values["native_coverage_rematch_infeasible_images"] += int(
            not bool(row["native_coverage_rematch_feasible"])
        )
        values["native_final_offgraph_edges"] += int(row["native_final_offgraph_edges"])
        values["s2_added_edges"] += int(row["s2_added_edges"])
        values["s2_added_outside_s0_edges"] += int(row["s2_added_outside_s0_edges"])
        for threshold in QUALITY_THRESHOLDS:
            values[f"gap_{threshold_key(threshold)}"] += int(row[f"gap_{threshold_key(threshold)}"])
        values["zero_quality_gap_units"] += int(row["zero_quality_gap_units"])
    output = []
    for key, values in groups.items():
        checkpoint, epoch, precision = key
        gap = values["gap_r000"]
        output.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_epoch": epoch,
                "precision": precision,
                **dict(values),
                "gap_rate": gap / values["valid_gt"],
                "r050_retention": values["gap_r050"] / gap if gap else None,
                "zero_quality_gap_share": (
                    None
                    if values["native_coverage_rematch_infeasible_images"]
                    else (values["zero_quality_gap_units"] / gap if gap else 0.0)
                ),
            }
        )
    return sorted(output, key=lambda row: (row["precision"], row["checkpoint_epoch"]))


def dense_tiny_trigger(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [
        row for row in summary_rows
        if row["precision"] == "BF16" and row["head"] == "O2O" and row["group_type"] == "size_x_competition"
    ]
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relevant:
        cells[row["group_value"]].append(row)
    passing = []
    for cell, rows in cells.items():
        checkpoint_passes = []
        for row in rows:
            gap = int(row.get("recoverable_gap_gt", 0))
            valid = int(row.get("valid_gt", 0))
            retention = row.get("recoverable_r_ge_050_share")
            if retention is not None and not 0.0 <= retention <= 1.0:
                raise RuntimeError("dense-tiny witness retention must be within [0, 1]")
            if valid and gap >= 50 and gap / valid >= 0.01 and retention is not None and retention >= 0.50:
                checkpoint_passes.append(row["checkpoint"])
        if len(checkpoint_passes) >= 4:
            passing.append({"cell": cell, "checkpoint_passes": sorted(checkpoint_passes)})
    return {
        "triggered": bool(passing),
        "passing_cells": passing,
        "quality_note": (
            "cell quality retention uses the r value of the same fixed identity-preserving unrestricted witness edge; "
            "it is conservative and not a separate subgroup-optimal r>=0.50 matching"
        ),
    }


def make_decision(
    image_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], geometry_hashes: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = bootstrap_pooled(image_rows)
    metrics = checkpoint_metrics(image_rows)
    bf16 = [row for row in metrics if row["precision"] == "BF16"]
    primary = [
        row for row in bf16
        if row["gap_rate"] >= GO_RATE and row["gap_r000"] >= 100
    ]
    quality_pass = [row for row in primary if row["r050_retention"] is not None and row["r050_retention"] >= 0.50]
    late_pass = any(row["checkpoint"] in {"epoch40.pt", "last.pt"} for row in quality_pass)
    pooled_gap = sum(row["gap_r000"] for row in bf16)
    pooled_zero = sum(row["zero_quality_gap_units"] for row in bf16)
    last_bf16 = next(row for row in metrics if row["checkpoint"] == "last.pt" and row["precision"] == "BF16")
    last_fp32 = next(row for row in metrics if row["checkpoint"] == "last.pt" and row["precision"] == "FP32")
    precision_same_side = (last_bf16["gap_rate"] >= GO_RATE) == (last_fp32["gap_rate"] >= GO_RATE)
    native_coverage_rematch_feasible = all(
        row["native_coverage_rematch_infeasible_images"] == 0 for row in metrics
    )
    zero_share = (pooled_zero / pooled_gap if pooled_gap else 0.0) if native_coverage_rematch_feasible else None
    gates = {
        "bootstrap_lower_ge_0_25pct": bootstrap["ci95_lower"] >= GO_RATE,
        "four_checkpoint_effect": len(primary) >= 4,
        "four_checkpoint_quality_retention": len(quality_pass) >= 4,
        "late_checkpoint_pass": late_pass,
        "zero_quality_share_lt_25pct": zero_share is not None and zero_share < 0.25,
        "last_bf16_fp32_same_side": precision_same_side,
        "geometry_identical": len(set(geometry_hashes)) == 1,
        "native_coverage_rematch_feasible": native_coverage_rematch_feasible,
    }
    go = all(gates.values())
    if go:
        verdict = "GO_TO_OFFLINE_PHASE1"
    elif not native_coverage_rematch_feasible:
        verdict = "INCONCLUSIVE_NATIVE_COVERAGE_NOT_REMATCHABLE"
    elif not precision_same_side:
        verdict = "PRECISION_SENSITIVE / INCONCLUSIVE"
    elif bootstrap["ci95_upper"] < GO_RATE or pooled_gap < 30:
        verdict = "NO-GO_ON_CLEAN_VOC"
    elif len(primary) >= 4 and len(quality_pass) < 4:
        verdict = "LOW_QUALITY / NO-GO"
    elif primary and all(row["checkpoint_epoch"] <= 21 for row in primary):
        verdict = "EARLY_ONLY / NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    dense = dense_tiny_trigger(summary_rows)
    if go:
        dense = {
            "triggered": False,
            "passing_cells": [],
            "quality_note": "Not applicable because the primary GO conditions already passed.",
        }
    elif not native_coverage_rematch_feasible:
        dense = {
            "triggered": False,
            "passing_cells": [],
            "quality_note": "Disabled because native-covered GT identities were not always rematchable in E_topk.",
        }
    o2m_images = [
        row for row in image_rows if row["precision"] == "BF16" and row["head"] == "O2M"
    ]
    o2m_valid = sum(int(row["valid_gt"]) for row in o2m_images)
    o2m_gap = sum(int(row["gap_r000"]) for row in o2m_images)
    o2m_rate = o2m_gap / o2m_valid if o2m_valid else None
    o2m_rematch_infeasible_images = sum(
        not bool(row["native_coverage_rematch_feasible"]) for row in o2m_images
    )
    o2o_rate = bootstrap.get("rate", bootstrap["gap"] / bootstrap["valid_gt"])
    sidecar = {
        "head": "O2M 10/10",
        "valid_gt": o2m_valid,
        "avoidable_gap": o2m_gap,
        "avoidable_gap_rate": o2m_rate,
        "native_coverage_rematch_infeasible_images": o2m_rematch_infeasible_images,
        "o2o_specific_wording_supported": bool(
            not o2m_rematch_infeasible_images and o2m_rate is not None and o2m_rate < o2o_rate
        ),
        "interpretation": (
            "O2M identity rematching failed on at least one image; the sidecar is raw-cardinality only and cannot "
            "support or refute O2O-specific wording."
            if o2m_rematch_infeasible_images
            else (
                "O2M gap is at least O2O gap; withdraw O2O-specific wording and treat this as a general TAL conflict-order issue."
                if o2m_rate is not None and o2m_rate >= o2o_rate
                else "O2M sidecar is weaker than O2O, but it remains a control and does not enter the O2O verdict."
            )
        ),
    }
    decision = {
        "audit": AUDIT_NAME,
        "verdict": verdict,
        "go": go,
        "gates": gates,
        "bootstrap": bootstrap,
        "checkpoint_metrics": metrics,
        "primary_effect_passes": [row["checkpoint"] for row in primary],
        "quality_passes": [row["checkpoint"] for row in quality_pass],
        "pooled_zero_quality_gap_share": zero_share,
        "dense_tiny_audit_trigger": dense,
        "o2m_sidecar": sidecar,
        "interpretation_limit": "GO only authorizes zero-training offline Phase 1; it does not establish AP gain or publishability.",
    }
    return decision, bootstrap


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(
    decision: dict[str, Any], quality: dict[str, Any], scans: list[dict[str, Any]], summary_rows: list[dict[str, Any]]
) -> str:
    bootstrap = decision["bootstrap"]
    zero_share_text = (
        "N/A (native-covered identities were not always rematchable in E_topk)"
        if decision["pooled_zero_quality_gap_share"] is None
        else f"{decision['pooled_zero_quality_gap_share'] * 100:.2f}%"
    )
    gap_label = (
        "avoidable gap"
        if decision["gates"].get("native_coverage_rematch_feasible", False)
        else "raw cardinality gap (not additive recoverable attribution)"
    )
    lines = [
        "# FCA-P0 final GT coverage audit",
        "",
        f"- Verdict: **{decision['verdict']}**",
        f"- Pooled O2O BF16 {gap_label}: {bootstrap['gap']}/{bootstrap['valid_gt']} "
        f"({bootstrap['rate'] * 100:.4f}%)",
        f"- Image-cluster bootstrap 95% CI: [{bootstrap['ci95_lower'] * 100:.4f}%, "
        f"{bootstrap['ci95_upper'] * 100:.4f}%]",
        f"- Zero-quality contribution share: {zero_share_text}",
        "",
        "## Frozen-checkpoint results",
        "",
        "| checkpoint | precision | valid GT | gap | gap rate | r>=0.50 gap | retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision["checkpoint_metrics"]:
        retention = "n/a" if row["r050_retention"] is None else f"{row['r050_retention'] * 100:.2f}%"
        lines.append(
            f"| {row['checkpoint']} | {row['precision']} | {row['valid_gt']} | {row['gap_r000']} | "
            f"{row['gap_rate'] * 100:.4f}% | {row['gap_r050']} | {retention} |"
        )
    lines.extend(["", "## Automatic gates", ""])
    for name, passed in decision["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## Native stage decomposition", ""])
    lines.extend(
        [
            "| checkpoint | precision | head | valid | S0 cov | S1 cov | S2 cov | S3 cov | S1→S2 lost | S1→S2 gained | added edges | off-graph final edges | S2→S3 lost | avoidable gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    overall = [row for row in summary_rows if row["group_type"] == "overall"]
    for row in sorted(overall, key=lambda item: (item["precision"], item["checkpoint_epoch"], item["head"])):
        lines.append(
            f"| {row['checkpoint']} | {row['precision']} | {row['head']} | {row['valid_gt']} | "
            f"{row['s0_coverage_rate'] * 100:.3f}% | {row['s1_coverage_rate'] * 100:.3f}% | "
            f"{row['s2_coverage_rate'] * 100:.3f}% | {row['s3_coverage_rate'] * 100:.3f}% | "
            f"{row['loss_s1_s2']} | {row['gain_s1_s2']} | {row['s2_added_edge_count']} | "
            f"{row['native_final_offgraph_edge_count']} | {row['loss_s2_s3']} | {row['recoverable_gap_gt']} |"
        )
    sidecar = decision["o2m_sidecar"]
    sidecar_rate = (
        "n/a" if sidecar["avoidable_gap_rate"] is None else f"{sidecar['avoidable_gap_rate'] * 100:.4f}%"
    )
    lines.extend(
        [
            "",
            "## O2M sidecar control",
            "",
            f"- O2M avoidable gap: {sidecar['avoidable_gap']}/{sidecar['valid_gt']} "
            f"({sidecar_rate}).",
            f"- {sidecar['interpretation']}",
            "",
            "## Evidence for and against continuing",
            "",
            f"- For: {sum(decision['gates'].values())}/{len(decision['gates'])} preregistered gates passed.",
            f"- Against: {sum(not value for value in decision['gates'].values())} gates failed; failed gates are "
            f"{[name for name, value in decision['gates'].items() if not value]}.",
            "- The dense-tiny cell trigger is explanatory only; its quality retention uses the same fixed "
            "identity-preserving unrestricted witness edge, not a unique subgroup-optimal r>=0.50 matching.",
        ]
    )
    lines.extend(
        [
            "",
            "## Quality evidence",
            "",
            f"```json\n{json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True)}\n```",
            "",
            "## Interpretation boundary",
            "",
            "This was a zero-training train-split assignment audit. It did not read VOC2007 test AP, update weights, "
            "or establish that any proposed matcher improves detection accuracy. A GO verdict authorizes only the "
            "pre-registered offline Phase 1 comparisons.",
            "Native S2 is reported as a non-monotone rewiring stage: the frozen implementation may introduce an edge "
            "outside E_topk via an all-GT overlap argmax. The main matching graph remains strictly E_topk; native "
            "off-graph pairs are never inserted into it.",
            "",
            "## Runtime scans",
            "",
        ]
    )
    for scan in scans:
        lines.append(
            f"- {scan['checkpoint']} {scan['precision']}: {scan['images']} images, {scan['gt']} GT, "
            f"{scan['elapsed_seconds']:.1f}s, geometry `{scan['geometry_sha256']}`"
        )
    return "\n".join(lines) + "\n"


def write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    global RUNTIME_LOG_PATH
    args = parse_args()
    protocol = protocol_record(args)
    if protocol["protocol_deviations"] and not args.smoke:
        raise RuntimeError(f"formal run has protocol deviations: {protocol['protocol_deviations']}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    RUNTIME_LOG_PATH = output_dir / "runtime.log"
    write_json(output_dir / "protocol.json", protocol)
    started_wall = time.time()
    started_iso = dt.datetime.now().astimezone().isoformat()
    object_handle = component_handle = image_handle = None
    try:
        kernel_preflight = kernel_fault_events(started_iso)
        write_json(output_dir / "kernel_health_start.json", kernel_preflight)
        if not kernel_preflight["valid"]:
            raise InvalidRunError(f"hardware fault appeared at audit start: {kernel_preflight['events']}")
        cpu_test_evidence = run_cpu_self_tests(output_dir)
        initialize_runtime(args.physical_device)
        source_root = Path(__file__).resolve().parents[1]
        data_yaml = Path(args.data).expanduser().resolve()
        source_provenance = validate_source_and_git(source_root, data_yaml)
        data_file_manifests = compute_data_file_manifests(EXPECTED_DATA_ROOT, args.skip_file_manifest_match)
        checkpoints = resolve_checkpoints(args.checkpoints, args.smoke)
        checkpoint_records = inspect_checkpoints(checkpoints)
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpu_count": torch.cuda.device_count(),
            "logical_gpu_name": torch.cuda.get_device_name(0),
            "inference_mode_required": True,
            "optimizer_forbidden": True,
            "backward_forbidden": True,
            **execution_affinity_and_gpu(),
        }
        provenance = {
            "source": source_provenance,
            "checkpoints": checkpoint_records,
            "runtime": runtime,
            "cpu_self_tests": cpu_test_evidence,
            "data_file_manifests": data_file_manifests,
        }
        write_json(output_dir / "provenance.json", provenance)

        object_handle = (output_dir / "objects.csv").open("w", newline="", encoding="utf-8-sig")
        component_handle = (output_dir / "components.csv").open("w", newline="", encoding="utf-8-sig")
        image_handle = (output_dir / "images.csv").open("w", newline="", encoding="utf-8-sig")
        writers = {
            "objects": csv.DictWriter(object_handle, fieldnames=OBJECT_FIELDS, extrasaction="raise"),
            "components": csv.DictWriter(component_handle, fieldnames=COMPONENT_FIELDS, extrasaction="raise"),
            "images": csv.DictWriter(image_handle, fieldnames=IMAGE_FIELDS, extrasaction="raise"),
        }
        for writer in writers.values():
            writer.writeheader()

        scans = []
        all_image_rows = []
        quality_values: dict[str, list[float]] = defaultdict(list)
        summary = SummaryAccumulator()
        scan_plan = [(record, "BF16") for record in checkpoint_records]
        if not args.smoke:
            scan_plan.append((next(record for record in checkpoint_records if record["filename"] == "last.pt"), "FP32"))
        for scan_index, (record, precision) in enumerate(scan_plan):
            log("checkpoint_start", checkpoint=record["filename"], precision=precision)
            scan, image_rows, quality = scan_checkpoint(
                record,
                precision,
                args,
                writers,
                summary,
                smoke_compare=args.smoke or scan_index == 0,
            )
            scans.append(scan)
            all_image_rows.extend(image_rows)
            for key, values in quality.items():
                quality_values[f"{record['filename']}:{precision}:{key}"].extend(values)
            object_handle.flush()
            component_handle.flush()
            image_handle.flush()
        object_handle.close()
        component_handle.close()
        image_handle.close()
        object_handle = component_handle = image_handle = None

        geometry_hashes = [scan["geometry_sha256"] for scan in scans]
        if len(set(geometry_hashes)) != 1:
            raise RuntimeError(f"ordered image/GT geometry changed across scans: {geometry_hashes}")
        summary_rows = summary.output()
        write_csv_rows(output_dir / "summary.csv", summary_rows)
        quality_summary = {key: quantiles(values) for key, values in sorted(quality_values.items())}
        write_json(output_dir / "quality_summary.json", quality_summary)
        write_json(output_dir / "scans.json", scans)

        if args.smoke:
            decision = {
                "audit": AUDIT_NAME,
                "verdict": "SMOKE_PASS",
                "go": None,
                "bitwise_control": scans[0]["bitwise_control"],
                "o2m_s2_s3_control": "PASS",
                "note": "Smoke does not evaluate FCA-P0 GO/NO-GO thresholds.",
            }
            bootstrap = None
        else:
            decision, bootstrap = make_decision(all_image_rows, summary_rows, geometry_hashes)
        write_json(output_dir / "decision.json", decision)
        write_csv_rows(
            output_dir / "decision.csv",
            [{"verdict": decision["verdict"], **decision.get("gates", {})}],
        )
        (output_dir / "report.md").write_text(
            report_markdown(decision, quality_summary, scans, summary_rows)
            if not args.smoke
            else "# FCA-P0 smoke\n\n**SMOKE_PASS**\n",
            encoding="utf-8",
        )
        # This is deliberately after bootstrap, summaries, decision and report generation. A fault at any earlier
        # point in the complete computation window invalidates the whole run before a completion marker is written.
        kernel_health = kernel_fault_events(started_iso)
        write_json(output_dir / "kernel_health.json", kernel_health)
        if not kernel_health["valid"]:
            raise InvalidRunError(
                f"new MCE/SIGBUS/EDAC event invalidated the entire run: {kernel_health['events']}"
            )
        completion = {
            "status": "COMPLETE",
            "verdict": decision["verdict"],
            "started_unix": started_wall,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started_wall,
            "geometry_sha256": geometry_hashes[0],
            "bootstrap": bootstrap,
            "kernel_health": kernel_health,
        }
        write_json(output_dir / "RUN_COMPLETE.json", completion)
        log("audit_complete", output_dir=str(output_dir), verdict=decision["verdict"])
        write_checksums(output_dir)
    except BaseException as error:
        for handle in (object_handle, component_handle, image_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()
        invalid = isinstance(error, InvalidRunError)
        failure = {
            "status": "INVALID" if invalid else "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "started_unix": started_wall,
            "failed_unix": time.time(),
        }
        marker = "RUN_INVALID.json" if invalid else "RUN_FAILED.json"
        write_json(output_dir / marker, failure)
        log("audit_invalid" if invalid else "audit_failed", error_type=type(error).__name__, error=str(error))
        write_checksums(output_dir)
        raise


if __name__ == "__main__":
    main()
