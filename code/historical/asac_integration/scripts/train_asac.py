"""Validate provenance and launch an isolated ASAC experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCHEMA = "ASAC-Phase1-v2"
ALLOWED_GPUS = "4,5,6,7"
PINNED_B50_ARGS_SHA256 = "d7a98308306ac9ca7d5691233cd80850cf5555d4452fd9926260b5239a5f46f8"
PINNED_MODEL_YAML_SHA256 = "afc4600876cc3a996aa65f5894d2a3005f25baa29b574bf46eff373779451b26"
PINNED_DATA_YAML_SHA256 = "5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40"
PINNED_DATA_ROOT = Path("/home/ubuntu/rz/yolo_study/datasets/VOC")
VOC_TRAIN_SPLITS = ("train2012", "train2007", "val2012", "val2007")
PINNED_VOC_TRAIN_MANIFEST = {
    "image_count": 16_551,
    "label_count": 16_551,
    "gt_lines": 40_058,
    "image_relative_path_space_size_sha256": "4a91fffa66e61125f414751d5bef2937457a5aeeb76a4e09fd6497c2be910cc0",
    "ordered_label_sha256sum_bundle": "39c8623c143706f2caede28dae7b5a248d9d2dd703aca12de043de942508c28a",
}
PINNED_CUSTOM_ASAC_SHA256 = {
    "freezer": "6edb0a71ba3706b5d75a93121301ddef7d196aa9c93335dda66dfdb6709424a1",
    "golden_script": "2897c908f23d5e630e8d998c1084ef56813389a7e91f865d1ff86f557f513402",
    "trainer": "e997d4fc3852203f487732ddc086d982231bf87044885265a33c9e5633f4fdf6",
    "runtime": "88a204beed3b18742bf4b03cd201252d499ef2da19827a7d31f8c1a78b5bc7aa",
}
PINNED_SOURCE_TREE_SHA256 = "135f65e164acc72d99f225b966a2da8c2bc754ec287e848b2db6a9ed7e4b8eee"
PINNED_P0_CHECKPOINT_SHA256 = "49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb"
PINNED_PANEL_CONTENT_SHA256 = "8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa"
PINNED_POPULATION_SHA256 = "f57cffdeb8bc29ef66a21c384ac0c3cd7ceff9552c48245263cb6c7c51b14926"
PINNED_GOLDEN_EXPECTED = {"O2M": 0.9545479413122414, "O2O": 0.9192160195274237}
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = (ROOT / "snapshot" / "ultralytics_research").resolve()
PINNED_CORE_HASHES = {
    "ultralytics/data/augment.py": "83e653664fa459beb66b9f245d90ca50ea25bc3c83caabfa9fab541d4abcb4b3",
    "ultralytics/data/dataset.py": "6c30496c691aefeb0646bea23a1519d2960c927b71dae10c370afcd9765f0e0e",
    "ultralytics/data/build.py": "1c3131adc60dc9763b9edcde853d3bc461ec908c2ed9d1965b2610bfd8c27847",
    "ultralytics/engine/trainer.py": "2c406877ac037a0b6691be1039f36750df3d7a675935627dafc4735a0b6decc7",
    "ultralytics/models/yolo/detect/train.py": "76d6b181259edc09b03f7cea1f9d59bd4eda2f7964146bbb2e76d4f7201dcf5f",
    "ultralytics/cfg/default.yaml": "1294dd04cdb28c7ecdbaf15dc01c9684b8365ba81be59f6ac47a5939bbd0e63a",
    "ultralytics/cfg/__init__.py": "ab0b03b1eb30bfe841fb551566d8b2e16c8cb7f6d38a3f21ef6bc81cc72f553e",
    "ultralytics/utils/loss.py": "6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9",
}
CUSTOM_ASAC_FILES = {
    "launcher": ROOT / "scripts" / "train_asac.py",
    "freezer": ROOT / "scripts" / "asac_protocol.py",
    "golden_script": ROOT / "scripts" / "smoke_asac_panel_golden.py",
    "trainer": EXPECTED_SOURCE / "ultralytics" / "models" / "yolo" / "detect" / "asac_train.py",
    "runtime": EXPECTED_SOURCE / "ultralytics" / "utils" / "asac.py",
}
SOURCE_MANIFEST_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SOURCE_MANIFEST_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(token: str):
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def load_json_strict(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )


def canonical_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def artifact_record(path: Path) -> dict:
    """Return an independently verifiable record for one required file."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required launch artifact not found: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}


def source_tree_manifest(root: Path) -> dict:
    """Hash every deterministic file in the isolated source tree using relative paths."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"isolated source tree not found: {root}")
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"isolated source tree must not contain symlinks: {path}")
        if (
            any(part in SOURCE_MANIFEST_EXCLUDED_PARTS for part in relative.parts)
            or path.suffix in SOURCE_MANIFEST_EXCLUDED_SUFFIXES
        ):
            continue
        if path.is_file():
            files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)})
    if not files:
        raise RuntimeError(f"isolated source tree is empty: {root}")
    return {
        "root": str(root),
        "policy": {
            "excluded_parts": sorted(SOURCE_MANIFEST_EXCLUDED_PARTS),
            "excluded_suffixes": sorted(SOURCE_MANIFEST_EXCLUDED_SUFFIXES),
        },
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "tree_sha256": canonical_sha256(files),
        "files": files,
    }


def voc_train_manifest(data_root: Path) -> dict:
    """Recompute the frozen VOC trainval path/size and label-content manifests."""
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"VOC data root not found: {data_root}")
    image_paths = sorted(
        (path for split in VOC_TRAIN_SPLITS for path in (data_root / "images" / split).glob("*.*") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    label_paths = sorted(
        (path for split in VOC_TRAIN_SPLITS for path in (data_root / "labels" / split).glob("*.txt") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    image_digest, label_digest = hashlib.sha256(), hashlib.sha256()
    for path in image_paths:
        relative = path.relative_to(data_root).as_posix()
        image_digest.update(f"{relative} {path.stat().st_size}\n".encode("utf-8"))
    gt_lines = 0
    for path in label_paths:
        relative = path.relative_to(data_root).as_posix()
        label_digest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
        with path.open("rb") as handle:
            gt_lines += sum(1 for line in handle if line.strip())
    return {
        "data_root": str(data_root),
        "train_splits_in_yaml_order": list(VOC_TRAIN_SPLITS),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "gt_lines": gt_lines,
        "image_relative_path_space_size_sha256": image_digest.hexdigest(),
        "ordered_label_sha256sum_bundle": label_digest.hexdigest(),
    }


def exact_output_directory(train: dict, formal: bool) -> Path:
    """Resolve and reserve the exact ASAC output name without allowing implicit name increments."""
    project, name = train.get("project"), train.get("name")
    if not project or not name:
        raise ValueError("ASAC requires explicit project and name")
    project_path, name_path = Path(str(project)), Path(str(name))
    if not project_path.is_absolute():
        raise ValueError("ASAC project must be an absolute path")
    if name_path.is_absolute() or len(name_path.parts) != 1 or name_path.name in {"", ".", ".."}:
        raise ValueError("ASAC name must be one non-empty path component")
    if train.get("exist_ok") is not False:
        raise ValueError("every ASAC launch requires exist_ok=false")
    lexical_target = project_path / name_path
    if os.path.lexists(lexical_target):
        raise FileExistsError(f"ASAC target already exists; refusing implicit name increment: {lexical_target}")
    target = lexical_target.resolve()
    return target


def reject_external_distributed_environment(environ: dict[str, str]) -> None:
    """Require this singleton launcher, rather than torchrun or a scheduler rank, to own DDP spawning."""
    inherited = {key: environ[key] for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if key in environ}
    if inherited:
        raise RuntimeError(f"ASAC launcher must not run inside an existing distributed job: {inherited}")


def persist_json_exclusive(path: Path, value: dict) -> None:
    """Atomically and exclusively publish a durable JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        if os.name == "posix":
            directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def collect_and_verify_code_evidence(formal: bool) -> tuple[dict, dict]:
    """Hash all custom/code inputs and enforce formal trust roots before importing any of them."""
    custom_files = {name: artifact_record(path) for name, path in sorted(CUSTOM_ASAC_FILES.items())}
    source_manifest = source_tree_manifest(EXPECTED_SOURCE)
    if formal:
        actual_custom = {name: custom_files[name]["sha256"] for name in PINNED_CUSTOM_ASAC_SHA256}
        if actual_custom != PINNED_CUSTOM_ASAC_SHA256:
            raise RuntimeError(
                f"formal ASAC custom source hash mismatch: expected={PINNED_CUSTOM_ASAC_SHA256}, "
                f"actual={actual_custom}"
            )
        if source_manifest["tree_sha256"] != PINNED_SOURCE_TREE_SHA256:
            raise RuntimeError(
                "formal ASAC isolated source tree hash mismatch: "
                f"expected={PINNED_SOURCE_TREE_SHA256}, actual={source_manifest['tree_sha256']}"
            )
    return custom_files, source_manifest


def resolve_golden_artifact(asac: dict, formal: bool, source_tree_sha256: str) -> tuple[dict, dict]:
    """Validate and remove the formal zero-update panel gate before constructing a trainer."""
    asac = dict(asac)
    golden_file = asac.pop("golden_artifact", None)
    if not formal:
        if golden_file is not None:
            raise ValueError("nonformal ASAC runs must not claim a formal golden artifact")
        return asac, {}
    if golden_file is None:
        raise ValueError("formal ASAC requires an explicit golden_artifact")
    golden_path = Path(golden_file).resolve()
    if not golden_path.is_file():
        raise FileNotFoundError(f"formal ASAC golden artifact not found: {golden_path}")
    golden = load_json_strict(golden_path)
    if not isinstance(golden, dict):
        raise TypeError("formal ASAC golden artifact must be a JSON object")
    if golden.get("schema") != "asac-panel-golden-smoke-v1" or golden.get("status") != "PASS":
        raise RuntimeError("formal ASAC golden artifact schema/status mismatch")
    pinned_top_level = {
        "checkpoint_sha256": PINNED_P0_CHECKPOINT_SHA256,
        "data_sha256": PINNED_DATA_YAML_SHA256,
        "absolute_tolerance": 1e-12,
        "physical_gpu": 4,
        "visible_gpu_count": 1,
    }
    actual_top_level = {key: golden.get(key) for key in pinned_top_level}
    if actual_top_level != pinned_top_level:
        raise RuntimeError(
            f"formal ASAC golden identity mismatch: expected={pinned_top_level}, actual={actual_top_level}"
        )
    if golden.get("expected") != PINNED_GOLDEN_EXPECTED:
        raise RuntimeError("formal ASAC golden expected stability values drifted")

    measurement, panel = golden.get("measurement"), golden.get("panel")
    if not isinstance(measurement, dict) or not isinstance(panel, dict):
        raise RuntimeError("formal ASAC golden artifact lacks measurement/panel mappings")
    if measurement.get("panel_content_sha256") != PINNED_PANEL_CONTENT_SHA256:
        raise RuntimeError("formal ASAC golden measurement panel hash mismatch")
    pinned_panel = {
        "panel_content_sha256": PINNED_PANEL_CONTENT_SHA256,
        "population_count": 624,
        "population_sha256": PINNED_POPULATION_SHA256,
        "dataset_length": 16_551,
        "loader_workers": 4,
        "loader_pin_memory": True,
    }
    actual_panel = {key: panel.get(key) for key in pinned_panel}
    if actual_panel != pinned_panel:
        raise RuntimeError(f"formal ASAC golden panel mismatch: expected={pinned_panel}, actual={actual_panel}")
    protocol = panel.get("protocol")
    if not isinstance(protocol, dict) or {
        "expected_panel_content_sha256": protocol.get("expected_panel_content_sha256"),
        "expected_population_count": protocol.get("expected_population_count"),
        "expected_population_sha256": protocol.get("expected_population_sha256"),
    } != {
        "expected_panel_content_sha256": PINNED_PANEL_CONTENT_SHA256,
        "expected_population_count": 624,
        "expected_population_sha256": PINNED_POPULATION_SHA256,
    }:
        raise RuntimeError("formal ASAC golden embedded panel protocol mismatch")

    heads, differences = measurement.get("heads"), golden.get("differences")
    if not isinstance(heads, dict) or not isinstance(differences, dict) or set(differences) != {"O2M", "O2O"}:
        raise RuntimeError("formal ASAC golden head/difference records are incomplete")
    for head, expected in PINNED_GOLDEN_EXPECTED.items():
        record = heads.get(head)
        difference = differences[head]
        if not isinstance(record, dict) or isinstance(difference, bool) or not isinstance(difference, (int, float)):
            raise RuntimeError(f"formal ASAC golden {head} record is invalid")
        actual = record.get("mean_stability")
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise RuntimeError(f"formal ASAC golden {head} stability is invalid")
        recomputed = float(actual) - expected
        if (
            not math.isfinite(float(difference))
            or abs(float(difference)) > 1e-12
            or recomputed != float(difference)
            or record.get("assigned_both") != 624
            or record.get("object_pairs") != 624
            or record.get("assignment_coverage") != 1.0
        ):
            raise RuntimeError(f"formal ASAC golden {head} locked measurement mismatch")
    autocast = measurement.get("autocast")
    expected_autocast = {
        "autocast_dtype": "torch.bfloat16",
        "autocast_enabled": True,
        "inference_mode": True,
        "input_dtype": "torch.float32",
        "o2m_boxes_dtype": "torch.bfloat16",
        "o2m_scores_dtype": "torch.bfloat16",
        "o2o_boxes_dtype": "torch.bfloat16",
        "o2o_scores_dtype": "torch.bfloat16",
    }
    if autocast != expected_autocast or measurement.get("strength") != 0.5:
        raise RuntimeError("formal ASAC golden BF16/inference contract mismatch")
    if measurement.get("assigner_oom_fallback") != {"O2M": False, "O2O": False}:
        raise RuntimeError("formal ASAC golden used or omitted an assigner OOM fallback")
    if not is_sha256(measurement.get("rng_sha256")):
        raise RuntimeError("formal ASAC golden RNG integrity hash is invalid")
    integrity = measurement.get("integrity")
    integrity_keys = {
        "parameters_sha256",
        "registered_buffers_sha256",
        "bn_buffers_sha256",
        "extra_tensor_attributes_sha256",
    }
    if (
        not isinstance(integrity, dict)
        or set(integrity) != integrity_keys
        or any(not is_sha256(integrity[key]) for key in integrity_keys)
    ):
        raise RuntimeError("formal ASAC golden model-integrity evidence is invalid")

    criterion = golden.get("criterion_contract")
    if not isinstance(criterion, dict) or {
        "criterion_class": criterion.get("criterion_class"),
        "head_class": criterion.get("head_class"),
        "head_runtime_end2end": criterion.get("head_runtime_end2end"),
        "dual_head_modules_present": criterion.get("dual_head_modules_present"),
    } != {
        "criterion_class": "ultralytics.utils.loss.E2ELoss",
        "head_class": "ultralytics.nn.modules.head.Detect",
        "head_runtime_end2end": False,
        "dual_head_modules_present": True,
    }:
        raise RuntimeError("formal ASAC golden criterion/head contract mismatch")
    branches = criterion.get("branches")
    expected_branches = {"one2many": (10, 10), "one2one": (7, 1)}
    if not isinstance(branches, dict) or set(branches) != set(expected_branches):
        raise RuntimeError("formal ASAC golden criterion branches mismatch")
    for name, (topk, topk2) in expected_branches.items():
        branch = branches[name]
        if not isinstance(branch, dict) or {
            "loss_class": branch.get("loss_class"),
            "assigner_class": branch.get("assigner_class"),
            "topk": branch.get("topk"),
            "topk2": branch.get("topk2"),
            "stride": branch.get("stride"),
        } != {
            "loss_class": "ultralytics.utils.loss.v8DetectionLoss",
            "assigner_class": "ultralytics.utils.tal.TaskAlignedAssigner",
            "topk": topk,
            "topk2": topk2,
            "stride": [8.0, 16.0, 32.0],
        }:
            raise RuntimeError(f"formal ASAC golden {name} criterion contract mismatch")

    golden_script = golden.get("golden_script")
    source_records = golden.get("source_records")
    isolated_manifest = golden.get("isolated_source_manifest")
    if not isinstance(golden_script, dict) or golden_script.get("sha256") != PINNED_CUSTOM_ASAC_SHA256["golden_script"]:
        raise RuntimeError("formal ASAC golden generator script hash mismatch")
    asac_source = source_records.get("ultralytics/utils/asac.py") if isinstance(source_records, dict) else None
    if not isinstance(asac_source, dict) or asac_source.get("sha256") != PINNED_CUSTOM_ASAC_SHA256["runtime"]:
        raise RuntimeError("formal ASAC golden runtime source hash mismatch")
    if not isinstance(isolated_manifest, dict) or isolated_manifest.get("tree_sha256") != source_tree_sha256:
        raise RuntimeError("formal ASAC golden isolated source tree hash mismatch")
    return asac, {"artifact": artifact_record(golden_path), "validated": True}


def build_launch_record(
    *, spec: dict, spec_path: Path, train: dict, asac: dict, base_path: Path, base_sha256: str,
    output_directory: Path, schedule_evidence: dict, golden_evidence: dict, formal: bool,
    prior_code_evidence: tuple[dict, dict] | None = None,
) -> dict:
    """Capture all code and configuration inputs needed to audit a launch."""
    model_path, data_path = Path(str(train.get("model", ""))), Path(str(train.get("data", "")))
    if not model_path.is_absolute() or not data_path.is_absolute():
        raise ValueError("ASAC model and data YAML paths must be absolute")
    custom_files, source_manifest = collect_and_verify_code_evidence(formal)
    if prior_code_evidence is not None and (custom_files, source_manifest) != prior_code_evidence:
        raise RuntimeError("ASAC code changed between pre-import verification and launch-record construction")
    base_record = artifact_record(base_path)
    if base_record["sha256"] != base_sha256:
        raise RuntimeError("B50 args changed after initial verification")
    model_record, data_record = artifact_record(model_path), artifact_record(data_path)
    pinned_yaml = {"model_yaml": PINNED_MODEL_YAML_SHA256, "data_yaml": PINNED_DATA_YAML_SHA256}
    actual_yaml = {"model_yaml": model_record["sha256"], "data_yaml": data_record["sha256"]}
    if formal and actual_yaml != pinned_yaml:
        raise RuntimeError(f"formal ASAC model/data YAML hash mismatch: expected={pinned_yaml}, actual={actual_yaml}")
    data_manifest = None
    if formal:
        if PINNED_DATA_ROOT.resolve() != PINNED_DATA_ROOT:
            raise RuntimeError(f"pinned VOC data root unexpectedly resolves elsewhere: {PINNED_DATA_ROOT}")
        data_manifest = voc_train_manifest(PINNED_DATA_ROOT)
        actual_data_manifest = {key: data_manifest[key] for key in PINNED_VOC_TRAIN_MANIFEST}
        if actual_data_manifest != PINNED_VOC_TRAIN_MANIFEST:
            raise RuntimeError(
                f"formal ASAC VOC trainval manifest mismatch: expected={PINNED_VOC_TRAIN_MANIFEST}, "
                f"actual={actual_data_manifest}"
            )
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": spec,
        "resolved_train": train,
        "resolved_asac": asac,
        "asac_payload_sha256": canonical_sha256(asac),
        "expected_output_directory": str(output_directory),
        "artifacts": {
            "config": artifact_record(spec_path),
            "base_args": {**base_record, "pinned_sha256": PINNED_B50_ARGS_SHA256},
            "model_yaml": {**model_record, "pinned_sha256": PINNED_MODEL_YAML_SHA256},
            "data_yaml": {**data_record, "pinned_sha256": PINNED_DATA_YAML_SHA256},
            "voc_train_manifest": data_manifest,
            "custom_asac_files": custom_files,
            "pinned_custom_asac_sha256": PINNED_CUSTOM_ASAC_SHA256,
            "frozen_schedule": schedule_evidence or None,
            "panel_golden_gate": golden_evidence or None,
        },
        "isolated_source_manifest": {**source_manifest, "pinned_tree_sha256": PINNED_SOURCE_TREE_SHA256},
    }
    return {"record": record, "record_sha256": canonical_sha256(record)}


def _load_freezer_module():
    path = ROOT / "scripts" / "asac_protocol.py"
    actual_sha256 = file_sha256(path)
    if actual_sha256 != PINNED_CUSTOM_ASAC_SHA256["freezer"]:
        raise RuntimeError(
            "refusing to execute an unpinned ASAC freezer: "
            f"expected={PINNED_CUSTOM_ASAC_SHA256['freezer']}, actual={actual_sha256}"
        )
    name = "_asac_protocol_launch_verifier"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load ASAC freezer: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def resolve_control_schedule(asac: dict, train: dict, evidence: dict | None = None) -> dict:
    """Validate a formal freezer artifact and replace file references with a strict epoch->q map."""
    asac = dict(asac)
    variant = asac.get("variant")
    formal = bool(asac.get("formal", False))
    control = variant in {"A1", "A3", "A4"}
    schedule_file = asac.pop("schedule_file", None)
    manifest_file = asac.pop("freeze_manifest_file", None)
    state_files = asac.pop("a2_state_files", None)
    trusted_manifest_sha256 = asac.pop("trusted_freeze_manifest_sha256", None)
    trusted_state_sha256 = asac.pop("trusted_a2_state_sha256_by_seed", None)
    if not control:
        if any(value is not None for value in (
            schedule_file, manifest_file, state_files, trusted_manifest_sha256, trusted_state_sha256
        )):
            raise ValueError(f"{variant} must not reference frozen control schedules")
        return asac
    if formal and any(value is None for value in (
        schedule_file, manifest_file, state_files, trusted_manifest_sha256, trusted_state_sha256
    )):
        raise ValueError(
            f"formal {variant} requires schedule/manifest/A2 files and explicit trusted manifest/state hashes"
        )
    if schedule_file is None:
        return asac

    schedule_path, manifest_path = Path(schedule_file).resolve(), Path(manifest_file).resolve()
    if not schedule_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen schedule artifacts: {schedule_path}, {manifest_path}")
    rows, manifest = load_json_strict(schedule_path), load_json_strict(manifest_path)
    actual_manifest_sha256 = file_sha256(manifest_path)
    if trusted_manifest_sha256 is not None and actual_manifest_sha256 != trusted_manifest_sha256:
        raise RuntimeError(
            f"freeze manifest differs from its explicit trust root: expected={trusted_manifest_sha256}, "
            f"actual={actual_manifest_sha256}"
        )
    seed = int(train["seed"])
    if not isinstance(rows, list) or len(rows) != 50 or not isinstance(manifest, dict):
        raise ValueError("frozen schedule must have 50 rows and a mapping manifest")
    if manifest.get("schema") != "asac-phase1-freezer-v2":
        raise ValueError("unexpected freezer manifest schema")
    epochs = [row.get("epoch") for row in rows if isinstance(row, dict)]
    if epochs != list(range(1, 51)):
        raise ValueError("frozen schedule epochs must be exactly ordered 1..50")
    if any(row.get("variant") != variant or int(row.get("seed", -1)) != seed for row in rows):
        raise ValueError("frozen schedule variant/seed does not match the experiment")
    expected_schedule_sha = manifest.get("output_schedule_sha256", {}).get(variant, {}).get(str(seed))
    actual_schedule_sha = file_sha256(schedule_path)
    if actual_schedule_sha != expected_schedule_sha:
        raise RuntimeError(
            f"frozen schedule hash mismatch: expected={expected_schedule_sha}, actual={actual_schedule_sha}"
        )
    freezer_path = ROOT / "scripts" / "asac_protocol.py"
    actual_freezer_sha = file_sha256(freezer_path)
    if actual_freezer_sha != manifest.get("freezer_source_sha256"):
        raise RuntimeError("freezer source no longer matches freeze_manifest.json")
    input_hashes = manifest.get("input_state_sha256_by_seed")
    if not isinstance(input_hashes, dict) or set(input_hashes) != {"0", "1", "42"}:
        raise RuntimeError("freezer manifest lacks all three A2 input hashes")
    if state_files is not None:
        if not isinstance(state_files, dict) or set(state_files) != {"0", "1", "42"}:
            raise ValueError("a2_state_files must map exactly seeds 0, 1, and 42")
        state_paths = {int(seed_key): Path(path).resolve() for seed_key, path in state_files.items()}
        if any(not path.is_file() for path in state_paths.values()):
            raise FileNotFoundError(f"missing A2 state artifact(s): {state_paths}")
        raw_by_seed = {state_seed: path.read_bytes() for state_seed, path in state_paths.items()}
        actual_state_sha256 = {
            str(state_seed): hashlib.sha256(raw).hexdigest() for state_seed, raw in sorted(raw_by_seed.items())
        }
        if trusted_state_sha256 is not None:
            if not isinstance(trusted_state_sha256, dict) or set(trusted_state_sha256) != {"0", "1", "42"}:
                raise ValueError("trusted_a2_state_sha256_by_seed must map exactly seeds 0, 1, and 42")
            if actual_state_sha256 != trusted_state_sha256:
                raise RuntimeError(
                    "A2 state artifacts differ from their explicit trust roots: "
                    f"expected={trusted_state_sha256}, actual={actual_state_sha256}"
                )
        freezer = _load_freezer_module()
        rebuilt = freezer.freeze_jsonl_bytes_by_seed(raw_by_seed)
        rebuilt_rows = list(rebuilt.schedules[variant][seed])
        if rebuilt.manifest != manifest:
            raise RuntimeError("freeze manifest differs from independent reconstruction of the three A2 states")
        if rebuilt_rows != rows:
            raise RuntimeError(f"{variant} schedule differs from independent reconstruction of the three A2 states")
        if evidence is not None:
            evidence.update({
                "schedule": artifact_record(schedule_path),
                "freeze_manifest": artifact_record(manifest_path),
                "a2_states": {
                    str(state_seed): artifact_record(path) for state_seed, path in sorted(state_paths.items())
                },
                "independent_reconstruction": True,
                "trusted_freeze_manifest_sha256": trusted_manifest_sha256,
                "trusted_a2_state_sha256_by_seed": trusted_state_sha256,
            })
    elif formal:
        raise ValueError(f"formal {variant} requires independent reconstruction from a2_state_files")
    schedule = {int(row["epoch"]): float(row["train_q"]) for row in rows}
    if any(not 0.25 <= value <= 1.0 for value in schedule.values()):
        raise ValueError("frozen schedule q outside [0.25,1.0]")
    schedule_projection = [{"epoch": epoch, "train_q": schedule[epoch]} for epoch in range(1, 51)]
    asac["schedule"] = schedule
    asac["schedule_manifest"] = {
        "schema": manifest["schema"],
        "variant": variant,
        "target_seed": seed,
        "schedule_file": str(schedule_path),
        "schedule_file_sha256": actual_schedule_sha,
        "schedule_q_sha256": canonical_sha256(schedule_projection),
        "freeze_manifest_file": str(manifest_path),
        "freeze_manifest_file_sha256": actual_manifest_sha256,
        "freezer_source_sha256": actual_freezer_sha,
        "input_state_sha256_by_seed": input_hashes,
    }
    return asac


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise TypeError("ASAC experiment spec must be a mapping")
    allowed = {"id", "hypothesis", "base_args", "train_overrides", "asac"}
    if set(spec) - allowed:
        raise ValueError(f"unexpected experiment keys: {sorted(set(spec) - allowed)}")
    if not isinstance(spec.get("train_overrides", {}), dict) or not isinstance(spec.get("asac"), dict):
        raise TypeError("train_overrides and asac must be mappings")
    return spec


def main() -> None:
    reject_external_distributed_environment(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.config.resolve()
    spec = load_spec(spec_path)

    base_path = Path(spec["base_args"]).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"pinned B50 args not found: {base_path}")
    actual_base_sha = file_sha256(base_path)
    if actual_base_sha != PINNED_B50_ARGS_SHA256:
        raise RuntimeError(
            f"B50 args hash mismatch: expected={PINNED_B50_ARGS_SHA256}, actual={actual_base_sha}"
        )
    train = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(train, dict):
        raise TypeError("B50 args must be a mapping")
    train.pop("save_dir", None)
    overrides = dict(spec.get("train_overrides", {}))
    formal = bool(spec["asac"].get("formal", False))
    preimport_code_evidence = collect_and_verify_code_evidence(formal)
    formal_allowed = {"name", "project", "exist_ok"}
    if formal and set(overrides) - formal_allowed:
        raise ValueError(f"formal run may override only {sorted(formal_allowed)}, got {sorted(overrides)}")
    train.update(overrides)
    if train.get("exist_ok") is not False:
        raise ValueError("every ASAC launch requires exist_ok=false")
    train["exist_ok"] = False
    if str(train.get("device")) != ALLOWED_GPUS:
        raise ValueError(f"ASAC may use only physical GPUs {ALLOWED_GPUS}")
    if int(train.get("workers", -1)) != 4:
        raise ValueError("ASAC requires workers=4")
    if train.get("pretrained") is not False or train.get("resume") is not False:
        raise ValueError("ASAC requires pretrained=False and resume=False to match B50")
    if os.environ.get("ULTRALYTICS_PIN_MEMORY", "1") == "0":
        raise RuntimeError("pinned memory is disabled by ULTRALYTICS_PIN_MEMORY=0")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "launch train_asac.py with CUDA_VISIBLE_DEVICES unset; the locked device=4,5,6,7 setting must select "
            "physical GPUs directly rather than an inherited logical remapping"
        )

    output_directory = exact_output_directory(train, formal)
    schedule_evidence = {}
    asac_input, golden_evidence = resolve_golden_artifact(
        dict(spec["asac"]), formal, preimport_code_evidence[1]["tree_sha256"]
    )
    asac = resolve_control_schedule(asac_input, train, schedule_evidence)
    if asac.get("schema") != SCHEMA:
        raise ValueError(f"expected ASAC schema {SCHEMA}")
    # Round-trip through duplicate-key rejecting JSON so the exact DDP environment payload is canonical and strict.
    payload = json.dumps(asac, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    asac = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    os.environ["YOLO_ASAC_CONFIG_JSON"] = json.dumps(
        asac, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )

    sys.path.insert(0, str(EXPECTED_SOURCE))
    import ultralytics
    from ultralytics.models.yolo.detect.asac_train import ASACTrainer

    imported_source = Path(ultralytics.__file__).resolve().parents[1]
    if imported_source != EXPECTED_SOURCE:
        raise RuntimeError(f"expected isolated source {EXPECTED_SOURCE}, imported {imported_source}")
    core_mismatches = {}
    for relative, expected in PINNED_CORE_HASHES.items():
        path = EXPECTED_SOURCE / relative
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected:
            core_mismatches[relative] = {"expected": expected, "actual": actual}
    if core_mismatches:
        raise RuntimeError(f"isolated source core drifted from the locked reference: {core_mismatches}")

    launch_record = build_launch_record(
        spec=spec,
        spec_path=spec_path,
        train=train,
        asac=asac,
        base_path=base_path,
        base_sha256=actual_base_sha,
        output_directory=output_directory,
        schedule_evidence=schedule_evidence,
        golden_evidence=golden_evidence,
        formal=formal,
        prior_code_evidence=preimport_code_evidence,
    )
    sidecar = output_directory.parent / f"{output_directory.name}.launch_record.json"
    persist_json_exclusive(sidecar, launch_record)
    sidecar_sha256 = file_sha256(sidecar)
    print(json.dumps({
        "launch_record": str(sidecar),
        "launch_record_file_sha256": sidecar_sha256,
        "record_sha256": launch_record["record_sha256"],
        "source_tree_sha256": launch_record["record"]["isolated_source_manifest"]["tree_sha256"],
    }, indent=2, sort_keys=True))
    exact_output_directory(train, formal)  # Close the preflight-to-constructor window as much as possible.
    trainer = ASACTrainer(overrides=train)
    actual_output_directory = Path(trainer.save_dir).resolve()
    if actual_output_directory != output_directory:
        raise RuntimeError(
            "Ultralytics changed the requested output directory; refusing to train: "
            f"expected={output_directory}, actual={actual_output_directory}"
        )
    trainer.train()
    if file_sha256(sidecar) != sidecar_sha256:
        raise RuntimeError(f"prelaunch record changed during training: {sidecar}")
    persist_json_exclusive(output_directory / "launch_record.json", launch_record)


if __name__ == "__main__":
    main()
