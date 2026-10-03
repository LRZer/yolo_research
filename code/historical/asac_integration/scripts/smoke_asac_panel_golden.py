"""Single-GPU zero-update golden check for the frozen ASAC training-time panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "snapshot" / "ultralytics_research"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SOURCE]
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402
import ultralytics  # noqa: E402

from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.utils.asac import ASACPanelEvaluator, ASACPanelProtocol, file_sha256  # noqa: E402
from train_asac import source_tree_manifest  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = "49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb"
EXPECTED_DATA_YAML_SHA256 = "5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40"
ABSOLUTE_TOLERANCE = 1e-12
EXPECTED_SOURCE_HASHES = {
    "ultralytics/utils/asac.py": "88a204beed3b18742bf4b03cd201252d499ef2da19827a7d31f8c1a78b5bc7aa",
    "ultralytics/utils/loss.py": "6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9",
    "ultralytics/utils/tal.py": "9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861",
    "ultralytics/data/build.py": "1c3131adc60dc9763b9edcde853d3bc461ec908c2ed9d1965b2610bfd8c27847",
    "ultralytics/models/yolo/detect/train.py": "76d6b181259edc09b03f7cea1f9d59bd4eda2f7964146bbb2e76d4f7201dcf5f",
    "ultralytics/nn/modules/head.py": "1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8",
}
EXPECTED = {
    "O2M": 0.9545479413122414,
    "O2O": 0.9192160195274237,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: dict) -> None:
    """Publish a complete result once without a shared-directory overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def class_path(value) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def validate_native_contract(model, criterion) -> dict:
    """Prove the same native YOLO26 dual-head/assigner contract frozen by ASAC-P0."""
    if class_path(criterion) != "ultralytics.utils.loss.E2ELoss":
        raise RuntimeError(f"expected E2ELoss, got {class_path(criterion)}")
    expected = {"one2many": (10, 10), "one2one": (7, 1)}
    branches = {}
    for name, expected_topk in expected.items():
        branch = getattr(criterion, name, None)
        assigner = getattr(branch, "assigner", None)
        if branch is None or assigner is None:
            raise RuntimeError(f"criterion lacks {name}.assigner")
        actual_topk = (getattr(assigner, "topk", None), getattr(assigner, "topk2", None))
        strides = [float(value) for value in branch.stride.detach().cpu().tolist()]
        if actual_topk != expected_topk or strides != [8.0, 16.0, 32.0]:
            raise RuntimeError(f"{name} native contract drift: topk={actual_topk}, strides={strides}")
        branches[name] = {
            "loss_class": class_path(branch),
            "assigner_class": class_path(assigner),
            "topk": actual_topk[0],
            "topk2": actual_topk[1],
            "stride": strides,
        }
    head = model.model[-1]
    dual = all(getattr(head, name, None) is not None for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"))
    # P0's training-format checkpoint contract keeps dual heads present while runtime inference selection is O2M
    # (`end2end=False`). Training mode still returns both O2M/O2O dictionaries.
    if class_path(head) != "ultralytics.nn.modules.head.Detect" or not dual or bool(head.end2end):
        raise RuntimeError("model lacks the native end-to-end O2M/O2O Detect head")
    return {
        "criterion_class": class_path(criterion),
        "branches": branches,
        "head_class": class_path(head),
        "head_runtime_end2end": bool(head.end2end),
        "dual_head_modules_present": dual,
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("golden smoke requires CUDA_VISIBLE_DEVICES=4 (physical GPU 4 only)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("golden smoke requires exactly one visible CUDA GPU")
    checkpoint = Path(args.checkpoint).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite golden smoke output: {output}")
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint drift: expected={EXPECTED_CHECKPOINT_SHA256}, actual={checkpoint_sha256}"
        )
    data_sha256 = file_sha256(data)
    if data_sha256 != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError(f"data YAML drift: expected={EXPECTED_DATA_YAML_SHA256}, actual={data_sha256}")
    imported_source = Path(ultralytics.__file__).resolve().parents[1]
    if imported_source != SOURCE.resolve():
        raise RuntimeError(f"expected isolated source {SOURCE.resolve()}, imported {imported_source}")
    source_records = {}
    for relative, expected_sha256 in EXPECTED_SOURCE_HASHES.items():
        path = SOURCE / relative
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"P0 source drift for {relative}: expected={expected_sha256}, actual={actual_sha256}")
        source_records[relative] = {"path": str(path.resolve()), "sha256": actual_sha256}

    runtime_project = output.parent / f".{output.stem}_trainer_runtime"
    if runtime_project.exists() or runtime_project.is_symlink():
        raise FileExistsError(f"refusing a reused golden trainer runtime: {runtime_project}")
    overrides = {
        "model": str(checkpoint),
        "data": str(data),
        "imgsz": 640,
        "batch": 64,
        "workers": 4,
        "epochs": 50,
        "device": "0",
        "pretrained": True,
        "save": False,
        "plots": False,
        "val": False,
        "project": str(runtime_project),
        "name": "panel_golden",
        "exist_ok": False,
        "seed": 42,
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
    expected_save_dir = (runtime_project / "panel_golden").resolve()
    if Path(trainer.save_dir).resolve() != expected_save_dir:
        raise RuntimeError(f"golden trainer output was renamed: expected={expected_save_dir}, actual={trainer.save_dir}")
    trainer.setup_model()
    if file_sha256(checkpoint) != checkpoint_sha256 or file_sha256(data) != data_sha256:
        raise RuntimeError("checkpoint or data YAML changed while the trainer was constructed")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("zero-update golden smoke unexpectedly created an optimizer")
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    trainer.ema = SimpleNamespace(ema=trainer.model)

    evaluator = ASACPanelEvaluator(trainer, ASACPanelProtocol())
    criterion_contract = validate_native_contract(trainer.model, evaluator.criterion)
    measurement = evaluator.measure()
    if file_sha256(checkpoint) != checkpoint_sha256 or file_sha256(data) != data_sha256:
        raise RuntimeError("checkpoint or data YAML changed during golden measurement")
    differences = {
        head: float(measurement["heads"][head]["mean_stability"]) - expected
        for head, expected in EXPECTED.items()
    }
    for head, difference in differences.items():
        if not math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
            raise RuntimeError(
                f"{head} golden mismatch: expected={EXPECTED[head]:.17g}, "
                f"actual={measurement['heads'][head]['mean_stability']:.17g}, delta={difference:.3g}"
            )
    result = {
        "schema": "asac-panel-golden-smoke-v1",
        "created_unix": time.time(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(data),
        "data_sha256": data_sha256,
        "golden_script": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        "imported_source": str(imported_source),
        "source_records": source_records,
        "isolated_source_manifest": source_tree_manifest(SOURCE),
        "criterion_contract": criterion_contract,
        "physical_gpu": 4,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "expected": EXPECTED,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "differences": differences,
        "measurement": measurement,
        "panel": evaluator.loader_metadata,
        "status": "PASS",
        "trainer_runtime_project": str(runtime_project.resolve()),
        "trainer_save_dir": str(expected_save_dir),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": getattr(ultralytics, "__version__", None),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "device_name": torch.cuda.get_device_name(0),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
            "device_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        },
    }
    write_json_exclusive(output, result)
    print(json.dumps({"status": "PASS", "output": str(output), "differences": differences}, sort_keys=True))


if __name__ == "__main__":
    main()
