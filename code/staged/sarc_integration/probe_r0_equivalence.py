"""Emit a deterministic CPU fingerprint for cross-snapshot R0 equivalence checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(repo))

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


def tensor_digest(named_tensors) -> str:
    """Hash ordered tensor names, dtypes, shapes, and raw CPU values."""
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(9192026)
cfg = YAML.load(repo / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml")
cfg["scale"] = "n"
model = DetectionModel(cfg, nc=3, verbose=False)
model.args = get_cfg(
    overrides={
        "epochs": 50,
        "o2o_kd_gain": 0.0,
        "tqgb_cls_m2o_gain": 0.0,
        "tqgb_cls_o2m_gain": 0.0,
        "tqgb_direction": "none",
    }
)
state_sha256 = tensor_digest(model.state_dict().items())
graph = [(type(layer).__name__, layer.f, layer.np) for layer in model.model]
image = torch.linspace(0, 1, 2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
batch = {
    "img": image,
    "batch_idx": torch.tensor([0, 1], dtype=torch.long),
    "cls": torch.tensor([[0.0], [1.0]]),
    "bboxes": torch.tensor([[0.30, 0.35, 0.22, 0.18], [0.68, 0.62, 0.25, 0.20]]),
}
loss, items = model(batch)
loss.sum().backward()
gradients = ((name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None)
payload = {
    "graph": graph,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "state_sha256": state_sha256,
    "loss": loss.detach().cpu().tolist(),
    "loss_items": {name: float(value.detach()) for name, value in items.items()},
    "gradient_sha256": tensor_digest(gradients),
}
print(json.dumps(payload, sort_keys=True))
