"""Compare YOLO26 checkpoints by O2O box, O2O class, and shared/O2M scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise KeyError(f"No model or ema object in {path}")
    return model.float().state_dict()


def scope(key: str) -> str:
    if ".one2one_cv2." in key:
        return "o2o_box"
    if ".one2one_cv3." in key:
        return "o2o_cls"
    return "shared_and_o2m"


def compare(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise KeyError(f"State keys differ. Missing={missing[:5]}, extra={extra[:5]}")
    groups = {
        name: {
            "tensors": 0,
            "elements": 0,
            "different_tensors": 0,
            "different_elements": 0,
            "max_abs": 0.0,
            "diff_sq": 0.0,
            "ref_sq": 0.0,
            "changed_keys": [],
        }
        for name in ("shared_and_o2m", "o2o_box", "o2o_cls")
    }
    for key, ref in reference.items():
        cand = candidate[key]
        group = groups[scope(key)]
        group["tensors"] += 1
        group["elements"] += ref.numel()
        if ref.is_floating_point() or ref.is_complex():
            delta = cand.to(torch.float64) - ref.to(torch.float64)
            different = delta != 0
            max_abs = float(delta.abs().max()) if delta.numel() else 0.0
            group["diff_sq"] += float(delta.square().sum())
            group["ref_sq"] += float(ref.to(torch.float64).square().sum())
        else:
            different = cand != ref
            max_abs = float((cand.to(torch.int64) - ref.to(torch.int64)).abs().max()) if ref.numel() else 0.0
        count = int(different.sum())
        if count:
            group["different_tensors"] += 1
            group["different_elements"] += count
            group["changed_keys"].append(key)
            group["max_abs"] = max(group["max_abs"], max_abs)
    for group in groups.values():
        group["relative_l2"] = (group.pop("diff_sq") / max(group.pop("ref_sq"), 1e-30)) ** 0.5
    return groups


def main():
    args = parse_args()
    reference = state_dict(args.reference)
    rows = []
    for path in args.candidates:
        rows.append(
            {
                "reference": str(Path(args.reference).resolve()),
                "candidate": str(Path(path).resolve()),
                "scopes": compare(reference, state_dict(path)),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(Path(row["candidate"]).name)
        for name, stats in row["scopes"].items():
            print(
                f"  {name}: {stats['different_tensors']}/{stats['tensors']} tensors, "
                f"{stats['different_elements']}/{stats['elements']} elements, "
                f"max_abs={stats['max_abs']:.8g}, rel_l2={stats['relative_l2']:.8g}"
            )


if __name__ == "__main__":
    main()
