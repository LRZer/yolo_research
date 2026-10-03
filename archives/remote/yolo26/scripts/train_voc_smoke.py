"""Run a short, from-scratch YOLO26n/VOC training smoke test.

This script is intentionally a pipeline check, not a formal experiment:
it trains on only 1% of the VOC training images for one epoch and does not
run per-epoch validation. The framework may still run one final validation
pass. Crucially, it builds the model from the YAML architecture with
``pretrained=False`` -- no .pt weights are loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "ultralytics_research"
VOC_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "datasets" / "VOC.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short YOLO26n/VOC training check from random weights.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for this check.")
    parser.add_argument("--fraction", type=float, default=0.01, help="Fraction of VOC training images to use.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="GPU ID, or 'cpu'.")
    parser.add_argument("--name", default="voc_yolo26n_smoke", help="Run name under the local runs directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or not 0 < args.fraction <= 1:
        raise ValueError("epochs, batch and imgsz must be positive; fraction must be in (0, 1].")
    if not VOC_YAML.is_file():
        raise FileNotFoundError(f"VOC configuration not found: {VOC_YAML}")

    print("=== YOLO26n / VOC smoke test ===")
    print(f"Data YAML: {VOC_YAML}")
    print("Model YAML: yolo26n.yaml (random initialization; no .pt weights)")
    print(f"Train fraction: {args.fraction:.1%}; epochs: {args.epochs}; per-epoch validation: disabled")
    print("Precision: BF16 AMP (native on this RTX 4060; no FP16 self-check)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and str(args.device) != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(int(args.device))}")

    model = YOLO("yolo26n.yaml", task="detect")
    model.train(
        data=str(VOC_YAML),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # Reliable on Windows.
        pretrained=False,
        amp="bf16",  # Native on RTX 4060; avoids the overly strict FP16 AMP self-check.
        val=False,  # Skip per-epoch validation; the framework may still perform one final validation pass.
        seed=42,
        deterministic=True,
        project=str(PROJECT_ROOT / "runs"),
        name=args.name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
