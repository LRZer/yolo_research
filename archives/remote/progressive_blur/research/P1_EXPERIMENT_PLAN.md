# P1 real-data experiment plan — frozen 2026-09-23

## Purpose

P1 is a mechanism-screening experiment, not the final paper result. It asks
whether the fully integrated learned progressive-blur path produces a credible
signal on a held-out VOC development split before spending a 100–125 epoch,
multi-seed budget.

## Fixed data

- Source images/labels: existing shared VOC tree, read-only.
- Train: 14,896 images from VOC07+12 trainval.
- Development validation: 1,655 images held out deterministically from trainval.
- Final test: 4,952 VOC2007 test images, not touched during P1 selection.
- Split seed: 2027.
- Maximum per-class image-fraction deviation from the 10% development target:
  0.004348.
- Exact manifests and SHA-256 values live in the remote `data_splits` audit.
- Train, development and test manifests traverse separate task-local symlink
  views so their Ultralytics `.cache` files cannot overwrite one another and are
  never created inside the shared VOC tree.

## Initialization decision

Use YOLOv8n source YAML with random initialization (`pretrained=false`) for both
groups, honoring the earlier project decision not to use pretrained detection
weights. Pretraining can later be tested as a separate robustness setting; it
must never be mixed between comparison groups.

## Paired P1 groups

- A0: unmodified Ultralytics YOLOv8n.
- D: YOLOv8n plus the 38,513-parameter learned progressive-blur front end.

Both use seed 2027, identical detector construction order, data order, optimizer,
input size, batch size, weak augmentations and 20-epoch budget. D uses map-loss
weight 1.0, final rho 0.5, five zero-rho epochs and a five-epoch linear ramp.

## Common training settings

- epochs: 20
- global batch: 64
- image size: 640
- physical GPUs: 4, 5, 6, 7 only
- NUMA: CPU and memory node 1
- workers: 4 per DDP rank
- AMP: BF16
- seed: 2027, deterministic mode
- optimizer: Ultralytics `auto`, resolved choice captured in logs
- no Mosaic, MixUp, CutMix, CopyPaste, geometric jitter, color jitter or flips
- validation every epoch on the fixed 1,655-image development set
- checkpoints every 5 epochs

The intentionally weak augmentation policy isolates the mechanism. Any positive
result must later survive a stronger standard-training baseline.

## Gates

1. Four-GPU real-VOC PGB smoke at 1% train fraction and one epoch.
2. Confirm no OOM, NaN, MCE/SIGBUS, invalid labels or missing map-loss columns.
3. Run A0 then D under the frozen 20-epoch configs.
4. Audit every results row and checkpoint before comparing metrics.
5. Do not use VOC2007 test to choose rho, lambda, epoch or architecture.

## Interpretation rule

P1 can justify continuing, but cannot establish publishable improvement. A useful
signal requires a later identity-wrapper control, at least two seeds, longer
training, stronger augmentation, final untouched test evaluation, per-class and
size/error analysis, and latency/memory reporting.
