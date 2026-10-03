# P1 strong Oracle blur on 50% of training images

Date: 2026-09-23 (Asia/Shanghai). Exploratory, one seed and 20 epochs.
The formal run was pre-registered in
`P1_STRONG_P50_TRAINONLY_PROTOCOL_2026-09-23.md` before training.

## Integrity and protocol

The task manager reported `verified_complete` at 03:47:26 CST: 20 finite,
contiguous epoch rows, readable best and last checkpoints. Physical GPUs
4–7, NUMA 0. Uncorrected EDAC counters for both NUMA nodes were 0 after
the run, without implying past hardware faults are fixed. The four-GPU
1%-fraction smoke and 20 unit/integration tests had passed beforehand.

The detector is scratch YOLOv8n on VOC P1 (14,896 train / 1,655 development),
seed 2027, 640 input, batch 64, workers 4 and 20 epochs. Its training
blur settings match the completed strong-100% run: GT-guided box protection,
Gaussian sigma levels 0/2/4/8/16, residual mix cap 0.5, five clean-start
epochs and five ramp epochs. The only planned parameter change is per-image
blur probability 1.0 → 0.5 after the clean start. Unchosen training images
remain exactly raw; all development images remain raw at every epoch and in
independent evaluation. The trained checkpoint is just YOLOv8n, with no
Oracle component at inference.

## Same-evaluator comparison on raw development images

| `best.pt` | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Original scratch baseline | 0.410728 | 0.295163 | 0.290927 | 0.161691 |
| Weak blur, 100% of training images | 0.359327 | 0.329336 | 0.288983 | 0.161312 |
| Strong blur, 100% of training images | 0.393578 | 0.305876 | 0.288812 | 0.157440 |
| Strong blur, 50% of training images | 0.405032 | 0.298566 | 0.289761 | 0.161610 |

The 50% model is **0.417 percentage points above** strong-100% in raw
mAP50–95, but **0.008 points below** the original baseline: essentially a
tie, not a demonstrated improvement. The built-in raw development best was
epoch 12 (mAP50–95 0.16178); standalone re-evaluation of `best.pt` gives
0.161610. The `last.pt` is only 0.143021, so checkpoint choice is material.

The result is consistent with the possibility that mixing raw views reduces
the train/validation appearance gap. It does **not** isolate that mechanism:
the probability change also halves the effective blur dose and consumes a
different random-number stream. It is one seed, short training, one dataset.
No method in this table convincingly outperforms the baseline. The
classwise raw re-evaluation is in
`P1_CLASSWISE_RAW_ANALYSIS_2026-09-23.md`: some classes gain while others
lose, including persistent train/aeroplane deficits. It does not identify a
causal mechanism.

## Research decision

Do not continue a grid search over blur probability, Gaussian sigma, and
mix strength on the VOC P1 development set. The current evidence shows a
harmful strong blur that can be neutralized by lower exposure, not a gain
worthy of a paper. The limited development split and prior art in
training-only background smoothing (CVPR 2025 SET) further argue against
claiming novelty from this augmentation alone. A more promising next gate
is a domain-specific, diagnosed failure mode, with clean official data
splits and conventional controls before inventing a new module.

Artifacts: `P1_STRONG_P50_TRAINONLY_RAW_VALIDATION.json`,
`P1_STRONG_P50_TRAINONLY_RESULTS.csv`,
`P1_STRONG_P50_TRAINONLY_STATE.json`; remote run
`/home/ubuntu/rz/progressive_blur_detection_research/runs/voc_p1/oracle_strong_p50_trainonly_s16_e20_b64_s2027`.
