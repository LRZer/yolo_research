# P1 weak Oracle blur, training only: matched control

Date: 2026-09-23 (Asia/Shanghai). This is a 20-epoch, one-seed exploratory
control, not evidence of a publishable improvement.

## Matched protocol

Scratch YOLOv8n, VOC P1 14,896 training / 1,655 development images,
seed 2027, 640 input, batch 64, 20 epochs, workers 4. Physical GPUs 4–7,
NUMA 0. Standard augmentation is off. Training boxes define Oracle background
blur; every validation and standalone evaluation uses the original, unmasked
images. The weak sigma bank is 0/1/2/4/8, `mix=0.5`, epochs 1–5 clean and
6–10 ramped. The only planned change against the strong train-only run is
the sigma bank (strong: 0/2/4/8/16). The manager audited exactly 20 finite,
contiguous CSV epochs and readable best/last checkpoints; its status is
`verified_complete` at 03:33:09 CST. Both EDAC uncorrected counters were 0
at the subsequent check; historical faults remain relevant.

## Matched raw-image evaluation

Same evaluator, unchanged development split and settings for all three rows:

| 20-epoch run / raw `best.pt` | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Scratch baseline | 0.410728 | 0.295163 | 0.290927 | 0.161691 |
| Weak Oracle, train only | 0.359327 | 0.329336 | 0.288983 | 0.161312 |
| Strong Oracle, train only | 0.393578 | 0.305876 | 0.288812 | 0.157440 |

The weak run's best raw mAP50–95 is 0.000379 below baseline (0.038
percentage points), i.e. essentially tied at this one-seed precision. The
strong run is 0.003872 below weak (0.387 percentage points) and 0.004251
below baseline (0.425 points). Under this matched protocol, increasing blur
strength hurts, rather than helps, raw-image AP. Weak blur raises recall and
lowers precision at the evaluator's operating point relative to the baseline;
that alone does not identify which classes or object sizes drove the AP result.
The weak run's `last.pt` is 0.143904 mAP50–95, so checkpoint choice matters.

Interestingly, the old weak Oracle experiment's checkpoints give identical
raw-image metrics despite its **masked** per-epoch validation; its checkpoint
files have different hashes and cannot be called identical artifacts. The
matching metric values are consistent with the same model trajectory and
the selected epoch coinciding here, but the matched train-only run is the
cleaner evidence because checkpoint selection explicitly used raw images.

## Interpretation and next decision

Simple target-protected training blur has not beaten the scratch baseline
on unmodified VOC development images. A plausible mechanism is a train/test
appearance gap, but this has not yet been causally isolated. The next
pre-registered diagnostic is the same strong blur applied to a random 50%
of training images, leaving the other 50% exactly original; validation stays
raw. This changes augmentation probability, not sigma or detector. If it
fails, stop tuning this Oracle family on VOC P1 and prioritize a diagnosed
failure mode in a domain dataset, rather than asserting blur novelty.

Artifacts: `P1_WEAK_TRAINONLY_RAW_VALIDATION.json`,
`P1_WEAK_TRAINONLY_RESULTS.csv`, `P1_WEAK_TRAINONLY_STATE.json`; remote run
`/home/ubuntu/rz/progressive_blur_detection_research/runs/voc_p1/oracle_weak_trainonly_s8_e20_b64_s2027`.
