# SARC assignment histogram probe specification

## Scientific question

Before implementing SARC, test whether YOLO26n's training-relevant P3/P4/P5 assignments contain a level-specific and temporally stable signal in

\[
u=\log_2\frac{\sqrt{wh}}{\mathrm{stride}_s}.
\]

If the distributions are indistinguishable, assignment-conditioned receptive-field allocation has no measured substrate. If they fluctuate strongly between adjacent checkpoints on the identical image panel, an EMA calibration target is unlikely to be trustworthy.

## Frozen protocol

- Checkpoints: B50 epoch 11, 21, 31, 41 and 50 (`epoch10.pt`, `epoch20.pt`, `epoch30.pt`, `epoch40.pt`, `last.pt`).
- Data: first eight batches (512 images at batch 64) from VOC trainval via the train split, loaded with validation-mode transforms to disable augmentation and shuffling; `rect=False` prevents aspect-ratio sorting.
- Runtime when executed later: one allowed physical GPU from 4–7, exactly workers=4, pinned memory required, BF16 autocast, inference mode, BN evaluation mode.
- Mutations: no optimizer, no backward pass, no weight update, no BN-stat update.
- Primary assignment: O2M TaskAlignedAssigner only. The YOLO26 Detect head detaches O2O feature inputs, so O2O does not train the shared backbone/neck targeted by SARC.
- Histogram: fixed `u` edges `[-4, 8]` with width `0.5`; under/overflow are counted and clipped into edge bins for probability mass conservation.
- Divergence: Jensen-Shannon with base-2 logarithms, range `[0,1]`.
- Panel identity: ordered image paths plus batch indices/classes/boxes are SHA-256 hashed; all checkpoints must match exactly or analysis aborts.

## Reliable P3/P4/P5 mapping

This mapping does not infer levels from target size or rely only on stride equality:

1. `Detect.forward_head` flattens `feats[i]` and concatenates levels in list order.
2. `make_anchors` walks the identical `feats` list and concatenates exactly `H_i*W_i` anchors per level.
3. The probe creates contiguous level IDs from the observed `H_i*W_i` values.
4. It independently obtains `stride_tensor` from `make_anchors` and requires every element in each block to equal the corresponding head stride.
5. It requires exactly three strides, exactly `[8,16,32]`, before naming them P3/P4/P5. Any disagreement is a hard error.

For a positive anchor, the assigner's returned pixel-space `target_bboxes` supplies `w,h`; no reconstruction from soft target scores is used.

## Two counting views

1. `positive_anchor` (primary): each positive TAL anchor contributes once. This matches the assignment-weighted signal that a training-time SARC histogram would see.
2. `unique_gt_per_image_level` (sensitivity): each `(image, local GT index, level)` contributes once. This removes TAL top-k multiplicity. If the primary view passes but this view is indistinguishable, the report warns that anchor multiplicity rather than object scale may drive the apparent structure.

For both views, report per checkpoint and level: sample count/share, mean/std, min/max, P10/P25/P50/P75/P90, histogram counts/probabilities, under/overflow, and per-batch count mean/std/CV.

## Pre-registered decision

The primary decision uses positive-anchor histograms:

- Pool the five fixed-checkpoint histograms separately for P3/P4/P5.
- Similarity NO-GO if **all** pooled layer pairs are nearly identical, operationalized as `max(JS(P3,P4), JS(P3,P5), JS(P4,P5)) < 0.02`.
- Instability NO-GO if **any** same-level JS between adjacent checkpoints is `>0.30`.
- If any pooled level has fewer than 100 positives, return INCONCLUSIVE rather than forcing GO/NO-GO.
- Otherwise return GO.

This exact interpretation resolves the prose “任意两层 JS <0.02”: calibration remains potentially useful when one level is distinct even if the other two overlap. Per-checkpoint between-level JS and all-pair cross-epoch JS remain in the output for auditing but do not silently alter the gate.

## Outputs

The script emits:

- one structured JSON containing method metadata, checkpoint summaries, every JS comparison and the decision;
- level-statistics CSV, JS CSV and one-row decision CSV;
- a human-readable Markdown report with mapping evidence, core distributions, stability and the gate outcome.

Script: `work/remote_research_source/scripts/probe_sarc_assignment_histograms.py`.
