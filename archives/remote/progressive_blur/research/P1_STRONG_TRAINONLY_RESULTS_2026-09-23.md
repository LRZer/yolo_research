# P1 stronger Oracle blur used only in training — first result

Date: 2026-09-23 (Asia/Shanghai). Status: training and independent raw-image
evaluation verified; further matched controls underway.

## Protocol and integrity

Scratch YOLOv8n, VOC P1 split (14,896 train / 1,655 development images),
seed 2027, 20 epochs, batch 64, 640 input, workers 4, disabled standard
augmentation. During training, GT boxes define a smooth strength map; all
validation/prediction images are unmodified. Compared with the previous Oracle
blur, only the Gaussian sigma bank changes from 0/1/2/4/8 to 0/2/4/8/16.
Mix maximum stays 0.5; epochs 1–5 are clean and epochs 6–10 ramp to 0.5.
The built-in trainer uses standard `DetectionValidator`, not the old
GT-conditioned Oracle validator. The model checkpoint contains only YOLOv8n,
not an inference-time blur front end.

Physical GPUs 4–7 only; CPU and memory bound to NUMA node 0. The manager
reported `verified_complete` at 03:13:54 CST, with exactly 20 CSV rows and
readable best/last weights. The current boot's `mc0` and `mc1` uncorrected
error counts stayed at zero for this run. This does not clear historical
hardware faults. A four-GPU 1%-fraction smoke and 19 unit tests passed before
formal training.

## Raw-development results

The same independent, unmasked `YOLO.val` evaluator was run on the scratch
baseline, the new stronger train-only model, and the old weak Oracle model.
Its primary comparison for the new model is:

| Checkpoint/input | P | R | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Scratch baseline `best.pt`, raw | 0.410728 | 0.295163 | 0.290927 | 0.161691 |
| Strong train-only blur `best.pt`, raw | 0.393578 | 0.305876 | 0.288812 | 0.157440 |
| Strong train-only blur `last.pt`, raw | 0.368317 | 0.293055 | 0.264570 | 0.143261 |

The new method's best checkpoint, selected at epoch 13 using **raw**
development images, is lower than the original scratch baseline by **0.425
percentage points** mAP50–95 and 0.211 points mAP50. The training CSV's
epoch-13 entry is mAP50–95 0.15717; the separate raw re-evaluation is 0.15744.
The small rerun discrepancy does not change the conclusion. P/R also vary
slightly between the in-training and standalone evaluators, so matched
standalone numbers are used in the table.

For context, the old weak Oracle's saved `best.pt` scores 0.161312
mAP50–95 on raw images. That checkpoint was chosen by **masked** validation,
so it is not yet a selection-matched weak control. A separate weak-blur,
raw-validation 20-epoch run has been launched to establish that comparison.

On 20 fixed VOC example images, doubling the sigma levels increased mean
absolute 8-bit pixel change from 1.768 to 2.429 intensity levels (+37.4%).
Thus the rendering change is real, but stronger background alteration did
not improve raw-image AP in this one-seed pilot. Because each image contains
large protected regions, global mean change is not a direct measure of
background-only blur strength.

## Interpretation and research boundary

The hypothesis “more aggressive target-protected blur during training helps
raw-image detection” is not supported by this setting. There is a small
recall gain and a larger precision loss against the baseline. This could
reflect background cues being useful, a train/predict appearance gap, or
augmentation strength; these explanations require controlled tests.
The result is one seed and only 20 epochs, not a definitive statement about
all blur strategies or full 125-epoch performance.

Two user-provided foveation papers were checked directly. The 2023
*Accelerating YOLO-based Real-time Object Detection via Foveated Rendering*
reports detection times on five selected images, not benchmark mAP; its
claimed speed change cannot establish that full-resolution pre-blurring
improves our task. The 2026 *Cost-Efficient Multi-Scale Fovea for
Semantic-Based Visual Search Attention* evaluates fixation/scanpath behavior
and changes the number of processed pixels. It is not an ordinary full-image
YOLO AP experiment; even its reported YOLOv11 latency improvement is modest
(0.144 to 0.115 s per attention iteration) compared with DETR. These papers
are useful conceptual precedents but do not validate the present mechanism.

## Artifacts

- Remote run: `/home/ubuntu/rz/progressive_blur_detection_research/runs/voc_p1/oracle_strong_trainonly_s16_e20_b64_s2027`
- Manager state: remote `research/p1_strong_trainonly_state.json`
- Independent raw result: `P1_STRONG_TRAINONLY_RAW_VALIDATION.json`
- Epoch curve: `P1_STRONG_TRAINONLY_RESULTS.csv`
- Strong visual manifest: `P1_STRONG_VISUAL_MANIFEST.csv`
- Protocol: `P1_STRONG_TRAINONLY_PROTOCOL_2026-09-23.md`
