# P1 extreme GT-box blackout: masked versus raw-input results

Date: 2026-09-23 (Asia/Shanghai)  
Status: 20-epoch training verified complete; independent raw-image evaluation complete  
Interpretation: privileged mechanism diagnostic, **not** a deployable detector

## Question and protocol

The extreme control keeps every pixel within the union of the current image's
ground-truth boxes unchanged and sets all pixels outside that union to exact
black. There is no protective margin, gradual transition or residual blend.
YOLOv8n is trained from its architecture (`yolov8n.yaml`, no pretrained
weights) on the fixed P1 VOC split: 14,896 training and 1,655 development
images, 20 classes. The settings match the P1 scratch baseline: seed 2027,
20 epochs, batch 64, 640-pixel input, workers 4, identical optimizer policy
and disabled geometric/color/mixed-image augmentation.

The training loader uses the GT-box mask. The training run's built-in
per-epoch validation **also** uses the GT-box mask. This deliberately shows
what the detector can do when box locations are revealed in its input. In a
separate evaluation, the saved blackout-trained weights are run on the
unaltered development images through the standard YOLO validator. This is the
requested train-masked / prediction-unmasked condition; the validation images
in that separate test are never masked or used to construct model input.

The model checkpoint does not contain an automatic Oracle preprocessor. A
plain standalone prediction from `best.pt` receives raw input unless someone
explicitly applies the external mask.

## Hardware and run integrity

Two formal attempts bound to NUMA node 1 failed before producing any epoch
result because the kernel reported new uncorrected hardware memory errors and
sent SIGBUS to Python processes. The third, bounded attempt used physical
GPUs 4–7 and bound CPU and memory allocations to NUMA node 0. Process
`numa_maps` confirmed `bind:0`. Its manager reported `verified_complete` and
20/20 epochs. The result CSV contains one header plus 20 epoch rows; best and
last weights were readable and evaluated. No new EDAC errors appeared during
this run: at final inspection, node-0 controller `mc0` remained 0 UE/0 CE,
while `mc1` remained 6 UE/0 CE. Both node-1 failures and the older evidence of
node-0 faults remain relevant. NUMA 0 is relatively safer in the current boot,
not proven fault-free. No system-wide hardware or memory settings were changed.

## Results

All metrics below are on the **development split**, not VOC2007 test. `P` and
`R` are the evaluator's aggregate precision and recall; mAP50–95 is COCO-style
IoU-threshold averaging as implemented by the installed Ultralytics version.

| Weights / evaluation input | P | R | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Original scratch baseline `best.pt`, raw images | 0.41073 | 0.29516 | 0.29093 | 0.16169 |
| Blackout-trained `best.pt`, GT-box-masked images | 0.52022 | 0.48237 | 0.49652 | 0.44016 |
| **Same blackout-trained `best.pt`, raw images** | **0.12021** | **0.10554** | **0.04408** | **0.01418** |
| Blackout-trained `last.pt`, raw images | 0.11463 | 0.08542 | 0.04222 | 0.01394 |

The masked-validation best checkpoint is epoch 14. Its raw-image mAP50–95 is
**14.751 percentage points below** the baseline re-evaluated with the exact
same raw-image settings (0.01418 versus 0.16169). The raw-image mAP50 deficit
is 24.685 percentage points. For the *same blackout-trained best checkpoint*,
removing the GT mask drops mAP50–95 from 0.44016 to 0.01418, or 42.598
percentage points. The masked result therefore cannot be reported as ordinary
target-detection performance: the mask leaks near-exact target-box locations
into the pixels seen by the detector.

### Additional raw-input checkpoints

| Saved epoch (one-based) | Raw mAP50 | Raw mAP50–95 |
|---:|---:|---:|
| 1 | 0.00065 | 0.00014 |
| 6 | 0.03080 | 0.00901 |
| 11 | 0.04308 | 0.01370 |
| 16 | 0.03993 | 0.01334 |
| Masked-selected best (epoch 14) | 0.04408 | 0.01418 |
| Last (epoch 20) | 0.04222 | 0.01394 |

All six available checkpoint types were independently re-evaluated on raw
images. The masked-selected best is the highest among these saved weights, but
not every one of the 20 individual epoch weights was saved. Thus this is not
an exhaustive search for the raw-selected epoch. In any case, the checked
weights are all far below the original scratch baseline.

The baseline re-evaluation (mAP50–95 0.161691) closely reproduces its
original logged value (0.16173), providing a control that the standard raw
evaluation uses the expected data and metric path.

## Interpretation and next decision

Pure GT-box blackout is useful as a stress test of whether background
suppression can make a box-aware input easier to detect. Its apparently large
masked-validation gain is largely uninterpretable as a detection improvement:
the black-background transitions encode information about *where* objects
are. When that privileged cue disappears at prediction time, the model's
performance collapses. This exact train-only extreme masking is therefore
not a promising practical intervention. It does not refute milder training
augmentation or a genuinely image-predicted mask, but those must be evaluated
without GT-derived input at validation/test time.

The current run did not use raw validation for *per-epoch checkpoint selection*;
the raw evaluations were performed after training. Repeating 20 epochs merely
to select a checkpoint by raw AP is not justified by the present severe gap.
If this line is revisited, a new protocol should make the train/validation
input asymmetry explicit from the start and compare against a matched
augmentation-only control.

## Files

- Remote project: `/home/ubuntu/rz/progressive_blur_detection_research`
- Run and checkpoints: `runs/voc_p1/oracle_blackout_scratch_e20_b64_s2027/`
- Manager state: `research/p1_blackout_state.json`
- Masked comparison: `research/p1_blackout_comparison.json`
- Independent raw evaluation: `research/p1_blackout_raw_validation.json`
- Raw baseline control: `research/p1_baseline_raw_validation.json`
- Evaluation program: `tools/evaluate_blackout_raw.py`
- Earlier failed attempt preserved under `runs/voc_p1/_failed_mce_oracle_blackout_20260923_022350/`
