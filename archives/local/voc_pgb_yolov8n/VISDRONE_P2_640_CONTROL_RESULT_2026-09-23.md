# Standard YOLOv8n-P2 / 640 VisDrone control — 2026-09-23

## Outcome

On the same 548-image official-source VisDrone val split, the **known
Ultralytics P2 architecture** improved scratch YOLOv8n/640 internal
converted-label mAP50–95 from **0.170547 to 0.198526** (+0.027979), but
remained below the conventional 960-input standard YOLOv8n at **0.232476**
(−0.033950). Every class improved versus 640 standard, and every class
remained below 960 standard. P2 is an architectural control, **not a novel
method**. Only one seed was run; these numbers are not official VisDrone
challenge AP and should not be compared to unrelated published protocols.

## Integrity and matched protocol

The task manager returned `verified_complete` at 05:37:22 CST. The saved
CSV has exactly 125 consecutive finite epoch rows; `best.pt` and `last.pt`
were loaded by the manager's checkpoint audit. The already-selected best
epoch is 115 (built-in mAP50–95 0.19824); epoch125 finished normally.
Independent `YOLO.val` on **raw** val images loaded `best.pt` and measured
the metrics below at `conf=.001`, NMS `.7`, `max_det=1000`, `half=False`.
The separate matched-box diagnostic completed on all 548 images and exactly
38,759 labels. No GT-guided Oracle processing occurred in evaluation.

| Property | Standard 640 | P2 640 | Standard 960 |
|---|---:|---:|---:|
| Detector | YOLOv8n scratch | Official YOLOv8n-P2 scratch | YOLOv8n scratch |
| Model outputs | P3/8–P5/32 | P2/4–P5/32 | P3/8–P5/32 |
| Train / val images | 6,471 / 548 | Same | Same |
| Seed / epochs / batch | 2027 / 125 / 64 | Same | Same |
| Device policy | Physical GPU4–7, NUMA0 | Same | Same |
| Best epoch | 115 | 115 | 111 |
| Independent val mAP50 | 0.309082 | **0.350077** | **0.403649** |
| Independent val mAP50–95 | 0.170547 | **0.198526** | **0.232476** |
| Cumulative CSV train time | 1,015.27 s | **1,674.71 s** | 1,573.05 s |
| Model summary at nc=10 | ~3.008 M, ~8.1 GFLOPs | **2.922 M, 12.2 GFLOPs** | Same structure as 640; higher spatial FLOPs at 960 |

The formal standard-640/P2-640 JSON configs differ only in `model` and
run `name`; train/val provenance, SGD, augmentation, image size, seed,
training budget and evaluator match. P2 took ~1.65× the 640 control's
cumulative CSV time and ~1.06× the 960 control's. This is **training
time**, not deployment latency. Its fewer parameters do not mean lower
compute: the P2 summary has more 640-input GFLOPs than standard 640.
End-to-end deployment timing has not yet been measured on a fixed device.

## Per-class independent AP50–95

| Class | Standard 640 | P2 640 | Standard 960 |
|---|---:|---:|---:|
| pedestrian | 0.1330 | 0.1778 | 0.2115 |
| people | 0.0851 | 0.1239 | 0.1340 |
| bicycle | 0.0244 | 0.0364 | 0.0632 |
| car | 0.4862 | 0.5319 | 0.5616 |
| van | 0.2359 | 0.2699 | 0.3264 |
| truck | 0.1542 | 0.1722 | 0.2129 |
| tricycle | 0.1029 | 0.1185 | 0.1442 |
| awning-tricycle | 0.0541 | 0.0672 | 0.0891 |
| bus | 0.2870 | 0.3137 | 0.3715 |
| motor | 0.1427 | 0.1737 | 0.2104 |

## Fixed-size GT recall diagnostic (not AP)

GT buckets were fixed once using the hypothetical **640-letterbox**
`sqrt(area)` of each GT, so object membership is identical across all
three runs. Predictions use low `conf=.001`, NMS `.7`, max1000 and
one-to-one same-class IoU matching. This tests a particular recall
operating point, not confidence ranking, precision or AP.

| 640-reference GT bucket | GTs | R@IoU .50, standard640 / P2-640 / standard960 | R@IoU .75, same order |
|---|---:|---:|---:|
| All | 38,759 | 0.6094 / **0.7328** / 0.7311 | 0.3299 / **0.4213** / 0.4299 |
| Tiny `<16 px` | 27,103 | 0.5065 / **0.6665** / 0.6605 | 0.1879 / **0.2921** / 0.2953 |
| Small `16–<32 px` | 8,724 | 0.8275 / **0.8757** / 0.8838 | 0.6064 / **0.6836** / 0.7056 |
| Medium `32–<96 px` | 2,897 | 0.9123 / **0.9216** / 0.9299 | 0.8208 / **0.8378** / 0.8561 |

P2 recovered roughly as many tiny GTs at IoU .50 as 960, yet had lower
AP50–95 and slightly lower tiny R@.75. **Inference:** P2's extra small-
object recall alone does not reproduce 960's overall quality; confidence
ranking/false positives, category ambiguity, localization quality or
training allocation could contribute. This is a hypothesis, not an
identified cause. Do not assume that a blur module will fix it.

## Hardware caveat and next decision

During the already-running P2 job, an unrelated process got SIGBUS from a
new NUMA1 memory fault at 05:23. P2 itself ran to completion and its
checkpoint was readable. A fresh gate after >10 quiet minutes allowed the
short independent validation above. At 05:43, **five additional deferred
uncorrectable memory errors** raised `mc1/ue_count` from 2 to 7; their EDAC
physical pages map to NUMA1 memory block71. There was no P2/validation
SIGBUS, but the machine is not healthy. Do **not** start the planned new
blur training while these faults recur; see
`HARDWARE_EVENT_2026-09-23_0456.md`. Publication-strength results must be
repeated on healthy hardware and across seeds.

Saved audit artifacts: `VISDRONE_OFFICIAL_P2_640_S2027_RESULTS.csv`,
`VISDRONE_OFFICIAL_P2_640_S2027_VAL_CLASSWISE.json`, and
`VISDRONE_OFFICIAL_P2_640_S2027_SIZE_RECALL.json`. Source YAML is the
official Ultralytics P2 definition copied into task-owned models with a
verified hash; source data and shared configuration were not modified.
