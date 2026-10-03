# Background-blur YOLOv8n research: status at 06:50 CST, 2026-09-23

This is a navigation snapshot, **not** a new result. Primary logs,
locked protocols, raw evaluator outputs and hardware incidents are
linked by file name below. The local copy lives under
`outputs/voc_pgb_yolov8n/`; a mirror is in the task-owned remote
`/home/ubuntu/rz/progressive_blur_detection_research/research/`.

## What has actually been measured

On VOC, same seed2027, YOLOv8n random initialization, 20 epochs, raw
original-image re-evaluation of selected checkpoints:

| VOC variant | mAP50–95 | Interpretation |
|---|---:|---|
| No blur | 0.161691 | Matched short-run baseline |
| Weak Oracle, train only | 0.161312 | No gain |
| Strong Oracle, all train images | 0.157440 | Decreased AP |
| Strong Oracle, half train images | 0.161610 | Nearly baseline, no demonstrated gain |

These are short experiments, not evidence that *all* forms of selective
background processing are ineffective. Full controls:
`P1_STRONG_P50_TRAINONLY_RESULTS_2026-09-23.md` and
`P1_CLASSWISE_RAW_VALIDATION.json`.

On official-source VisDrone train6471/val548, 38,759 converted val
boxes, random-initialized seed2027, 125 epochs, matched optimizer/batch:

| Conventional control | Raw-val internal mAP50–95 | Role |
|---|---:|---|
| Standard YOLOv8n / 640 | 0.170547 | Conventional baseline |
| Official P2-head YOLOv8n / 640 | 0.198526 | Known high-resolution-head control, **not innovation** |
| Standard YOLOv8n / 960 | **0.232476** | Current strong conventional baseline |

Increasing input size from 640 to 960 produced a +0.061929 absolute
one-seed internal AP gain, concentrated in tiny-GT recall diagnostics;
P2 recovered similar very-low-threshold GT counts but remained below
960 AP. Its epoch100→125 curve gained only ~0.00109, so the observed
gap is not credibly explained solely by an unfinished 125-epoch run.
CPU-only paired prediction diagnostics found P2's IoU-.50 matched GTs
at confidence .001 about equal to 960 (28,399 vs 28,349), but P2 had
more unmatched candidates (439,265 vs 352,433) and fewer high-score
TPs at confidence .5 (11,629 vs 13,591). This suggests a ranking/
quality component, **not a proved causal mechanism**. Converted labels
omit ignored regions; all of these are internal, not official VisDrone
challenge scores. See `VISDRONE_P2_640_CONTROL_RESULT_2026-09-23.md`,
`VISDRONE_640_VS_960_RESULT_2026-09-23.md`, and
`VISDRONE_P2_VS_960_CPU_QUALITY_DIAGNOSTIC_2026-09-23.md`.

## Current experiment and locked next analysis

At 06:46, one formal YOLOv8n scratch/960/125ep train-only strong
GT-protected blur run started on physical GPU4–7, NUMA0. Seed2027,
batch64, same source train/val, optimizer and detector as the standard
960 control; only the task run name and `oracle_*` treatment differ.
Training images are blurred with probability0.5 after a five-epoch
zero/five-epoch ramp, with sigma bank 0/2/4/8/16, mix0.5. Validation
is always on raw images without GT preprocessing. The earlier 1%-data
one-epoch 4-GPU real-path smoke passed; it has no AP interpretation.
Formal manager PID67146 was alive at initial check; **no formal AP is
known as of this snapshot**. Locked plan:

1. Monitor at roughly 10-minute intervals; never duplicate a live
   manager or use GPUs0–3. Audit 125 consecutive finite CSV rows and
   readable best/last after manager completion.
2. Independently re-evaluate the selected best checkpoint on all 548
   raw val images; report mAP50, mAP50–95, all ten classes and the same
   fixed 640-reference size buckets. Compare the matched standard960
   checkpoint first; test-dev is not a tuning set.
3. Use the **pre-result locked secondary** paired CPU score/TP/FP
   diagnostic to inspect why any AP difference occurred. Do not call a
   secondary win a primary success. Positive one-seed screening signals
   need a second seed; negative results should trigger failure analysis,
   not post hoc blur-strength fishing.

Protocol: `VISDRONE_TRAIN_ONLY_BLUR_960_PROTOCOL_2026-09-23.md`.
Manager state/events/log: remote research directory, tag
`visdrone_oracle_blur_p50_960_s2027`.

## Novelty and hardware limits

The exact Oracle intervention is **training augmentation**, not an
inference-time learned front end. Existing IA-YOLO, DENet, IAT, SET,
foveation, QueryDet and UGS literature blocks a broad claim of
“adaptive preprocessing/background smoothing/high-resolution small
objects” as novel. A publishable direction still needs a reproducible,
specific failure mode, distinct mechanism, matched strong controls,
multiple seeds and ideally another dataset/model. The Desktop-paper
readings and primary-source caveats are in
`DESKTOP_LITERATURE_REVIEW_2026-09-23.md` and
`DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`.

The shared server has suffered multiple uncorrectable-memory bursts;
EDAC was `mc0=0/mc1=12` at the latest 06:43 gate. The user had the
owner's permission to keep using it, but that is not evidence of
healthy hardware. The strict no-new-error gate failed; a documented
narrower NUMA0/GPU4–7 continuation decision preceded smoke and formal
launch after 35/45 quiet minutes respectively. See
`HARDWARE_EVENT_2026-09-23_0456.md`. Even a successful experiment here
is provisional until rerun on healthy hardware. No global memory,
daemon, dataset or other-user process was altered by this task.
