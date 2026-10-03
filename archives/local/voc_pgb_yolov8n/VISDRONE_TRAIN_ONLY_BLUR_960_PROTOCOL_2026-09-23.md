# Pre-registered VisDrone train-only Oracle blur at 960 — 2026-09-23

Written before the proposed experiment's smoke or formal result. This is a
diagnostic continuation of the target-protected-background-blur hypothesis,
**not a deployable learned front end**: GT boxes generate the training-only
mask; raw images enter every validation/test and eventual independent eval.

## Motivation and locked comparison

VOC/20-epoch strong-blur train-only controls yielded no AP gain. VisDrone
has much smaller GTs, denser scenes and a strong conventional 960-input
scratch YOLOv8n baseline; we ask whether train-only protected blur helps
there, rather than assuming a VOC negative universally transfers.

- Baseline: official-source VisDrone train 6,471 / val 548, standard YOLOv8n
  scratch, 960 input, 125 epochs, seed 2027, batch 64, SGD, default
  augmentation, physical GPU4–7/NUMA0. The existing best checkpoint's
  independently recomputed raw-val internal converted-label mAP50–95 is
  0.232476; mAP50 is 0.403649; fixed-reference `<16 px` GT IoU-.50 recall
  is 0.6605. These numbers and caveats precede this protocol.
- Candidate: exactly the same detector, lists, seed, 125 epochs, image size,
  batch, optimizer and validation, but apply box-protected blur to each
  training image with probability 0.5. GT is used **only** inside train batch
  preprocessing. The selected blur uses Gaussian sigma 0/2/4/8/16 pixels,
  mix 0.5, 10% box margin, 0.5 box-size-normalized transition and minimum
  8-pixel transition radius. Epochs 1–5 have zero mix, epochs 6–10 ramp to
  0.5; remaining epochs stay at 0.5. This schedule/strength is inherited
  from the earlier VOC p50 diagnostic, not selected on VisDrone val.
- Only config differences from the baseline are run name and `oracle_*`
  fields. The latter are popped before Ultralytics base trainer argument
  validation. No change to labels, source dataset, test-dev or standard
  YOLO architecture. This is a training-augmentation intervention, not an
  image enhancement module at inference.

## Gate and verification

1. Finish and audit the currently active P2/640 control first; do not
   overlap task-owned GPU jobs. Confirm physical GPU4–7 idle and no new
   task/non-task collision. Apply the documented narrower hardware gate:
   no further MCE/SIGBUS after the 04:56 node1 event, same boot, mc0
   unchanged, task strictly bound to NUMA0. If it fails, stop/reassess;
   do not hide the fault or change shared system settings.
2. Copy only task-owned configs, verify hash, run the distinct 1%-fraction
   four-GPU smoke at 960 with blur probability 1 and no 5-epoch warmup so
   the actual blur path is exercised. Verify DDP, memory, batch alignment,
   checkpoints, CSV and no new hardware fault. Smoke AP has no value.
3. If smoke passes and resources are idle, launch formal via task-owned
   manager tag `visdrone_oracle_blur_p50_960_s2027`. The manager must
   audit 125 contiguous finite CSV rows and readable best/last weights.
   Check at ~10-minute intervals unless a meaningful event requires action.
   Resume only from a validated checkpoint after a genuine interruption.
4. Re-evaluate best checkpoint independently on the same 548 raw val images
   at 960, `conf=.001`, NMS `.7`, `max_det=1000`, `half=False`. Report
   mAP50, mAP50–95, all 10 class APs and the **fixed 640-reference**
   size-bucket IoU-.50/.75 recall diagnostic. Report training time/memory
   and later end-to-end inference cost if considering a deployable method.

## Decision rule and limits

- A positive one-seed AP change, particularly with better tiny recall, is
  a screening signal only. Repeat with seed 2005 and compare an appropriate
  960 control before claiming reproducibility; test-dev is not a tuning set.
- A non-positive result weakens this *specific* train-only blur recipe on
  VisDrone. Inspect per-class and per-size failures before changing one
  mechanism at a time; do not fish over blur intensity on the val set.
- Even if positive, train-only background smoothing has close precedent in
  SET (CVPR 2025) and detection-driven small front ends in IA-YOLO, DENet
  and IAT. Novelty would require a distinct failure mechanism and controls,
  not simply an AP point gain. This internal metric is not official
  VisDrone challenge AP because converted labels omit ignored regions.

## Pre-result hardware-gate amendment at 06:34 CST

The original gate in step 1 above, “no further MCE/SIGBUS after the
04:56 event,” **failed**: additional node1-mapped uncorrectable events
occurred at 05:23, 05:43 and 05:58, leaving `mc1/ue_count=12`. This
amendment does not pretend that gate passed or that the hardware is
healthy. Given the server owner's prior permission to continue and the
user's requirement to use only NUMA0/physical GPU4–7, a **single
bounded 1%-fraction, one-epoch smoke only** is permitted if a fresh
read-only check after at least 30 quiet minutes shows stable counters,
same boot, no new MCE/SIGBUS, idle GPUs4–7, no task trainer/manager,
and no existing smoke run/state. Any new fault or task SIGBUS stops
further **new** GPU work pending reassessment; do not restart in a loop.
Formal 125-epoch training requires a separate post-smoke hardware,
resource, checkpoint/log and code-path audit. The method settings,
datasets, seed and matched-control comparison remain locked.

The 06:33 check (about 35 minutes after 05:58) met only this *narrower
smoke* preflight: same boot `03:01:16`, EDAC `mc0=0/mc1=12`, no new
matching kernel errors after `05:58:27`, physical GPUs4–7 each showed
2 MiB used/0% utilization, no task train/manager process, smoke run
and state absent. Remote config SHA256 matches the previously recorded
smoke `355ae0fe...` and formal `b097ccac...`. This does not authorize
formal launch automatically. All results on this host remain provisional
until replication on healthy hardware.

## Separate formal-run decision at 06:44 CST (before launch)

The smoke returned `verified_complete` at 06:35:39 with 65 train
images, 2/2 DDP batches, readable best/last, raw 548-image validation,
no task SIGBUS and unchanged post-smoke EDAC `0/12`. Entrypoint,
train-only subclass, nonzero epoch0 blur schedule, probability1 and
standard-validator path were inspected; see
`VISDRONE_ORACLE_960_SMOKE_RESULT_2026-09-23.md`.

At the separate 06:43 check, about **45 minutes** after the last
05:58 memory error, the same boot and EDAC `mc0=0/mc1=12` persisted,
no new matching kernel faults after 06:33, physical GPUs4–7 were again
2 MiB/0%, no task trainer/manager was live, formal run/state paths
were absent, and formal JSON SHA256 was still
`b097ccacb6a08438fe141a343a999ec4b773f5d2a06e792032ad27c162f946f0`.
Given the owner's permission and strict NUMA0/GPU4–7 isolation, this
permits **one** formal launch of the original locked 125-epoch
train-only Oracle blur candidate. It does not retroactively pass the
original strict hardware gate. Monitor at ~10-minute intervals, do
not duplicate the manager, do not touch other users' processes, and
do not blindly restart after a fault. If the host reboots, resume
only from a separately verified checkpoint and resource/hardware gate.
Any result remains provisional on this faulty host and must be
reproduced on healthy hardware before strong claims.

## Pre-result mechanism diagnostic locked at 06:49 CST

The already completed standard960/P2-640/standard640 CPU-only error
audit (`VISDRONE_P2_VS_960_CPU_QUALITY_DIAGNOSTIC_2026-09-23.md`)
found that P2 retrieves similar numbers of very-low-confidence GTs to
960 but retains fewer high-score TPs and more low-score unmatched
predictions. This is an *existing control observation*, not evidence
about the Oracle-blur candidate that has just started.

After the candidate finishes and passes all 125-epoch/checkpoint audits,
the primary decision remains its independent raw-val mAP50–95 versus
the matched standard960 best checkpoint, with mAP50 and all ten classes
reported. A secondary mechanism diagnostic will use the **same**
task-owned CPU `analyze_visdrone_prediction_quality.py` script on its
best checkpoint, same 548 val images, 38,759 converted-label GTs,
`conf=.001`, NMS `.7`, `max_det=1000`, IoU .50/.75, and confidence
thresholds .001/.01/.05/.1/.25/.5. Compare TP/FP, precision/recall,
high-confidence TP retention and detection-cap incidence to the
already cached standard960 CPU diagnostic. Also repeat the same fixed
640-reference size-bucket recall check. These diagnostics are *not*
official VisDrone AP, and converted-label ignored regions can inflate
unmatched counts. Do not declare success from an attractive secondary
metric if the pre-registered primary AP does not improve. If AP rises
in one seed, test repeatability before a paper claim; if AP falls,
inspect whether background suppression lost context or whether the
training distribution mismatch is more plausible, without changing
the current result post hoc.

## Pre-result optional GT-crowding stratification locked at 07:19 CST

Before the formal Oracle candidate's final result, an existing-control
re-aggregation (`VISDRONE_GT_CROWDING_DIAGNOSTIC_2026-09-23.md`)
identified a possible high-density failure mode: P2/640 retrieves
slightly more low-threshold GTs than standard960 in the three less
crowded quartiles but fewer in the most crowded group, while producing
more unmatched candidates in all groups. This is **not** evidence that
Oracle blur will help. As a secondary analysis after independent AP,
fix four groups of exactly 137 images ordered by (converted GT count,
filename), and use the same CPU paired prediction script at
`conf=.001`, NMS .7, max_det1000, IoU .50/.75. Compare candidate versus
standard960 within each frozen group. Do not use the groups to select
blur hyperparameters or rescue a negative primary AP.
