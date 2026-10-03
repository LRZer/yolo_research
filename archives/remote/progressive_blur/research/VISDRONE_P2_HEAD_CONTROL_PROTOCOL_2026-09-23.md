# Pre-registered standard P2-head control — 2026-09-23

Purpose: test whether a conventional stride-4 detection output improves
the official-provenance VisDrone val split under the same 640-pixel input.
This is **not** a novel method or evidence that target-protected blur works.
It is a strong architectural control before claiming a new small-object
mechanism. The 960-input standard YOLOv8n control was already launched
before this protocol was written; no P2 results have been seen or tuned.

## Hypothesis and fair comparison

- Baseline A: existing clean YOLOv8n scratch, 640, seed 2027, 125 epochs,
  best checkpoint chosen on 548-image official-source val split. Independent
  internal converted-label val mAP50-95 = 0.170547.
- Baseline B: standard YOLOv8n scratch, 960, otherwise matched; in progress.
- Control C: official Ultralytics YOLOv8 P2/4–P5/32 architecture at **n**
  scale, 640, seed 2027, 125 epochs, same train/val lists, SGD, batch 64,
  default geometric/color augmentations, max_det 1000 and checkpoint rule.
  The source YAML at
  `/home/ubuntu/rz/yolo_study/ultralytics/ultralytics/cfg/models/v8/yolov8-p2.yaml`
  had SHA256 `ba54363e9f283e8f60b0fa0843eb37115093338d6f9273ccd4f74f60639754f4`.
  A task-owned copy is named `models/yolov8n-p2.yaml` to select n scale;
  verify its content, scale, model summary and parameters during smoke.
  A read-only CPU structural dry run on the task-owned copy confirmed
  `scale=n` and output strides `[4, 8, 16, 32]`; it reported 3,354,144
  parameters at the YAML's default `nc=80`. Training will override `nc` to
  the dataset's 10 classes, so record the actual run's model summary rather
  than presenting this dry-run count as experimental complexity.

Only the model architecture changes from A to C. Because it changes
parameter count/FLOPs and uses more high-resolution activations, report
memory, training time, and eventual inference cost; do not frame an AP gain
as cost-free. Comparing B vs C is a different trade-off, not an isolated
architecture ablation, because input size differs. No test-dev is used for
model selection or parameter tuning. Plain Ultralytics AP on converted
labels is an **internal** measure, not official VisDrone DET toolkit AP.

## Execution gate

1. Wait until the formal 960 manager is terminal and its run is audited.
   Do not overlap our GPU jobs or interrupt other users.
2. Confirm task GPU/process occupancy is empty, physical GPUs 4–7 are
   available, EDAC UE counts have not newly increased and no recent
   MCE/SIGBUS appears. Bind physical GPUs 4–7 through
   `CUDA_VISIBLE_DEVICES=4,5,6,7` and NUMA0 via
   `numactl --cpunodebind=0 --membind=0`.
3. Run task-owned 1-epoch, 1%-fraction 4-GPU smoke. Confirm n scale,
   outputs P2/4–P5/32, no OOM, no source-cache writes, checkpoint and
   CSV readable. Smoke AP has no evidentiary value.
4. Only if smoke passes, launch formal task manager using
   `configs/visdrone_official_yolov8n_p2_scratch_640_s2027_e125.json`
   and tag `visdrone_official_p2_640_s2027`. Manager must audit strictly
   125 contiguous finite result rows and readable best/last checkpoints.
   On interruption, resume only from a validated checkpoint.
5. Re-evaluate best checkpoint independently on official-source val with
   the same internal evaluator, imgsz 640, max_det 1000, report all 10
   per-class AP values and timing. Repeat seed before publication-strength
   claims; one seed is only a diagnostic screen.

## Decision rule

If C improves mostly classes with tiny median boxes and costs less than
960 for similar AP, this supports a feature-stride bottleneck *hypothesis*.
If 960 dominates both at acceptable cost, prioritize a conditional
high-resolution mechanism or tiling comparison. If neither helps,
investigate annotations/evaluator and training optimization before adding
modules. Any new method must beat appropriate resolution/P2/tiling controls
and show a mechanism rather than only a one-run AP delta.

## Hardware-gate addendum after the 04:56 node1 event

The original execution gate's "EDAC UE counts have not newly increased"
condition **failed** after 960 completed: `mc1` changed 0→1 at 04:56,
and the affected physical page maps to NUMA node1. This was a kernel
SIGBUS to the unrelated `update-notifier`, not a task training error.
No P2 GPU work was started at that point. See the separate incident
record `HARDWARE_EVENT_2026-09-23_0456.md` for exact evidence and the
new, explicitly narrower continuation gate (10-minute stability,
node0 unchanged, no new kernel faults, GPUs idle, strict node0 binding).
This addendum does not erase or claim satisfaction of the original gate.

At 05:07 CST, after the 10-minute stability window, the limited P2
smoke completed without OOM or a new MCE/SIGBUS. Its DDP launch used
four processes; task-owned view cache, one CSV epoch and readable
best/last were verified. The 10-class model summary reported 90 fused
layers, 2,922,360 parameters and 12.2 GFLOPs, output strides 4/8/16/32.
These are structural/operational smoke results, **not AP evidence**.
Following a fresh idle-device and EDAC preflight, the formal P2
125-epoch run started at 05:09 with task manager tag
`visdrone_official_p2_640_s2027`.

Post-launch config integrity: the 640 standard-YOLO and 640 P2 formal
JSON files differ only in `model` and run `name`; the task-owned P2
config SHA256 is
`f9054203450a8dc2a960753c7e1a4ebc53a789dac125179c61d8fd91f9f75412`
locally and on the server. This does not make P2 compute-equivalent:
the smoke's n/10-class fused model showed 2.922 M parameters and
12.2 GFLOPs versus the baseline's 3.008 M and ~8.1 GFLOPs at 640.
