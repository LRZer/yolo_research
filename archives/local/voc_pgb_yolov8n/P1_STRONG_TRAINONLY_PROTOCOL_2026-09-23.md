# P1 strong Oracle background blur, training-only protocol

Date: 2026-09-23; status: smoke passed, formal experiment pending at the time of this protocol.

Question: can a stronger GT-guided progressive background blur help YOLOv8n
when the validation and prediction inputs remain entirely unmodified?

- Model/data: scratch `yolov8n.yaml`, same strict VOC P1 seed-2027 split
  (14,896 train / 1,655 dev), 20 epochs, batch 64, input 640, workers 4;
  same optimizer/augmentations as the previous baseline and blur Oracle.
- Physical GPUs 4–7 only, `CUDA_VISIBLE_DEVICES=4,5,6,7`; NUMA node 0 CPU
  and memory binding. NUMA 0 had no new errors in the current boot at launch,
  but historical errors mean it is not certified healthy. Do not change
  system-wide hardware or other users' processes.
- Training input: GT boxes construct the same continuous protection-strength
  map as the original Oracle: margin 0.10, transition 0.50, minimum radius
  8 pixels. The sole blur-strength change is Gaussian sigma levels
  **0/2/4/8/16** versus the old **0/1/2/4/8**. Residual blend maximum
  remains 0.5. Epochs 1–5 use mix 0, epochs 6–10 ramp 0.1–0.5, and
  epochs 11–20 stay at 0.5.
- Every per-epoch validation uses the unmodified development image through
  standard `DetectionValidator`, with GT boxes used only as evaluation
  labels. Best checkpoint is chosen by raw-image development mAP50–95.
  Post-training raw-image re-evaluation should corroborate the logged result.
- This is a *training augmentation diagnostic*, not an inference-time Oracle
  detector. At prediction, no blur module or GT is needed. Report precision,
  recall, mAP50, mAP50–95, best and last epochs against the original raw-input
  baseline. The earlier Oracle with masked validation is not a directly
  comparable deployable metric.
- Stop on this task's SIGBUS or invalid output; do not blindly retry. The
  manager audits exactly 20 rows and readable best/last checkpoints.

Implementation: `pgb/oracle_integration.py`, `tools/run_experiment.py`,
`tools/p1_strong_trainonly_manager.py`, config
`configs/voc_oracle_strong_trainonly_p1_e20.json`.

Preflight: 19 unit tests passed and a four-GPU, one-epoch 1%-train-fraction
smoke completed, using the standard unmasked validator. NUMA 1 recorded two
new uncorrected memory errors during the preflight period, involving other
processes; NUMA 0's error count stayed at zero. This is noted as hardware
risk, not attributed to the smoke run.
