# P1 extreme Oracle blackout protocol

Date: 2026-09-23  
Status: pre-registered diagnostic

## Intervention

For every train and development-validation image, form the union of all exact
ground-truth detection rectangles. Preserve RGB pixels whose centers lie inside
that union and set every other pixel to exact zero. There is no margin,
transition, feathering, blur or original-image residual.

The hard rectangular edges expose label locations. This is an intentional
label-leaking stress test, not a deployable method and not ordinary detector
accuracy. VOC2007 test remains untouched.

## Controlled settings

- Same 14,896/1,655 fixed VOC train/development split as P1.
- YOLOv8n from source YAML, no pretrained weights.
- 20 epochs, seed 2027, global batch 64, image size 640, BF16.
- Physical GPUs 4–7, NUMA node 1, workers 4 per rank.
- Same optimizer policy and disabled augmentation settings as the baseline.
- Hard blackout is active from epoch 1 in both training and validation.
- Primary selection is maximum development mAP50–95.

## Interpretation boundary

This experiment asks how a detector behaves when background is removed and
the exact target rectangles are visible as crop boundaries. Any gain may come
from background removal, from direct box-boundary leakage, or both. A drop
would indicate that target context/background remains important despite the
privileged locations.

## Preflight

- Seventeen unit/integration tests passed, including exact pixel identity inside
  boxes, exact zero outside boxes, multi-box union and empty-label behavior.
- A one-epoch four-GPU real-VOC smoke completed with readable checkpoints.

