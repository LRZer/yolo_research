# P1 Oracle-PGB protocol

Date: 2026-09-23  
Purpose: mechanism upper-bound diagnostic; not a deployable model result.

## Question

If the progressive blur renderer is given the true object boxes during both
training and development validation, can ideal object protection plus
background blur improve YOLOv8n detection over the P1 baseline?

## Controlled comparison

Oracle-PGB retains the exact P1 baseline detector, split, initialization,
seed, optimizer policy and training budget. The only intervention is replacing
each normalized batch image with the multi-scale Gaussian rendering produced
from the ground-truth continuous distance map.

- VOC train: 14,896 images; development validation: 1,655 images.
- VOC2007 final test: not used.
- YOLOv8n from `yolov8n.yaml`, no pretrained weights.
- 20 epochs, seed 2027, batch 64, image size 640, BF16.
- Physical GPUs 4–7 only; NUMA node 1; workers 4 per rank.
- All strong/geometric/color augmentations disabled, matching P1.
- Blur levels: sigma 0, 1, 2, 4, 8; final residual mix 0.5.
- Schedule: mix 0 for epochs 1–5, linear ramp in epochs 6–10, then 0.5.
- Geometry: 10% box margin, transition scale 0.5 of box width/height,
  minimum radius 8 pixels.
- Selection: maximum development mAP50–95.

The true boxes are intentionally used during validation. Therefore the result
is labelled Oracle and may only diagnose the blur mechanism's upper bound. It
must not be reported as ordinary inference accuracy or a deployable method.

## Decision rule

- Clear AP improvement: the blur mechanism has headroom; improve learned map
  prediction and cost.
- No AP improvement: the current blur intervention itself lacks value on VOC;
  do not spend a long multi-seed budget on the learned front end unchanged.
- Mixed precision/recall result: inspect confidence calibration, per-class AP
  and processed images before choosing a direction.

## Preflight

- Fifteen unit/integration tests passed.
- A real-VOC four-GPU smoke used 149 train images and all 1,655 development
  images for one scratch epoch and completed with readable checkpoints.
- Zero AP in that one-epoch 1% smoke is expected and is not a scientific result.

