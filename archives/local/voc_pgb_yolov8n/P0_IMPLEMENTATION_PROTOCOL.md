# P0 implementation protocol — frozen 2026-09-23

## Research question

Can an image-only learned front end, supervised during training by a continuous
distance field derived from detection boxes, protect target neighborhoods while
progressively suppressing distant high-frequency background content and thereby
improve YOLOv8n detection?

This is a hypothesis, not a demonstrated result. The method is not described as
semantic background segmentation, restoration, acceleration, or optical depth
of field.

## P0 scope

1. Preserve and independently reproduce the supplied PyTorch core.
2. Verify geometry, interpolation, gradients, empty labels, ignored pixels,
   checkpoint round-tripping, odd image sizes and 640-pixel input.
3. Integrate the front end into Ultralytics 8.4.146 without losing it in
   optimization, EMA, checkpoint/resume, validation or prediction paths.
4. Run tiny synthetic integration tests before touching VOC or a remote GPU.
5. Only after integration audit, prepare deterministic VOC development splits
   and controlled baseline/method experiments.

## Frozen initial method

- Predictor: supplied 38,513-parameter convolutional encoder/decoder.
- Box protection margin: 0.10 of box width/height per side.
- Transition scale: 0.50 of box width/height, with 8-pixel minimum at 640 input.
- Blur bank: sigma = 0, 1, 2, 4, 8 input pixels.
- Residual blur mix target: rho = 0.5.
- Warm-up proposal: rho 0 for epochs 1–5, linear 0→0.5 for epochs 6–10.
- Auxiliary loss: protected/outer-region balanced Smooth L1, beta 0.1.
- Inference input: image only; boxes never control inference-time processing.

## Mandatory causal controls

- A0: standard YOLOv8n baseline.
- A1: identity wrapper with the same engineering/checkpoint path.
- B: ground-truth-guided blur as training-only augmentation; raw image at test.
- C: learned map with one fixed blurred image and soft mixing.
- D: learned map with the full multi-scale Gaussian bank.
- D-no-map: D without auxiliary map supervision.
- D-shuffled: spatially shuffled predicted map diagnostic.
- Oracle: ground-truth map at evaluation, reported only as a non-deployable
  diagnostic and never as the proposed method.

These controls are needed because improvement could arise from extra capacity,
box-supervised objectness, regularization, or engineering differences rather
than the progressive blur mechanism itself.

## Data and evaluation boundary

- Development: a saved, image-level split made only from VOC07+12 trainval.
- Final test: VOC2007 test, not used for selecting parameters or epochs.
- All groups share initialization source, data order, transforms, optimizer,
  training budget and evaluation code.
- Report mAP50, mAP50–95, precision, recall, per-class AP, seeds, parameters,
  FLOPs where measurable, total latency and peak memory.
- Ultralytics AP and historical VOC2007 11-point AP must not be conflated.

## Initialization decision still open

Earlier work preferred scratch training, while the supplied proposal recommends
identical COCO-pretrained initialization. The implementation will support both.
The main-paper protocol will be chosen before full training and applied equally
to baseline and method; no result-dependent switching is allowed.

## Execution and server safety boundary

Per the user's 2026-09-23 decision, formal training, integration performance
tests and accuracy experiments run on the remote server, not on the laptop GPU.
Use only physical GPUs 4–7 and bind this task to the known healthy NUMA node 1.
Local execution is limited to fast code-level checks that do not count as
experimental evidence. Do not change global server settings, other users'
processes, or shared credentials.
