# VisDrone 960 Oracle target-strength audit — no training

Date: 2026-09-23. This is a **CPU-only**, task-owned, read-only audit of
the already pre-registered target-protected background-blur geometry.
It uses the exact `pgb.core.make_targets` function, 960×960 square
LetterBox coordinates and original image dimensions for all 548
official-source VisDrone **val** images. GT boxes are used here only to
describe the Oracle map, not to process validation images for detector
evaluation. No GPU, detector inference, training, image rewriting or
test-dev data was involved. The audit script and raw per-image JSON are
`tools/audit_oracle_strength_val.py` and
`VISDRONE_ORACLE_STRENGTH_AUDIT.json`.

## Integrity

- Read 548/548 images and their labels, 38,759 usable boxes after
  coordinate clipping, 0 empty images. Per-image box counts sum to
  38,759, agreeing with the earlier val provenance audit.
- All original val images are 16:9: 408 at 1360×765, 121 at 960×540,
  19 at 1920×1080. At square 960 LetterBox, each image has
  960×540=518,400 valid image pixels; padding is excluded from all
  fractions. Total valid pixels: 284,083,200.
- The 20-bin strength histogram sums exactly to the valid-pixel count.
  The independently computed fully protected fraction here, 13.2637%,
  agrees closely with the previous rectangle-union area audit's mean
  13.2528%; small differences reflect pixelization/LetterBox rounding.

## Strength distribution under the locked recipe

The target definition is `margin=0.1`, `transition=0.5`,
`min_radius=8`, with planned Gaussian levels σ=0/2/4/8/16 and final
image mix 0.5. The formal training recipe would apply this transform
to a random 50% of training images after its warmup. On **val geometry**
only, the exact reference strength `S*` gives:

| Quantity, valid image area only | Fraction |
|---|---:|
| Fully protected (`S*=0`) | 13.2637% |
| `S*≥0.1` | 78.6785% |
| `S*≥0.5` | 67.3924% |
| `S*≥0.9` | **54.9202%** |
| `S*≥0.99` | 45.7968% |

The image-wise mean strength is 0.6701 (median 0.6722; p10 0.4696;
p90 0.8635). Since every image has the same valid area after
LetterBox, the mean of image means equals the valid-pixel-weighted
mean. At `S*≥0.9`, the interpolated blur level is at least σ=14.4
pixels; however the image still keeps 50% original content when the
training-time mix is 0.5. Do **not** mistake strength 0.9 for “90% of
the original image removed.”

## Interpretation and limits

This confirms the locked VisDrone recipe is a substantial background
intervention rather than a near-no-op: most valid pixels are outside
hard box protection and over half receive near-maximal *target* blur
strength when that sample is selected for augmentation. Dense targets
occupy all four coarse 2×2 sectors in many images, but their actual
protected boxes occupy a small image fraction; the two audits are
consistent. Such wide high-frequency suppression could also remove
context and create a train–val distribution gap. That is a
**hypothesis**, not a measured harm or benefit.

Val-image geometry is only a proxy for training batches: Mosaic,
random perspective, scaling and other augmentations can change box
sizes and spatial coverage. The audit does not render Gaussian-blurred
images, measure actual image information loss, evaluate mAP, benchmark
GPU cost, or prove that a learned predictor can recover the GT-guided
map at inference. We are **not** retuning the pre-registered strength
based on this val audit. The planned short real-path smoke and matched
125-epoch train-only comparison remain deferred while new uncorrectable
server memory faults continue.
