# Locked Oracle blur on real VisDrone images — qualitative audit

2026-09-23 07:05 CST. This is **not** a model result or a tuning pass.
The already-registered train-only Oracle renderer (`sigma=0/2/4/8/16`,
`mix=0.5`, 10% box margin, 0.5 transition, 8 px minimum radius) was
applied CPU-only to three **val** images for visualization. Val images
are never inserted into the training set by this audit. These samples
were selected deterministically at the 10th/50th/90th percentiles of
converted-label box count, not by their apparent treatment effect.

Each 2×2 panel shows: upper left original with green GT boxes; upper
right target strength (blue protected, red more blurred); lower left
processed image; lower right pixel-difference heatmap. The full panels
are task-owned files locally and under remote
`research/visdrone_oracle_visuals_960/`.

| Sample | Converted boxes | Mean target strength on valid pixels | Fully protected valid pixels | Mean absolute RGB change (0–255) | Panel |
|---|---:|---:|---:|---:|---|
| Low-count (10th percentile) | 16 | 0.7267 | 10.11% | 3.78 | `01_016boxes_val_519_0000359_03529_d_0000712.jpg` |
| Median-count | 65 | 0.5894 | 17.09% | 5.22 | `02_065boxes_val_328_0000289_05001_d_0000836.jpg` |
| High-count (90th percentile) | 122 | 0.6435 | 13.51% | 6.11 | `03_122boxes_val_88_0000155_00401_d_0000001.jpg` |

Visual inspection of the median-count panel: rectangles receive low
blur strength while distant road/building/tree texture changes. The
processed image remains recognizable and is **not** a black-mask-like
intervention; `mix=0.5` inherently retains half the original image.
Mean pixel change is spatially averaged and cannot prove or refute
feature-level impact on tiny targets. Moreover, the actual training
batch includes Mosaic and other transformations; this single-image
audit establishes the renderer's behavior on real source images, not
the exact distribution of all augmented training batches.

The fixed 125-epoch experiment must determine whether this processing
helps raw-image detection. No strength or schedule was changed after
viewing these panels.
