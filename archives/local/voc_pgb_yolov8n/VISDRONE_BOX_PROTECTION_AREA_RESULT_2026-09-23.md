# VisDrone target-box protection-area audit — 2026-09-23

This is a **GPU-free, label-only geometry audit**, not an AP experiment or a
deployable selector. The task-owned script
`tools/audit_visdrone_box_protection_area.py` read every converted-label
file in the 548-image official-source VisDrone val view (38,759 GT boxes),
formed the exact union of axis-aligned rectangles, and repeated it after
expanding each box by 10% of its width/height on each side. It did not
alter source labels or images. Hand-checked unit tests covered overlapping,
disjoint and empty rectangle sets. Full per-image output is in
`VISDRONE_OFFICIAL_BOX_PROTECTION_AREA.json`.

| Image-area fraction | Mean | Median | 10th–90th percentile |
|---|---:|---:|---:|
| Union of original GT boxes | 9.84% | 9.16% | 2.91–17.50% |
| Union of 10%-margin protection rectangles | 13.25% | 12.34% | 4.02–23.33% |
| Outside those protection rectangles | 86.75% | 87.66% | 76.67–95.98% |

In 506/548 images, the 10%-margin protection rectangles occupy under 25%
of image area. This coexists with the earlier finding that 436/548 images
contain GT centers in **all four** fixed 2×2 cells: occupied coarse tiles
do not imply that GT boxes fill much of the image. Thus the simple
quarter-image skipping route has poor GT-center coverage, while finer
feature-level sparsity might still exploit large background area. That
general mechanism is already covered by [QueryDet (CVPR 2022)](https://arxiv.org/pdf/2103.09136),
not a new contribution here.

For the proposed train-only Oracle blur, this audit shows that most image
area lies beyond the *hard protection rectangles*, not that 86.75% would
receive maximum blur: the distance-based transition softens nearby pixels,
and the formal setting uses 0.5 residual mix and applies blur to only half
the training images after warmup. Nor does low box-area occupancy prove
that far-background context is dispensable. The 312 originally clipped
and 1,719 expanded/clipped boxes are expected edge intersections with the
image boundary; no annotations were changed. Converted labels omit ignored
regions; box unions include box-internal background and therefore are not
foreground segmentation area. The next AP test remains on hold after the
05:23 recurring memory fault until a fresh safety decision is documented.
