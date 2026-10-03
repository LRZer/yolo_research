# Fixed-grid high-resolution tile selection: GT-Oracle feasibility — 2026-09-23

This read-only, GPU-free audit tested whether a small number of
fixed image sectors could cover most VisDrone target centers. It
used the **548 official-source validation label files and 38,759
converted GT boxes**. Each image was divided in normalized
coordinates into 2×2 or 3×3 equal cells; cells were ranked by the
number of GT centers inside them. Choosing those top cells requires
ground truth, so the results are an **optimistic Oracle bound**, not
a deployable algorithm, detection result, AP, or speed measurement.

| Grid | Occupancy / 548 images | GT-center coverage of Oracle top-k cells |
|---|---|---|
| 2×2 | 436 images (79.6%) contain GT centers in all four cells; 77 in three, 28 in two, 7 in one | top1 (25% image area): 50.4%; top2 (50%): 79.0%; top3 (75%): 93.1% |
| 3×3 | 76 images occupy all nine cells; 321 (58.6%) occupy at least seven | top1 (11.1% area): 42.5%; top2 (22.2%): 64.0%; top3 (33.3%): 77.6%; top4 (44.4%): 86.8%; top5 (55.6%): 92.8%; top6 (66.7%): 96.6% |

These are micro-averages over all boxes, not per-class AP. Raw JSON
also includes mean per-image coverage and complete occupancy
histograms. There are **no empty-label validation images** in this
converted split.

Interpretation: object centers are concentrated enough that an
impossible GT Oracle could cover ~93% with 5 of 9 cells, but most
images have objects spread broadly. An exhaustive fixed-grid detector
cannot skip the remaining cells without missing annotated targets;
an ordinary learned selector would perform worse than this Oracle.
Processing five separate high-res crops also has overhead and may not
be cheaper than one 960-pixel pass. Thus this result does **not**
justify a naive foveated-speed claim. If a dynamic-resolution route is
pursued, it needs an accurate image-only selector, explicit latency
comparison to whole-image 960 and standard tiling, and recall/AP
accounting for skipped targets and boundary/context effects.

Exact limitations: cell membership used box centers only; box extents,
overlap, crops crossing borders, ignored/other VisDrone annotations,
detector effectiveness and inference cost were not evaluated. This
is an architectural feasibility screen, not an experiment on images.
