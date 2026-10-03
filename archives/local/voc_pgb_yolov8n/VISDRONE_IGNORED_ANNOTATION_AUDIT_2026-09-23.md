# VisDrone ignored/other annotation audit — 2026-09-23 07:02 CST

Read-only, CPU-only check of the **official-source raw annotation text** for
the exact official train and val splits used in the locked 960 experiment.
The project's earlier YOLO conversion code keeps categories 1–10 and
excludes category 0 (`ignored regions`) and 11 (`others`). This audit reads
the originals; it does not edit source or derived labels and does not run
the official VisDrone evaluator.

| Source split | Images | Category-0 rows / images containing one | Category-11 rows / images containing one | Sum of clipped category-0 box areas / all image pixels |
|---|---:|---:|---:|---:|
| Official train | 6,471 | 8,813 / 2,648 | 1,532 / 791 | 0.3362% |
| Official val | 548 | 1,378 / 340 | 32 / 29 | 0.9996% |

All image/annotation pairs were present and all parsed rows were well
formed. On val, ignored regions occur in **340/548 images (62.0%)**, yet
their summed box area is about **1.0%** of total pixels. These figures are
not exact *union* coverage: overlapping ignored boxes are double-counted
in the area sum. The numbers also say nothing about the fraction of
predictions that fall inside these regions. A small spatial fraction
cannot prove a small AP effect, especially with dense detections near
ignore zones.

Interpretation: our converted-label internal AP and FP counts remain useful
**matched** comparisons among models under the same evaluator, but are
not official VisDrone challenge AP. The 1,378 ignored val annotations
make that qualification concrete; one must not relabel category 0/11 as
ordinary negatives or modify the raw annotations to improve a score.

The audit script is `tools/audit_visdrone_ignored_annotations.py` in this
task's source tree and remote task tools. It uses only PIL image-header
dimensions plus original annotation text and reports per-category counts
and summed clipped areas. It did not request any GPU.
