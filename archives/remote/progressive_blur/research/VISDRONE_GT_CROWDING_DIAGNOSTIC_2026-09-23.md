# GT-crowding diagnostic on existing VisDrone controls — 2026-09-23 07:19 CST

No new model inference or GPU use. This is a paired re-aggregation of
previously frozen CPU predictions for standard YOLOv8n/960 and
YOLOv8n-P2/640 on the same 548 raw val images. Images were sorted
by converted-label GT box count, with deterministic filename tie-break,
into four equal groups of 137. IoU-.50 classwise confidence-greedy
matching uses `conf>=.001`, NMS .7, cap1000, exactly as in the earlier
`VISDRONE_*_CPU_QUALITY.json` diagnostics. These are **not official AP**.

| GT-count quartile | GT/image range | GT boxes | Standard960 IoU-.50 recall | P2/640 recall | P2−960 matched GT | P2−960 unmatched predictions | Images hitting cap: 960 / P2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 1–39 | 2,885 | 79.03% | 79.83% | +23 | +25,016 | 2 / 14 |
| Q2 | 40–65 | 7,231 | 75.58% | 76.73% | +83 | +30,315 | 10 / 66 |
| Q3 | 65–93 | 10,807 | 74.95% | 75.58% | +68 | +23,098 | 29 / 124 |
| Q4 | 93–317 | 17,836 | 70.11% | 69.41% | −124 | +8,403 | 82 / 133 |

At IoU .75, P2 recall is at or below standard960 in every quartile;
the Q4 values are 37.10% vs 38.48%. The descriptive pattern is that
P2/640 yields more low-score/low-threshold candidates and slightly
more IoU-.50 matches in Q1–Q3, but not the most crowded Q4. More P2
images hit the max1000 output cap, especially Q3–Q4. This is consistent
with candidate flooding and weaker localization/ranking, **not** a
causal explanation or a reason to alter the running Oracle experiment.

The quartiles depend only on GT counts, not candidate predictions.
Before the current Oracle candidate produces its final AP, we lock an
optional *secondary* question: when its own best checkpoint is available,
apply these exact image groups and matching thresholds versus the
standard960 reference to see whether any AP change co-occurs with
crowding-specific TP/FP behavior. An AP loss cannot be reframed as a
success by cherry-picking a quartile. Converted labels omit official
ignore regions; the FP counts are especially sensitive to that choice.

Raw computed table: `VISDRONE_P2_VS_960_GT_DENSITY.json`. Reusable
grouping code: `tools/summarize_pair_density.py`.
