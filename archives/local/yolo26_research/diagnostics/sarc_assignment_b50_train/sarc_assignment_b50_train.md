# SARC assignment histogram probe

**Decision: GO** — level separation exists and consecutive-checkpoint distributions remain within the stability bound.

> Read-only fixed-checkpoint diagnostic: train split, deterministic non-augmented panel, no optimizer, no backward pass, no weight or BN-stat updates. JS uses log base 2 and is bounded by [0,1].

## Reliable P3/P4/P5 mapping

`Detect.forward_head` flattens each feature map and concatenates levels in `feats` order. `make_anchors` iterates over the identical list and concatenates the identical HxW blocks. The probe reconstructs these block indices and then requires every anchor's generated stride to match the corresponding model stride (8/16/32); mismatch aborts instead of silently assigning a level.

Only O2M/TAL assignments are primary because YOLO26 detaches O2O inputs from the shared backbone/neck. `positive_anchor` is the SARC training-signal view; `unique_gt_per_image_level` de-duplicates TAL top-k repetitions as a sensitivity check.

Panel: 512 images in 8 batches; fingerprint `0586e541229571cbe4849dc1efab4008f6e3a051a262dba6892f902893818472`.

## Positive-anchor distributions

| epoch | level | count | share | mean u | p10 | p50 | p90 | per-batch CV |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | P3 | 2549 | 20.38% | 2.822 | 2.107 | 2.858 | 3.471 | 0.226 |
| 11 | P4 | 3861 | 30.86% | 2.970 | 2.267 | 2.954 | 3.660 | 0.234 |
| 11 | P5 | 6100 | 48.76% | 3.214 | 2.569 | 3.221 | 3.866 | 0.070 |
| 21 | P3 | 2259 | 18.06% | 2.751 | 2.057 | 2.797 | 3.350 | 0.239 |
| 21 | P4 | 3795 | 30.35% | 2.865 | 2.195 | 2.841 | 3.525 | 0.246 |
| 21 | P5 | 6452 | 51.59% | 3.179 | 2.502 | 3.189 | 3.855 | 0.061 |
| 31 | P3 | 2041 | 16.27% | 2.697 | 2.002 | 2.713 | 3.319 | 0.248 |
| 31 | P4 | 3747 | 29.88% | 2.791 | 2.156 | 2.776 | 3.443 | 0.267 |
| 31 | P5 | 6753 | 53.85% | 3.151 | 2.450 | 3.164 | 3.845 | 0.059 |
| 41 | P3 | 1942 | 15.49% | 2.677 | 1.982 | 2.700 | 3.267 | 0.252 |
| 41 | P4 | 3748 | 29.90% | 2.756 | 2.123 | 2.753 | 3.414 | 0.263 |
| 41 | P5 | 6847 | 54.61% | 3.140 | 2.431 | 3.156 | 3.845 | 0.051 |
| 50 | P3 | 1906 | 15.18% | 2.675 | 1.980 | 2.700 | 3.282 | 0.247 |
| 50 | P4 | 3808 | 30.34% | 2.746 | 2.119 | 2.751 | 3.414 | 0.274 |
| 50 | P5 | 6839 | 54.48% | 3.146 | 2.444 | 3.161 | 3.845 | 0.051 |

## Pooled between-level JS

| weighting | P3 vs P4 | P3 vs P5 | P4 vs P5 |
|---|---:|---:|---:|
| positive_anchor | 0.0166 | 0.1234 | 0.0687 |
| unique_gt_per_image_level | 0.0205 | 0.0350 | 0.0216 |

## Consecutive-checkpoint JS (positive anchors)

| epochs | level | JS |
|---|---|---:|
| 11→21 | P3 | 0.0058 |
| 21→31 | P3 | 0.0030 |
| 31→41 | P3 | 0.0004 |
| 41→50 | P3 | 0.0001 |
| 11→21 | P4 | 0.0079 |
| 21→31 | P4 | 0.0054 |
| 31→41 | P4 | 0.0012 |
| 41→50 | P4 | 0.0012 |
| 11→21 | P5 | 0.0013 |
| 21→31 | P5 | 0.0009 |
| 31→41 | P5 | 0.0002 |
| 41→50 | P5 | 0.0001 |

## Pre-registered gate

- Similarity NO-GO: maximum pooled positive-anchor between-level JS is strictly below `0.02`. Observed: `0.1234`.
- Instability NO-GO: any consecutive-checkpoint positive-anchor JS is strictly above `0.3`. Maximum observed: `0.0079`.
- Sample sufficiency: at least `100` pooled positive assignments per level; otherwise the result is INCONCLUSIVE.
