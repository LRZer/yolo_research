# P1 Oracle-PGB results

Date: 2026-09-23  
Status: verified complete  
Scope: privileged mechanism diagnostic, not deployable inference

## Primary result

Selection for every experiment is the checkpoint with maximum development
mAP50–95. The baseline, learned PGB and Oracle all selected epoch 12.

| Metric | Baseline | Learned PGB | Oracle-PGB | Oracle - baseline |
|---|---:|---:|---:|---:|
| Precision | 0.42630 | 0.39286 | 0.38119 | -0.04511 |
| Recall | 0.29020 | 0.31732 | 0.33975 | +0.04955 |
| F1 from reported P/R | 0.34532 | 0.35107 | 0.35928 | +0.01396 |
| mAP50 | 0.29178 | 0.29113 | 0.30918 | +0.01740 |
| mAP50–95 | 0.16173 | 0.15921 | 0.17042 | +0.00869 |

Oracle improves mAP50 by 1.740 percentage points and mAP50–95 by 0.869
percentage points over the baseline. It improves recall by 4.955 points while
precision falls by 4.511 points. This supports a real but modest upper-bound
signal: ideal object protection makes the present blur intervention useful on
the development split, primarily by increasing sensitivity/recall.

Relative to learned PGB, Oracle adds 1.805 points mAP50 and 1.121 points
mAP50–95. Therefore the current learned map path fails to realize the available
upper bound; map localization/optimization is a more plausible bottleneck than
the blur renderer being entirely ineffective.

## Per-class best-checkpoint results

| Class | Baseline AP50 | Oracle AP50 | Delta | Baseline AP50–95 | Oracle AP50–95 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.559 | 0.566 | +0.007 | 0.347 | 0.345 | -0.002 |
| bicycle | 0.422 | 0.447 | +0.025 | 0.251 | 0.266 | +0.015 |
| bird | 0.163 | 0.176 | +0.013 | 0.0805 | 0.0920 | +0.0115 |
| boat | 0.114 | 0.176 | +0.062 | 0.0545 | 0.0709 | +0.0164 |
| bottle | 0.0507 | 0.0606 | +0.0099 | 0.0236 | 0.0309 | +0.0073 |
| bus | 0.475 | 0.522 | +0.047 | 0.334 | 0.379 | +0.045 |
| car | 0.407 | 0.408 | +0.001 | 0.237 | 0.251 | +0.014 |
| cat | 0.344 | 0.367 | +0.023 | 0.184 | 0.205 | +0.021 |
| chair | 0.0906 | 0.112 | +0.0214 | 0.0365 | 0.0425 | +0.0060 |
| cow | 0.147 | 0.150 | +0.003 | 0.0772 | 0.0830 | +0.0058 |
| diningtable | 0.208 | 0.208 | 0.000 | 0.0824 | 0.0720 | -0.0104 |
| dog | 0.371 | 0.371 | 0.000 | 0.195 | 0.202 | +0.007 |
| horse | 0.279 | 0.324 | +0.045 | 0.167 | 0.184 | +0.017 |
| motorbike | 0.384 | 0.397 | +0.013 | 0.230 | 0.235 | +0.005 |
| person | 0.510 | 0.499 | -0.011 | 0.243 | 0.238 | -0.005 |
| pottedplant | 0.0589 | 0.0861 | +0.0272 | 0.0250 | 0.0329 | +0.0079 |
| sheep | 0.253 | 0.282 | +0.029 | 0.112 | 0.131 | +0.019 |
| sofa | 0.186 | 0.185 | -0.001 | 0.106 | 0.0951 | -0.0109 |
| train | 0.441 | 0.427 | -0.014 | 0.241 | 0.230 | -0.011 |
| tvmonitor | 0.371 | 0.410 | +0.039 | 0.204 | 0.218 | +0.014 |

AP50–95 improved in 15 of 20 classes. The largest AP50–95 gain was bus
(+0.045); the largest reductions were train (-0.011), sofa (-0.0109),
diningtable (-0.0104), person (-0.005) and aeroplane (-0.002). This is not a
uniform benefit and should motivate context/scale analysis rather than a broad
claim that blur always helps.

## Cost, integrity and data boundary

- Oracle elapsed train+validation time: 464.542 s.
- Baseline elapsed time: 312.328 s; Oracle ratio: 1.487x (+48.7%).
- Learned PGB elapsed time: 738.908 s; its learned front end adds further cost.
- Oracle finished exactly 20 finite rows; `best.pt` and `last.pt` reload with
  finite parameters. GPU 4–7 returned idle; no new MCE/SIGBUS event appeared.
- The shared VOC cache currently contains the full 16,551 labels. Its mtime is
  00:55, well before the 01:45 Oracle start; Oracle wrote only task-local caches.
  Raw images and labels were unchanged.
- The final VOC2007 test set was not used.

## Scientific interpretation

This result does not prove a practical method because validation received true
boxes. It does show that the renderer and distance rule possess measurable
headroom on this one split and seed. The learned front end currently leaves
that headroom unused.

The next highest-value work is to measure predicted-map failure against the
Oracle map, then introduce intermediate controls between perfect labels and the
current 38.5K predictor. Recommended order:

1. Quantify protected-region leakage, background strength error and distance-
   binned map error for learned PGB.
2. Evaluate corrupted Oracle maps (box jitter, missed objects and false boxes)
   to determine which localization errors destroy the gain.
3. Test lower-cost single-scale rendering and mix strengths 0.25/0.5 under the
   Oracle control before enlarging the learned predictor.
4. Only after choosing a mechanism with a reproducible margin, run multiple
   seeds and a longer/stronger training protocol.

