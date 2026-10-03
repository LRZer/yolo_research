# VisDrone 640 vs 960 conventional resolution control — 2026-09-23

## Bottom line

On the task-owned, official-provenance VisDrone train/val split, raising
scratch YOLOv8n input from 640 to 960 produced an independently
recomputed **internal converted-label** val mAP50–95 of **0.232476 vs
0.170547**, an absolute gain of **0.061929** (6.19 percentage points)
in this one-seed experiment. All ten classes' AP50–95 increased. A
pre-registered matched-box diagnostic attributes the largest recall
gain to GTs in the `<16 px` 640-reference size bucket. The result is a
strong conventional baseline and a scale-bottleneck *hypothesis*, not
a new method, a causal proof, or official VisDrone leaderboard AP.

## Locked comparison and run integrity

| Item | 640 control | 960 control |
|---|---:|---:|
| Model | YOLOv8n scratch, pinned YAML | Same pinned YAML |
| Official-source train / val | 6,471 / 548 | Same image lists |
| Val GT boxes in converted labels | 38,759 | 38,759 |
| Initialization / seed | random / 2027 | random / 2027 |
| Training | 125 epochs, SGD, batch 64, workers 4 | Same |
| Input | 640 | 960 |
| Device policy | physical GPU 4–7, NUMA0 | Same |
| Best checkpoint selected on | val, epoch 115 | val, epoch 111 |
| `results.csv` rows | 125 | 125 |
| Cumulative CSV train time | 1,015.27 s (16.92 min) | 1,573.05 s (26.22 min) |
| Largest `GPU_mem` shown in rank-0 train log | 10.1 G | 14.3 G |

The two formal JSON configs differ only in `imgsz` and run `name`;
remote SHA256 matches each local source. The 960 manager finished at
04:51:50 CST with `verified_complete`: 125 contiguous finite result
rows plus readable `best.pt` and `last.pt`. Independent raw-image
`YOLO.val` used the **existing best checkpoint**, same official-source
val images, `conf=.001`, NMS IoU `.7`, `max_det=1000`, `half=False`,
and the corresponding training image size. No GT-guided Oracle or blur
was applied in evaluation. The 960 run's cumulative training time is
about **1.55×** that of 640; training time is not deployment latency.
The log-reported peak rank-0 `GPU_mem` was ~1.42× higher at 960;
this is a training-process PyTorch memory indicator, not a synchronized
device-wide `nvidia-smi` peak or a deployment benchmark.

## Independent raw-validation metrics

| Metric | 640 | 960 | Absolute change |
|---|---:|---:|---:|
| mAP50 | 0.309082 | 0.403649 | +0.094568 |
| mAP50–95 | 0.170547 | 0.232476 | +0.061929 |

Built-in best-epoch rows, for audit only: 640 epoch115 precision
0.42334, recall 0.32519, mAP50 0.30918, mAP50–95 0.17051;
960 epoch111 precision 0.49777, recall 0.40092, mAP50 0.40347,
mAP50–95 0.23240. Those precision/recall fields are Ultralytics'
operating-point summaries, **not** the fixed-threshold diagnostic recall
in the table below. The 960 final epoch125's built-in mAP50–95 was
0.23167, close to its selected best; that does not remove the need
for repeated seeds.

| Class | AP50–95 at 640 | AP50–95 at 960 | Δ |
|---|---:|---:|---:|
| pedestrian | 0.1330 | 0.2115 | +0.0785 |
| people | 0.0851 | 0.1340 | +0.0489 |
| bicycle | 0.0244 | 0.0632 | +0.0388 |
| car | 0.4862 | 0.5616 | +0.0754 |
| van | 0.2359 | 0.3264 | +0.0904 |
| truck | 0.1542 | 0.2129 | +0.0587 |
| tricycle | 0.1029 | 0.1442 | +0.0413 |
| awning-tricycle | 0.0541 | 0.0891 | +0.0350 |
| bus | 0.2870 | 0.3715 | +0.0845 |
| motor | 0.1427 | 0.2104 | +0.0678 |

## Pre-registered size-stratified matched-box diagnostic

The task-owned evaluator consumed all **548** identical val images and
exactly **38,759** GT boxes in both runs. Each GT was assigned once
using `sqrt(box area)` after a *hypothetical 640 letterbox resize*,
so the bucket membership is identical at 640/960. Detection used
`conf=.001`, NMS `.7`, max 1000 boxes/image; predictions were matched
one-to-one to same-class GTs by descending confidence. This is
diagnostic GT recall, **not AP**. The fractions depend on this low
confidence operating point and the converted-label protocol.

| Fixed 640-reference GT bucket | GTs | R@IoU .50: 640 → 960 | Δ | R@IoU .75: 640 → 960 | Δ |
|---|---:|---:|---:|---:|---:|
| All | 38,759 | 0.6094 → 0.7311 | +0.1217 | 0.3299 → 0.4299 | +0.1000 |
| `<16 px` | 27,103 | 0.5065 → 0.6605 | +0.1540 | 0.1879 → 0.2953 | +0.1074 |
| `16–<32 px` | 8,724 | 0.8275 → 0.8838 | +0.0563 | 0.6064 → 0.7056 | +0.0993 |
| `32–<96 px` | 2,897 | 0.9123 → 0.9299 | +0.0176 | 0.8208 → 0.8561 | +0.0352 |
| `>=96 px` | 35 | 0.8286 → 0.8571 | +0.0286 | 0.6857 → 0.6571 | −0.0286 |

The `<16` group is approximately 70% of val GTs and has the
largest IoU-.50 recall change. The largest bucket has only 35 GTs;
its apparent small gain/drop is too noisy to interpret. These results
are consistent with an object-pixel bottleneck, but resizing changes
more than a single physical mechanism (feature sampling, augmentation
geometry, optimization and compute). A standard P2/4 detection head
at 640 was therefore pre-registered as a further control, not as
our claimed contribution.

For transparency, the same **diagnostic IoU-.50 recall** within the
`<16 px` bucket by class is below. All ten class-specific deltas are
positive, but counts vary greatly and small classes are noisy.

| Class | Tiny GTs | R640 | R960 | Δ |
|---|---:|---:|---:|---:|
| pedestrian | 7,790 | 0.498 | 0.668 | +0.169 |
| people | 4,656 | 0.342 | 0.517 | +0.175 |
| bicycle | 1,076 | 0.221 | 0.406 | +0.185 |
| car | 7,292 | 0.742 | 0.856 | +0.114 |
| van | 966 | 0.106 | 0.260 | +0.154 |
| truck | 307 | 0.202 | 0.296 | +0.094 |
| tricycle | 592 | 0.223 | 0.380 | +0.157 |
| awning-tricycle | 296 | 0.095 | 0.260 | +0.166 |
| bus | 116 | 0.147 | 0.250 | +0.103 |
| motor | 4,012 | 0.565 | 0.733 | +0.168 |

## Limits and next decision

1. **One seed and one dataset/split.** Repeat independent seeds before
   calling the gain stable. The much larger change makes resolution a
   credible conventional control, but not publication evidence alone.
2. **Internal metric.** Our YOLO-formatted labels omit VisDrone
   ignored/other annotations, while the official DET toolkit handles
   them differently. Do not compare these AP numbers to the official
   challenge leaderboard without implementing and checking its exact
   evaluator. The earlier queue historically looked at test-dev; this
   new protocol does not select epochs on it, but cannot retroactively
   make it historically unseen.
3. **Compute trade-off.** 960 took ~1.55× the training time. Measure
   full inference latency and peak memory before an efficiency claim.
4. **Research direction.** First run the pinned P2-head control at 640
   after 4-GPU smoke. Then evaluate whether a conditional allocation
   mechanism can approach 960's AP at materially lower deployment
   cost, compared against P2 and standard tiling/resizing controls.
   Existing foveation/region-adaptive work narrows novelty; no new
   method has yet been shown to beat these controls.

Raw local artifacts: `VISDRONE_OFFICIAL_640_S2027_RESULTS.csv`,
`VISDRONE_OFFICIAL_960_S2027_RESULTS.csv`, both `*_VAL_CLASSWISE.json`,
both `*_SIZE_RECALL.json`, both state JSONs. Remote originals live in
`/home/ubuntu/rz/progressive_blur_detection_research/research/` and
`runs/visdrone_official/`. Source images/labels were not modified.
