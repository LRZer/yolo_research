# VisDrone official-provenance split: 640 scratch YOLOv8n baseline

Date: 2026-09-23 (Asia/Shanghai). Status: 125-epoch training and independent
official-val-*split* re-evaluation verified. The metric is the internal
Ultralytics metric on converted YOLO labels, **not** the official VisDrone
toolkit score; see `VISDRONE_EVALUATION_AUDIT_2026-09-23.md`.

## Protocol and integrity

- Official source membership recovered without modifying original images or
  annotations: 6,471 train, 548 val, 1,610 public test-dev. Test-dev is
  not used by this run for per-epoch evaluation or checkpoint choice.
- Random-initialized pinned YOLOv8n architecture (YAML SHA256
  `44c9efc76c9dad388ad1ef6c4efb37e47ec885fe2ad12f110c76f6cdb750889d`),
  Ultralytics `8.4.146`, seed 2027, SGD, 125 epochs, batch 64, input 640,
  workers 4, standard mosaic/color/flip settings, `max_det=1000`.
- Physical GPUs 4–7, NUMA 0. Task-owned symlink data view keeps label caches
  in this research project. The manager returned `verified_complete` at
  04:16:46 CST: exactly 125 finite, contiguous result rows with readable
  `best.pt` and `last.pt`. Training CSV cumulative time 1015.27 s
  (16.92 min). EDAC uncorrected counters for both memory nodes were 0 after
  completion; this does not negate historical hardware errors.

## Same-protocol independent official-val-split score

`best.pt` was selected at epoch 115 by raw official-val-split images. A
separate `YOLO.val` pass at 640, confidence 0.001, NMS IoU 0.7 and
`max_det=1000` yielded mAP50 **0.309082** and mAP50–95 **0.170547**.
The built-in epoch-115 mAP50–95 was 0.17051; the small re-evaluation
difference does not affect interpretation. The fixed epoch-125 row has
mAP50–95 0.16999, close to the best row.

| Class | AP50 | AP50–95 |
|---|---:|---:|
| pedestrian | .3298 | .1330 |
| people | .2582 | .0851 |
| bicycle | .0721 | .0244 |
| car | .7397 | .4862 |
| van | .3571 | .2359 |
| truck | .2479 | .1542 |
| tricycle | .2019 | .1029 |
| awning-tricycle | .0911 | .0541 |
| bus | .4311 | .2870 |
| motor | .3619 | .1427 |

The existing YOLO labels give the official val split 38,759 positive boxes.
At 640 letterbox scale, median box width/height are about 9.88×12.67 input
pixels; 92.44% are smaller than 32² pixels, 45.18% have a shortest edge
below 8 pixels, and 14.29% below 4 pixels. At 960, the shortest-edge-below-4
fraction would fall to 3.56% purely geometrically. These measurements
motivate the pre-registered 960 conventional control but do not prove it
will increase AP. Full train/val/test geometry is in
`VISDRONE_OFFICIAL_BOXSIZE_AUDIT.json`.

The older two-seed 640 queue used a different train and checkpoint-selection
split (merged train+val, per-epoch public test-dev). Its 0.1361±0.0015
test-dev mAP50–95 is **not directly comparable** with this 0.1705 val score.
Nor can the present internal metric be compared directly with publications
using the official ignore-region evaluator.

Artifacts: `VISDRONE_OFFICIAL_640_S2027_STATE.json`,
`VISDRONE_OFFICIAL_640_S2027_RESULTS.csv`,
`VISDRONE_OFFICIAL_640_S2027_VAL_CLASSWISE.json`, and remote checkpoint
directory `/home/ubuntu/rz/progressive_blur_detection_research/runs/visdrone_official/yolov8n_scratch_official_640_s2027_e125`.
