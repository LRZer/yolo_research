# Conditional next direction: VisDrone official-split tiny-object diagnostic

Prepared 2026-09-23 after the VOC P1 blur pilots did not beat their raw-image
scratch baseline. This is a platform and conventional-control experiment,
**not** a claim of an innovative detector. Proceed only if physical GPUs
4–7 are free and the current VOC P1 diagnostic has finished.

## Why this direction

The earlier 125-epoch YOLOv8n scratch queue (two seeds) gave VisDrone2019-DET
mAP50–95 about 0.1361±0.0015, but it merged original train+val and used
test-dev for every epoch. It is useful exploratory evidence, not a clean
paper evaluation. In its 640-input audit, 93.2% of test boxes were below
32² pixels and the median was around 9×10.5 pixels; tiny-scale information
loss is a plausible bottleneck that can be tested with standard controls.
Simple background smoothing for tiny objects is already studied in CVPR
2025 SET, so it is not a novelty premise.

## Official provenance split (created, read-only input data)

An independent script recovered original membership from the processed
dataset's `source_map.tsv`. It generated lists referencing, but not copying
or changing, existing YOLO image/label files. A task-owned symlink view keeps
Ultralytics-generated label caches under this research project rather than
the shared processed dataset directory. Counts are exactly original
train 6,471, val 548, test-dev 1,610; all 8,629 images and labels exist.
Test-dev appears only in `test.txt`, not train or val. Existing image and
annotation content and old baseline runs are unchanged. The first smoke
did generate a task-derived cache in the old processed directory, as noted
below. Checksums and paths
are in `VISDRONE_OFFICIAL_SPLIT_AUDIT.json` and the remote
`data_splits/visdrone_official/audit.json`.

## First baseline and decision gate

Train pinned YOLOv8n architecture **from random initialization** on official
train, seed 2027, 125 epochs, batch 64, input 640, workers 4, SGD and the
same default mosaic/color/flip schedule as the previous scratch queue.
Set `max_det=1000` consistently for the dense scene validation and later
test evaluation; the short smoke observed up to 317 objects in a val image,
and the older full-train audit has an image with 902 boxes.
Select `best.pt` by **official val only**, evaluate test-dev once after
deciding the protocol. Use physical GPUs 4–7 and NUMA 0. Configuration:
`configs/visdrone_official_yolov8n_scratch_640_s2027_e125.json`. The existing
older baseline cannot be treated as a numeric comparator because it trained
on 548 more images and used a different selection split.

After the clean baseline, measure AP by object scale and visualize misses.
Then test conventional higher-resolution input and/or a P2 head with matched
training/selection before proposing an image-processing mechanism. Any
putative method should improve over these simple controls, not just over an
under-resolved 640 baseline. A second seed and final held-out test are
required before claiming a robust improvement.

Metric caveat: the derived YOLO labels exclude VisDrone's category-0 ignore
regions and category-11 `others`. The official
[VisDrone DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)
ignores detections in those regions, whereas a plain Ultralytics AP pass on
the converted labels may count some as false positives. Thus the current
YOLO-val AP is an **internal matched metric**, not an official leaderboard
score. Before any paper claim or cross-paper comparison, export predictions
in official format and validate with the official evaluator or a carefully
checked port. Also, the team's older queue already inspected test-dev
metrics; it is held out from *this new protocol's* checkpoint selection,
not historically unseen to the researchers.

No VisDrone efficacy result is asserted here; the above is an experimental
commitment and a data-integrity audit.

The first smoke using the source-tree paths completed but Ultralytics wrote
a regenerable `labels/train.cache` in the shared processed dataset directory;
the original images and annotation text files were not changed. The final
data YAML now uses a task-owned symlink view. A second four-GPU, one-epoch
smoke confirmed separate train and val caches are generated inside this
research project's `data_splits/visdrone_official/view/labels/`. Formal
training must use this latter YAML only.
