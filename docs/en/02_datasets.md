# 2. Dataset preparation and 22 scratch YOLOv8n baselines

[Contents](../../README.md) · [中文](../zh-CN/02_datasets.md) · [All 22 result rows](tables.md)

## Scope, transfer and validation

The cross-domain stage broadened the research beyond generic VOC. Ten downloaded source packages yielded eleven detection views because LLVIP infrared and visible images were evaluated separately. They cover terrestrial animals, underwater organisms/litter, darkness, thermal UAV images, SAR ships, pedestrians, semantic open-vocabulary categories, aerial traffic and waste. **JODD was investigated but its official package was not obtained; no JODD baseline was trained.** Other shortlisted datasets were research candidates, not completed experiments.

Preparation included source/archive integrity checks, image decoding, actual counts rather than website headline counts, annotation semantics, original splits, bounding-box clipping, empty labels, duplicates, class coverage, object density and image/box-size statistics. Data were transferred and organized on the research server, converted into YOLO labels and exposed via views/symlinks without overwriting source images. A resumable dedicated manager ran the experiments and an independent audit checked CSV continuity, finite metrics, checkpoint readability and initialization settings.

There are **98,414 image-view entries and 789,474 retained boxes** across all eleven views. This is not a count of independent scenes: LLVIP modalities reuse the same scenes/annotations. The full historical audit contains per-class image incidence, object counts, sizes and empty-label statistics: [comprehensive report](../../archives/local/yolo26_research/COMPREHENSIVE_DATASET_AND_YOLOV8N_BASELINE_REPORT_2026-09-22.md), [155 class rows](../../archives/local/yolo26_research/COMPREHENSIVE_CLASS_COUNTS_2026-09-22.csv), [4,203 dimension/split rows](../../archives/local/yolo26_research/COMPREHENSIVE_IMAGE_DIMENSIONS_2026-09-22.csv).

| View | Classes in configuration | Train images / boxes | Evaluation images / boxes | Evaluation-split policy |
|---|---:|---:|---:|---|
| AP-10K | 54 | 8,018 / 10,394 | 1,997 / 2,634 | Fixed derived detection view; all names retained even if a class has no samples |
| DUO | 4 | 6,671 / 63,997 | 1,111 / 10,517 | Official train/test |
| ExDark | 12 | 4,800 / 16,593 | 2,563 / 7,117 | Author train+val merged; author test evaluated |
| HIT-UAV | 4 | 2,319 / 19,971 | 579 / 4,780 | Train+val merged; test evaluated; axis-aligned boxes only |
| HRSID-JPG | 1 | 3,642 / 11,047 | 1,962 / 5,922 | Author split-specific JSON; not PNG version |
| LLVIP infrared | 1 | 12,025 / 34,135 | 3,463 / 8,302 | Official paired train/test |
| LLVIP visible | 1 | 12,025 / 34,135 | 3,463 / 8,302 | Same scenes/labels, separate visible-image training |
| ORCA class-level seen | 24 | 9,290 / 27,320 | 2,744 / 8,954 | Seen-class subset; not the full open-vocabulary task |
| SeaClear | 40 | 6,888 / 25,349 | 1,722 / 6,200 | Locked local split manifest |
| VisDrone2019-DET | 10 | 7,019 / 381,959 | 1,610 / 75,102 | **Official train6471+val548 merged**, test-dev used for evaluation |
| ZeroWaste-f | 4 | 3,574 / 21,689 | 929 / 5,055 | Author train+val merged; test evaluated |

This suite's VisDrone split is **different** from the later blur/HBS/foveation split of train 6471/val 548. Its much lower 640 AP does not indicate that the later model suddenly improved on the same evaluation set.

## Resolved data issues

| Source | Finding | Actual handling |
|---|---|---|
| AP-10K | Pose data used for detection; some configured classes absent | Retained bbox semantics and names; zero-count classes documented, not fabricated |
| DUO | 953 out-of-bounds boxes; one clipped box had no area | Clipped953, skipped one invalid box; retained original74,515 source boxes separately |
| ExDark | Documentation image initially mistaken for a sample; suffix case difference | Corrected sample accounting and validator; one label/image suffix mismatch handled explicitly |
| HIT-UAV | `DontCare` and rotated annotation branch | Excluded148 `DontCare` boxes; used four valid axis-aligned classes only |
| HRSID | Combined JSON16,951 boxes differed from split JSON16,969 | Used authoritative split files, did not mix object sets |
| LLVIP | Five one-pixel-width boxes; labels use closed VOC coordinates | Converted consistently; retained valid one-pixel objects; paired modalities not treated as independent samples |
| ORCA | `category_id=-1` semantics and seen subset | Separated caption/open-vocabulary semantics from malformed annotation claims |
| VisDrone | Ignore/others categories and three zero-height boxes | Filtered per converted-label protocol; not the official challenge evaluator |
| ZeroWaste | 138 out-of-bounds objects | Applied explicit bbox clipping during conversion |
| Archive metadata | AppleDouble `._` files looked like bad image files | Excluded metadata from image scans without deleting actual source data |

The original audits include all conversion counts, class histograms, min/median/P90 object density, 640-letterbox dimensions and relative archive/storage sizes. Dataset acquisition and cleanup are part of the completed work, not merely background context.

## Frozen baseline protocol

Each view used **YOLOv8n from scratch**, seeds **2005 and 2027**, **125 epochs**, **640 input**, **batch 64** and the same locked optimizer/augmentation configuration, with a dataset-specific class count. Original run manifests and `args.yaml` are under [server baseline runs](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines). There are22 completed runs, not a pretraining checkpoint transferred between datasets. All44 best/last files were checked on the server; model binaries are not redistributed here.

The test/evaluation view was inspected every epoch by explicit project choice. Thus it was not blind. **Fixed epoch 125** is the primary endpoint; test-selected best is shown only as secondary. The two seeds estimate a narrow seed sensitivity, not comprehensive confidence intervals.

![Both seeds, fixed125 and secondary best](../../figures/dataset_22_baseline_endpoints.png)

## Fixed-125 AP95 results

| View | seed2005 | seed2027 | Two-seed mean |
|---|---:|---:|---:|
| AP-10K | 0.6142 | 0.6079 | 0.61105 |
| DUO | 0.6151 | 0.6196 | 0.61735 |
| ExDark | 0.3561 | 0.3516 | 0.35385 |
| HIT-UAV | 0.5561 | 0.5602 | 0.55815 |
| HRSID-JPG | 0.6357 | 0.6248 | 0.63025 |
| LLVIP infrared | 0.6111 | 0.6136 | 0.61235 |
| LLVIP visible | 0.4558 | 0.4728 | 0.46430 |
| ORCA class-level seen | 0.2845 | 0.2849 | 0.28470 |
| SeaClear | 0.5740 | 0.5600 | 0.56700 |
| VisDrone merged-train / test-dev | 0.1372 | 0.1351 | 0.13615 |
| ZeroWaste-f | 0.2059 | 0.2198 | 0.21285 |

Values above retain the original rounded training CSV precision. [Machine-readable endpoints](../../results/dataset_22_run_endpoints.csv) also include P/R/AP50, derived H, best epoch/AP, last train/val box-cls-dfl losses and cumulative time. [All epochs](../../results/all_canonical_epoch_curves.csv) contain every original field. No per-class AP unavailable from these runs has been invented; the per-class **dataset counts** are distinct from model AP.

LLVIP's infrared advantage was checked against shared scenes and annotations, not explained by different labels. Lower ORCA/VisDrone/ZeroWaste scores were investigated through domain/task mismatch, crowding, scale, ignored regions and class structure; they were not asserted to be bugs solely because AP was low. These baselines support later domain-specific selection but do not establish a universal cross-dataset ranking or an improved detection method.
