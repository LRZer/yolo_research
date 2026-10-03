# 5. CUB-200-2011: image representations, A–F

[Contents](../../README.md) · [中文](../zh-CN/05_cub.md) · [Complete epoch/endpoints](tables.md)

## Simplified research question

The project simplified from multi-object detection to single-subject fine-grained classification: **must the whole image retain uniform high-frequency detail?** GT boxes fixed the protected region initially, removing learned search, scanpaths and stopping policies from the experiment. Oracle representation quality was to be established before any predicted fixation or actual nonuniform compute allocation.

The initial discussion considered ResNet18/224, radial hard/gradual/random fovea, radius and blur sweeps, Oxford-IIIT Pet, learned fixation and multiple glimpses. The user instead selected **scratch ResNet50 and native500**. The actual completed experiment was box-protected peripheral processing, not a comprehensive radial-foveation/radius sweep. Oxford Pet and those later active-vision stages were **not run**.

## Dataset audit

CUB-200-2011 has **200 classes,11,788 images,5,994 official train /5,794 official test**, one main-object bbox per image,15 part types and 312 attributes. Local full profiling verified176,820 part rows,141,407 visible parts,794,258 click rows,3,677,856 image-attribute rows and 371,047 positive attributes. Two images had an extra attribute-field formatting anomaly, retained and documented; every image still had312 attribute rows. The class attribute matrix is200×312.

Per-class totals range41–60. The image width min/median/max is121/500/500 and height120/375/500;1,152 distinct dimensions. RGB images number11,780; eight images are grayscale and are converted to RGB in the loader. Bbox area/image area P10/median/P90=0.1391/0.3174/0.5923. Thirty supplied boxes exceed image bounds by up to32pixels and are consistently clipped. [Full profile JSON](../../archives/local/foveated_classification/CUB_DATASET_FULL_PROFILE_2026-09-25.json), [200-class table](../../results/cub_all_200_class_counts.csv).

Reference segmentation masks were downloaded/validated but **not used** by A–F. There are11,786 L masks and two LA masks; values0/51/102/153/204/255 show that they must not be assumed to be strict binary masks without a stated conversion. Part/attribute annotations likewise were not classification inputs. The download/audit preserved the official split and documented the dataset's noncommercial research/education restrictions and potential ImageNet overlap; scratch training avoided silently importing ImageNet pretraining.

![All200 classes, train/test counts](../../figures/cub_class_distribution.png)

Common native dimensions:500×333(1,820),500×375(1,192),500×400(366),500×334(364),333×500(305),500×357(239),500×332(235),500×500(205),400×500(166),500×335(157). These motivate preserving valid original pixels and padding rather than distorting every image to a square. Padding500 does not imply every source is500×500; compute still covers the padded canvas.

## Common training contract

- Torchvision ResNet50, random initialization,200-way cross-entropy, seed **2005**.
- Original valid RGB pixels are **not resized** for A–F. Center pad to500×500; normalize using official-train valid-RGB statistics. No test-derived normalization or target crop in A/B/E/F.
- Training augmentation: horizontal flip and common mild RGB brightness/contrast/saturation jitter0.9–1.1. No RandomResizedCrop, no rotation/large geometry/mix augmentation in this experiment. C/D box geometry follows the common flip.
- SGD base LR0.025, minimum LR1e−5, warmup5epochs, momentum0.9, weight decay0.0005.
- Effective batch 64, microbatch16×accumulation4, workers 4;94optimizer updates/epoch including the remainder. Every complete epoch sees5,994 train and 5,794 test images.
- Originally planned200epochs; after test progress was inspected, the user changed actual length to **125**. Configuration/hash and **the first 125 entries of the 200-epoch cosine schedule** were retained. Last LR0.0080742119364. These are **not** cosine schedules reaching their endpoint at 125.
- A–E ran sequentially on physicalGPU0/NUMA0. F later ran on sharedGPU6/NUMA0 in a different existing software environment. This is a disclosed comparability limitation, not a silent identical-runtime claim.

Each epoch evaluated the entire official test set by user choice. There was no separate development split; test was not blind and the length revision followed its inspection. Fixed125 last is primary. Best is secondary and test-selected.

## Exact transformations

Let $L$ be the original long side,$s=L/500$. All transforms operate on valid image pixels **after common RGB training augmentation, before padding/normalization**.

| Group | Actual input construction | GT needed at test? |
|---|---|---|
| A original | Native RGB, mean-color center padding | No |
| B global blur | `scipy.ndimage.gaussian_filter` float32,sigma8s,reflect,truncate3; round back to uint8 | No |
| C context removal | GT box expanded10% of its width/height **per side**, protected pixels unchanged; outside smooth transition to official-train mean RGB over70s pixels | **Yes** |
| D progressive blur | Same protected region/transition as C; continuous interpolation of sigma0/2/4/8×s Gaussian images | **Yes** |
| E grayscale | PIL `RGB→L→RGB`, three replicated grayscale channels; no blur, same spatial extent | No |
| F Scharr magnitude | Luminance/255, Scharr dx/dy scale1/32, reflect101; magnitude,specified fixed gain4,clip/round to uint8; repeat three channels | No |

For C/D, outside distance$d$ from the expanded rectangle is mapped through $t=\operatorname{clip}(d/(70s),0,1)$ and $a=t^2(3-2t)$. C uses $I'=(1-a)I+a\mu_{train}$; D maps the smooth strength to the Gaussian bank. This is **not hard crop deletion**, exact foreground segmentation or a foveal disk centered on the bbox. Box interior background is also protected. Native colors/details inside the protected rectangle remain unchanged.

F computes $E=\operatorname{round}(255\operatorname{clip}(4\sqrt{g_x^2+g_y^2},0,1))$. Derivatives precede padding so there are no artificial canvas edges; its edge padding is **black**, unlike mean RGB image padding. It removes color and most low-frequency appearance, not simply “adding texture features.” Per-image normalization is not used to amplify weak edges. F's transform was approved through previews before training. [Exact trainers](../../archives/remote/classification/src), [frozen configs](../../archives/remote/classification/configs), [preview dimensions/manifests](../../archives/local/foveated_classification/previews).

## Fixed125 results, including training accuracy

| Group | Train Top1 | Train Top5 | Train loss | Test Top1 | Test Top5 | Test loss | Test Top1−A |
|---|---:|---:|---:|---:|---:|---:|---:|
| A original |100%|100%|0.014418|49.965482%|76.216776%|2.062928|—|
| B global blur |100%|100%|0.016150|29.668623%|57.386952%|3.210199|−20.296859pp|
| C GT context removal |100%|100%|0.013626|53.331032%|79.064550%|1.900559|+3.365551pp|
| D GT progressive blur |100%|100%|0.014064|53.451847%|79.703141%|1.887603|+3.486365pp|
| E grayscale |100%|100%|0.014786|27.096997%|55.799103%|3.247952|−22.868485pp|
| F Scharr |100%|100%|0.015649|17.742492%|41.491198%|4.088853|−32.222989pp|

All rows use the **last** checkpoint. F best was epoch 79 Top1=19.054194%, not the primary 17.742492%. Every group's best epoch/Top1/Top5/loss is available in the [endpoint CSV](../../results/cub_primary_and_secondary_endpoints.csv); no best-only replacement hides late degradation.

![Complete learning curves: accuracy and both losses](../../figures/cub_A_F_learning_curves.png)

## Cost and interpretation

| Group | Mean train seconds | Mean full-test seconds | Mean total/epoch seconds |
|---|---:|---:|---:|
| A |37.996|11.864|49.860|
| B |37.988|28.316|66.304|
| C |37.853|22.480|60.333|
| D |75.245|67.426|142.672|
| E |37.973|11.868|49.842|
| F |79.566|24.507|104.073|

These include data processing and full-set loops, not isolated model latency. F shared a GPU and changed software; its timing is not a fair complexity benchmark against A–E. Every group still inputs500×500 to ResNet; blur/grayscale/edges do not reduce backbone FLOPs.

Findings are conditional: B strongly blurs the **bird itself** and loses20.30 points; E shows the value of color under this training regime; F's stand-alone edge representation loses32.22 points and does not show that adding edges as an auxiliary signal would fail. C/D's3.37/3.49 point increases support target-protected peripheral representations **when location is given**. D exceeds C by only 0.120815 point (~seven correctly classified test images) while total epoch cost is2.36×; no single-seed evidence establishes a stable advantage for multiscale blur over mean-color attenuation.

All six training accuracies reach 100% with low training loss, whereas test accuracies/losses remain much worse. This is a strong generalization gap, not a proof that every error is memorization or that a different epoch alone fixes it. No classwise model accuracy/confusion matrix was recorded for these runs; dataset per-class counts must not be mislabeled as per-class performance.

## Audits, recovery and delivered artifacts

A–E each passed `verified_revised_e125`: exactly 125 continuous finite epochs, readable last, correct source/config/stats, full image counts, expected LR/update counts and task stop. D's server reboot after complete 15 epochs was audited before resuming from 16; no partial epoch was counted. F completed 125 and its CPU audit accepted the recorded source/runtime change. Startup BERT/EDAC records and prior uncorrectable memory incidents remain limitations; zero new counts after reboot do not prove repaired hardware.

[A–E final report](../../archives/local/foveated_classification/CUB_RESNET50_FIVE_GROUP_E125_FINAL_REPORT_2026-09-25.md), [D recovery](../../archives/remote/classification/reports/CUB_D_REBOOT_RESUME_2026-09-25.md), [F final report](../../archives/remote/classification/reports/CUB_RESNET50_F_SCHARR_E125_RESULT_2026-10-01.md), [individual server runs and audits](../../archives/remote/classification/runs).

The completed output is a reproducible representation benchmark and an Oracle localization/appearance signal, not a segmentation→classification deployable system. No segmentation model or predicted bird-localization pipeline was trained in this stage.
