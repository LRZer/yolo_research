# 4. VisDrone: feature smoothing, Oracle blur and foveated inference

[Contents](../../README.md) · [中文](../zh-CN/04_visdrone.md)

## Why the dataset changed

Clean VOC contained relatively few tiny targets. The later VisDrone view preserved official **train 6,471 / val 548 / test-dev1,610**, rather than the earlier merged-train baseline suite. It contained343,200 training boxes,38,759 validation boxes and 75,102 test-dev boxes after the documented conversion. Ignore/others semantics were not fully reproduced by the internal YOLO evaluator; scores below are **not official VisDrone challenge scores**.

Resolution/head controls used scratch YOLOv8n, seed 2027,125 epochs, batch 64. Development-selected independent best results were AP50/AP95=0.309082/0.170547 for standard640 (best 115),0.350077/0.198526 for P2-640, and 0.403649/0.232476 for standard960 (best 111). All ten classes improved with 960 over standard640. Standard960 training took1,573.05s vs1,015.27s at 640. P2 architecture, size recall, ignored-region and protection-area audits were saved; P2 and 960 are different cost interventions, not matched-FLOP ablations.

The 125-epoch strong **training-only** Oracle960 run (probability0.5, raw validation) completed, but training-validator best AP95≈0.22687 was below the standard960≈0.23240 and took5,274.23s. Built-in best and independent standalone AP are labeled separately. [Resolution evidence](../../archives/local/voc_pgb_yolov8n/VISDRONE_OFFICIAL_960_S2027_VAL_CLASSWISE.json), [original runs](../../archives/remote/progressive_blur/runs/visdrone_official).

## Frozen-model, three-strength test-time blur

The mature detector was frozen. Light/moderate/heavy Gaussian sigma3/8/16 were applied at input scale. Raw and sham-processing controls were evaluated in the same run protocol.

| Frozen model / evaluation split | Raw | Light3 | Moderate8 | Heavy16 |
|---|---:|---:|---:|---:|
| YOLOv8n960 best125, predicted-box protection, val548 |0.232551|0.221578|0.213893|0.211288|
| Same checkpoint, **GT-box Oracle protection**, val548 |0.232551|0.261151|0.260525|0.256711|
| YOLO26n640 O2O last150, **GT Oracle**, VOC test4,952 |0.577006|0.602777|0.614255|0.628525|

![Three strengths, with model/split boundaries](../../figures/testtime_blur_GT_vs_prediction.png)

For the VisDrone GT arm, AP50 was0.403680/0.449207/0.448962/0.442570. For YOLO26 O2O it was0.778530/0.796847/0.801477/0.812868. All20 VOC classes increased under heavy Oracle. The O2M Oracle series was **not completed**; only its raw/sham evidence was retained. This avoids claiming three-strength results for a head that was not actually fully evaluated.

GT processing uses test annotations, embeds target geometry in the input and cannot be deployed without a localization source. It is a privileged diagnostic, not an annotation-free improvement and not necessarily a mathematical upper bound. Predicted and GT protection also differed in margins/feathering; their difference does not isolate annotation accuracy alone. A historical MC1 uncorrectable-error window is retained in the report and limits hardware confidence. [GT VisDrone series](../../archives/local/voc_pgb_yolov8n/testtime_groundtruth_blur_960), [predicted series](../../archives/local/voc_pgb_yolov8n/testtime_predicted_blur_960), [YOLO26 series](../../archives/local/yolo26_research/voc_yolo26n_testtime_gt_oracle_640).

## SET HBS-only transfer: a controlled negative result

The project adopted the **training-only HBS component** from SET, not its complete method. At each of three detection scales, a residual two-convolution denoiser modified background features while restoring GT foreground features. An additional detector loss supervised the modified branch. The main comparator used the same extra loss with unmodified features (**identity-extra-loss**), so a benefit could not be attributed merely to a second supervised pass. SET's API component was not implemented.

Both arms were scratch125, seed 2027,960, batch 64, SGD, common augmentations and val 548, on GPU0–3/NUMA0. Initial detector tensors, branch gradients, annotation-free inference equivalence, checkpoint/resume and four-card1%/1epoch smoke were checked. Inference only used the original detector path; GT and denoiser were training-only. [Protocol](../../archives/local/voc_pgb_yolov8n/SET_HBS_ONLY_VISDRONE_960_E125_PROTOCOL_2026-09-24.md).

| Independent last125, FP32 | P | R | AP50 | AP95 |
|---|---:|---:|---:|---:|
| Identity-extra-loss |0.492073|0.403812|0.396232|0.229070|
| HBS-only |0.508415|0.378393|0.380833|0.214733|

AP95 difference−0.014337, nine classes decreased: **no-go**. Tiny objects (<16px sqrt-area in a fixed hypothetical 640 letterbox reference) numbered27,103/38,759. Their IoU.5 recall was0.6283 HBS vs0.6570 identity; IoU.75 recall0.2655 vs0.2987. These are diagnostic recalls, not COCO AP_small. Training3,523.28 vs2,150.44s (~1.64×). Forward-only inference1.384/1.382ms was effectively unchanged, but the saved model still contained denoiser parameters (~4.05M vs3.01M): no computation overhead is not the same as zero checkpoint storage overhead.

![HBS identity-matched, all10 classes](../../figures/visdrone_hbs_classwise.png)

## Foveated crop prototype and evaluator correction

After reviewing attention/foveation papers, a frozen960 last 125 detector first processed the full image, scored cells of a4×4 grid by coarse predicted confidence, selected four cells with 10% halo, processed each crop at 960 and fused classwise boxes. Fixed central four cells provided a weak equal-call comparator. No GT was used to select adaptive cells. Every full+four-crop condition uses **five 960 model inputs**, not fewer FLOPs/pixels.

An initial `multi_label=False` full AP95=0.215110 differed from the validator candidate policy. Comparing crop results to that number would falsely indicate improvement. The corrected common `multi_label=True`, conf.001, IoU.7, max_det1000 protocol gave:

| val548 | P | R | AP50 | AP95 | Retained predictions at evaluation threshold |
|---|---:|---:|---:|---:|---:|
| Full image |0.495904|0.403318|0.401083|0.230872|492,024|
| Full + fixed central4 |0.460974|0.392365|0.376783|0.217609|529,844|
| Full + predicted top4 |0.463859|0.409765|0.392441|0.226803|541,555|

Adaptive crops beat a weak fixed-cell comparator but lost to the cheaper full pass. Stage timing was5.439s full-batch,77.968s fixed crop stage,107.496s adaptive crop stage. These stages differ in batching and exclude some loading; they are **not** a controlled deployment latency claim. [Unified pilot](../../archives/local/voc_pgb_yolov8n/FOVEATED_VISDRONE_CROP_PILOT_VAL548_MULTILABEL.json).

## Coverage, localization and false positives

GT-center tile analysis found top4 coverage≈73.02% for an Oracle density selector vs39.98% fixed central; predicted selection≈74.13%, and eight cells≈92.51%. Coverage is not detection AP and counts neither full containment nor classification correctness. Fixed tiles can truncate large vehicles: bus29/251 (11.55%) and truck98/750 (13.07%) were not fully contained in any eligible grid crop, vs pedestrian≈2.25%. This motivated boundary/scale hypotheses but did not prove them.

At a common confidence0.25/IoU0.5 working point, paired provenance showed:

| Class | GT | Full TP→adaptive TP | Lost full TP | New TP | Full FP→adaptive FP |
|---|---:|---:|---:|---:|---:|
| pedestrian |8,844|3,746→5,078|13|1,345|2,428→7,105|
| people |5,125|1,605→1,984|5|384|1,180→2,122|
| truck |750|236→308|0|72|235→784|
| bus |251|113→147|0|34|69→252|

Thus truck/bus AP loss was not primarily explained by losing original full-pass TP at this threshold. Additional local false positives and ranking/calibration were important observed changes. More final boxes is not automatically more correct detection. Entire-crop light blur worsened AP95 from 0.226803 to0.211691; it blurred foreground as well and was rejected.

## Background-only crop blur, mask safety and transfer

The revised renderer left the full branch raw and blurred only predicted-safe background in selected crops. A fixed mask policy used coarse boxes≥0.01,20% per-side expansion, minimum8pixels at full960, sigma3 in crop960 and feather8. No GT entered deployed preprocessing.

On val 548, of **28,031 GT fully inside selected crops** (not all 38,759),27,732 (98.93%) had≥95% protected pixels; small<8 reference-scale targets4,602/4,673 (98.48%). Protected pixel area47.90%, blur-permitted52.10%. Confidence.001 protected99.36% but retained only 45.75% processable area; confidence.05 protected95.67%. This is a coverage/area trade-off, not AP. On test-dev, protection declined to49,426/51,388=96.18%, tiny11,870/12,832=92.50%, with 62.82% area permitted for blur. The full branch remained unmodified for targets outside the selected-crop denominator.

| test-dev1,610 | P | R | AP50 | AP95 | Retained predictions |
|---|---:|---:|---:|---:|---:|
| Full |0.450111|0.343734|0.322967|0.179863|1,366,078|
| Full + adaptive4 |0.429730|0.358506|0.317616|0.178484|1,595,348|
| Adaptive4, keep only local candidates≤64px at full960 |0.434143|0.360188|0.321098|0.180785|1,593,195|
| Adaptive4, predicted-background blur |0.429113|0.355263|0.314890|0.176827|1,597,902|

The small-only rule removed53,369/3,228,926 crop candidates (1.65%), preserved five calls, and added only 0.0922 AP95 point over full while AP50 stayed below it. It was not an established practical gain. Test-dev had been historically inspected, so this is not blind generalization.

## Explicit GT-mask diagnostic and final decision

One fixed val 548 diagnostic replaced only the local mask with GT protection:

| Common frozen model/candidate protocol | AP50 | AP95 |
|---|---:|---:|
| Full |0.401083|0.230872|
| Full + raw adaptive4 |0.392441|0.226803|
| Full + predicted-background-blur4 |0.390437|0.225561|
| Full + GT-background-blur4 |0.396973|0.228695|

GT improved over the predicted mask by 0.3134 AP point but remained0.2178 point below full. At confidence.25/IoU.5, raw/predicted/GT crop totals TP=24,047/24,101/24,091 and FP=24,846/25,839/24,405. GT reduced FP relative to predicted blur without improving the cheap full baseline. The predicted mask used296,082 boxes vs38,759 GT boxes and protected47.90% vs35.02% crop area; the change is not only correction of missed targets.

![Matched val/test-dev AP](../../figures/foveation_val_and_testdev_endpoints.png)

![Per-class true and false positives, common working point](../../figures/foveation_TP_FP_by_class.png)

All10-class AP, total predictions, candidate counts, masks and working-point counts are in [foveation endpoints](../../results/foveation_same_protocol_endpoints.csv), [fixed-threshold class counts](../../results/foveation_fixed_threshold_classwise_TP_FP.csv), [diagnostic ledger](../../results/diagnostic_counts_ledger.csv) and source JSONs. Cropping, threshold/size filtering, all-crop blur, predicted-safe blur and GT-safe blur were all investigated; none established a robust compute-efficient detection improvement. Annotations remained available for diagnostics only, and the current crop route was rejected rather than repeatedly tuned on the same val set.
