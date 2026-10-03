# 6. Global/local classification and normalization

[Contents](../../README.md) · [中文](../zh-CN/06_global_local_bn.md)

## G4/G5: exact two-view model

The next hypothesis was that a classifier could combine global context and enlarged target detail. For each image, the global branch used the native500 canvas. The local branch cropped the **GT bird box**, no extra margin, floor left/top and ceil right/bottom, clipped to nonempty bounds; its long side was bicubically scaled to500 and centered on the same canvas. All11,788 boxes were checked. Crop dimensions and enlargement factors were previewed before training. This step truly enlarges the bird, unlike A–F's unscaled protected region.

G4 received **original global + local crop**. G5 received **strict normalized-zero global canvas + the same local crop**. G5 did not load an original global image or convey its dimensions, but the zero branch still went through the backbone and normalization. It was not a single-crop model with the branch removed.

One shared ResNet50 processed `[global_batch; crop_batch]`. Global-average-pooled2048-dimensional vectors were concatenated to4096 and fed into one 200-way linear classifier. Parameters=24,327,432. Shared convolution/BN layers do not mean half as many input views: there are two processed500×500 views per pair. The model was scratch seed 2005, SGD and augmentations as before, same200-epoch LR horizon/actual125. GT boxes were needed in train **and test**.

Training used physicalGPU2/3, NUMA0, DDP: each rank microbatch8pairs (16 views), two ranks×accumulation4 gives64pairs/update. SyncBN pooled32views per synchronized microstep. Workers2/rank, total4; evaluation16pairs/rank. Common RGB augmentation was deterministic from seed, epoch, image ID and aligned across views; flipped half-open bbox geometry was tested. The exact sampler visited5,994/5,794 images once without padded duplicate/drop samples. There were94updates; final42-image update comprised16/16/10 global microsteps. Loss scaling used rank sumCE×world_size/current-global-update-count to preserve the correct batch mean.

Initial AMP scale 65,536 smoke skipped an update and **blocked launch**; its diagnostic was retained. Scale1,024 passed the full smoke with zero skipped updates. Source/config/bbox/stats hashes and checkpoint controls were recorded. [Protocol](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_DDP23_E125_PROTOCOL_2026-10-01.md), [32-bit numerical/AMP controls](../../archives/remote/classification/reports), [preview sizes](../../archives/local/foveated_classification/previews/cub_global_local_gt_long500_v1_2026-10-01/dimensions.csv).

## Complete paired results

| Fixed125 | Train Top1/Top5 | Train loss | Test Top1 | Test Top5 | Test loss | Secondary best |
|---|---|---:|---:|---:|---:|---|
| G4 global+crop |100% /100%|0.009257|52.278219%|78.736624%|1.960690|epoch72,Top1 53.883328%|
| G5 blank+crop |100% /100%|0.009684|64.687608%|86.555057%|1.456469|epoch118,Top1 65.429755%|

G5 exceeds G4 by **12.409389 Top1 percentage points and 7.818433 Top5 points**. Both passed `verified_native_e125_ddp_pair`: exact125 finite rows, source/config/stats, both checkpoint epochs and all train/test sample/update counts. [Independent final audit](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_DDP23_FINAL_2026-10-01.json).

G4's mean epoch was131.999s vs G5's69.258s, with changing shared-resource load. This is a historical observation, **not a controlled architectural speedup**. Comparing G5 to A's49.9655% also changes crop magnification, privileged input, two-view processing, distributed normalization and runtime, so it cannot isolate the global context effect. The G4/G5 pairing is the narrower useful comparison.

## Why shared BN can confound this test

Batch normalization computes

$$\mu_B=\frac1m\sum_jx_j,\qquad v_B=\frac1m\sum_j(x_j-\mu_B)^2,\qquad z_j=\gamma\frac{x_j-\mu_B}{\sqrt{v_B+\epsilon}}+\beta.$$

During training, G4 mixes full-image features (small bird plus context) with crop features (enlarged bird) in the same batch moments. G5 mixes crop features with a uniform blank-input feature stream. Both kinds of branch can change means/variances and running evaluation statistics. A blank image supplies no scene content but still supplies values to BN. Therefore G5 is **not** a clean single-local-image baseline, and G5>G4 does not prove that natural background is always harmful. Distributions, magnification, correlated views and normalization are competing hypotheses, not verified causes.

## G6: split running statistics, shared weights

G6 was designed as a minimal G4 ablation:53BN layers keep separate global/local running means, var and counters, but **share convolution and BN gamma/beta parameter objects**. Parameters remain24,327,432; detector/classifier initialization was checked tensor-identical to G4. This is split-statistics BN, not two independent encoders or two entirely independent affine BN sets.

Each branch now has16views per synchronized microstep instead of a mixed32. Separate statistics therefore also change effective BN sample count; it cannot be attributed solely to “removing distribution mixing.” CPU tests checked finite gradients, serialization/parameter aliasing and evaluation-state separation. DDP normalization checks and a170train/34test smoke completed 3 updates with zero skipped steps and identical rank model-state hashes. [Protocol](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_SPLIT_BN_E125_PROTOCOL_2026-10-02.md).

## Exact current state: user stopped at 50

G6 launched2October15:24CST onGPU2/3. At the user's explicit stop request, it was terminated at **18:20:47CST after 50 complete epochs**; manager, training ranks and task-owned processes were scoped and checked. Last50 and best 46 were CPU-readable, finite and provenance-consistent. It is **not running, not completed 125, and not silently resumed**. The subsequent GPU6 request led to resource inspection, but **no GPU6 training was launched** in the collected record.

| G6 incomplete endpoint | Value |
|---|---:|
| Completed epochs |50/125|
| Train Top1 /Top5 |91.508175% /99.449449%|
| Train loss |0.321017|
| Test Top1 /Top5 |43.182603% /73.023818%|
| Test loss |3.051473|
| Best observed Top1 |43.803935%,epoch46|

[Authoritative stop record](../../archives/remote/classification/reports/G6_splitstats_DDP23_2026-10-02_USER_STOP.json), [original50-row CSV](../../archives/remote/classification/runs/resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23/results.csv). Earlier live-run Markdown remains historical and is superseded by this stop record.

![G4/G5 full curves; G6 dashed incomplete prefix](../../figures/cub_global_local_and_incomplete_bn.png)

The 50-epoch prefix can motivate further diagnosis but cannot be compared to125-epoch G4/G5 as a completed BN verdict. No separately trained crop-only model, comprehensive BN batch-count control, multiple seeds or blind confirmation has been completed. The completed result is G4/G5's conditional pairing; BN causality remains **unresolved**.
