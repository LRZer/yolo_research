# 7. Engineering, evaluation and reproducibility

[Contents](../../README.md) · [中文](../zh-CN/07_reproducibility.md)

## What can be reproduced from this repository

The repository supplies archived research source/configs, original result tables, numerical diagnostics, model/data/config fingerprints, launch/stop/recovery records, tests, strict audits, and scripts to regenerate the publication tables/figures. It does not distribute original datasets, model binaries or complete third-party framework checkouts. Reproduction requires separately obtaining authorized data and matching the pinned framework/source state. Historical absolute paths are provenance, **not portable paths or instructions to run jobs on someone else's server**.

| Stage | Original runtime evidence | Important changes |
|---|---|---|
| YOLO26 dual-head | `/home/ubuntu/rz/yolo_study/.conda/yolo26`, pinned Ultralytics research checkout | Dirty research source hashes matter; version string alone is insufficient |
| VOC/VisDrone frontend/HBS | Isolated project implementations/configs and manager manifests | Custom module/trainer imports required for checkpoint loading; framework loss reduction mattered |
| CUB A–E | Recorded `/home/ubuntu/miniconda3/envs/LZCyolo/bin/python` | This path later disappeared; do not claim A–E used the current environment |
| CUB F/G4/G5/G6 | `/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python` | PyTorch2.6.0+cu118,torchvision0.21.0+cu118; F runtime change disclosed |
| Local prototype/tests | Local conda PyTorch and archived CPU reports | CPU tests are not real-model GPU/AP evidence |

No shared environment was changed for this publication. Switching from single GPU accumulation to DDP/SyncBN can change batch moments, augmentation order and floating-point reduction even at equal effective batch and seed. Exact numerical reproduction was proved for particular YOLO26 controls, **not universally promised across card counts and environments**.

### Research source coverage

The [362-file Python index](../../results/code_index.csv) includes data acquisition, conversion, integrity auditing, resumable transfer and the 22-run queue in [dataset engineering](../../code/dataset_engineering), not only the newest classifier. The original YOLO26 baseline launcher/config is in [baseline launcher](../../code/historical/baseline_launcher). Historical local analyses, GPU smoke, training transforms and task managers are retained in their original project directories.

[Selected framework overlays](../../code/framework_overlays) preserve the actual recorded loss/config/loader/trainer modifications for quality transfer and ASAC, plus SARC's candidate convolution/task/model definitions. These are **separate historical source states**, not one combined installable framework: choose the matching run protocol and source hashes. SARC overlays remain candidate-only; their presence does not imply formal GPU training. Full unrelated vendor documentation/checkouts are excluded, and [the upstream AGPL text](../../LICENSES/Ultralytics-AGPL-3.0.txt) is preserved.

## Metric dictionary

Detection AP50 is class-average precision at IoU.5; AP95 averages IoU.50:.05:.95. The Ultralytics/internal implementation uses its specified interpolation/matching/NMS candidate policy; it is not automatically the original VOC2007 11-point protocol or VisDrone official toolkit. Reported P/R usually correspond to the evaluator's selected operating point; they are not counts at a common fixed confidence. $H=2PR/(P+R)$ is explicitly a derived number and not the native F1 maximum.

Fixed-threshold TP/FP diagnostics use a declared confidence and IoU, one-to-one class-compatible matching. FN=GT−TP; predictions=TP+FP only within that same counted working point. Full AP prediction counts at conf.001 differ from conf.25 counts. A size-recall diagnostic is not AP_small, and a protection/coverage rate requires its eligible-GT denominator.

Classification Top1/Top5 are percentages of test images whose correct class appears in the first 1/5 predictions. Cross-entropy measures probability assignment, not just correctness; decreasing train loss with worsening test loss can indicate increasingly confident errors. Model per-class accuracy/calibration/confusion matrices were not saved for the current CUB experiments and are not reconstructed from dataset class counts.

All primary results keep split, model/head, seed, training budget, initialization, checkpoint and evaluator visible. **Never compare** YOLO26 O2M to O2O as if they were identical; VisDrone merged-train test-dev to official-val; training CSV to standalone validation without saying so; best to last silently; G6 epoch 50 to G4/G5 epoch 125 as a complete ablation.

## Audit standards used in the experiments

1. Verify configuration, training/data/statistics/source hashes and model initialization before launch.
2. Use true process/GPU checks; a stale state file or static `nvidia-smi` display does not prove a live/idle job. Early GPU display staleness was detected and direct NVML used.
3. Require exactly the target number of continuous finite CSV epochs, correct samples/update counts/LR, and readable matching checkpoints.
4. Independent matched validation for key long detection pairs; preserve train-validator discrepancies rather than overwriting them.
5. Include auxiliary modules in optimizer, EMA, serialization, predict and resume; avoid ndarray/uint8/detach between learned frontend and detector.
6. Resume only after checkpoint/epoch/hash/hardware consistency; retain invalid/failed starts and never resume a corrupt or empty checkpoint.
7. Stop/clean only verified task-owned processes. Original other-user jobs are outside the research scope.

Examples: T0/B50 and seed 0 same-boot repeat passed exact equivalence; SARC fusion preserved frozen gamma outside SiLU; ASAC A0 failed equivalence and was relabeled honestly; blur aux reduction had a3× broadcast defect; CUB D resumed only from complete 15; G4/G5 exact-DDP tail/AMP tests gated launch; G6 stopped at complete 50 with authoritative stop record.

## Hardware and runtime limitations

The eight-RTX3090 server had recurring uncorrectable-memory/MCE/SIGBUS history, including NUMA-associated failures and whole-server reboots. A boot's EDAC counters resetting to zero is **not proof of repair**. Historical NUMA1 use, later NUMA0 placement and subsequent GPU allocation changes are documented. Some windows had no new decoded errors; other windows included undecoded BERT records or new MC1 events. These are exploratory-host qualifications, not grounds to fabricate corrupted measurements or silently discard unfavorable runs.

Dataset preprocessing, inference batching, card sharing, NUMA placement and competing loads differ across stages. Full test-loop seconds, forward-only milliseconds, predecoded end-to-end prediction and rough FLOPs are different cost scopes. Gaussian/multiview transformations at unchanged input dimensions do not intrinsically save CNN computation. A five-call960 crop pipeline is not an efficiency model merely because each crop covers a smaller source area.

## Public evidence and validation

The [manifest](../../results/evidence_manifest.csv) records original-source SHA256, published SHA256, size and any normalization/redaction or compression. Repeated copies retain separate provenance; [canonical run registry](../../results/training_run_registry.csv) prevents counting snapshots as experiments. Large gzip records have uncompressed original SHA in the manifest. The public publication validator checks all manifest bytes, generated tables, complete/incomplete CUB epochs, link targets and credential patterns.

Rebuild derived outputs only:

```bash
python -m pip install -r tools/requirements-report.txt
python tools/build_report.py
python tools/validate_publication.py
```

These commands are local analysis/publication commands; they do not train or resume a model. To reproduce training, use the relevant archived protocol and code with adjusted local data paths and matching source/runtime, rather than copy an absolute historical server launcher blindly.

## Limits on scientific claims

- Many test/dev sets were viewed repeatedly; CUB length was revised after test inspection. They are not pristine blind confirmations.
- Most method studies and all current CUB representations use one seed; YOLO26 T1's second seed contradicted its first. Two baseline seeds are not a broad robustness study.
- GT Oracle preprocessing grants location information unavailable to ordinary inference and can alter the image via box geometry cues.
- Added E/F/G5 diagnostics were not all preregistered before A. Planned controls not run are explicit in the progress chapter.
- Architecture/normalization/compute/runtime changes confound cross-family comparisons.
- No best-selected snapshot, attention-looking visualization or lower auxiliary loss substitutes for an appropriate matched control.
- There is no established novel state-of-the-art algorithm, completed deployment pipeline or accepted paper in the recorded evidence. The delivered assets are experimental benchmarks, code, audits, mechanism analyses and bounded research findings.
