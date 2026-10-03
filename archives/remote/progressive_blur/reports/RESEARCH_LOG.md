# VOC PGB + YOLOv8n research log

## 2026-09-23 20:15 CST — test-time predicted-background blur, all levels negative

- User requested light/medium/heavy background blur **only during inference**
  using the existing raw-trained VisDrone YOLOv8n/960 best.pt. Implemented
  a two-pass, image-only protocol: first-pass predicted boxes define fully
  protected regions and a feathered halo; no GT is used to preprocess images.
  Fixed sigma levels 3/8/16. Baseline and every level use the same 548 official-
  provenance val images, checkpoint, evaluator, batch16, conf=.001, NMS=.7 and
  max_det1000; results are internal converted-label AP, not challenge AP.
- Full-set raw mAP50–95=0.232551; light=0.221578 (−1.097 AP points),
  medium=0.213893 (−1.866), heavy=0.211288 (−2.126). Each level lowered
  AP50–95 for all ten classes. A post-result sham control ran the first pass
  and CPU/GPU roundtrip without pixel change, reproducing raw AP exactly.
  The blur groups changed mean pixels by 6.01/9.26/12.26 of 255;
  mean protected area=20.05%, so this is not a no-op.
- GPU experiment used only physical GPU4, NUMA0 pinning, and task-specific
  Ultralytics config. EDAC remained mc0=0/mc1=44 through completion; GPU7's
  other workload was untouched. Source, strict CPU renderer tests, JSON
  metrics and the full interpretation are in
  `VISDRONE_TESTTIME_PREDICTED_BLUR_RESULTS_2026-09-23.md`.

## 2026-09-23 19:40 CST — formal Oracle960 finished, discussion checkpoint

- Task manager reports `verified_complete` at 08:14:47 with exactly
  125 CSV epochs and saved best/last weights. Built-in best-epoch
  mAP50–95 is 0.22687 versus standard960 0.23240 (−0.00553);
  mAP50 is 0.39614 versus 0.40347. This is an initial negative
  comparison, **not** the locked independent raw-val best.pt audit.
  CSV training time 5274.23 versus 1573.05 seconds (~3.35×).
- The server's mc1 UE count rose from 12 at the five-hour handoff to
  38, with a new other-process SIGBUS around 19:37. No new GPU work
  or server-side evaluation was started. Recommended healthy-hardware
  independent audit before deciding on further training.
- Read-only comparison and decision gates are documented in
  `PROGRESS_AND_NEXT_DECISION_2026-09-23.md`. Current recommendation:
  do not blindly sweep blur parameters; after audit, likely prioritize
  small/dense target detection, localization and confidence ranking
  against the strong standard960 baseline. This entry was recorded
  locally only because the shared server currently has new memory faults.

## 2026-09-23 08:04 CST — five-hour cutoff and handoff

- At the agreed five-hour endpoint the formal VisDrone train-only
  Oracle960 manager PID 67146 and trainer PID 67332 remained alive.
  `results.csv` had 109 contiguous epoch rows of the planned 125;
  therefore no final AP or verified-complete claim is possible yet.
  Physical GPUs 4–7 were active, GPUs 0–3 idle; EDAC remained
  mc0=0/mc1=12 and the final interval showed no new MCE/SIGBUS.
- Left the healthy-running task in place without changing its process,
  config, checkpoints, data or server environment. Paused the ten-minute
  heartbeat at deadline; no further GPU experiment or independent
  evaluation was started. The full interim findings, negative controls,
  source-literature limits and post-run acceptance steps are in
  `FIVE_HOUR_RESEARCH_REVIEW_2026-09-23.md`.
- No publishable AP improvement has been established. The pending
  125-epoch result must pass strict artifact audit and independent raw
  evaluation against standard960 before deciding whether to replicate
  or retire this particular blur recipe.

## 2026-09-23 07:41 CST — five-hour review draft landed

- Consolidated verified VOC controls, standard VisDrone 640/P2/960
  controls, the still-running formal Oracle960 candidate, novelty
  limits, CPU-only engineering findings and hardware caveats in
  `FIVE_HOUR_RESEARCH_REVIEW_2026-09-23.md`. It is explicitly a
  **pre-deadline draft**, not a claim that the pending 125-epoch run
  finished. Update with authoritative job status at 08:04 before
  reporting the five-hour outcome.

## 2026-09-23 07:31 CST — isolated selected-only rendering candidate

- Static code review found the p=.5 train-only path renders/maps all
  batch images then discards the processing for unselected half. Added
  an **unconnected** selected-only variant and 3 CPU tests; all pass,
  including fixed-seed image agreement within 1e-6 and exact identity
  for clean images. No training code, current DDP process or config was
  modified. No whole-training speed/AP claim. See
  `SELECTED_ONLY_ORACLE_RENDER_NOTE_2026-09-23.md`.
- The same 3/3 tests also passed CPU-only on the remote pinned conda
  environment/NUMA0 with no visible GPU. This does not alter the
  formal run or establish DDP/training speed.
- A separate four-image real VisDrone CPU960 batch (127/16/65/122 boxes)
  yielded identical chosen-mask positions and bitwise-equal processed
  image tensors for registered vs selected-only paths. One CPU call was
  1.883s vs 0.854s; this is not a replicated or end-to-end GPU timing.
  Report updated in `SELECTED_ONLY_ORACLE_RENDER_NOTE_2026-09-23.md`.

## 2026-09-23 07:23 CST — sharper prior-art and context risk

- A closer read of author-hosted Scale-Aware AutoAug found its
  *test-time* all-context-removal COCO diagnostic (small AP 25.2→18.0,
  large AP 53.0→56.1) and its Gaussian-weighted soft boundary blend.
  Neither is the current train-only soft blur, but they further narrow
  any novelty claim and make context loss a concrete risk. Added the
  exact setup distinctions to the literature addendum. No active
  experiment parameter was changed.

## 2026-09-23 07:22 CST — context-versus-clutter literature boundary

- Cross-checked primary CVPR 2021 scale-aware augmentation and ECCV 2018
  context-aware augmentation sources. They offer evidence that surrounding
  context may help small/mid object detection, while CVPR 2025 SET offers
  prior for suppressing high-frequency background clutter. Neither side
  proves the current train-only RGB blur will help on VisDrone. Added
  cautious source links and the two-sided failure hypothesis to
  `DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`; no method change was made.

## 2026-09-23 07:19 CST — frozen GT-crowding diagnostic

- Re-aggregated **existing**, CPU-only standard960 and P2/640
  548-image prediction JSONs into four equal GT-count groups without
  new inference. P2 has +23/+83/+68/−124 extra IoU-.50 matches from
  Q1→Q4 and +25,016/+30,315/+23,098/+8,403 unmatched candidates;
  more images hit the max1000 cap. IoU-.75 P2 recall is not higher
  in any group. See `VISDRONE_GT_CROWDING_DIAGNOSTIC_2026-09-23.md`
  and raw JSON. Added a **pre-result optional secondary** crowding
  analysis to the Oracle960 protocol; primary raw-val AP remains decisive.

## 2026-09-23 07:11 CST — bounded-local target map on real images

- Ran a CPU/NUMA0 exact-vs-bounded target-map check on the same three
  16/65/122-box VisDrone images used for qualitative auditing. The
  local algorithm was 8.06–13.19× faster **for map construction alone**
  in best-of-three CPU calls, with max strength error <1e-4, zero pixels
  above the bound, and exactly equal protected/valid masks. Report:
  `VISDRONE_REAL_TARGET_MAP_COST_AUDIT_2026-09-23.md`. The alternative
  remains disconnected from the formal trainer; no claim of equivalent
  GPU training time, AP or bitwise behavior is made.

## 2026-09-23 07:06 CST — formal run alive; same-epoch training-cost signal

- At the scheduled ~10-minute check, formal manager PID 67146 and training
  child PID 67332 remained live; `results.csv` contained contiguous rows
  through epoch 29 (30 lines including header). Physical GPU4–7 were
  active, GPU0–3 idle; manager `/proc/.../numa_maps` showed `bind:0`.
  Same boot (`03:01:16`), EDAC mc0=0/mc1=12 unchanged, and the post-06:43
  kernel query found no new matching MCE/SIGBUS. No restart/action taken.
- At **the same epoch 29**, the candidate CSV cumulative `time` is
  1132.33 s versus the existing standard960 baseline's 371.678 s, or
  ~3.05× observed training time. The candidate's *built-in interim* AP50–95
  is 0.17215 versus baseline 0.17496; these are not independent best-checkpoint
  raw-val results and should **not** be used for method selection. The
  3.05× observation is a cost warning, not a component-level timing
  attribution: it may include augmentation, target generation, data and
  concurrent-system effects. Finish the pinned run, then audit total cost
  and independent AP before deciding anything.

## 2026-09-23 07:05 CST — real-image Oracle visualization, no retuning

- Applied the locked 960px Oracle rule CPU-only to three official-val
  samples selected by 10th/50th/90th percentile box count. Saved
  original/strength/processed/difference panels and mean pixel-change
  measurements in
  `VISDRONE_ORACLE_960_QUALITATIVE_AUDIT_2026-09-23.md` and three JPGs.
  The median 65-box sample changed by mean 5.22/255 RGB levels across
  valid pixels; its processed image remains recognizably similar to the
  original. This is a qualitative/physical-behavior check only, not AP,
  not representative of Mosaic batches, and not a reason to retune the
  already running formal candidate.

## 2026-09-23 07:02 CST — raw VisDrone ignored-annotation check

- During the formal GPU job, ran only a CPU/NUMA0, read-only audit of
  official-source train/val annotations. Val has 1,378 category-0 ignored
  boxes in 340/548 images, category 11 has 32 boxes in 29 images;
  clipped category-0 box areas sum to 0.9996% of val pixels. Train has
  8,813 category-0 boxes in 2,648/6,471 images. The earlier converter
  omits both categories. This quantifies but does not remove the caveat
  that internal converted-label AP differs from official VisDrone AP.
  No labels were changed. Full report:
  `VISDRONE_IGNORED_ANNOTATION_AUDIT_2026-09-23.md`.

## 2026-09-23 06:57 CST — user Desktop paper pass and running-job check

- Inventoried all 20 PDFs in the user's four indicated Desktop folders. Read
  the three remaining unsummarized `图像增强` papers through the pertinent
  methods/results and verified the principal REGDet ablation table visually.
  Added source-specific evidence and limitations to
  `DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`. The most relevant warning is
  that detection-guided preprocessing is prior art and overly smooth face
  regions can suppress small-object evidence. Restoration PSNR is not AP.
- The first approximately 10-minute read-only check of formal
  `visdrone_oracle_blur_p50_960_s2027` found manager PID 67146 and its
  training child live, 16 completed CSV data rows (17 lines including
  header), and only physical GPUs 4–7 active; GPUs 0–3 remained idle.
  State still says `new` because the manager updates it on terminal events.
  No new MCE/SIGBUS matched the post-06:43 kernel query. This is progress,
  **not** an AP result. Continue approximately 10-minute checks; do not
  duplicate or interrupt the run.

## 2026-09-23 — Project start and P0 freeze

- Started a new, isolated local implementation at `work/voc_pgb_yolov8n`.
- Kept the supplied Desktop prototype unchanged as source material.
- Confirmed the local conda environment `pytorch`:
  - Python 3.11.9
  - PyTorch 2.6.0+cu118
  - CUDA runtime 11.8; CUDA available
  - one NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
  - Ultralytics 8.4.146
- Inspected all supplied core files. The predictor contains 38,513 trainable
  parameters and the existing test record reports passing tensor/gradient tests.
- Independently noted that the multiple pasted research summaries are duplicates,
  not independent experimental evidence.
- Froze the P0 question, method defaults, causal controls, data boundary and
  remote-server safety constraints in `P0_IMPLEMENTATION_PROTOCOL.md`.
- Implemented a package-form core and expanded its tests. One behavior-preserving
  optimization was added: when blur mix is exactly zero, the renderer returns the
  original tensor without computing the Gaussian bank. This is important for the
  proposed rho warm-up and does not change numerical output.

Next gate: run the expanded tests locally, record exact results, and compare the
workspace implementation against the supplied source implementation under fixed
weights and inputs.

## 2026-09-23 — Execution-location correction

- User specified that training and experimental testing must remain on the
  remote server.
- The local 8-test pass is retained only as a code sanity check and will not be
  presented as training, GPU performance, or accuracy evidence.
- Formal runs will use only remote physical GPUs 4–7 with NUMA node 1 binding.
- Before syncing or running anything, the remote environment, free GPUs and
  task-specific destination will be checked read-only.

## 2026-09-23 — Remote research root created

- Created the dedicated remote research root requested by the user:
  `/home/ubuntu/rz/progressive_blur_detection_research`.
- Moved only this task's freshly created files out of the old `yolo_study`
  subdirectory. Existing datasets, earlier research, shared environments and
  other users' files were not changed.
- Verified that the core source, tests, P0 protocol, CPU test log and CUDA smoke
  result are present at the new root.

## 2026-09-23 — Integration defect 1: constructor-time forward

- First Ultralytics integration test failed before training: `DetectionModel`
  performs a dummy tensor forward inside its constructor to infer feature-map
  strides, but the subclass front end is attached only after that constructor
  returns.
- Root cause was confirmed from the traceback; it was not a CUDA or data error.
- Added a narrow construction-stage bypass: if `frontend` is not attached yet,
  prediction dispatches directly to the parent detector. All post-construction
  tensor inference still traverses the front end.
- Gate remains open until the corrected integration tests pass.
- The next test attempt exposed a test-fixture error rather than a model error:
  Ultralytics loss expects attribute-style hyperparameters, while the synthetic
  test supplied a plain dict. The fixture was corrected to a namespace matching
  the trainer's real model contract.

## 2026-09-23 — Integration defect 2: checkpoint-loader API

- The first remote tiny-trainer command stopped before constructing a trainer or
  using a GPU because the audit utility imported `attempt_load_one_weight`, which
  is not exported by the remote Ultralytics tree despite the matching reported
  version number.
- Inspected the remote source and replaced that audit-only call with the actual
  `load_checkpoint(weight, device, inplace, fuse)` entry point.
- No training state or result was produced by the failed attempt.

## 2026-09-23 — Integration defect 3: custom options in validator config

- The next tiny-trainer attempt built the model and dataloaders but stopped
  before the first training batch. Ultralytics' validator rejected the custom
  `pgb_*` keys that had been appended to its strictly validated standard config.
- Removed all PGB-only keys from the standard Ultralytics argument namespace.
- PGB configuration is now carried by `PGBDetectionModel.pgb_config()` inside
  the serialized custom model. On resume, `PGBDetectionTrainer.get_model()`
  reads this configuration from the checkpoint model before rebuilding and
  loading weights. This preserves custom settings without changing any shared
  or package-global configuration.

## 2026-09-23 — P0 trainer/EMA/checkpoint/resume gate passed

- Generated a deterministic audit-only dataset: 12 training images with 16
  boxes and 6 validation images with 8 boxes, two synthetic classes at 128 px.
- Ran the custom trainer on remote physical GPU 4 with CPU and memory bound to
  NUMA node 1. This audit used scratch YOLOv8n initialization and AMP.
- Completed epoch 1, saved an unstripped `last.pt`, then intentionally terminated
  the process immediately after the checkpoint callback to emulate interruption.
- Verified the intermediate checkpoint before resume:
  - epoch index 0;
  - optimizer state present;
  - EMA class `PGBDetectionModel`;
  - exact PGB configuration present in the model.
- Resumed from `last.pt` and completed epoch 2. The final audit returned
  `verified_complete`:
  - exactly two `results.csv` rows;
  - train and validation columns for box, cls, dfl and map loss;
  - readable `best.pt` and `last.pt`;
  - checkpoint contains `frontend.*` tensors;
  - image-only checkpoint inference returns finite predictions;
  - restored evaluation mix is 0.5;
  - checkpoint Ultralytics version is 8.4.146.
- No new kernel MCE, SIGBUS or memory-failure event appeared in the preceding
  30-minute window.
- The synthetic mAP was zero and is intentionally not a scientific result; the
  tiny run exists only to audit the end-to-end engineering path.

Next gate: read-only audit of existing VOC assets, data splits and reusable
YOLOv8n baseline artifacts before defining the first real experiment matrix.

## 2026-09-23 — VOC asset audit and padding-mask requirement

- Confirmed the existing VOC root is complete enough for this project and uses
  paired image/YOLO-label files:
  - train2007: 2,501 images and labels;
  - val2007: 2,510;
  - train2012: 5,717;
  - val2012: 5,823;
  - test2007: 4,952;
  - total disk use: about 2.2 GB.
- Existing formal VOC runs are YOLO26n experiments, so they cannot substitute
  for the required YOLOv8n baseline.
- Inspected an actual 640-pixel VOC training batch. It retains `ori_shape`,
  realized resize gains in `ratio_pad`, and `resized_shape`, which are sufficient
  to reconstruct the non-padding rectangle when geometry-changing augmentation
  is disabled.
- Added `valid_pixels_from_batch()` and wired it into map-target construction so
  LetterBox padding is excluded from auxiliary blur supervision.
- Restored PGB keys on trainer arguments for exact DDP/checkpoint propagation,
  while overriding validator construction to pass a sanitized standard-only
  argument copy. This avoids the prior validator rejection without modifying
  Ultralytics global configuration.
- Limitation recorded: the available converted YOLO tree does not include raw
  VOC XML annotations, so `difficult` boxes skipped during conversion cannot yet
  be reconstructed as auxiliary ignore regions.

## 2026-09-23 — Strict split and DDP gate passed; P1 frozen

- Created a deterministic image-level 90/10 split inside the 16,551-image
  VOC07+12 trainval pool using rarest-present-class strata and SHA-256 seeded
  ordering (seed 2027).
- Split audit passed with 14,896 train, 1,655 development and 4,952 untouched
  VOC2007-test images. Maximum class-image-ratio deviation from 10% is 0.004348.
- Generated absolute-path manifests, a dataset YAML and SHA-256 digests under the
  remote `data_splits` directory; the shared VOC tree was not modified.
- Re-ran 12 remote unit/integration tests after padding-mask changes: all passed.
- Completed a two-GPU DDP audit on physical GPUs 4 and 5, NUMA node 1:
  - DDP launched two ranks successfully;
  - train and validation logged box/cls/dfl/map losses;
  - final best/last checkpoints are readable and contain the front end;
  - checkpoint preserved the full custom PGB configuration;
  - no recent MCE/SIGBUS/memory-failure events appeared.
- Froze `P1_EXPERIMENT_PLAN.md`: one 1%-fraction four-GPU smoke, followed on
  success by paired 20-epoch scratch A0 and D runs on the strict development
  split. P1 deliberately uses weak augmentation to isolate mechanism behavior.

## 2026-09-23 — Real-VOC smoke passed; cache-isolation defect corrected

- Four-GPU real-VOC PGB smoke completed on physical GPUs 4–7 with NUMA node 1:
  - 1% train fraction (149 images), full 1,655-image development validation;
  - global batch 64, 640 input, BF16, scratch initialization;
  - one epoch completed without OOM, NaN or invalid samples;
  - per-rank peak shown by trainer was about 3.34 GB;
  - train and validation map-loss paths ran and best/last checkpoints were saved.
- The zero mAP from one epoch on 1% scratch data is expected and is not treated
  as a method result.
- Detected a filesystem side effect immediately after the smoke: Ultralytics
  chose the first shared label directory for its derived `.cache`, so the task's
  subset scan replaced the existing full-dataset `labels/train2007.cache`.
  Original images and `.txt` labels were never changed.
- Restored the shared derived cache by rebuilding it from the original full
  `voc_server.yaml`; verified it again contains 16,551 labels with full-dataset
  hash `04bb051208ae52fc8fce7344b29265b8a28ae49df96cf42a54f223c0c36abf65`.
- Updated the split builder to create task-local `voc_view/images/*` and
  `voc_view/labels/*` symlinks. Manifests now use those paths, so future caches
  land under this research root. No long P1 run will start until this isolation
  is regenerated and verified.
- Refined this before regeneration into separate `voc_view/train`,
  `voc_view/dev` and `voc_view/test` namespaces so each split also gets its own
  cache file instead of making train and development invalidate each other.

## 2026-09-23 — Isolation revalidation passed; P1 queue prepared

- Regenerated the identical 14,896/1,655/4,952 membership through separate
  task-local symlink namespaces.
- Built both full train and development datasets through the new view and
  verified independent task caches contain 14,896 and 1,655 labels.
- Rechecked the shared cache after this operation: it remains the restored
  16,551-label full cache with hash
  `04bb051208ae52fc8fce7344b29265b8a28ae49df96cf42a54f223c0c36abf65`.
- Repeated the four-GPU 1%-fraction VOC smoke using isolated paths. It completed
  successfully, reproduced the prior smoke loss values, saved valid custom-model
  checkpoints and left the shared cache unchanged. No recent memory errors were
  recorded.
- Implemented an idempotent P1 manager for sequential A0 then D execution. It
  writes atomic queue state/events, resumes only checkpoints that still contain
  optimizer state, and accepts completion only with exactly 20 finite result
  rows plus readable finite best/last weights. A preliminary comparison JSON is
  generated only after both audits pass.

## 2026-09-23 — P1 manager launch defect fixed before training

- First background manager process exited before constructing a trainer because
  its event helper used the parameter name `kind`, which collided with the
  experiment field `kind=baseline`.
- GPU 4–7 remained idle; both queue tasks stayed in `new` state with zero epochs
  and no run directories/results created.
- Renamed the helper parameter to `event_type`; the execution/audit logic and
  frozen experiment configurations were unchanged.

## 2026-09-23 — P1 paired run started

- Froze SHA-256 values for package code, tools, configs and split manifests plus
  the Ultralytics Git commit in remote `research/P1_CODE_MANIFEST_SHA256.txt`.
- Started the corrected task-local manager under `nohup`, with all descendants
  bound to NUMA node 1 and `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Verified the live process tree contains the manager, A0 runner and a four-rank
  torch distributed job. No process is mapped to physical GPU 0–3.
- First completed baseline epoch wrote a valid result row; development metrics
  were precision 0.707, recall 0.0204, mAP50 0.00487 and mAP50–95 0.00113. These
  very early scratch metrics are recorded only as progress, not interpreted.
- At the last low-frequency check, baseline epoch 2 was in progress, per-rank
  memory was about 2.3 GB, throughput about 16 iterations/s, and no new kernel
  memory-error event was present.

## 2026-09-23 — P1 baseline completed; PGB run in progress

- The A0 YOLOv8n scratch baseline completed exactly 20 epochs and passed the
  queue manager's integrity audit: finite `results.csv`, readable finite
  `best.pt` and `last.pt`, and configuration/checkpoint agreement.
- Baseline best development mAP50–95 was `0.16173` at epoch 12, with mAP50
  `0.29178`, precision `0.42630` and recall `0.29020`. Its best mAP50 was
  `0.29254` at epoch 13. The epoch-20 values declined to mAP50 `0.26371` and
  mAP50–95 `0.14023`, so P1 comparisons must use the predeclared best-checkpoint
  criterion rather than treating the final epoch as the best model.
- The complete PGB run started automatically after baseline verification. At
  the latest check it had completed 15/20 epochs; its current best development
  mAP50–95 was `0.15921` at epoch 12 and best mAP50 was `0.29606` at epoch 11.
- At matched epoch 15, PGB versus baseline changed mAP50 by `+0.00576` and
  mAP50–95 by `+0.00366`, while precision changed by `-0.00612` and recall was
  essentially unchanged (`-0.00005`). This is a provisional within-run signal,
  not the final P1 comparison and not evidence of generalization.
- PGB map loss was stable near `0.108` on development data. The four target
  GPUs remained active, no GPU 0–3 was used, and no new MCE/SIGBUS/memory-
  failure event appeared in the latest 30-minute kernel-log window.

## 2026-09-23 — P1 verified complete; no AP improvement

- Both A0 and D/PGB completed exactly 20 epochs. The queue reached
  `verified_complete` after finite CSV checks and successful loading of all four
  best/last checkpoints.
- By the predeclared maximum-development-mAP50–95 rule, both best checkpoints
  are epoch 12. PGB minus baseline was precision `-0.03344`, recall `+0.02712`,
  mAP50 `-0.00065`, and mAP50–95 `-0.00252`.
- Thus PGB increased recall but reduced precision and did not improve either AP
  metric. P1 is negative/inconclusive and does not justify a claim of improved
  detection.
- Recorded elapsed training/validation time was 312.328 s for A0 and 738.908 s
  for PGB (2.366x). This is not an inference-latency measurement.
- Corrected a reporting-only off-by-one error: Ultralytics' CSV epochs are
  already one-based. Best-epoch labels changed from 13 to 12; no metric,
  checkpoint or training result changed.
- A standalone re-audit initially lacked the research root on `sys.path`, which
  triggered Ultralytics to install an unrelated PyPI package named `pgb`. It
  was immediately uninstalled. The manager now prepends the research root,
  resolves the task-local package, and the full independent audit passes again.
- Full interpretation and proposed causal controls are recorded in
  `P1_RESULTS_2026-09-23.md`.

## 2026-09-23 — Oracle-PGB implemented and preflight passed

- Added a label-guided Oracle control that constructs the exact continuous
  blur map from the current batch boxes during both training and development
  validation. It uses a plain YOLOv8n detector and contains no learned map
  predictor.
- Oracle is explicitly non-deployable and will not be evaluated on the final
  VOC2007 test set. Its only purpose is to estimate whether the current blur
  intervention has a useful upper bound.
- Matched P1 controls: scratch initialization, seed 2027, 20 epochs, batch 64,
  640 input, weak augmentation, mix schedule 0 -> 0.5, and identical map
  geometry.
- Fifteen unit/integration tests passed. A four-GPU real-VOC smoke using 149
  training images and the full 1,655-image development split completed one
  epoch, exercised training and label-guided validation, and saved readable
  checkpoints. Its zero AP is expected for a one-epoch 1% scratch run and is
  not interpreted.
- Fixed a pre-training script-entry path defect exposed by the smoke. The first
  attempt failed during imports before creating a trainer or using a GPU; the
  project root is now inserted explicitly. No experimental result was affected.

## 2026-09-23 — Oracle-PGB verified complete; positive upper-bound signal

- Oracle completed exactly 20 epochs and passed finite-results plus best/last
  checkpoint audits. The final VOC2007 test set remained untouched.
- The predeclared best checkpoint was epoch 12. Against the baseline best
  checkpoint, Oracle changed precision by `-0.04511`, recall by `+0.04955`,
  mAP50 by `+0.01740`, and mAP50–95 by `+0.00869`.
- This corresponds to +1.740 points mAP50 and +0.869 points mAP50–95. The
  mechanism has modest headroom when supplied perfect boxes, but it shifts the
  detector toward higher recall and lower precision.
- Oracle exceeded learned PGB by +1.805 points mAP50 and +1.121 points
  mAP50–95, identifying learned-map quality/optimization as a likely bottleneck.
- AP50–95 improved for 15/20 classes. The largest gain was bus (+0.045); train,
  sofa, diningtable, person and aeroplane declined slightly.
- Oracle elapsed time was 464.542 s versus 312.328 s for baseline (1.487x).
  This is pipeline training/validation time, not deployable inference latency.
- GPU 4–7 returned idle and no new memory-hardware event appeared. The shared
  VOC cache was verified to contain all 16,551 labels and had last changed well
  before Oracle started; only task-local cache files were written.
- Full metrics and interpretation are in `P1_ORACLE_RESULTS_2026-09-23.md`.

## 2026-09-23 — Real Oracle image visualization set exported

- Created remote `oracle_visualizations/` with one distinct real development
  image selected for each of the 20 VOC classes, preferring multi-object scenes.
- For every sample exported the clean 640x640 letterboxed input, Oracle strength
  heatmap, clean processed detector input, absolute-difference heatmap and a
  2x2 annotated comparison panel; 100 JPEGs decoded successfully.
- All images use the registered experiment parameters: margin 0.10, transition
  0.50, minimum radius 8 px, sigma levels 0/1/2/4/8 and residual mix 0.5.
- The manifest records source paths, classes, object counts, protected fraction,
  mean strength and mean absolute pixel change. Across these 20 selected scenes,
  mean absolute pixel change ranged from 0.0253 to 7.4873 intensity levels.
- Visual inspection of sparse, crowded and large-object panels confirmed that
  target boxes remain sharp, the transition is continuous outside boxes, and
  distant texture is blurred. Clean detector-input files contain no drawn boxes.

## 2026-09-23 — Extreme ground-truth blackout control prepared

- Implemented a user-requested privileged control that preserves the union of
  exact ground-truth box interiors and sets every outside pixel to exact black.
  It has no margin, gradient, blur or residual and is active from epoch 1.
- This directly reveals rectangular label boundaries and is therefore recorded
  as an extreme label-leaking stress test, never as a deployable method result.
- Matched the P1 baseline split, source initialization, seed 2027, 20 epochs,
  batch 64, 640 input, optimizer policy and disabled augmentations.
- Seventeen tests passed, including pixel-exact inside/outside behavior. A
  one-epoch 1% real-VOC four-GPU smoke completed successfully and saved readable
  weights. Its zero AP is not interpreted.

## 2026-09-23 — Blackout formal run blocked by new hardware-memory SIGBUS

- Both formal attempts were launched with physical GPUs 4–7 and explicit
  `numactl --cpunodebind=1 --membind=1` binding.
- Attempt 1 stopped in epoch 1 before any result row/checkpoint after three new
  uncorrected memory errors; SIGBUS was sent to two PyTorch data workers and one
  pinned-memory thread. The partial directory was preserved as
  `_failed_mce_oracle_blackout_20260923_022350`.
- A single controlled retry after the kernel reported page recovery also failed
  before any result row. Three additional poisoned pages sent SIGBUS to Python
  processes at 02:25:59–02:26:00.
- No scientific metric or resumable checkpoint exists. Further automatic retry
  was stopped to avoid repeatedly stressing shared hardware.
- Experimental semantics were re-audited: training batches and every
  development-validation prediction are blacked out using their ground-truth
  boxes. A plain standalone `best.pt` prediction would receive raw input unless
  the external Oracle preprocessing is explicitly applied; the privileged mask
  is not embedded in the standard YOLO checkpoint.

## 2026-09-23 — NUMA error attribution and training decision

- The current boot began 2026-09-22 22:48 CST. Current EDAC counts are
  `mc0: 0 UE, 0 CE` and `mc1: 6 UE, 0 CE`; the six new blackout-run failures
  are on `MC1`, with firmware locator `P1MG0`.
- The server uses 2 GiB memory blocks. New failing PFNs around `0x30be2ee`
  resolve to memory block 97, which belongs to NUMA node 1. The old
  2026-09-18 PFNs around `0x11c6f36` resolve, under the current memory map, to
  memory block 35, which belongs to NUMA node 0.
- The current journal contains only the current boot, so its zero `mc0` count
  cannot rule out the September 18 node-0 errors. The previous messages
  supplied by the user are the evidence for that older fault.
- Accordingly neither NUMA node can be established as fault-free across the
  observed history. Node 0 has no observed errors since this reboot and is
  relatively safer than currently failing node 1, but is not certified healthy.
- The blackout 20-epoch run remains at 0/20, stopped, with no accepted
  checkpoint. No system-wide memory offlining or hardware configuration was
  attempted on this shared server.

## 2026-09-23 — NUMA-0 blackout run completed; raw-image evaluation exposes failure

- After the two NUMA-1 SIGBUS failures, a bounded retry used physical GPUs
  4–7 and `numactl --cpunodebind=0 --membind=0`. Process `numa_maps` confirmed
  the binding. The task manager verified exactly 20 completed epochs and
  readable best/last checkpoints at 02:38:54 CST. EDAC counters remained
  `mc0 UE=0, CE=0; mc1 UE=6, CE=0` during this run. This does **not** certify
  NUMA 0 as healthy: older user-provided faults map to it under the current
  physical memory layout.
- The training images were black outside the union of GT boxes. The built-in
  per-epoch validation also used GT-box blackout; its best checkpoint is epoch
  14, with P=0.52022, R=0.48237, mAP50=0.49652 and mAP50–95=0.44016.
  These are privileged, label-leaking diagnostic numbers, not deployable AP.
- Per the user's request, separately evaluated the same saved checkpoints
  using the *untouched* VOC development images and a standard YOLO validator:
  no Oracle mask or GT information was used in the model input. Blackout
  `best.pt` gave P=0.120211, R=0.105540, mAP50=0.044079 and
  mAP50–95=0.014179. `last.pt` gave mAP50=0.042216 and
  mAP50–95=0.013939. Saved epoch 1/6/11/16 checkpoints were also checked;
  none exceeded the masked-selected best checkpoint on raw-image AP.
- Re-evaluated the scratch baseline `best.pt` with the *same raw-image
  validator*: P=0.410728, R=0.295163, mAP50=0.290927 and
  mAP50–95=0.161691, consistent with its original 0.16173 development AP.
  Blackout-trained best therefore loses 0.246848 mAP50 and 0.147512
  mAP50–95 absolute (24.685 and 14.751 percentage points) relative to the
  baseline under the realistic raw-input condition.
- Checkpoint selection during training was based on *masked* validation;
  only the separate post-training evaluation was unmasked. The comparison
  does not claim that every epoch was re-ranked by raw AP. All saved periodic
  checkpoints, best and last were re-evaluated. No further training was
  started: the severe raw-input degradation already rejects this exact
  train-only blackout method as a practical detector.
- Full protocol, figures, limitations and artifact paths are recorded in
  `P1_BLACKOUT_RESULTS_2026-09-23.md`.

## 2026-09-23 — Provisional next-direction discussion (no new experiment)

- Current VOC evidence disfavors the present image-level background treatment
  as the main paper route: learned PGB lost 0.252 percentage points mAP50–95
  versus its scratch baseline; perfect-box progressive blur gained only 0.869
  points in a single-seed 20-epoch pilot; extreme train-only blackout failed
  catastrophically under raw-image evaluation. None of these individually
  proves all mild blur methods impossible, but further parameter sweeps on VOC
  have low expected research value.
- Recommend a **diagnostic-first small-object direction**, initially using the
  already-prepared VisDrone2019-DET baseline. In the existing 640-input audit,
  93.2% of test boxes are smaller than 32² pixels and the two-seed fixed-epoch
  mAP50–95 is 0.1361±0.0015. Test whether the dominant bottleneck is input
  resizing/information loss, feature-stride mismatch, dense-object assignment,
  or background clutter *before* proposing a module.
- First gate: size-binned recall/AP and failure visualization, followed by
  matched no-new-method controls (640 vs higher resolution, conventional P2,
  and tiling) under the same data split/metric/compute reporting. Preserve
  official train/validation/test separation in any new publishable protocol;
  the existing 125-epoch queue used test for per-epoch selection and is
  exploratory, not a final unbiased result.
- Only if a repeatable, specific failure remains after those controls should
  the team design an image-only, inference-valid mechanism. Simple background
  smoothing and high-resolution shallow features already appear in CVPR 2025
  SET and ICCV 2025 DM-EFS, respectively, so neither alone is an adequate
  novelty claim. Literature: https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html ;
  https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html .
- Alternate domain-specific path: paired LLVIP visible/infrared can support a
  visible-only inference with infrared privileged training-information study,
  but the current paired-scene split and single-class scope demand protocol
  care. This is a backup idea, not a launched experiment.
- Direction remains provisional pending discussion with the user. No GPU job
  or dataset modification was started.

## 2026-09-23 — Stronger Oracle blur, training-only five-hour research started

- User requested a stronger Oracle-background-blur run with *training-only*
  image processing, raw development validation and 20 epochs; subsequently
  authorized a five-hour period of further controlled ideas and experiments.
- Prespecified a one-variable strength increase: Gaussian sigma bank
  0/1/2/4/8 → 0/2/4/8/16. The mix cap stays 0.5, and the original five-epoch
  clean start plus five-epoch ramp, box margin, transition and other settings
  remain unchanged. This avoids confounding sigma strength with mix strength.
- Added `OracleTrainOnlyBlurTrainer`, which applies GT-guided blur only in
  `preprocess_batch` and uses the stock unmasked `DetectionValidator` for every
  epoch. Nineteen tests passed; the four-GPU 1%-train-fraction smoke completed
  one epoch and validated the untouched 1,655-image development split.
- The server briefly became unreachable immediately after preflight. On
  recovery, a new boot ID was observed and EDAC counters had reset to zero;
  that reset is not evidence that either memory node is repaired. The full
  run started at 03:05:24 CST in
  `runs/voc_p1/oracle_strong_trainonly_s16_e20_b64_s2027`, with physical
  GPUs 4–7 and verified `bind:0` NUMA policy. No previous run directory or
  same-task process existed before launch. At the first observation it had
  4/20 epoch rows and was actively using only GPUs 4–7.
- A same-thread, ten-minute-interval continuation was set with a strict
  08:04 CST stop for initiating new runs. It must remain quiet on ordinary
  progress, audit outputs before claiming completion, and stop itself at the
  time limit. This is an execution mechanism, not an experimental result.
- Detailed pre-registered conditions are in
  `P1_STRONG_TRAINONLY_PROTOCOL_2026-09-23.md`; no efficacy conclusion yet.

## 2026-09-23 — Strong train-only Oracle fails to beat raw-input baseline

- Formal strong-blur run finished 20/20 epochs at 03:13:54 CST and passed
  checkpoint and finite-CSV audit. The epoch-13 raw-development best row has
  mAP50–95 0.15717; independent raw re-evaluation of `best.pt` gives 0.157440.
- The same independent evaluator gives scratch baseline `best.pt` 0.161691,
  so the stronger blur loses 0.425 percentage points mAP50–95 on raw images.
  Its standalone P/R/mAP50 are 0.393578/0.305876/0.288812 versus baseline
  0.410728/0.295163/0.290927. Strong blur traded precision for recall.
- The old weak-Oracle `best.pt` is 0.161312 under a *post hoc* raw eval, but
  it was chosen by masked validation. Launched a clean, 20-epoch weak-blur
  train-only control at 03:25:17 CST to isolate sigma strength under the same
  raw-validation checkpoint-selection protocol.
- A 20-image visualization audit confirms the stronger sigma bank changes
  pixels more: mean absolute 8-bit change 1.768 → 2.429 intensity levels.
- Directly read two user-provided foveation PDFs. The short 2023 YOLO paper
  times five chosen images without benchmark AP; the 2026 multi-scale fovea
  paper studies scanpaths and lower-resolution crops, not full-image detector
  AP. These do not supply evidence that our current augmentation should work.
  Full results and caveats are in `P1_STRONG_TRAINONLY_RESULTS_2026-09-23.md`.

## 2026-09-23 — Weak train-only control closes the sigma comparison

- Weak-sigma (0/1/2/4/8) GT-guided blur, training only, finished and passed
  20-epoch/checkpoint integrity audit at 03:33:09 CST. On the same untouched
  development images its best checkpoint is P=0.359327, R=0.329336,
  mAP50=0.288983, mAP50–95=0.161312. The matched scratch baseline is
  0.161691 mAP50–95; strong sigma (0/2/4/8/16) is 0.157440. Thus the
  weak run is near-tied with baseline (-0.038 percentage points) while the
  stronger blur is worse than weak by 0.387 points. The 20-epoch, single-seed
  test does not support a gain from increasing blur strength.
- Independent raw evaluation of weak best/last and periodic checkpoints is
  stored in `P1_WEAK_TRAINONLY_RAW_VALIDATION.json`; full comparison and
  limitations in `P1_WEAK_TRAINONLY_RESULTS_2026-09-23.md`.
- User pointed to Desktop research folders. Direct review of two foveation
  PDFs, IA-YOLO, DENet and a low-light enhancement empirical study is in
  `DESKTOP_LITERATURE_REVIEW_2026-09-23.md`. The literature contains clear
  learned-preprocessing prior art and a caution that better-looking images
  can hurt detection; it does not validate our current blur strategy.
- Pre-registered one more diagnostic: strong blur with per-image probability
  0.5 versus the completed 1.0 run, all else fixed, to test the train/raw
  distribution-gap hypothesis. Twenty integration/unit tests and a 4-GPU
  1%-fraction smoke passed. Protocol is
  `P1_STRONG_P50_TRAINONLY_PROTOCOL_2026-09-23.md`; formal result pending.
- The formal 50%-probability run was launched at 03:39:13 CST through the
  task-specific manager in `runs/voc_p1/oracle_strong_p50_trainonly_s16_e20_b64_s2027`.
  The manager and child trainer were observed alive, GPUs 4–7 active while
  0–3 idle, and the run directory had been absent before launch. The
  continuation monitor was updated to audit this exact run rather than
  relaunching already-complete comparisons.
- A read-only classwise raw-development evaluator was prepared at
  `tools/evaluate_classwise_raw.py` for the existing best checkpoints. Use it
  only once the current GPU training is finished, then inspect per-class AP
  changes to test whether particular categories account for the aggregate
  decline. This diagnostic does not choose new checkpoints or use the test
  split.
- A primary-source literature cross-check identified CVPR 2025 SET's
  training-time hierarchical background smoothing for tiny objects. Its
  official repository advertises no extra inference burden. This is a close
  precedent for any generic training-only smoothing novelty claim and is
  recorded in `DESKTOP_LITERATURE_REVIEW_2026-09-23.md`.

## 2026-09-23 — 50% clean views recover strong-blur loss, not a gain

- Strong-blur 50% training-image run passed 20-epoch/checkpoint audit at
  03:47:26 CST. Independent evaluation on untouched VOC P1 development
  images gave best P=0.405032, R=0.298566, mAP50=0.289761,
  mAP50–95=0.161610. Built-in raw validation selected epoch 12; standalone
  `last.pt` is 0.143021 mAP50–95. Both EDAC uncorrected counters were 0
  after this run; historical faults remain a caveat.
- The 50% condition is 0.417 percentage points above strong-100% blur
  (0.157440), but 0.008 points below the original scratch baseline
  (0.161691). Reduced train/raw appearance mismatch is one explanation;
  halved effective blur dose is another, not separated by this experiment.
  No blur control shows a convincing baseline gain. Do not tune further on
  this VOC P1 development set. Full result in
  `P1_STRONG_P50_TRAINONLY_RESULTS_2026-09-23.md`.
- Prepared a **non-destructive** alternative domain protocol by recovering
  original VisDrone2019-DET train/val/test-dev membership from the existing
  `source_map.tsv`: 6,471 official train, 548 official val, 1,610 test-dev.
  All 8,629 derived images and YOLO labels exist; no source image belongs
  to two splits. New manifests and YAML are under remote
  `data_splits/visdrone_official/`, preserving the older merged dataset tree.
  This is a candidate platform, not yet a new training result. Local audit:
  `VISDRONE_OFFICIAL_SPLIT_AUDIT.json`.
- A matched raw-development per-class AP re-evaluation of baseline, weak,
  strong-100%, and strong-50% best checkpoints completed read-only. No
  selection of new checkpoints or test split was involved. Strong-100% loses
  about 0.06 AP for `train` and 0.05 for `aeroplane`, while gaining about
  0.04 for `sheep` and 0.03 for `bus`; the 50% condition still loses about
  0.05 for `train`. This nonuniformity rules out a simple claim that blur
  consistently helps all categories. Exact 20-class values and caveats are
  in `P1_CLASSWISE_RAW_ANALYSIS_2026-09-23.md` and JSON.

## 2026-09-23 — Conditional VisDrone clean-split pilot prepared

- Given no VOC P1 blur gain, prepared a separate domain pilot rather than
  continuing a broad blur hyperparameter search. Existing VisDrone 640
  baseline used merged train+val and per-epoch test-dev; a source-provenance
  reconstruction now produces official 6,471/548/1,610 train/val/test-dev
  lists without changing the original processed dataset. Pinned model YAML
  SHA256 is `44c9efc76c9dad388ad1ef6c4efb37e47ec885fe2ad12f110c76f6cdb750889d`;
  remote Ultralytics version `8.4.146`.
- Protocol in `VISDRONE_OFFICIAL_PILOT_PROTOCOL_2026-09-23.md`: first train
  ordinary scratch YOLOv8n 125 epochs at 640, official val checkpoint
  selection, test-dev untouched until protocol decided. Higher resolution
  and P2 are conventional controls to check before method claims. The prior
  two-seed merged-split baseline is not numerically comparable to this one.
- Physical GPUs 4–7 were idle, task manager absent, both EDAC uncorrected
  counters were 0, and the new smoke/formal run directories did not exist.
  Started a one-epoch, 1%-fraction four-GPU smoke for data/DDP validation;
  no full VisDrone job has been launched yet. That initial smoke completed,
  but the training framework created a regenerable `labels/train.cache` in
  the shared processed dataset tree. Original images and annotation text
  files were not changed. To prevent further source-tree cache writes, a
  task-owned symlink view was built: 6,471/548/1,610 image-and-label links
  under this project's `data_splits/visdrone_official/view/`. A second
  four-GPU smoke completed and generated distinct task-local train and val
  caches; the official val scan covered all 548 images with no corruption.
  The formal config was switched to the view YAML. Dense-scene evaluation
  uses explicit `max_det=1000` to avoid truncating images with hundreds of
  objects. Both EDAC UE counters stayed zero after the smokes.
- Full official-split 640 scratch baseline launched at 03:59:33 CST with
  seed 2027, 125 epochs and task-specific manager tag
  `visdrone_official_640_s2027`. Verified the manager and child trainer
  alive, physical GPUs 4–7 active while 0–3 idle, task-local view YAML in
  the trainer arguments, `max_det=1000`, and no pre-existing formal run
  directory. The 10-minute continuation was updated to monitor this exact
  process, not to relaunch completed VOC P1 runs or touch test-dev for model
  selection.
- Pre-registered a conventional 960-input control in
  `VISDRONE_960_RESOLUTION_CONTROL_PROTOCOL_2026-09-23.md`. It will change
  only input size versus the new clean 640 baseline and must pass a
  four-GPU, 1%-fraction smoke first. No 960 job was started while 640 is
  running. Report compute/memory/latency as well as AP; a standard
  resolution gain alone is not a novel method.
- Evaluation-protocol check: the official VisDrone DET toolkit excludes
  detections in ignored/other regions. Our converted YOLO labels omit those
  rows, so plain Ultralytics AP is a matched internal metric, not directly
  the official VisDrone score. Earlier work already looked at test-dev;
  this new protocol keeps it out of new per-epoch selection but cannot call
  it historically unseen. Both distinctions are in the pilot protocol.
- Read-only inspection of a third-party Python port revealed an apparent
  ignored-box IoF error (plus incompatible `np.int` use), so it was **not
  installed or used** as an official evaluator. Initial suspicion that its
  repeated `eval_class` list was an additional port error was corrected
  after directly reading the official MATLAB `calcAccuracy.m`: the official
  code has the same image-frequency-weighted class indexing. MATLAB/Octave
  are not available on the server. Exact caveats and validation plan are in
  `VISDRONE_EVALUATION_AUDIT_2026-09-23.md`; until verified, reported
  VisDrone AP remains an internal YOLO metric only.

## 2026-09-23 — Official-split VisDrone 640 scratch baseline verified

- The 125-epoch, seed-2027, batch-64, 640-input VisDrone scratch baseline
  completed at 04:16:46 CST; manager audited contiguous/finite 125 rows
  and readable best/last weights. Cumulative CSV time 1015.27 s. Physical
  GPUs 4–7 became idle and both EDAC UE counters remained 0 at check.
- Best epoch 115 on the official 548-image val split. Independent matched
  YOLO evaluation of that checkpoint gave mAP50 0.309082 and mAP50–95
  0.170547 (built-in best row 0.17051). This is an internal converted-label
  metric, not the official VisDrone toolkit score; test-dev has not been
  used in this new run's checkpoint selection.
- Direct geometric audit of all task-owned label links: official val has
  38,759 boxes; after 640 letterboxing, median box size about 9.88×12.67 px,
  92.44% under 32² area, 45.18% shortest edge under 8 px, 14.29% under
  4 px. At 960, the last fraction would geometrically fall to 3.56%.
  Neither this nor weak per-class AP proves the cause of misses. Exact
  metrics, per-class AP and limitations are in
  `VISDRONE_OFFICIAL_640_BASELINE_2026-09-23.md`.
- The 960-input conventional control's four-GPU, 1%-fraction smoke
  completed successfully after the 640 manager exited. Peak reported
  GPU memory in the short smoke was 5.08 GB per process; this is not a
  full-run peak estimate. EDAC UE counters remained 0. At 04:25 CST,
  after checking no task/training or GPU process was active, the full
  125-epoch seed-2027 960 run was launched under the task manager on
  physical GPUs 4–7 and NUMA0. Its state is
  `research/visdrone_official_960_s2027_state.json` on gpu8-team.

## 2026-09-23 — Additional desktop literature screen

- Read methods, primary tables and conclusions from six further PDFs in
  the user's Desktop folders. The detailed source-by-source appraisal is
  `DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`.
- FeatEnHancer and FE-YOLO are further prior art for jointly optimized
  preprocessing/feature enhancement plus detection. Their gains are
  dataset/detector-dependent, so they do not justify another generic
  "enhancement helps detection" claim.
- The 2026 YOLO-TaMoE manuscript already studies region-adaptive
  computation and VisDrone, limiting novelty claims for spatial routing.
  Its 100-epoch batch-16 scratch numbers are not directly comparable to
  our 125-epoch batch-64 internal converted-label metric.
- The 2025 Self-Supervised YOLO preprint's own table shows a slight
  YOLOv5 regression despite prose claiming consistent gains, so it is
  weak support for changing our scratch baseline protocol. The more
  established hierarchical SSL paper concerns transfer/pretraining,
  a separate research question from our current scratch controls.

## 2026-09-23 — Standard P2-head small-object control prepared

- Pre-registered `VISDRONE_P2_HEAD_CONTROL_PROTOCOL_2026-09-23.md` while
  the 960 conventional input-size control is active. The official
  Ultralytics P2 architecture was copied unchanged to a task-owned
  `models/yolov8n-p2.yaml`; local SHA256 exactly matches the remote
  official source (`ba54363e9f283e8f60b0fa0843eb37115093338d6f9273ccd4f74f60639754f4`).
  A 1-epoch 4-GPU smoke and 125-epoch formal config are prepared but
  **not launched**. P2 is a known architectural control, not novelty.
- Gate: finish/audit 960, free task GPU processes, hardware preflight,
  then run P2 smoke; only after a pass run the same-seed/budget P2 formal
  experiment. Compare AP, per-class behavior, compute and memory on
  the same official-source val split using internal converted-label AP.
- CPU-only structural dry run of the pinned task-owned P2 YAML confirmed
  `scale=n`, strides `[4, 8, 16, 32]`, and 3,354,144 parameters at default
  YAML `nc=80`; the actual VisDrone training model will set `nc=10` and
  must report its own parameter count. Remote copy's SHA256 matches the
  official source exactly. No GPU P2 run started while 960 is active.
- Pre-registered `VISDRONE_SIZE_STRATIFIED_ERROR_PROTOCOL_2026-09-23.md`
  while 960 is still training. A task-owned evaluator will run the
  selected 640/960 checkpoints on identical raw val images and report
  class-aware, one-to-one GT recall at IoU .50/.75 by *fixed* 640-reference
  size buckets. It is a diagnostic, not AP nor the official VisDrone
  toolkit score. Synthetic helper checks for duplicate matching and
  bucket edges passed; full model-inference validation is still pending.

## 2026-09-23 04:43 CST — Intermediate 960 observation (not final)

- Formal 960 manager and training process are alive, CSV has 82
  contiguous completed rows, physical GPUs 4–7 are occupied by this
  task, and EDAC UE counters remain 0/0. At epoch 82, 960's built-in
  official-source val **internal** mAP50-95 is 0.22605; the previously
  saved 640 CSV at the same epoch is 0.16511. The +0.06094 gap is a
  preliminary within-seed, within-epoch observation, not a final AP
  claim nor proof that object sampling alone caused the gain. No job
  intervention was needed; continue to 125 and independent re-evaluation.
- While 960 remained active, a CPU-only, single-image preflight of the
  independent 640 checkpoint predictor returned the expected task-view
  path and 567 low-threshold predicted boxes at `conf=.001`, NMS `.7`,
  `max_det=1000`; no GPU was used. This validates path handling and
  inference invocation for the planned size diagnostic, not its full
  548-image metric or accuracy.
- Key-by-key comparison of pinned formal configs confirmed only `imgsz`
  and run `name` differ; both remote config SHA256 values match local
  copies. This strengthens the 640-vs-960 contrast, but does not imply
  identical stochastic augmentation outcomes at different resolutions.

## 2026-09-23 — 960 control completed; independent size diagnosis

- Remote 960 manager finished at 04:51:50 CST as `verified_complete`:
  strict 125 CSV rows and readable best/last. No task processes or GPU
  allocations remained; both EDAC UE counters were 0 and recent kernel
  check found no new MCE/SIGBUS. A brief SSH forwarding refusal during
  observation was transient; the same task was re-queried, not relaunched.
- Best checkpoint epoch111 on the 548-image official-source val split.
  Independent raw-image Ultralytics evaluation yielded internal mAP50
  0.403649 and mAP50–95 0.232476 versus 640's 0.309082/0.170547.
  All 10 per-class AP50–95 values rose. CSV time 1573.05 vs1015.27s
  (~1.55×). This is a conventional resolution effect, not a novel method.
- A read-only scan of both rank-0 training logs found displayed
  `GPU_mem` maxima of 10.1 G (640) and 14.3 G (960), a ~1.42×
  increase; not a full-system GPU memory or inference-speed measure.
- Pre-registered size diagnostic successfully ran on both saved best
  checkpoints, matched all 548 images and 38,759 labels, and found
  IoU-.50 recall on fixed `<16 px` GTs increased 0.5065→0.6605
  (n=27,103), compared with 0.9123→0.9299 for 32–<96 px GTs
  (n=2,897). This supports a size bottleneck hypothesis, not causal
  proof. Full metrics, class table and caveats are in
  `VISDRONE_640_VS_960_RESULT_2026-09-23.md`.
- Diagnostic `<16 px` IoU-.50 recall rises in all ten classes (e.g.
  pedestrian 0.498→0.668, bicycle 0.221→0.406, motor
  0.565→0.733); the report includes all class counts and deltas,
  avoiding selective examples as the only evidence.
- P2 preflight then found a **new** 04:56:39 CST uncorrectable memory
  error: EDAC mc1 0→1, SIGBUS to unrelated `update-notifier`.
  Physical page address maps via memory block77 to NUMA node1;
  mc0 remains 0. The original no-new-UE gate therefore failed,
  and P2 was **not** launched. Exact read-only evidence, residual
  risks and a documented ≥10-minute node0-only continuation gate are
  in `HARDWARE_EVENT_2026-09-23_0456.md`. The server owner had earlier
  authorized continued use despite faults, but we do not call the
  hardware healthy or hide this event.
- At 05:07 CST the documented ≥10-minute stability check passed:
  same boot, mc0=0/mc1=1, no further MCE/SIGBUS, physical GPUs4–7
  idle. P2 1-epoch/1%-fraction four-GPU smoke completed with task-view
  cache, readable best/last and no new fault. DDP used four processes;
  10-class P2 model summary: 2,922,360 params, 12.2 GFLOPs,
  strides4/8/16/32. Smoke AP is not research evidence.
- Fresh preflight again found no task/GPU process or new hardware
  fault, and formal P2/640 scratch 125epoch seed2027 started at
  05:09 CST using physical GPU4–7/NUMA0, manager tag
  `visdrone_official_p2_640_s2027`. This is a known standard P2
  architectural control and remains subject to strict audit and
  independent val evaluation before any conclusion.
- Post-launch key-by-key config diff found only `model` and run `name`
  differ between 640 standard YOLO and P2. Local/remote P2 config
  SHA256 matches. Despite fewer parameters (2.922M vs standard
  ~3.008M in model summaries), P2's smoke summary has higher
  theoretical GFLOPs (12.2 vs ~8.1 at 640); measure actual latency
  rather than claiming it is cheaper from parameter count.
- Pre-registered and ran a GPU-free, GT-Oracle fixed-grid tile
  coverage audit on all 548 VisDrone val labels/38,759 GTs while P2
  trained. In 436/548 images, object centers occupy **all four** 2×2
  cells. Even an impossible top-2 GT-selected 2×2 strategy covers
  only 79.0% of centers; top-3 covers 93.1%. For 3×3, top-5 of 9
  GT-selected cells covers 92.8%. Full occupancy histograms and
  limitations are in `VISDRONE_TILE_COVERAGE_ORACLE_RESULT_2026-09-23.md`
  and raw JSON. Naive fixed-grid selective inference is not a clear
  low-cost path to exhaustive detection here; no image inference or
  AP claim arises from this audit.
- At the user's request, further read three Desktop papers not yet covered
  in detail: ICPR 2021 dynamic low-light enhancement, ICCV 2023
  DynamicISP, and BMVC 2022 IAT (including its supplementary joint-training
  table). The former's fixed bilateral/guided filtering underperformed
  its raw Faster R-CNN on ExDark, while its dynamic joint training improved
  author-reported AP; its small-object AP is inconsistent across tables.
  The latter is a RAW/prior-frame feedback ISP and reports four-seed
  human-detection results and end-to-end latency, including a much slower
  multi-function variant. IAT uses a ~91K-parameter RGB front end and
  reports ExDark/YOLOv3 joint-training gains that differ by initialization;
  its main table's pre-enhancement result is not the joint result. These
  narrow novelty claims for generic
  task-driven preprocessing and reinforce that blur/AP/speed must each be
  empirically demonstrated. Details: `DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`.
- Read primary QueryDet (CVPR 2022) paper to bound the VisDrone/P2 pivot:
  conditional high-resolution feature computation is prior art, and the
  published VisDrone protocol uses quarter-image training rather than our
  whole-image YOLOv8n protocol. Its P2 ablation also cautions that extra
  high-resolution features can reduce AP absent training-loss adaptation.
  Our fixed-grid tile-coverage audit does not invalidate fine feature-level
  sparsity. Details/source URL in the literature addendum.
- Pre-registered the next diagnostic, **not yet launched**:
  `VISDRONE_TRAIN_ONLY_BLUR_960_PROTOCOL_2026-09-23.md`. It compares
  standard YOLOv8n scratch/960/125ep seed2027 against a matched
  train-only, GT-box-protected Gaussian background blur with p=0.5,
  inherited VOC strong settings. Raw official-source val remains untouched.
  A key-by-key JSON comparison found only run `name` plus intended
  `oracle_*` controls differ from the existing 960 baseline. The distinct
  smoke must exercise the actual blur path, and P2 must finish first.
  This is a controlled question about domain transfer, not a claimed
  novelty or deployable model.
- At 05:29 low-frequency P2 observation found epoch 88 and a live manager,
  but a second **new** uncorrectable memory event had occurred at 05:23:26.
  SIGBUS went to unrelated `teamviewerd`, EDAC mc1 rose 1→2 while mc0
  stayed 0; the faulting physical page maps to memory block77/NUMA1.
  The existing P2 process was not touched. A **new** blur run is held
  pending hardware reassessment, not launched on an already tripped
  recurrence gate. Exact evidence and decision are in
  `HARDWARE_EVENT_2026-09-23_0456.md`.
- A CPU-only audit of the three existing VisDrone 125-epoch CSV curves
  found mature-epoch rank order standard960 > P2-640 > standard640
  on 114/116 epochs 10–125 (exceptions only epochs 12 and 14 for
  P2/standard640). Epoch100→125 AP50–95 gains were only 0.00063,
  0.00109 and 0.00110 in that model order; at epoch125 960 still led
  P2 by 0.03478. This weakens an "undertrained P2" explanation but
  cannot prove a longer schedule would never help or identify the
  precision/localization cause. Full method and numbers:
  `VISDRONE_LATE_EPOCH_CURVE_AUDIT_2026-09-23.md`.
- A CPU-only exact Oracle target-strength audit used all 548 official-
  source VisDrone val images/38,759 boxes at 960 square LetterBox and
  the **already locked** margin/transition/min-radius recipe. All
  counts/histogram reconciled. Among valid image pixels, 13.2637% are
  fully protected, 54.9202% have target strength ≥0.9, and 45.7968%
  ≥0.99; the mean image strength is 0.6701. This proves the proposed
  mask is a strong intervention geometrically, not that it helps AP;
  training mosaics differ and inference uses no GT. It does not change
  the pre-registration or justify starting GPU work amid recurrent UE.
  Details and raw per-image JSON:
  `VISDRONE_ORACLE_STRENGTH_AUDIT_2026-09-23.md` and
  `VISDRONE_ORACLE_STRENGTH_AUDIT.json`.
- Under a 06:23 stable-but-still-faulty EDAC read (`mc0=0/mc1=12`), ran
  **CPU-only**, four-thread, NUMA0-bound paired prediction-quality
  diagnostics on the existing P2/640 and standard/960 best checkpoints;
  no GPU work, new training, data edits or test-dev access. Both covered
  the same 548 official-source val images/38,759 converted GTs and
  passed per-point TP+FP integrity. At IoU.50/conf.001 P2 had 28,399
  TPs versus 960's 28,349 but 439,265 vs 352,433 unmatched predictions;
  at conf.1 P2 had 23,239 vs 24,634 TPs and 42,040 vs 35,935 FPs.
  At conf.5 P2 had 11,629 vs 13,591 TPs. This supports a confidence/
  discrimination or precision contribution to the AP gap beyond mere
  low-floor recall, without establishing its cause; VisDrone ignored
  regions and max_det censoring remain caveats. Full protocol, table
  and raw JSON: `VISDRONE_P2_VS_960_CPU_QUALITY_DIAGNOSTIC_2026-09-23.md`.
- Extended the same CPU-only prediction-quality diagnostic to the
  existing standard640 best checkpoint. At IoU.50/conf.001 standard640
  had 23,604 TPs/299,857 unmatched predictions and only 33/548 images
  reached max_det1000, versus P2's 28,399 TPs/439,265 unmatched and
  337/548 capped. At conf.1 precision/recall standard640 38.30%/52.56%,
  P2 35.60%/59.96%, standard960 40.67%/63.56%. The three-way pattern
  suggests P2 trades many more candidates for recall while 960 also
  improves score/precision quality. This remains a converted-label
  diagnostic, not official AP or causal proof. Raw data is
  `VISDRONE_STANDARD_640_CPU_QUALITY.json` and the comparison report
  above.
- The 06:33 read-only preflight found the same boot and stable
  `mc0=0/mc1=12` for ~35 minutes after the last UE, no new matching
  kernel errors, GPUs4–7 each at 2 MiB/0%, no task training process,
  task-owned smoke run/state absent, and pinned remote config hashes
  unchanged. The original “no new UE after 04:56” gate remains failed.
  Before any run, documented a narrower, single 1%-data/one-epoch
  NUMA0/GPU4–7 smoke-only allowance under prior server-owner permission;
  formal training needs a separate post-smoke decision. This is a
  risk-managed experiment on **unhealthy hardware**, not a claim of
  repair. See protocol amendment and hardware incident record.
- The pre-registered 1%-fraction/one-epoch, 4-GPU/NUMA0 VisDrone
  Oracle train-only **smoke** returned `verified_complete` at 06:35:39.
  Its log shows 65 training images, 2/2 batches, 548 raw-val images,
  38,759 val GTs, ~5.1 GiB rank-0 GPU memory, readable best/last.
  Entrypoint mapping, inherited batch preprocess, zero-warmup/ramp
  schedule (`mix_for_epoch(0)=0.5`), probability1 and standard
  validator were checked; DDP serializes the concrete subclass and
  custom args. This gives code-path evidence of real blur application,
  not a pixelwise batch trace. Post-smoke EDAC still mc0=0/mc1=12,
  no new kernel fault and GPUs returned idle. AP from this smoke has
  no value; formal 125-epoch training needs a separate gate. Details:
  `VISDRONE_ORACLE_960_SMOKE_RESULT_2026-09-23.md`.
- At a **separate** 06:43 formal preflight, ~45 minutes after the last
  05:58 UE, the same boot and EDAC mc0=0/mc1=12 persisted; no new
  matching kernel faults since06:33, GPUs4–7 idle, no task trainer/
  manager, formal run/state absent and formal config hash unchanged.
  The completed 1%-data smoke had not changed EDAC and had validated
  DDP/real blur code path. Before launching, recorded a one-attempt
  formal 125-epoch train-only Oracle blur decision with physical
  GPU4–7/NUMA0 binding; the original strict no-new-UE gate remains
  failed and healthy-hardware replication remains mandatory. See
  protocol amendment and hardware incident record.
- Formal pre-registered VisDrone train-only Oracle blur 960/125ep
  seed2027 was launched once at 06:46 with task manager PID **67146**,
  tag `visdrone_oracle_blur_p50_960_s2027` and original exact-target
  config SHA256 `b097ccac...`. Initial authoritative checks found
  manager alive, `started` event and state `new`, with no preexisting
  run. `/proc/67146/environ` had `CUDA_VISIBLE_DEVICES=4,5,6,7`;
  `/proc/67146/numa_maps` showed `bind:0` with N0 pages; physical
  GPUs4–7 held the job, not GPUs0–3. This is **running**, not a result.
  Monitor at ~10-minute intervals, do not duplicate launch, and audit
  125 epochs/checkpoints and independent raw val before reporting AP.
- Before the running Oracle candidate yielded an AP result, locked an
  additional **secondary** mechanism analysis: independent raw-val AP
  remains primary; then use the same CPU prediction-quality script and
  fixed size-recall evaluator at standard960 settings to compare
  candidate versus existing standard960 on TP/FP across confidence
  thresholds, IoU .50/.75 and cap incidence. This derives from the
  P2/960 diagnostic but is not itself an assumption that blur works.
  Ignored-region and one-seed caveats remain; secondary improvements
  cannot rescue a negative primary AP. Details added to the protocol.
- Created a concise navigation snapshot
  `RESEARCH_STATUS_SNAPSHOT_2026-09-23_0650.md` consolidating the
  verified VOC20ep negative controls, VisDrone640/P2/960 controls,
  paired CPU error evidence, currently running Oracle960 formal
  experiment, locked evaluation and novelty/hardware limits. It does
  not replace raw reports or claim a result from the running job.
- While the already-running P2 task continued, a separate CPU-only exact
  union-area audit read all 548 VisDrone val labels/38,759 boxes without
  using GPU or changing source data. Original GT rectangles occupy mean
  9.84% of an image, 10%-margin protection rectangles 13.25%, leaving
  86.75% outside the *hard* protection zones. This does **not** imply
  maximum blur there or prove background can be discarded. It reconciles
  dense 2×2 tile occupancy with sparse box area; QueryDet already covers
  finer feature-level sparse computation. Full per-image JSON and caveats:
  `VISDRONE_BOX_PROTECTION_AREA_RESULT_2026-09-23.md`.
- P2/640 formal control returned `verified_complete` at 05:37:22 after
  strict125 finite CSV rows and readable best/last. Fresh resource and
  >10-minute post-05:23 hardware preflight found GPUs idle, mc0=0/mc1=2
  and no later MCE; the short independent raw-val and size diagnostics
  used only physical GPU4/NUMA0. Internal mAP50–95 is **0.198526**:
  +0.027979 vs standard640 but −0.033950 vs standard960. Every class
  follows that ranking. Tiny fixed-bucket R@.50 P2=0.6665 is similar
  to 960=0.6605, while AP remains lower. P2 CSV time 1674.71s is
  ~1.65× standard640 and ~1.06× standard960. This is a known control,
  not innovation; full class/size/cost report and artifacts are in
  `VISDRONE_P2_640_CONTROL_RESULT_2026-09-23.md`.
- At 05:43:19–20 the kernel logged **five deferred uncorrectable memory
  errors**, increasing EDAC mc1 2→7 (mc0 still0). Their EDAC page
  addresses independently map to memory block71/NUMA1; no new userspace
  SIGBUS appears in this burst. New GPU training is **held** rather
  than applying the earlier minimum quiet gate mechanically after a
  cluster of faults. Exact records and decision are appended to
  `HARDWARE_EVENT_2026-09-23_0456.md`; no shared process/config was
  changed. Healthy-hardware replication remains necessary.
- During the GPU hold, reviewed primary ICCV 2025 UGS results as a
  distinct small-object prior: scale-sensitive localization-gradient
  stabilization exists and its reported YOLOv5-x/VisDrone protocols are
  not comparable to ours. Added a stand-alone bounded-error candidate
  target-map implementation (`pgb/fast_oracle_target.py`) plus CPU
  equivalence tests against the original exact Oracle map; it is **not
  connected** to any trainer or formal config and has no throughput/AP
  claim. Details in `FAST_ORACLE_TARGET_ENGINEERING_NOTE_2026-09-23.md`.
- A task-owned synthetic CPU microbenchmark at 960×960/70 boxes found
  exact target creation 0.0699 s vs bounded-local 0.00676 s best-of-two,
  with max map difference 7.17e-5 and identical protection masks.
  This **does not** estimate full server-GPU training time; the optimized
  path remains unconnected pending actual-profile evidence.
- During the GPU hold, the exact Oracle renderer passed an additional
  **CPU-only** one-image smoke at 960×960 with 70 synthetic boxes,
  sigma bank 0/2/4/8/16 and mix=0.5. It returned finite in-range output,
  strength range 0–1, mean absolute input change 0.1128 and local CPU
  runtime 0.163 s/image. This tests the code path, not GPU memory,
  full data loading, AP or training speed; no formal run was started.
- Read three more user-provided Desktop PDFs: Dark-ISP (RAW low-light
  detection), YOLO in the Dark (RAW feature-domain model fusion), and
  SNR-Aware Low-light Image Enhancement (restoration-only). Recorded
  their methods, key tables, evaluation limitations and current-project
  implications in `DESKTOP_LITERATURE_ADDENDUM_2026-09-23.md`. In
  particular, Dark-ISP uses a pretrained RetinaNet and true Bayer RAW,
  while YOLO in the Dark evaluates SID detections against teacher-created
  long-exposure boxes. Neither is evidence that our RGB blur improves AP.
- The 06:03 scheduled read-only server check found five additional
  deferred uncorrectable ECC events at 05:58:26, bringing `mc1` UE
  count from 7 to **12** (`mc0=0`), same boot. Their five EDAC pages
  resolve to memory block71, previously mapped to NUMA1; no new
  userspace SIGBUS was in this burst. New task GPU work remains on
  hold because the fault trend is continuing, regardless of earlier
  server-owner permission. Details in
  `HARDWARE_EVENT_2026-09-23_0456.md`.
- 2026-09-23 20:28 CST: Completed the user-requested **true-GT-box
  test-time blur Oracle** on VisDrone val548 with a frozen scratch
  YOLOv8n/960/125ep checkpoint. Pre-registered no-margin GT protection,
  σ=3/8/16 background blur, raw and sham controls. CPU geometry and
  protected-pixel tests passed 5/5; smoke8 raw/sham AP matched. Full548
  raw/sham mAP50–95 both 0.232551; light 0.261151 (+2.860 AP points),
  medium 0.260525 (+2.797), heavy 0.256711 (+2.416). These are **not
  deployable gains**: evaluation labels create the protected rectangles
  and leak location/shape to the model. Predicted-box experiment was
  negative, but different halo widths prevent a clean causal comparison.
  During the run mc1 UE count increased 44→52 (eight errors at 20:25),
  mc0 stayed 0; no SIGBUS to this task observed. Full metrics and
  limitations: `VISDRONE_TESTTIME_GT_ORACLE_RESULTS_2026-09-23.md`; raw
  JSON in `testtime_groundtruth_blur_960/`.

## 2026-09-23 — 安全背景门控 v0 实现与首次真实 VOC 联调

- 按用户决定启动中等容量、保守目标保护的图像前端研究。实现 959,953 参数的多尺度 CNN、单尺度可微高斯混合、非对称保护损失及 YOLOv8n 训练/验证/推理集成；旧 38.5K PGB 未被冒充为新结果。
- 8 项单测/集成测试通过。两次合成数据全链路冒烟通过。真实 VOC 1%/单轮 BF16 map-only 冒烟在三次保留的工程失败后修复，第四次完整结束且权重可读；冻结 YOLO 参数和零模糊预测均与起始 scratch 基线逐位相同。许可图均值约 0.018，尚非有效门控，更无 AP 提升结论。
- 用户将 GPU 选择改为 1–4，允许只用 2/3 卡。拓扑显示 GPU1–3 属 NUMA0，GPU4 属 NUMA1；当前 node1 新近发生硬件错误，node0 本次窗口无新错误但历史亦不健康。冒烟仅用 GPU1/NUMA0，不触碰 GPU7 的其他用户进程。完整报告：`SAFE_BACKGROUND_V0_IMPLEMENTATION_2026-09-23.md`。

## 2026-09-24 — 安全背景门控 v0 阶段实验及负结果

- 物理 GPU1–3/NUMA0 完成全 train90 5ep map-only 门控；物理 GPU1–2/NUMA0 完成 10% train90 2ep 冻结检测器反馈、3ep 联合训练及同起点/同预算原图对照。阶段和 best/last 审计通过，冻结期检测器逐 tensor 未变，联合期检测器和门控确有更新。所有推理图像的门控均由图像自身预测，未使用测试框。
- 同一 dev1655/YOLO.val 评估：map-only 门控 mix0/0.5/1.0 的 mAP50–95 分别为 0.142530/0.143155/0.142569；联合组 mix0.5 为 0.137052，同预算裸 YOLO 0.139551。增益未保留，且裸 YOLO 推理约 0.695 ms/图、联合组约 2.564 ms/图（单次 batch64/RTX3090 统计）。
- 前 256 张 dev 的 612 实例中，满足“框内 95% 像素 p≤0.1”的比例从 map-only 93.63% 降至联合训练 92.16%；远背景 p>0.1 比例由 55.50% 升至 58.56%。随后联合组全 dev1655/3882 实例审计完成：总体安全率 88.92%，小目标 86/101=85.15%；远背景 p>0.1 为 56.41%。类别级联合组 20 类中 16 类低于同预算对照。**不将本版放大为正式论文实验或宣称 AP 提升。**
- 这轮 EDAC `mc0 UE=0`、`mc1 UE=52` 未增，NUMA0 仍不能称健康。完整方法、运行路径、表格、局限和后续门槛见 `SAFE_BACKGROUND_V0_PILOT_2026-09-24.md`。

## 2026-09-24 — 125 轮完整 VOC 配对长训预注册与四卡联调

- 用户要求尝试超过100轮、用同一 NUMA 的四卡检验短训负结果。GPU0–3 当前空闲且属 NUMA0；GPU7 被他人使用，NUMA1 最近有不可纠正错误且本地可用内存很低。预检 EDAC `mc0=0`、`mc1=52`，NUMA0 本地可用约59GB，项目磁盘余约365GB；不宣称节点健康。
- 固定 `safe_background` 与原图 YOLO 的125轮配对实验，完整 VOC train90/dev10 seed2027、四卡0–3/NUMA0、batch64、workers4/rank、同一 scratch20 YOLO 起点及相同增强/优化器。模糊组先前5轮只训门控，未额外更新检测器。39项共同配置逐项一致；第125轮权重主比较、best权重次比较，不用 VOC07 test 选参。详见 `SAFE_BACKGROUND_LONG_E125_PROTOCOL_2026-09-24.md`。
- 正式开始前，四卡/NUMA0 上通过 1%/1轮真实 Mosaic 与实际模糊、联合梯度的 smoke；CSV恰一轮，best/last可读，增强设置核实，EDAC未增。任务专属长训管理器 `tools/run_safe_background_long_pair.py --check` 通过，核验配置 SHA、空闲 GPU0–3、NUMA0 内存和磁盘。正式结果尚未产生。
- 01:09 CST，任务专属管理器 PID 124989 启动模糊组子进程 PID 125004，DDP 四进程分别占用物理 GPU0–3；GPU7 原有 PID30205 保持不动。启动状态显示 NUMA0 本地空闲57,866MB、磁盘空闲363.8GiB、EDAC仍0/52；`taskset` 亲和力及 `/proc/.../numa_maps` 的 `bind:0` 验证 NUMA0 内存策略。训练日志确认 train14,896 / dev1,655、batch64、16名总 dataloader workers、125轮计划。此时仅第一轮进行中，不宣称长训完成。
- 01:12 CST 初检：模糊组第1轮 CSV 已写出，best/last 各约50MB且可读取；checkpoint `train_args` 明确记录125轮、batch64、GPU0–3、`sbg_map_only=False`、推理mix0.5、前5轮零混合再10轮渐增。管理器与四个DDP进程仍存活，EDAC仍0/52。第1轮 dev mAP50–95=0.0954 为过渡期暂值，不能与终局/对照比较。旧的已暂停监测已更新为本任务十分钟静默复查。
- 约02:45 CST 应用户询问审计联合损失：当前检测损失是三元素向量，而辅助 `map_loss` 标量被广播到每个元素后由训练器求和；配置0.25在联合阶段实际有效系数为0.75。第56轮显示map=0.06817，对实际标量总损失贡献约0.20451；这不等于门控梯度占比。运行中不修改训练；本轮结果必须标注真实系数，不能冒充预注册lambda=0.25。详见 `SAFE_BACKGROUND_LONG_LOSS_SCALE_AUDIT_2026-09-24.md`。
- 约04:05 CST 评估门控损失设计：非对称保护/远背景监督、区域平衡及检测反馈构成合理首版原型；但远离已标框不保证安全，difficult 尚未接入忽略框，保护项未逐实例均衡，旧短训小目标保护率仅85.15%。先完成第125轮配对，再以完整dev目标安全率、远背景处理、梯度冲突和AP/延迟判断；不在运行中调权重。详见 `SAFE_BACKGROUND_LOSS_DESIGN_REVIEW_2026-09-24.md`。
- 约04:10 CST 针对用户提出“将模糊性质融入检测网络”做文献核对与候选路线：SET/CVPR2025 已有特征背景平滑，BlurPool 与双流融合也有先例。提出保留原特征直通、以低成本平滑特征作检测条件化增量，并用逐实例风险而非全图均值保护小目标；此为待验证假设，不宣称新颖性或结果，当前长训结束前不新开训练。详见 `BLUR_INTEGRATED_DETECTOR_DIRECTION_2026-09-24.md`。
- 04:32:47 CST 模糊组完成全部125轮、训练进程退出码0，任务管理器报告 `job_verified_complete`；`results.csv` 为表头+连续125轮，第125轮 dev P/R/mAP50/mAP50-95=`0.76262/0.61668/0.69688/0.49529`。`best.pt`/`last.pt` 已落盘；这仍是训练期每轮验证值，正式结论须等两组完成后独立重评。同一管理器于04:32:48通过下一作业预检并启动原图对照组 PID150816、四个 DDP 进程占物理 GPU0–3/NUMA0；EDAC `mc0=0, mc1=52` 未增，GPU7他人进程未触碰。联合阶段实际辅助损失系数0.75的协议偏差继续保留标注，不能以此单组结果宣称优于基线。
- 04:35 CST 将方法组第125轮 P/R/AP、train/val 四项损失、最高开发集轮次及训练结束后 `best.pt` 的20类表逐项从 CSV/日志复核，完整归档于 `SAFE_BACKGROUND_LONG_E125_METHOD_RESULT_2026-09-24.md`。明确逐类表并非第125轮 `last.pt` 的独立验证，日志里的0.6ms/image也不能直接算含前端的端到端延迟；基线和保护率仍待验收。
- 约04:45 CST 用户要求最晚持续研究至09:00。将十分钟自动复查加上09:00截止、配对独立评估、门控风险/延迟、负结果与文献分析条款；到点不杀正常运行的本任务。准备收尾用独立评估脚本：固定 conf=.001、IoU=.7、max_det=300、FP32、rect=True，加入 checkpoint/data SHA256 与20类 P/R/AP50/AP50-95 输出，并拒绝覆盖已有 JSON；本地语法检查通过，脚本同步远端。运行中的训练脚本与配置未修改，也没有新启动GPU任务。
- 04:48:49 CST 原图对照组已完成59/125轮，管理器及四个 DDP 进程正常占 GPU0–3，EDAC仍0/52；第59轮 dev P/R/mAP50/mAP50-95=`0.75052/0.60463/0.68228/0.48106`，仅中途值。最近5轮约15.9秒/轮，按此估计剩余66轮约17.5分钟；前端组全程 `12196.8/125≈97.6` 秒/轮，约为原图组训练轮耗时的6.1倍。此比值是训练吞吐的早期对照，不等于推理延迟，更不能凭中途AP判断最终性能。
- 04:55 CST 按 CSV **列名**而非固定列序比较同轮进度，原图基线在 epoch25/50/59/66/75/85/86 的 dev mAP50-95 均高于安全背景模糊组，差值分别为 `+0.00282/+0.00627/+0.00674/+0.00725/+0.00688/+0.00512/+0.00439`（概率尺度）。至 epoch87 基线为0.49309，距方法组末轮0.49529仅0.00220，但跨轮次不是最终因果比较。初步证据强烈指向原图基线占优且训练成本远低；仍须等第125轮与独立复评，不用“肯定”代替验收。
- 约04:57 CST 应用户询问预先写下负结果后的决策框架：先完成正式末轮配对与独立验证，再分辨门控漏保护、实际背景处理不足、推理模糊反伤、训练干扰及Oracle几何泄露；若无明确可修复机制，停止在VOC上追loss/σ，优先审查VisDrone小目标与强常规分辨率/P2/tiling控制。核对SET、IA-YOLO、DM-EFS一手论文，避免把已有前端/特征平滑/浅层特征当成新颖性。详见 `NEGATIVE_OUTCOME_DECISION_PLAN_2026-09-24.md`，本轮未开新训练。
- 约05:04 CST 用户明确放宽研究路线：允许已有论文方法的认真移植、场景适配和可解释改进，不强求全新范式。记录务实候选：先完成VOC对照，再在规范VisDrone划分与强常规控制下，把SET的HBS作为忠实移植基线；仅对已诊断的尺度/密度失效做一个适配，与纯移植和等成本原图模型配对，多seed/跨集验证。SET/DM-EFS/IA-YOLO一手来源及发表边界见 `PRACTICAL_METHOD_ADAPTATION_ROUTE_2026-09-24.md`。尚未实施或声称新颖性。
- 约05:05 CST 用户进一步授权自主开展后续实验与研究至09:00，不需等其逐项批准。十分钟复查任务已更新为：两组严格审计后，先查门控/图像变化/AP/端到端成本；仅据明确假设、经过smoke和资源预检再启动至多一个低成本、有同预算对照的诊断/适配试验，继续限制 GPU0–3/NUMA0，绝不碰其他用户进程和共享环境；09:00停止复查并报告实情，不中断安全运行中的训练。
- 约05:33 CST 配对正式审计与分析完成：两组 CSV 均连续125轮，best/last均可加载、20类和共同配置一致。第125轮相同独立 dev1655 复评：安全背景组 P/R/mAP50/mAP50-95=`0.758914/0.618161/0.696393/0.495201`，原图组=`0.746594/0.633446/0.701927/0.500451`。方法 mAP50-95 **低0.005251（0.525 AP点）**，Recall亦低；20类中7升13降。方法混合系数0的诊断AP95=0.494941，启用0.5仅增0.000260；均低于原图。门控全dev3882框安全率90.78%，小框93/101安全，远背景均值许可0.382；正方形LetterBox诊断与正式rect评估几何不同，不冒充精确逐图保护率。训练耗时比6.14×；预解码图上的相同GPU0 `YOLO.predict` 短基准 batch16 为4.400 vs2.333ms/图。新增内核MCE/SIGBUS未见，EDAC仍mc0=0/mc1=52。结论为该v0 no-go，不将负结果追调为“提升”。完整证据与局限见 `SAFE_BACKGROUND_LONG_E125_FINAL_RESULT_2026-09-24.md`，原始JSON/CSV与严格审计同步保存。
- 约05:37 CST 新增固定阈值配对诊断，先做32图 smoke 再跑完整 dev1655；原图输入、相同预测设置后才用标签匹配。3882个标注中双方均检出2576、仅原图147、仅方法139、双方均未检出1020（同类 IoU≥0.5、conf≥0.25、一对一贪心）；小目标仅原图4、仅方法3。因此无法把总AP损失简单解释为“小目标大面积被模糊导致漏检”，排名/定位和阈值行为仍可能有影响。完整 JSON 保存为 `SAFE_BACKGROUND_LONG_E125_PAIRED_RECALL_FULL_DEV.json`，并增补至最终报告。
- 约05:47–05:54 CST 转向已发表 SET 的训练期 HBS 子模块（**没有**实现其 API，不称完整 SET 或创新），将 FCOS 官方代码中的逐尺度两卷积残差背景特征平滑移植到 YOLOv8n 的三个 Detect 输入特征。保留原 YOLO 推理路径；额外检测损失用原特征再过同一头的 identity-extra-loss 组控制。原检测器权重同源且逐 tensor 相等、无标签推理输出逐 tensor 相同、HBS 模块反传梯度有限非零；两组各完成1% VisDrone/1轮真实四卡 DDP smoke，CSV/权重落盘且无新 EDAC/MCE。预注册 full VisDrone official train6471/val548、960/125轮、scratch/seed2027、GPU0–3/NUMA0、两组仅HBS模式不同，详见 `SET_HBS_ONLY_VISDRONE_960_E125_PROTOCOL_2026-09-24.md`。05:53:55 CST 任务专用管理器经预检启动 HBS 正式组，PID169059、子进程169129；NUMA0空闲约52GB、磁盘约363GiB、EDAC0/52，物理GPU0–3四进程占用，GPU7他人PID30205未触碰。管理器计划按顺序验收HBS后再启identity；当前**没有正式AP结论**。
- 约06:57 CST，HBS 首组训练和任务管理器严格验收完成：CSV 连续125轮、best/last权重可加载且配置一致，末轮训练时 val P/R/mAP50/mAP50–95=`0.50634/0.37930/0.38069/0.21474`。管理器已自动通过新一轮预检并顺序启动 identity-extra-loss 对照，四卡 GPU0–3/NUMA0，EDAC仍0/52、近10分钟无新增 MCE/SIGBUS；GPU7他人进程未触碰。当前没有配对独立复评，不能根据 HBS 单组数字下优劣结论。首组记录见 `SET_HBS_ONLY_VISDRONE_960_E125_METHOD_RESULT_2026-09-24.md`。
- 约07:35 CST，identity 对照也完成连续125轮并通过 checkpoint/配置审计，任务管理器记录 `pair_verified_complete`。两组末轮权重在相同 val548、数据 YAML 哈希和独立 `YOLO.val` 配置下复评：HBS P/R/mAP50/mAP50–95=`0.508415/0.378393/0.380833/0.214733`，identity=`0.492073/0.403812/0.396232/0.229070`，HBS 主指标低 **1.434 AP点**。逐类9降1升；在38,759标注的配对尺寸召回中，27,103个 tiny 框 R@IoU.5 为 HBS 0.6283 vs identity 0.6570。HBS 训练累计耗时3523秒 vs2150秒；推理模型前向单次测时相近。判 HBS-only 移植 no-go，不冒充完整 SET 或原创改进；详细指标、数据限制和后续机制审计计划见 `SET_HBS_ONLY_VISDRONE_960_E125_FINAL_RESULT_2026-09-24.md`。
- 约07:50 CST，按用户强调逐页阅读 Itti/Koch/Niebur 1998、YOLO+foveated rendering 2023、SemBA-FAST 2025、Multi-Scale Fovea 2026 四份 PDF，并复核图表。关键判断：后两篇主任务是给定目标类别且目标存在的**人类注视序列预测**，2026 版借多个裁剪缩小每次检测像素预算，不报告全图全部目标的 AP；2023 YOLO 短文只有5图计时，不能证明常规 mAP 提升或卷积 FLOPs 降低；1998 显著性仅适合作为候选点对照，不能保证低对比 tiny 目标召回。2026 表 II `5×64` 的 0.01% 像素比例与其尺寸公式不符，重算约1.16%。由此将下一道研究门槛改为 GT-free 的选区覆盖率与同等计算预算下的 VisDrone AP/真实时延，同 AutoFocus、ClusDet、SAHI 建立关联；尚未新增 GPU 实验或宣称可发表创新。详见 `FOVEATED_ATTENTION_FOUR_PAPERS_REVIEW_2026-09-24.md`。
- 约07:55 CST 实施只读 VisDrone 4×4 网格目标密度上界审计：train6471/343,200框的中位占用格数9/16，val548/38,759框为10/16；中央固定4格仅覆盖各约40%，按真实框事后挑最密4格也仅覆盖 train74.55%、val73.02%，8格分别94.41%、93.84%。这不是部署方案也不是任意形状裁剪的AP上界，只是固定格/框中心统计；它反对直接套用“少量注视点硬裁剪、仍要找到全图全部目标”，支持下一步研究全图粗检+少量补充高分辨率复检，并检查补充区域是否覆盖粗检漏检。脚本及解释见上述四论文研究文档；无GPU、无训练、原始标签未改。
- 约07:59 CST 预注册的冻结模型语义选区覆盖试验已完成 VisDrone val548/38,759框：不读GT选格，先用原图 YOLOv8n/960 预测框的单格置信度之和选4格；在粗检未匹配到的10,303个GT中覆盖75.67%框中心，8格覆盖93.87%。预测路径顺序和模型/清单SHA已核对。这不是重新检测的AP或速度改进，只达到继续研究的覆盖率门槛。详细协议 `FOVEATED_VISDRONE_REGION_PILOT_PROTOCOL_2026-09-24.md`，原始JSON在远端 `diagnostics/foveated_region_coverage_20260924/full548_order_checked.json`。
- 08:04 CST 尝试下一道裁剪推理代码的远端**CPU自测**时，内核新一轮MC1不可纠正内存错误，EDAC `mc1=52→58`，记录 Python 进程被MCE杀掉/SIGBUS，另波及teamviewerd与apport；两次自测均退出码1且无成功输出，与内核PID时间高度吻合。**暂停所有新增GPU训练/推理和该自测**，没有完整裁剪AP结果，不能把已完成的选区覆盖当成检测增益。完整故障事实和边界见 `FOVEATED_PILOT_HARDWARE_HOLD_2026-09-24.md`。
- 08:10 CST 进一步检索原始发表来源确认新颖性边界：GOIS/Neurocomputing 2025 与 DAHI/Pattern Recognition 2026 已分别覆盖动态密度切片、粗检密度导向裁剪和边缘感知合并；AutoFocus、ClusDet、SAHI 更早已覆盖相关范式。故当前“预测框热度选格”只算机制诊断或现有路线的务实适配，不能命名为新发明或已可发论文。名含 Focal Region 的 CVPR2026 UAVGen 是数据合成，不是同一推理机制。来源链接与当前试验具体边界记入四论文评述；硬件暂停期间只作离线分析。
- 08:09服务器重启后，保留上一boot MC1 UE52→58与SIGBUS事实；新boot安静观察、NUMA0/GPU0空闲核验后，远端CPU自测、16图GPU冒烟及548图冻结裁剪推理完成。首次548图原图AP95=0.215110与独立YOLO.val/last.pt=0.231578不符，源码定位是普通predict默认NMS `multi_label=False` 而验证器用True；首次JSON只保留诊断，不用于比较正式基线。
- 08:39 CST 三组统一 `multi_label=True` 重算：VisDrone val548同一last.pt/转换标签AP50–95原图0.230872、原图+固定中央4格0.217609、原图+预测热度top4格0.226803；自适应比同预算弱固定对照+0.009194，却比原图−0.004069，且每图5次960推理，判当前原型no-go。bus/truck分别比原图下降0.0409/0.0310 AP，具体失效机制尚须逐图核验，不预设是NMS/裁断。三组完整P/R/mAP50/逐类/时长、首次口径修正、原始JSON及SHA见 `FOVEATED_VISDRONE_CROP_PILOT_RESULT_2026-09-24.md`。试验完毕EDAC新boot mc0/mc1仍0且近窗无新MCE；不能称硬件已修复。未新增训练、未动其他用户任务或GPU4–7。
- 约08:50 CST 离线复核SAHI官方实现及AutoFocus/ClusDet/GOIS/DAHI一手论文，形成 `FOVEATED_PRIOR_ART_AND_FAILURE_ANALYSIS_2026-09-24.md`。当前C优于B只说明优于**中央固定4格弱对照**，而非优于正规固定重叠切片。我们的所有裁剪均放大到960且按普通classwise NMS合并，不是2026论文的低像素fovea。C最终约542k框/548图接近每图1000框上限，bus/truck损伤可能涉及排序截断、裁边或NMS替换，但尚无逐图因果证据。下一道工作是保存分支来源的离线TP/FP转移审计，再做同预算SAHI/高分辨率控制；先不在已反复查看的val上调融合阈值或启长训。
- 约09:00 CST 将“真正等预算”的下一轮候选写成 `FOVEATED_EQUAL_BUDGET_NEXT_PROTOCOL_2026-09-24.md`（讨论稿，未运行）：640全图+最多2个512局部输入共933,888模型输入像素，约为单次960的1.013倍，而当前full+4×960是5倍。必须同权重比较full640/full960、固定与自适应两裁剪，并单列正规SAHI的真实预算；以前960粗检top2覆盖率不能冒充640的结果。另写出NMS可误压原图TP的纯数学反例，说明一种可能机制而非实际归因。仍待逐框证据、安全预检与用户后续研究决定，不在本轮加跑。
- 约09:07 CST 继续核对GOIS期刊关联的作者公开仓库：README部分比较在VisDrone 6471训练图上评估，公开issue明确询问是否同图微调和评估而未见答复；故不把仓库宣称的大幅增益直接视为未见数据证明，也不指称论文全部实验已证泄漏。其页面“MIT”标签与实际专有`LICENSE`内容不一致，后续避免直接复制改写GOIS代码；SAHI实际`LICENSE`为MIT，优先作为实现可核对的强对照。链接和限定词写入 `FOVEATED_PRIOR_ART_AND_FAILURE_ANALYSIS_2026-09-24.md`。
- 约09:15 CST 对2026多尺度fovea论文的计时作**明确标注为推论**的任务转换：YOLO11每次fovea0.115秒、全图0.144秒，两次注视仅检测就约0.230秒，未算扫描决策与去重；因此不能从其单次注视加速推出本项目全图全部目标加速。补记于四论文评述和先验/失效审计，强调每张图完整任务总预算。
- 约09:25 CST 对冻结裁剪试验的P/R口径做源码复核：当前Ultralytics 8.4.146 `ap_per_class`在各组各自平滑平均F1峰值的置信度处读取摘要P/R；C相对B的R+0.017399不是共同固定置信度下的实例召回提升。已修正结果报告和预注册门槛表述：AP主结果及no-go不变；固定工作点召回尚待严格一对一GT配对核验。本轮无新GPU作业或训练。
- 约09:35 CST 仅用已封存全量JSON拆分逐类AP门槛：bus整图→自适应AP50 `0.5088→0.4642`、AP95 `0.3646→0.3237`；truck AP50 `0.3291→0.2820`、AP95 `0.2151→0.1841`。宽松IoU0.5也下降，不能仅归结为高IoU定位精度；car AP50基本不变而AP95小升。仍无法从聚合值确认NMS、裁边或分数排序哪个为主；记录于先验/失效分析。另把路线文档中08:15的“尚无AP”标为历史状态，避免和08:39已完成试验相矛盾。无GPU作业。
- 约09:45 CST 起草10:00交接`FOVEATED_RESEARCH_10AM_HANDOFF_2026-09-24.md`，将两项125轮负结果、冻结裁剪三组指标、四篇重点论文的任务差异、相关既有方法、当前可归因与未归因证据、下一道等预算/盲测门槛统一收束。文档明确是09:45草稿，最终时间状态须在10:00补记；未启动训练、未把诊断结果称创新。
- 10:00 CST 按用户截止要求暂停本任务每10分钟自动复查，应用返回状态`PAUSED`；没有终止任何服务器进程，也没有新GPU训练/推理。截止前09:56只读检查新boot EDAC mc0/mc1均0、过去30分钟未见新MCE/SIGBUS；历史上一boot MC1不可纠正错误/SIGBUS仍需健康硬件复现，不能因新计数清零称修复。完成`FOVEATED_RESEARCH_10AM_HANDOFF_2026-09-24.md`最终结论与未完成事项，结果保持负面如实报告。
- 10:00后应用户“研究并分析”，只读审计VisDrone train6471/val548原始转换标签与既有4×4/10% halo裁剪几何，不启动模型/GPU、不恢复自动复查。val中任意小格无法完整包含bus 29/251=11.55%、truck 98/750=13.07%，而pedestrian 2.25%、people 0.84%；train中bus13.65%、truck11.06%，风险在两集均存在。val严重截断（保留<80%）bus5.58%、truck4.13%，但平均被截断框仍保留约82%面积，故这只是裁边风险而非AP因果解释。统一将约30%宽高的小格放大到960，目标线性尺寸约3.33倍亦为待验证尺度失配假说。自测及脚本SHA、清单SHA、逐类表、反例与后续逐框归因门槛见`FOVEATED_VISDRONE_CROP_GEOMETRY_AUDIT_2026-09-24.md`；远端CPU读测后EDAC仍0/0。
- 同一用户请求下新增**一次定向冻结推理诊断**，预检GPU0空闲/NUMA0可用/EDAC0/0/近窗无MCE，仅使用物理GPU0且不碰GPU7他人任务；16图smoke与原融合框逐元素一致，随后val548完成。固定conf0.25、同类IoU0.5一对一匹配下，bus原图TP113→融合147、丢失0、新增34，FP69→252；truck TP236→308、丢失0、新增72，FP235→784。pedestrian/people各丢13/5但新增1345/384 TP，且这18个丢失GT完整落在已选裁剪内。结论：bus/truck AP50降分不能在此工作点归咎于原图TP被裁断框替换；新增FP负担与跨尺度排序更值得优先审计，但固定点不等于完整AP归因。代码、checkpoint/manifest/结果SHA和全部限制见`FOVEATED_VISDRONE_PROVENANCE_AUDIT_2026-09-24.md`，原始`FOVEATED_CROP_PROVENANCE_VAL548.json`本地/远端保留；运行后新boot EDAC仍0/0、近窗无新MCE/SIGBUS。未训练、未恢复自动复查。

## 2026-09-24 12:56–13:15 CST — 小框专属融合单变量试验及模糊假说

- 应用户继续试验，在运行前固定`FOVEATED_SMALL_ONLY_FUSION_PROTOCOL_2026-09-24.md`：同一冻结VisDrone scratch YOLOv8n/960/125轮末轮权重、整图热度top4格和五次960前向不变，仅过滤回映整图960后最长边>64像素的局部分支框，整图框不动。选择64受AutoFocus小尺度范围启发但非其复现。test-dev 1,610图历史项目看过，不称完全盲测；内部转换标签AP不称官方指标。代码纯函数自测及16图冒烟通过，新程序整图/原融合与旧程序同口径结果完全一致。
- 完整test-dev75,102框内部AP50–95：单次整图0.179863，整图+四格原融合0.178484，整图+四格小框专属融合0.180785。新规则相对原融合+0.230 AP百分点，但相对**单次整图仅+0.092点**，mAP50反而低0.187点；bus/truck虽比原融合部分恢复，仍分别低于单次整图3.115/1.610 AP点。局部低阈值候选仅删1.65%，融合最终框仅少0.135%；保持五次960输入，成本-效果不合格。故作为机制诊断保留，不称稳健改进或论文成果。完整逐类/P/R/哈希/限制在`FOVEATED_SMALL_ONLY_TESTDEV_RESULT_2026-09-24.md`及原始JSON，本地与远端同步。
- 用户提出模糊减误报：查旧VisDrone冻结模型推理期预测框保护背景模糊，σ3/8/16均使AP95从原图0.232551降至0.221578/0.213893/0.211288，故不能直接采用“多模糊”。局部裁剪轻模糊仍是不同假说，必须与原裁剪同预算单变量配对，并查真阳性损失；暂未运行，研究边界与可证伪设计写入`FOVEATED_CROP_BLUR_HYPOTHESIS_2026-09-24.md`。本轮未重训、未重开已暂停自动复查。完成后GPU0空闲、EDAC新boot mc0/mc1=0、近窗无新增MCE/SIGBUS，但硬件历史故障依旧。

## 2026-09-24 — 局部裁剪轻度模糊单变量试验

- 为直接检验“模糊能否压局部误报”，在运行前固定`FOVEATED_CROP_LIGHT_BLUR_PROTOCOL_2026-09-24.md`：同一冻结YOLOv8n/960末轮权重、VisDrone val548、相同top4裁剪和原融合，仅把局部裁剪送模型前按960输入等效σ3做高斯模糊；整图输入不动，不叠加64像素过滤或其它修改。全裁剪处理会同时模糊目标，并非目标保护式背景模糊。自测、16图smoke通过，原图与未模糊裁剪对照和旧程序完全相等；实际像素平均差6.282/255。
- 全548图内部AP50–95：单次整图0.230872、整图+原裁剪0.226803、整图+轻模糊裁剪0.211691；模糊比未模糊低**1.511 AP点**，比单次整图低1.918点。固定conf0.25/IoU0.5下pedestrian FP7105→2962但TP5078→3843；people FP2122→1414、TP1984→1649；truck FP784→305、TP308→243；bus FP252→286、TP147→117。模糊确实压部分FP，但严重抑制TP，bus反而更差，故本σ3局部模糊no-go，不继续在val扫σ。逐类、P/R口径、限制与SHA见`FOVEATED_CROP_LIGHT_BLUR_RESULT_2026-09-24.md`，原始JSON和脚本本地/远端保留。GPU0/NUMA0独占本任务，跑后GPU0空闲、新boot EDAC0/0且近窗无新MCE/SIGBUS；历史故障未修复。

## 2026-09-24 — 用户澄清目标保护意图、可行性审计

- 用户明确其设想是**尽量只模糊目标之外的背景**。已承认并区分上一轮“整张裁剪均匀模糊”并未检验该设想；原计划的逐目标均匀模糊损失重跑草案标为未执行。参考旧VOC学习式保护门控：小框93/101达到框内≥95%像素许可≤0.1，但方法AP仍低于基线，且同权重推理时关掉模糊只变0.000260；因此“模糊模块不够准确”不是旧实验已证的唯一原因。
- 新的**零模糊覆盖审计**使用整图冻结检测器低置信预测框外扩20%/最少8整图960像素、按原热度top4裁剪；不让GT影响掩膜。val548共38,759框，其中28,031能完整落入至少一张选中裁剪。conf≥0.01时，在这些GT中27,732/28,031（98.93%）框内≥95%面积受预测掩膜保护，小于8像素的4,602/4,673（98.48%）安全；所选裁剪47.90%像素硬保护，52.10%仍可处理。conf0.001/0.05给出保护—面积曲线，不作AP搜参。高覆盖可能来自大量假阳性框偶然保护，仍有299个eligible GT不安全，且框内保护不等于上下文保真。详见`FOVEATED_TARGET_PROTECTION_FEASIBILITY_RESULT_2026-09-24.md`及原始JSON/SHA；这仅是必要条件，无新AP。随后将conf0.01、20% halo、局部背景σ3锁进`FOVEATED_PREDICTED_BACKGROUND_BLUR_PROTOCOL_2026-09-24.md`，在test-dev仅作一次无GT推理对照；当时正式结果尚未完成。

## 2026-09-24 13:56 CST — 目标保护式背景模糊全量测试（负结果）

- 按运行前协议，冻结同一VisDrone YOLOv8n/960末轮权重，仅凭**原图预测框**保护局部裁剪中的候选目标，四张裁剪中只模糊预测掩膜外的背景，整图分支仍原图；真实框只事后评估。纯函数与16图冒烟确认硬保护像素逐位不变。test-dev 1,610图75,102框一次性评估：单次整图mAP50–95 **0.179863**、整图+未模糊四裁剪**0.178484**、整图+目标保护背景模糊四裁剪**0.176827**。后者较未模糊四裁剪低0.166 AP百分点，较单次整图低0.304点；mAP50亦更低，no-go。整图与未模糊四裁剪控制的全部指标与旧程序逐值一致。
- test-dev可完整落入选中裁剪的GT中49,426/51,388（96.18%）至少95%框内像素受保护；小于8整图960等效像素桶11,870/12,832（92.50%），低于先前val的98.93%/98.48%。保护掩膜覆盖37.18%所选裁剪像素，62.82%可处理，实际平均通道改变量3.20/255。故“只模糊目标外背景”已较大程度做到，却仍有小目标漏保及AP下降；不能将损失全部归咎于漏保。固定工作点pedestrian TP+120但FP+345、people TP+5但FP+288，指向误报/排序等仍须逐框审计。
- 全部条件、逐类AP、工作点TP/FP、保护率、代码/JSON/权重SHA、限制和硬件核验见`FOVEATED_PREDICTED_BACKGROUND_BLUR_TESTDEV_RESULT_2026-09-24.md`。结果JSON SHA256 `d9b629ae15bf165a166b2161f8b3a4fc06394a6422a6d8e634268d83829e70c4`，本地与服务器一致。test-dev曾被历史项目查看，不称盲测；内部转换标签不称官方VisDrone指标。只用物理GPU0/NUMA0，13:56后GPU0空闲、新boot EDAC0/0且近20分钟无新MCE/SIGBUS；历史硬件错误未消失。未启动新训练，也不继续在同一test-dev扫参数。

## 2026-09-24 14:19–14:33 CST — 真实框保护上限 Oracle 与预测框同图配对

- 按用户同意继续研究，在运行前落盘`FOVEATED_PREDICTED_VS_GT_PROTECTION_ORACLE_PROTOCOL_2026-09-24.md`：冻结同一VisDrone YOLOv8n/960/125轮末轮权重，仅在val548比较A单次整图、B整图+原始四裁剪、C整图+预测框保护背景σ3模糊四裁剪、D整图+GT框保护背景σ3模糊四裁剪。D在测试输入中使用真实标注，仅是不可部署的机制诊断。所有组同选格、同融合、同评价；不触碰已被历史分析的test-dev，也不调参。纯函数自测、16图smoke及控制组与封存程序逐值一致；全548图A/B全部指标与旧独立程序逐值相同。
- 全量内部mAP50–95：A **0.230872**，B **0.226803**，C **0.225561**，D **0.228695**。D较B高0.189 AP百分点、较C高0.313点，但较成本更低的A仍低0.218点；mAP50亦低于A。Oracle完美保护eligible GT 28,031/28,031；C为27,732/28,031（98.93%），然而C保护面积47.90%，D仅35.02%，表明差异不是只修复299个漏保GT：预测低分框还广泛改变背景处理区域，D又包含位置泄漏。
- 固定conf0.25/IoU0.5全10类合计：B TP24,047/FP24,846，C TP24,101/FP25,839，D TP24,091/FP24,405。D相对B真阳性+44、误报−441；C相对B真阳性+54、误报+993。工作点不是完整AP因果证明。结论为**小的条件性Oracle信号、当前可部署预测掩膜失败、且整体不及单次整图**；暂不扩训更大前端、不在val扫σ/阈值。报告、逐类AP/TP/FP、限制及原始JSON见`FOVEATED_PREDICTED_VS_GT_PROTECTION_ORACLE_RESULT_2026-09-24.md`。代码SHA256 `406e219e6ea897666a09bd1297c5f01fd97c2079d28a0680af67e62e2b17414f`，全量JSON SHA256 `c92d65cb464e083490abec6b1ce44144d129d38b8fe5bcfe19fb92de4fc2ce1a`，本地与服务器同步。只用GPU0/NUMA0，运行后GPU0空闲、新boot EDAC0/0、近窗无新MCE/SIGBUS；历史故障仍须健康硬件复现。
