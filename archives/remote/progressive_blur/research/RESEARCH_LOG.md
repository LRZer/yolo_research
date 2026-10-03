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
