# IGB-P0 isolated implementation audit

Date: 2026-09-19  
Status: **CPU implementation verified; real YOLO/CUDA execution not yet performed**  
Scope: local staging only; no remote access, remote write, CUDA call, optimizer, checkpoint mutation, or core-source edit.

## Outcome

IGB-P0 is implemented as an external zero-update probe. It consumes the live native O2M/TAL assignment interface, isolates each matched GT's assigned-class positive BCE plus complete CIoU + DFL/L1 localization loss, and obtains exact independent VJPs at the three tensors entering Detect (P3/P4/P5). It records target mass/count/area/quality correlations, Gini, top-20 share, and effective number, then applies the four conjunctive gates frozen in `BOLD_IDEA_SCREEN_2026-09-19.md`.

No file under `ultralytics/` was changed. The probe depends on the current `get_assigned_targets_and_loss(..., return_details=True)` and `bbox_loss` interfaces rather than reimplementing assignment or box loss.

## Artifacts and SHA256

| artifact | SHA256 |
|---|---|
| `work/remote_research_source/scripts/probe_igb_gradient_budget.py` | `a70cf1269c43fecada7b7087ec54094cc67c01640e90889ec29f912b9425dd75` |
| `work/remote_research_source/scripts/test_probe_igb_gradient_budget.py` | `80ce0f69e225d329e46ae6b4021129ead2accc85f1ee9217ca7a30d59709f9d2` |
| `work/yolo26_research/IGB_P0_SPEC.md` | `cbf27e8a2d0b75e190e580340828127661b074acebbb2fb9cc95b5ca95f11964` |

Pinned dependency/interface evidence:

| exact input | SHA256 |
|---|---|
| strict B50 provenance helper `probe_asac_assignment_stability.py` | `b0d764fdb00c54a926e27a0a5b8ffd8f016c626998c75321fc1697f4ed444bf9` |
| exact B50/TQGB `ultralytics/utils/loss.py` | `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9` |
| current `ultralytics/utils/tal.py` | `9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861` |
| current `ultralytics/nn/modules/head.py` | `1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8` |
| current `ultralytics/nn/tasks.py` | `c1da24859fd56b9869759bd4bfb753456c9074bfc3238b51a08ea3c6cbeda1ae` |
| current `ultralytics/data/build.py` | `1c3131adc60dc9763b9edcde853d3bc461ec908c2ed9d1965b2610bfd8c27847` |
| current detection trainer | `76d6b181259edc09b03f7cea1f9d59bd4eda2f7964146bbb2e76d4f7201dcf5f` |

The executable fail-fast checks these seven core/helper hashes. The loss pin was corrected after review showed that the first implementation had accidentally pinned the separately staged T1-BRC source (`e4dd...`) rather than the exact completed B50/TQGB source (`6dae...`). It additionally records—but does not pre-pin—the model YAML and dataset YAML hashes, because their authoritative remote bytes have not been copied into this staging subset. The strict B50 helper separately pins all five checkpoint bytes, the complete training manifest, checkpoint graph structure, run identity, and config paths.

## Frozen execution contract

- Checkpoints: exact pinned B50 `epoch10.pt`, `epoch20.pt`, `epoch30.pt`, `epoch40.pt`, `last.pt` (completed epochs 11/21/31/41/50). This is the trusted frozen-panel alternative allowed by the request; arbitrary e0/e10/e20/e40/last files are not silently accepted.
- Data: first 24 sequential, non-augmented train-split batches; batch 4; image size 640; seed 42.
- Panel: deterministic SHA256 selection across the complete 24 batches; exactly 128 GT on VOC.
- Loader: actual `num_workers == 4`, `pin_memory is True`, sequential sampler, `shuffle=False`, `augment=False`, `rect=False` are asserted.
- Runtime: exactly one visible CUDA device and active CUDA BF16 autocast are asserted. Norm accumulation/statistics use detached FP64.
- State: every BN stays eval; no optimizer exists; every parameter `.grad` remains `None`; all parameters and all persistent buffers hash identically before/after; checkpoint file hashes are re-read after the run.
- Decision validity: all 24 batches, 128 selected GT, at least 96 matched GT, at least 0.75 assignment coverage, at least 12 informative Gini batches, and five finite mass correlations are required before GO is possible.
- Scientific GO requires all: median-batch Gini `>=0.35`; median-checkpoint top-20 share `>=0.45`; median-checkpoint `rho(gradient,mass)>=0.40`; positive rho at three or more checkpoints. Otherwise `NO-GO`.

## CPU verification

Executed with `F:\Anaconda\envs\pytorch\python.exe`:

```text
python -m unittest -v work/remote_research_source/scripts/test_probe_igb_gradient_budget.py
7 tests, 7 passed

python -m py_compile \
  work/remote_research_source/scripts/probe_igb_gradient_budget.py \
  work/remote_research_source/scripts/test_probe_igb_gradient_budget.py
passed
```

The tests cover per-GT positive-loss additivity, exact independent three-feature VJPs, level mass/count and area/quality bookkeeping, boundary-detach equivalence, Gini/top-20/effective-number/Spearman (including ties), conjunctive GO/NO-GO behavior, non-finite correlation rejection, and incomplete-panel rejection.

An independent runtime/semantics review identified three strictness gaps (incomplete evidence could GO, undefined rho values could be dropped, source was only fingerprinted). All three were fixed: protocol sufficiency is now a hard precondition, all five mass correlations must be finite, and current live core/helper hashes are pinned. A subsequent main-task review also caught and fixed the BRC-vs-B50 loss pin described above. The IGB tests remain 7/7 after that correction; the revised ASAC provenance helper separately passes 37/37 tests.

## Exact semantics and explicitly flagged shortcuts

1. **Boundary detach is semantics-preserving for the requested derivative.** Equal-valued Detect inputs are detached and made leaves. This computes exact `dL_g/dP3,P4,P5` and avoids retaining the backbone graph. It cannot estimate upstream parameter gradients; using these numbers for that purpose would be invalid.
2. **The common E2E multiplier is omitted.** The ledger differentiates the raw native O2M component and omits `batch_size * prog_o2m_w`. This changes absolute gradient-norm magnitude by one checkpoint-common scalar, but cannot change within-checkpoint Gini, top-20 share, effective number, or Spearman gates. Absolute cross-checkpoint norm comparisons are not valid unless that scalar is restored.
3. **Only GT-attributable classification is included.** Assigned-class BCE on positive anchors is included. Background and non-assigned-class BCE are excluded because they have no unique GT owner. The per-GT sum therefore equals positive cls + complete positive localization, not the full native classification loss.
4. **No gradient proxy is used.** There is no summed-GT backward, loss-magnitude proxy, per-anchor approximation, random projection, or Hutchinson estimator. Each matched GT incurs its own `autograd.grad` call. Expected formal cost is up to 640 VJPs (128 GT × 5 checkpoints), so runtime may be material despite zero updates.
5. **BF16 is the formal numerical regime.** The VJP follows live CUDA BF16 autocast rather than an FP32 reference. FP64 is used only after gradients are detached for norm/statistic accumulation.

## Unverified risks / smoke blockers

- The real source snapshot, CUDA BF16 path, dataloader pinning, checkpoint deserialization, and 640-VJP memory/time have not been exercised locally. Synthetic CPU success is not a real-model smoke.
- The remote model/data YAML bytes must be logged and inspected on the first run; they are runtime-hashed but not locally content-pinned.
- Exact source pins intentionally fail if TQGB/ASAC work changed any live core file. Do not weaken the check. The formal isolated source must be copied from the exact B50/TQGB reference snapshot, not from the T1-BRC staging tree.
- O2M assignment can leave selected GT unmatched. Rows are not fabricated with zero gradients; coverage is reported and a low-coverage panel cannot GO.
- The formal command creates only diagnostic output files. Trainer initialization may create normal run-directory scaffolding, so it must execute in an isolated source/project context, never alongside an active seed run.

## Minimal safe isolated staging set

Only these two executable files need to enter a fresh source snapshot:

```text
scripts/probe_igb_gradient_budget.py
scripts/probe_asac_assignment_stability.py   # exact pinned helper dependency
```

Keep the specification, test, and audit outside the runtime source as review evidence. Do not copy any `ultralytics/` overlay: the probe is deliberately external and the pinned source must already match.

## Future authorized smoke/formal command

First perform a CPU-only preflight in a **new, separately named source snapshot**:

```bash
cd /home/ubuntu/rz/yolo_study/ultralytics_research_igb_p0
python scripts/probe_igb_gradient_budget.py \
  --checkpoints /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/{epoch10.pt,epoch20.pt,epoch30.pt,epoch40.pt,last.pt} \
  --data /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml \
  --source-root /home/ubuntu/rz/yolo_study/ultralytics_research_igb_p0 \
  --output-dir /tmp/igb_p0_preflight \
  --dry-run
```

After checking all pinned SHA256 values and confirming GPUs 4–7 are idle, the real read-only diagnostic is:

```bash
cd /home/ubuntu/rz/yolo_study/ultralytics_research_igb_p0
python scripts/probe_igb_gradient_budget.py \
  --checkpoints /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/{epoch10.pt,epoch20.pt,epoch30.pt,epoch40.pt,last.pt} \
  --data /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml \
  --source-root /home/ubuntu/rz/yolo_study/ultralytics_research_igb_p0 \
  --physical-device 4 \
  --output-dir /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/diagnostics/igb_p0_b50 \
  --name igb_p0_b50_s42
```

This audit did not run either command and did not create or modify the proposed remote paths.
