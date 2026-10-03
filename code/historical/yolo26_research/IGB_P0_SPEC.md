# IGB-P0: native O2M per-instance gradient-budget probe

Status: implementation candidate, 2026-09-19. This document freezes a read-only diagnostic. It does **not** authorize an IGB training loss, a checkpoint update, a remote write, or a validation/test-set read.

## 1. Question and decision

IGB-P0 asks whether the current YOLO26 O2M/TAL positive supervision lets a small set of ground-truth instances dominate the gradient arriving at the shared P3/P4/P5 tensors. It measures the native assignment; it does not change TAL, O2O, TQGB, SARC, ASAC, KD, or any core source file.

The following gates are copied from `BOLD_IDEA_SCREEN_2026-09-19.md` and are conjunctive:

1. median batch Gini of per-GT joint gradient norm is at least `0.35`;
2. the median, over checkpoints, of the pooled top-20% GT gradient share is at least `0.45`;
3. the median finite checkpoint Spearman correlation `rho(gradient norm, target-score mass)` is at least `0.40`;
4. that correlation is positive at at least three checkpoints.

Any failed or non-finite gate is `NO-GO`. `GO` permits only an isolated IGB implementation followed by the already preregistered 10-epoch I0/I1/I2 screen; it is not evidence of an AP improvement.

Before those scientific gates can pass, every checkpoint must satisfy protocol validity: all 24 batches processed, exactly 128 GT selected on VOC, at least 96 matched selected GT, assignment coverage at least 0.75, at least 12 batches with two or more matched selected GT, and finite mass correlation at all five checkpoints. These checks cannot turn a failed scientific gate into GO; they only prevent incomplete evidence from doing so.

The originally proposed checkpoint schedule cannot be reconstructed honestly from filenames alone. The executable therefore uses the already frozen and content-pinned B50 panel `epoch10.pt`, `epoch20.pt`, `epoch30.pt`, `epoch40.pt`, and `last.pt`. Their completed epochs are 11, 21, 31, 41, and 50 under the repository's zero-based checkpoint metadata. The existing ASAC provenance verifier checks exact checkpoint SHA256 values, the complete 124-key training manifest, graph identity, run `args.yaml`, `results.csv`, config hashes, and common trajectory. This is the “frozen panel appropriate” alternative to an unavailable e0/e10/e20/e40/last panel. `best.pt` and arbitrary substitute checkpoints are rejected.

## 2. Frozen data and execution protocol

- Train split only, deterministic non-augmented loader (`mode="val"`, `rect=False`, `shuffle=False`). No VOC07 test or validation AP is read.
- Exactly 24 sequential batches, batch size 4, image size 640, seed 42.
- Exactly four loader workers and pinned host memory are runtime assertions, not advisory settings.
- The complete raw 24-batch panel is fingerprinted. Up to 128 raw GT identities are selected once by the smallest seeded SHA256 keys over `(image path, local GT index, class, box)`, so the cap is spread over the complete panel rather than truncating early batches.
- The raw panel fingerprint, selected-GT fingerprint, batch membership, class, and box must be identical for every checkpoint.
- One logical CUDA device is exposed from physical device 4/5/6/7. CUDA BF16 autocast is mandatory and checked at runtime. Scalar norms and statistics are accumulated in FP64 after detaching.
- The model is placed in training-output mode only so Detect returns raw O2M/O2O dictionaries. Every BatchNorm module remains in eval mode. The native training behavior that detaches O2O from shared features is consequently retained.
- No optimizer is constructed, no `.backward()` is called, no `.step()` is possible, and checkpoint files are never opened for writing. Parameter and BN-buffer hashes, BN modes, optimizer absence, and the fact that every parameter `.grad` remains `None` are asserted before and after each checkpoint.

## 3. Exact per-GT quantity

For an image-local GT `g`, native O2M assignment supplies foreground anchors `A_g`, assigned class `c_g`, and soft TAL target `q_a`. Define:

```text
L_g = hyp.cls * sum[a in A_g] BCEWithLogits(s[a,c_g], q_a) / M
    + hyp.box * native_CIoU_loss(A_g) / M
    + hyp.dfl * native_DFL_or_L1_loss(A_g) / M
M   = max(sum over the entire batch of native target_scores, 1)
```

The box terms are evaluated by the live `one2many.bbox_loss` with a mask containing only `A_g`; anchors, strides, decoded boxes, image normalization, class weights, hyperparameter gains, and the global native denominator are unchanged. For YOLO26 `reg_max=1`, the third term is the repository's normalized L1 term; the code also remains valid if the live model uses DFL.

“Positive cls” deliberately means only assigned-class BCE on O2M foreground anchors. Native BCE for background anchors and non-assigned classes has no unique GT owner and is excluded rather than arbitrarily attributed. Thus `sum_g L_g` equals the selected GTs' positive cls plus full positive localization component, not the complete native O2M loss. The script checks this additive decomposition for every probed batch.

For captured shared feature tensors `F_l`, `l in {P3,P4,P5}`:

```text
n_{g,l} = || dL_g / dF_l ||_2
n_g     = sqrt(sum_l n_{g,l}^2)
```

`torch.autograd.grad` is invoked once per matched selected GT. This is an exact per-GT vector-Jacobian product; losses are never summed before differentiation. The raw O2M branch loss is used without E2E's common `batch_size * prog_o2m_w` multiplier. That omitted scalar changes every norm at one checkpoint by the same factor and therefore cannot change Gini, top-20 share, effective number, or any within-checkpoint Spearman gate. Results must not be described as absolute full-training gradient magnitudes.

### Boundary detach optimization

A Detect forward-pre-hook replaces each incoming P3/P4/P5 tensor by an equal-valued detached leaf with `requires_grad=True`. This avoids retaining the backbone graph. It preserves the requested local derivative `dL_g/dF_l` exactly for the fixed boundary values; it intentionally makes upstream-parameter gradients unavailable. A synthetic test compares the boundary VJP against an uncut graph. No proxy such as loss magnitude, feature activation, summed-GT backward, Hutchinson estimator, or per-anchor norm is permitted.

## 4. Per-GT ledger and statistics

Each matched selected GT row records:

- checkpoint and completed epoch; fixed batch/image/local-GT identity;
- positive count and assigned target-score mass `m_g = sum q_a` overall and by P3/P4/P5;
- normalized box area;
- mean assigned-target quality `m_g/count`;
- mean matched prediction quality `mean sqrt(sigmoid(s[a,c_g]) * IoU(pred_box_a, GT))`;
- positive cls, CIoU, and DFL/L1 loss components;
- P3/P4/P5 and concatenated joint gradient norms.

Unmatched selected GTs are counted as assignment-coverage evidence but receive no artificial zero-gradient row. Statistics use matched rows:

- batch Gini over joint norms, followed by a median across all checkpoint-batches containing at least two matched selected GTs;
- per-checkpoint pooled top-20 share (`ceil(0.2*N)` largest norms), effective number `(sum n)^2/sum(n^2)`, and normalized effective number;
- per-checkpoint Spearman correlations of norm with mass, positive count, area, matched prediction quality, and mean target quality; ties receive average ranks;
- decision top-20 and mass-correlation values are medians over checkpoints, avoiding arbitrary cross-epoch gradient-scale pooling.

## 5. Files, outputs, and future run

- Executable: `work/remote_research_source/scripts/probe_igb_gradient_budget.py`
- CPU tests: `work/remote_research_source/scripts/test_probe_igb_gradient_budget.py`
- Local implementation audit: `outputs/yolo26_research/IGB_P0_IMPLEMENTATION_AUDIT.md`

The runtime writes `<name>.json`, `<name>_per_gt.csv`, and `<name>.md` under the requested output directory. It also records checkpoint hashes, trajectory signature, panel hashes, live source paths and SHA256 hashes (probe, ASAC provenance helper, loss, TAL, Detect head, model task graph, dataset builder, detection trainer, model YAML, and dataset YAML), Python/PyTorch/CUDA/cuDNN/GPU details, and the complete frozen protocol. The helper and seven live core files are content-pinned; a compatible but modified source fails before a verdict until the pins are explicitly re-audited.

Example future smoke command is intentionally documented only in the audit report. It must run in a fresh isolated source snapshot, never in the source tree used by active TQGB/ASAC training.
