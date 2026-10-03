# Selected-only stochastic Oracle rendering candidate — 2026-09-23 07:31 CST

Static review of the current `apply_stochastic_oracle_blur` found a
direct cost inefficiency: when `oracle_train_probability=0.5`, it first
constructs strength maps and renders Gaussian blur for **every** image
in the batch, then selects half the images with a Bernoulli mask and
restores the others from the original batch. Thus approximately half
of the expensive map/render work is discarded; this is a code-path
fact, **not** proof that it accounts for the observed ~3.05× whole-train
slowdown.

Implemented a separate, **unconnected** candidate module
`pgb/selected_oracle_blur.py`: draw the same per-image Bernoulli mask,
build maps/render only for selected images, scatter results back, and
leave all unselected images exactly unchanged. The formal training
entrypoint and registered `oracle_integration.py` remain untouched.

Local CPU sanity tests (`tests/test_selected_oracle_blur.py`, 3/3 pass)
check fixed-seed mask identity, rendered image agreement to absolute
1e-6, exact cleanliness of unselected images, zero-treatment identity,
and full-probability equivalence. These tests do not establish GPU/DDP
performance, handling of all augmented metadata, whole-epoch
determinism, checkpoint compatibility or AP. In particular, the
unselected images intentionally have no diagnostic strength map in the
candidate, whereas the old implementation computes one. The detector
loss does not consume that diagnostic field.

The same three tests also passed on the remote server's pinned conda
environment with `CUDA_VISIBLE_DEVICES` empty and CPU/memory bound to
NUMA0. They remain CPU sanity checks, not 4-GPU trainer verification.

An additional **single real-image CPU batch** used four letterboxed
official-val VisDrone frames at 960 pixels with 127/16/65/122 boxes.
Under fixed seed 2027, both paths selected the same two images and
produced **bitwise equal** output tensors (`max_abs_difference=0.0`).
The registered full-batch path took 1.883 s, while selected-only took
0.854 s in this one call. This is neither a repeated benchmark nor a
whole-epoch GPU speed measurement; the batch did not include Mosaic
or actual Ultralytics dataloader metadata. The val frames were used
only for this read-only processing audit, not for training or model
selection. No current run was modified.

Decision: no hot patch of the running 125-epoch job. If the final
independent AP warrants continuation, separately verify the new path
on real transformed VisDrone batches (including Mosaic), 4-GPU smoke,
epoch time and matched AP. If AP is non-positive, first revisit the
scientific treatment rather than merely making an ineffective method
faster.
