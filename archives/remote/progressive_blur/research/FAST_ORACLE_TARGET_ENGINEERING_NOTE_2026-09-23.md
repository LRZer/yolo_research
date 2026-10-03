# Optional bounded-error Oracle target builder — 2026-09-23

The existing exact training-only blur target in `pgb/core.py` evaluates
every GT box against every image pixel. VisDrone val averages about 71
annotated boxes/image, so extending the exact per-box/per-pixel loop to
960 input may be expensive. While new GPU runs are held by hardware
faults, an **unconnected candidate optimization** was added in
`pgb/fast_oracle_target.py`; it changes no completed run or pre-registered
formal configuration.

For a desired absolute strength error `epsilon`, a box cannot change
`1-exp(-D²/2)` by more than `epsilon` where
`D² >= -2 log(epsilon)`. The candidate computes each box's distance only
inside a conservatively bounded rectangular influence region and sets
the rest to strength 1. It preserves exact zero-strength protected-box
pixels and ignore/valid masks. For `epsilon=1e-4`, the analytic maximum
truncation error is 0.0001 in a unit strength map; a 0.5 residual image
mix bounds the corresponding coefficient error to 0.00005 before the
blurred-versus-raw pixel difference is considered. The method still uses
the same images and boxes after augmentation; no labels are read at val.

On the user's local `pytorch` conda environment, direct CPU tests against
the original exact implementation passed for empty images, overlapping
small boxes, clipping at boundaries, a large box, and 32 random crowded
boxes at 96×128. The tests checked identical protected and valid masks
and max strength difference ≤ `epsilon + 1e-6` for epsilon values
0.01, 0.0001 and 0.000001. These are **numerical equivalence checks**,
not a measured 960 GPU throughput gain or validation AP. The candidate
must be benchmarked and audited on actual VisDrone-shaped batches before
it is wired into any experiment; if used, the formal protocol/config
would need an explicit pre-result amendment. Current formal blur config
remains on the exact existing code path.

A reproducible **synthetic CPU-only** microbenchmark
(`tools/benchmark_oracle_target.py`, one 960×960 image, 70 seeded random
boxes, 4 CPU threads, 2 repetitions) measured best-of-two target-build
times of **0.0699 s exact** versus **0.00676 s local** on the user's
computer. Maximum map difference was `7.17e-5` and protected/valid masks
matched exactly. This roughly 10× isolated CPU speed difference is *not*
a server-GPU training-speed prediction: the Gaussian blur bank, dataloading,
augmentation, DDP and validation may dominate actual runtime. The exact
path may already be acceptable, so do not change the formal implementation
without a full-batch smoke/profile.
