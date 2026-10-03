# Real-image Oracle target-map engineering audit — 2026-09-23 07:11 CST

Motivated by the ongoing formal run's ~3.05× **observed same-epoch
training time**, this is a CPU-only timing/equivalence study of one
possible component, **not** a measurement of end-to-end GPU training.
The existing formal candidate still uses the exact target map and was
not changed. Three official-val images were chosen deterministically
by box-count percentile as in the visual audit; this is not method
selection on val AP.

Both implementations operate at 960×960 and use identical current
letterboxed YOLO boxes, valid mask, margin, transition and minimum
radius. The bounded-local implementation ignores per-box distances
only where their effect on output strength is proven ≤1e-4; it is a
candidate engineering optimization and not yet connected to any trainer.
Each timing is the minimum of three post-warmup calls, 4 CPU threads,
NUMA0, with no visible GPU.

| Boxes | Exact map min | Bounded-local min | Local speedup | Max strength-map difference | Protection/valid masks |
|---:|---:|---:|---:|---:|---|
| 16 | 0.03797 s | 0.002878 s | 13.19× | 8.91e-5 | Exactly equal |
| 65 | 0.04716 s | 0.005852 s | 8.06× | 7.95e-5 | Exactly equal |
| 122 | 0.08019 s | 0.008455 s | 9.48× | 8.33e-5 | Exactly equal |

There were **zero pixels** with strength difference >1e-4 in these
three images. These observations suggest the dense exact-distance
construction has avoidable work on crowded 960-pixel images. They do
**not** establish an 8–13× end-to-end speedup, a GPU speedup, bitwise
equivalence, or any AP effect. Gaussian-bank convolution, data loading
and detector training are excluded. Also, wall time may fluctuate
because another task-owned GPU train is concurrently active.

Decision: leave the pinned formal experiment untouched. If its final
independent AP justifies further work, first test bounded-local target
on **training-format transformed batches** including Mosaic/empty
boxes, then compare whole-epoch GPU times and AP using a separate
registered run. If AP is non-positive, algorithmic blur changes take
priority over optimizing this implementation.
