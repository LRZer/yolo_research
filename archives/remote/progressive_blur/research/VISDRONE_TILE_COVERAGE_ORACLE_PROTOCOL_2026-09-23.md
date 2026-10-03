# GT-oracle tile coverage feasibility audit — 2026-09-23

Motivation: the 640→960 conventional control gained 0.06193 internal
mAP50–95 but cost ~1.55× training time. Before proposing a dynamic
high-resolution tile selector, test an optimistic geometry-only upper
bound: do objects occupy just a few image sectors, or almost all of
them? This audit is specified before inspecting its output.

Use the exact 548-image official-source val labels, 38,759 converted
GT boxes. Divide each original image into 2×2 and 3×3 equal fixed
cells in normalized coordinates; count object centers per cell. For
each image choose the top-k cells by GT count (**Oracle**, impossible
at inference without labels). Aggregate micro fraction of GT centers
covered by k cells and mean per-image fraction, plus occupancy
histograms. No GPU, no model inference or modifications to data.

This is a deliberately generous upper bound for fixed-grid selection:
it assumes perfect knowledge of target centers, ignores box extent,
context, tile overlap/border effects, ignored annotations, and how
well a detector performs within tiles. If even the GT Oracle needs
most cells for high coverage, a learned selector cannot plausibly
save much compute on this fixed-grid strategy. If few cells suffice,
it merely warrants a real proposal/latency experiment; it does not
establish AP gains or publication novelty. Existing tiling and active
vision methods are mandatory comparators for any proposed method.
