# Pre-registered VisDrone size-stratified diagnostic — 2026-09-23

Purpose: distinguish a small-object sampling explanation from a generic
gain in the 640-vs-960 experiment. This protocol was written while the
960 run was in progress, before observing its final checkpoint result.

The task-owned `tools/evaluate_visdrone_size_recall.py` computes matched
GT-box recall, **not AP**. It runs raw-image inference with the selected
best checkpoint on the same 548 official-source validation images for
each model. Prediction settings: confidence 0.001, NMS IoU 0.7,
`max_det=1000`, no test-time augmentation. Match predictions to GTs
within class, greedy by confidence, one-to-one, at IoU 0.50 and 0.75.
Converted YOLO labels omit official ignored/other regions, so this is
an internal diagnostic, not the official VisDrone evaluator.

To make size groups independent of inference resolution, assign every
GT box according to `sqrt(width*height)` after a **hypothetical 640
letterbox resize of its original image**. Fixed groups are: `<16 px`,
`16–<32 px`, `32–<96 px`, `>=96 px`. Preserve the same image IDs and
labels for both 640 and 960 results. Report all-target, group-level,
and group-by-class target counts and recalls. Never choose thresholds or
buckets after inspecting final result to exaggerate a gain.
The earlier independent label audit counted 38,759 val GT boxes; the
new evaluator must reproduce that denominator exactly or fail.

Interpretation: if 960 improves disproportionately for `<16` and
`16–<32`, that supports a scale/sampling bottleneck hypothesis, not
causal proof. P2 at 640 provides a follow-up architectural control.
If gains occur uniformly across all sizes, investigate broader
optimization/augmentation effects. Compare exact AP separately via
the existing independent Ultralytics evaluator. Do not call a
single-seed result publication-strength; repeat seeds for any candidate.
