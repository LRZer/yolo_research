# Pre-registered standard control: VisDrone YOLOv8n at 960 versus 640

Recorded 2026-09-23 while the official-split 640 baseline is running. This
is a conventional resolution control, **not a proposed innovation**. Do not
launch before the 640 baseline has fully finished and the same four physical
GPUs are free.

Hypothesis: VisDrone objects are usually extremely small after 640 resize,
so increasing input to 960 will restore useful object pixels and improve
official validation AP at a measurable cost in runtime and memory. The
earlier test-dev audit found 93.2% of boxes below 32² pixels at 640 and
median 9×10.5 pixels. The observation motivates the experiment but does
not guarantee a gain.

One-variable contrast: official train/val membership, scratch YOLOv8n YAML,
seed 2027, 125 epochs, batch 64, SGD, augmentation schedule, workers 4,
`max_det=1000`, and official-val checkpoint selection stay fixed. Only
`imgsz` 640→960 changes. Report mAP50–95, mAP50, precision, recall,
per-class AP, training time, peak memory and inference timing. 960 has 2.25×
as many input pixels, so an AP gain alone is not a free improvement. The
test-dev split remains untouched during this control selection.

First run a one-epoch 1%-fraction four-GPU smoke to verify memory, distributed
execution and the task-owned data view. The formal 960 run can start only if
the smoke succeeds without affecting other GPU users or hardware safety.
If 960 is substantially better, compare any new method against that stronger
baseline (or justify a compute-matched tradeoff). If not, inspect whether
feature stride/assignment rather than input resizing is the bottleneck.

Configs: `configs/visdrone_official_yolov8n_scratch_960_smoke_4gpu.json`
and `configs/visdrone_official_yolov8n_scratch_960_s2027_e125.json`.
# Config integrity check (2026-09-23)

The local 640 and 960 formal JSON configs were compared key by key.
The only differences are `imgsz` (`640` vs `960`) and the run `name`.
Remote SHA256 files match the local copies: 640
`f4aecbf0307e590acff370fb3d9b328ed0a84c5e6a3c068f9975142dc27c849e`,
960 `cc4358ad595d361d6cbd6110fc787a804c78b70b951e1cdf282446c8068bbb4a`.
This confirms the configured comparison, but does not by itself prove
identical effective stochastic augmentation trajectories at different
resolutions; report one-seed results as diagnostic.
