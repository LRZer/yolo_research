# VisDrone evaluation protocol audit

Date: 2026-09-23. Scope: evaluation integrity, not a training result.

The [official VisDrone DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)
states that detections inside ignored regions or labeled `others` are not
counted. It uses the original eight-column annotations/submission format and
reports AP over IoU 0.50:0.95 plus recall at specified maximum detections.
Our current `processed_yolo/VisDrone2019-DET` labels contain only the ten
positive classes, so Ultralytics `val` cannot enforce that exact ignore
protocol. Its score is suitable for **matched internal controls only**;
neither its value nor its per-class values should be presented as an
official VisDrone benchmark score.

The server has neither MATLAB nor Octave. A third-party
[Python port](https://github.com/tjiiv-cprg/visdrone-det-toolkit-python)
was inspected in a task-local copy but **not installed or used**. Its current
source has an apparent implementation error requiring repair and reference
testing: in `viseval/calc_accuracy.py`, the IoF intended for ignored GT
boxes is computed against `gt_format`, rather than `ig_format`. The
resulting overlap matrix may have the wrong number of ignored-box columns.
It also uses deprecated `np.int`, which is incompatible with modern NumPy.
These findings are from source inspection, not a demonstrated metric
discrepancy on an official fixture; do not silently substitute this port
for the official MATLAB evaluator.

The port's `eval_class` list repeats a class for every image containing it
before averaging AP, which initially looked like a class-mean error.
Comparison with the **official** `utils/calcAccuracy.m` shows the same
behavior there. It may produce an image-frequency-weighted aggregate rather
than a conventional equal-class mean, but it is not a port deviation and
must not be "fixed" without explicitly changing and renaming the metric.
This is another reason to distinguish official-toolkit AP, a conventional
COCO-style AP, and the internal Ultralytics AP.

Plan before a paper claim: (1) export model detections back to original
image coordinates and official eight-column format with class IDs 1–10;
(2) run the official toolkit in a verified MATLAB/Octave environment, or
cross-check a corrected Python port against the official toolkit on a
documented fixture; preserve its documented/implemented aggregation rather
than assuming equal-class averaging; (3) preserve full original
ignore/others annotations;
(4) state whether scoring is on official val or public test-dev; and
(5) keep the internal YOLO metric clearly separate from official AP.

The new official-provenance split still has scientific value: original
train/val/test-dev membership is correct, and all future model variants can
be compared with identical internal validation and preprocessing. Earlier
work already inspected test-dev; it is not historically unseen, although
this clean protocol keeps it out of new per-epoch checkpoint selection.
