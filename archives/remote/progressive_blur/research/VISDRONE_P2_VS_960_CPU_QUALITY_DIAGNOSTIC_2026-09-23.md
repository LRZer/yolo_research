# Why does P2/640 trail standard/960 despite similar low-threshold recall?

Date: 2026-09-23. This is a **CPU-only paired error diagnostic**, not new
training and not official VisDrone AP. It reuses the already verified
seed-2027 best checkpoints: official Ultralytics YOLOv8n-P2 at 640 and
standard YOLOv8n at 960. Both were run through the same task-owned
`tools/analyze_visdrone_prediction_quality.py` on all 548 identical
official-source val images and 38,759 converted-label GT boxes.

## Protocol and integrity

- Each model's own training image size, CPU inference, batch 4,
  `conf=0.001`, NMS IoU .7, `max_det=1000`, `half=False`. Prediction
  matching is greedy by descending confidence within class and image,
  one prediction per GT, at IoU .50 or .75. Scores are then filtered to
  several operating points. Both models use exactly the same script and
  validation image/label IDs.
- Server process used `CUDA_VISIBLE_DEVICES=` (empty), four CPU threads,
  and NUMA0 CPU/memory binding. The read-only 06:23 preflight had
  `mc0=0/mc1=12`, unchanged boot and no post-05:58 kernel fault; **no
  GPU work was launched**. The continuing ECC history still blocks new
  GPU training.
- Both outputs contain 548 per-image records, 38,759 GTs; at every
  operating point `TP+FP=predictions`, and the classwise GT bucket counts
  sum to 38,759. P2 checkpoint SHA256 begins `5646adcf...`, standard960
  `450ee16a...`; full hashes are in the raw JSON files.

## Paired diagnostic at IoU .50

| Confidence ≥ | P2/640 TP | Standard/960 TP | P2/640 FP | Standard/960 FP | P2/640 precision | Standard/960 precision | P2/640 recall | Standard/960 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .001 | 28,399 | 28,349 | 439,265 | 352,433 | 6.07% | 7.44% | 73.27% | 73.14% |
| .01 | 27,711 | 27,739 | 224,956 | 158,031 | 10.97% | 14.93% | 71.50% | 71.57% |
| .05 | 25,362 | 26,153 | 82,874 | 64,838 | 23.43% | 28.74% | 65.44% | 67.48% |
| .1 | 23,239 | 24,634 | 42,040 | 35,935 | 35.60% | 40.67% | 59.96% | 63.56% |
| .25 | 18,309 | 20,342 | 10,138 | 10,321 | 64.36% | 66.34% | 47.24% | 52.48% |
| .5 | 11,629 | 13,591 | 1,555 | 1,752 | 88.21% | 88.58% | 30.00% | 35.07% |

At the very low .001 floor the matched-TP counts differ by only **50
of 38,759 GTs** in P2's favor, reproducing the earlier approximate
recall parity (0.7327 vs 0.7314 here). Yet P2 emits **86,832 more
unmatched low-score predictions** at that floor and **6,105 more** at
confidence .1, where standard960 has 1,395 more TPs. At confidence .5,
standard960 has 1,962 more TPs; P2 no longer has more FPs there.
Among the IoU-.50 TPs found at the .001 floor, 40.95% survive the .5
score threshold for P2, compared with 47.94% for standard960. This is
a score-distribution diagnostic, not a calibrated probability claim.

At IoU .75, the low-floor matched counts are 16,320 P2 versus 16,674
standard960 (recall 42.11% versus 43.02%); at confidence .1 they are
14,397 versus 15,387 (37.14% versus 39.70%). Higher-IoU localization
therefore also differs, though this summary cannot attribute the cause.

The `max_det=1000` cap was reached in **337/548** P2 images versus
**123/548** standard960 images at the .001 floor. This can censor
additional low-confidence predictions and complicates interpreting
floor-level recall/FP differences. Higher-score operating points are
less affected because NMS/capping keep detections in confidence order.
The paired per-image IoU-.50 floor TP count favored P2 on 254 images,
standard960 on 206, and tied on 88; the near-equal aggregate recall is
not that every image behaves identically.

## Inference for the research route — not a proved mechanism

The known P2 head recovers roughly the same *low-confidence* GT set as
960, but standard960 turns more of those into higher-confidence correct
detections and has fewer low-score unmatched predictions. A plausible
gap is **discrimination/confidence ranking or class ambiguity**, plus
some higher-IoU localization difference, rather than a pure absence of
candidate small-object boxes. We have **not** identified whether
unmatched boxes are true background, duplicate detections, class
confusions, objects in converted-label ignored regions, or annotation
omissions. A background-blur augmentation might alter any of these, but
this audit is not evidence that it will help.

The official VisDrone evaluator handles ignored/other regions
differently; these converted-label FPs and precision values are **not**
official challenge scores. The script's CPU `model.predict` pathway is
also not guaranteed pixel-identical to `model.val`; compare only the
paired models within this script. These diagnostic operating points
must not be misnamed AP, and they are one seed on one dataset.

Raw outputs (including per-image records):
`VISDRONE_P2_640_CPU_QUALITY.json` and
`VISDRONE_STANDARD_960_CPU_QUALITY.json`.

## Additional same-script standard-640 anchor

A third **CPU-only** run using the same script and official-source val
view loaded the already verified standard YOLOv8n/640 best checkpoint;
its full raw output is `VISDRONE_STANDARD_640_CPU_QUALITY.json`.
This adds a useful anchor for the P2 trade-off without new training.

| IoU .50 operating point | Standard 640 | P2 640 | Standard 960 |
|---|---:|---:|---:|
| TP at confidence ≥ .001 | 23,604 | 28,399 | 28,349 |
| FP at confidence ≥ .001 | 299,857 | 439,265 | 352,433 |
| Images reaching 1,000-detection cap | 33/548 | 337/548 | 123/548 |
| Precision at confidence ≥ .1 | 38.30% | 35.60% | 40.67% |
| Recall at confidence ≥ .1 | 52.56% | 59.96% | 63.56% |
| TP at confidence ≥ .5 | 10,523 | 11,629 | 13,591 |

P2 does recover many more low-confidence candidates than standard640,
consistent with its higher AP, but it also generates many more
unmatched/capped proposals. Standard960 achieves similar floor-level
recall to P2 with a better precision/score distribution. This narrows
the current *diagnostic* from “small targets are simply unseen” to a
multi-part problem of candidate recovery **and** ranking/quality.
It does not establish the mechanism causing either effect or imply that
background blur is the correct remedy.
