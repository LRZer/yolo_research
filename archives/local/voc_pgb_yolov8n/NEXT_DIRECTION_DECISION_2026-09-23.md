# Next research direction — proposal for discussion

Date: 2026-09-23  
Status: provisional; no new training launched

## Recommendation

Deprioritize image-level progressive background blur on VOC as the main paper
route. Pivot to diagnosing and mitigating **tiny-object information loss under
resolution and feature downsampling**, with VisDrone2019-DET as the initial
experimental domain. Keep YOLOv8n as the controlled detector, not as the
scientific novelty itself.

## Evidence already in hand

- VOC learned PGB versus matched 20-epoch scratch baseline: mAP50–95 0.15921
  versus 0.16173 (−0.252 percentage points).
- Perfect-box progressive blur, one-seed 20-epoch diagnostic: 0.17042 versus
  0.16173 (+0.869 points). It is privileged and not an upper bound for every
  possible treatment; it only measures this exact renderer with ideal boxes.
- Extreme box-interior-only blackout: 0.44016 with GT-masked validation, but
  0.01418 on raw images versus 0.16169 for the same raw-input baseline. The
  apparent masked gain relies on GT-derived box boundaries.
- The existing VisDrone audit found 93.2% of evaluated boxes below 32² pixels
  at 640 input and fixed-epoch two-seed mAP50–95 0.1361±0.0015. This is a
  much clearer measured bottleneck than "background might be distracting."

## Decision gates, not yet a new method

1. Re-establish the standard VisDrone split: train on official train, choose
   settings using official validation, reserve test-dev for final comparison.
   Existing exploratory runs used test each epoch and cannot alone support a
   clean publication claim.
2. Audit recall/AP by target size, density, class and image; inspect false
   negatives and downsampled images. Establish whether target pixels disappear
   in the input or survive input resize but vanish within network features.
3. Run matched conventional controls: 640 versus 960/1280 inputs; P2/high-
   resolution head; tiled training/inference. Report AP by size, latency,
   FLOPs/VRAM and training budget, not just one overall mAP number.
4. Only after identifying a remaining repeatable failure, formulate a method
   that addresses that failure without GT information at inference. A possible
   hypothesis is **budget-limited retention of fine features where tiny-object
   evidence is at risk**, but this is not yet an innovation claim. Compare it
   directly with the controls above and with close literature.
5. Confirm any signal under at least two seeds, a second suitable dataset,
   and a held-out test protocol before discussing paper readiness.

## Novelty warning and alternative

Simply adding background smoothing is too close to [CVPR 2025 SET](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html),
which explicitly includes hierarchical background smoothing for tiny-object
detection. Simply adding shallow high-resolution features overlaps with
[ICCV 2025 DM-EFS](https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html).
These papers do not make small-object work impossible; they require us to
identify a more specific failure and show why standard fixes do not solve it.

An alternate, more domain-specific route is LLVIP's paired visible/infrared
pedestrian data: infrared as privileged *training-only* information with
visible-only inference. The existing baseline gap is informative, but the
single-class problem and paired-scene leakage risks make this a separate
project with careful evaluation requirements, not an immediate replacement.

Decision requested from the user: whether to adopt the VisDrone diagnostic
track, keep exploring mild blur with a bounded gate, or pursue paired-modality
LLVIP. No implementation or training has begun for any of these options.
