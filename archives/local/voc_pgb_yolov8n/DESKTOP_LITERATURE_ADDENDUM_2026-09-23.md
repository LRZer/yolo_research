# Desktop paper reading addendum — 2026-09-23

Scope: read the methods, main numeric tables, and conclusions of the
following PDFs supplied on the user's Desktop. The files were not modified.
Numbers below are transcribed from authors' tables, not independently
reproduced. Do not compare them directly to this project's VOC/VisDrone AP:
datasets, detector, training budgets, and evaluators differ.

| Paper and source | Evidence read | Implication for this project |
|---|---|---|
| *FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision* (2023 arXiv PDF, `图像增强`, pp.1–8, Tables 2 and 6) | Three image/feature scales, task-loss-driven feature enhancement and attention/skip fusion. ExDark RetinaNet AP 46.3→46.4, but FQ R-CNN AP 47.0→56.5 with COCO-pretrained fine-tuning. | Task-aligned enhancement is established; benefit can vary dramatically by detector. A local module's visual appeal or gain on one backbone is insufficient. Table 6 labels some 72.6 figures as mAP even though Table 2 gives 72.6 as mAP50, so cite metric definitions carefully. |
| *FE-YOLO: Fourier enhancement YOLO for end-to-end object detection in low-light conditions* (Digital Signal Processing 166, 2025, 105355, DOI 10.1016/j.dsp.2025.105355; `图像增强`, pp.1–12, Tables 5,6,9) | Trainable Fourier front end plus detection loss and amplitude/phase constraints. ExDark Table 5 YOLOv3 mAP@0.5 76.4 vs joint FE-YOLO 78.5; separately trained FENet+YOLOv3 77.6. | Another close joint-processing precedent, albeit low-light-specific and distinct from background blur. The text says ExDark has 7,363 images but also claims its 80/20 split has 9,600/2,400; that count is internally inconsistent. Verify dataset/protocol before using reported AP as a comparator. |
| *YOLO-TaMoE: Token-Aware Sparse Mixture-of-Experts for Region-Adaptive Object Detection* (2026 PDF in `YOLO增强`, pp.1–25, Tables 1–5; publication venue not verified here) | Token-level routing, shared local/frequency expert, shallow-detail stem. Reports VisDrone-2019 mAP50:95 18.4 for YOLO-Master-n and 19.0 for TaMoE at 25% sparsity (Table 1); 100 epochs, batch 16, 640, scratch (pp.19–20). Reports only Params/GFLOPs, not an end-to-end latency gain in Table 1. | Region-adaptive computation is already an active line of work, so novelty cannot rest on "different regions get different processing." Its published protocol differs from ours (125 epochs, batch 64, task-owned official provenance split, internal converted-label AP); no direct numeric leaderboard comparison. |
| *Human Scanpath Prediction in Target-Present Visual Search with Semantic-Foveal Bayesian Attention* (2025 arXiv PDF in `目标增强`, pp.1–8, Fig.3, Table I) | An explicit foveated sensor plus YOLOv5 guides sequential visual search, not exhaustive one-pass detection. Fig.3 shows narrower foveae worsen peripheral localization/classification. Search evaluation uses scanpath metrics and an oracle stopping criterion when the gaze reaches the target. | Important caution: suppressing peripheral detail can itself damage detector evidence. Its scanpath scores, repeated observations, and oracle stop cannot be converted into our mAP or deployment throughput. It does not validate the original full-image background blur idea. |
| *Self-Supervised Pretraining Improves Self-Supervised Pretraining* (2022 PDF in `自监督预训练`, pp.1–4) | Hierarchical pretraining adapts an existing representation to target-domain unlabeled images; authors report up to 80× faster convergence in their broader study. | Relevant if the project explicitly changes to pretraining/transfer or scarce labels. It does not address the current agreed random-initialization baseline and cannot be introduced into just one arm of a fair comparison. |
| *Self-Supervised YOLO: Leveraging Contrastive Learning for Label-Efficient Object Detection* (2025 arXiv v1, explicitly "Preprint. Under review," `自监督预训练`, pp.1–9, Table 1) | SimCLR backbone pretraining followed by one-class cyclist detection. Table 1 at epoch 30 gives YOLOv8s scratch/SSL mAP50:95 0.7652/0.7663, a +0.0011 difference; YOLOv5s scratch/SSL 0.7486/0.7467, a −0.0019 difference. | The prose's "consistent gains" is contradicted by its own YOLOv5 table; no multiple seeds/uncertainty shown there. Also uses YOLOv8s and a different one-class dataset, not YOLOv8n/VisDrone. Low-priority evidence for current route. |

The earlier reading memo `DESKTOP_LITERATURE_REVIEW_2026-09-23.md`
covers foveation, IA-YOLO, DENet, and a negative low-light enhancement
study. Together with the official CVPR 2025 SET paper's train-only
hierarchical background smoothing, the literature makes a broad
background-suppression or learned-front-end novelty claim untenable
without a narrower measured failure mechanism and independent gains.

Current evidence-driven decision remains: finish the standard 640-vs-960
VisDrone control first, diagnose class/size-specific errors, then compare
any method against the appropriate strong conventional baseline with the
same initialization, data split, seed/budget and evaluator. Increased
resolution alone is a control, not the proposed contribution.

## 05:18 CST — further Desktop reading prompted by the user

Three previously unsummarized PDFs in `C:\Users\Lenovo\Desktop\图像增强`
were read through their methods, experiments, tables and conclusions. These
are author-reported results under different datasets/protocols, not direct
comparators to our YOLOv8n/VOC or VisDrone runs.

| Paper | Verified detail | Research implication |
|---|---|---|
| Guo, Lu and Wu, *Dynamic Low-Light Image Enhancement for Object Detection via End-to-End Training*, ICPR 2021, 8-page local PDF | Uses multiple image-specific dynamic-filter/exposure branches with enhancement targets and a Faster R-CNN/FPN detector. The detector sees the original image **as well as** enhanced views during training; its RPN classification losses are weighted, while RoIs come from the original-image feature maps. On ExDark, Table V reports Faster R-CNN AP 30.4, fixed bilateral filtering 27.8, fixed guided filtering 25.8, and the HE+sharpening based proposed method 32.1. | Further precedent for detection-supervised processing and train-time multi-view supervision. More importantly, simply suppressing image detail can damage detection. Its Table I/III small-object AP for the same headline method is 3.6, whereas Table V reports 5.4; do not quote a unique small-object gain without resolving that inconsistency. |
| Yoshimura et al., *DynamicISP: Dynamically Controlled Image Signal Processor for Image Recognition*, ICCV 2023 official-paper PDF, 11 pages | Controls differentiable RAW ISP functions using prior-frame detector features. For static images, it feeds the same image twice, not a free single pass. The human detection evaluation averages four seeds. Table 3 reports AP 48.3 for differentiable static gamma tuning versus 48.9 for dynamic gamma; Table 6 sequential setting reports 49.1 at 8.3 ms versus static gamma 48.3 at 8.1 ms on its hardware. Its multi-function variant reaches 49.6 AP but 27.7 ms because denoising/sharpening implementation is unoptimized. | Strong prior for task-driven adaptive front ends, plus a model of honest speed reporting. The RAW/video/feedback setup differs from our RGB still-image blur. Neither its AP nor latency can be transplanted to our setting; an added front end requires end-to-end timing, and extra functions can erase purported efficiency. |
| Cui et al., *You Only Need 90K Parameters to Adapt Light: a Light Weight Transformer for Image Enhancement and Exposure Correction*, BMVC 2022/arXiv v4 local PDF, main text pp.1–10 plus supplement pp.18–20 | IAT uses a pixelwise local multiplicative/additive branch and a global color-matrix/gamma branch. It has ~91K parameters. On its ExDark/YOLOv3 protocol, main Table 4 reports original mAP 76.4 vs separately pre-enhanced IAT 77.2, with per-image latency 0.033 vs 0.040 s. Supplement Table 6 reports jointly optimized IAT initialized from LOL at 77.8, versus scratch-initialized IAT at 77.1; this is not the same as the pre-enhancement result. On ACDC, main Table 4 pre-enhancement mIoU 62.1 is below the original 63.3; joint IAT initialized from LOL reaches 63.8 in the supplement. The paper labels detection metric simply `mAP`, so do not assume an IoU definition. | A close precedent for a lightweight RGB-image front end jointly optimized with detection; a 38.5K predictor plus YOLO cannot claim novelty from small parameter count or joint training alone. It also highlights initialization and domain dependence. Even a ~7 ms per-image overhead matters and must be compared with our full pipeline rather than front-end parameter count. |

In light of these papers and earlier IA-YOLO/DENet/SET reading, a publishable
claim cannot be only “learn a small front end” or “blur background to improve
YOLO.” The next decision still depends on our clean VisDrone P2 control,
error decomposition and an appropriately matched baseline. The new papers
provide mechanisms and negative controls, **not** evidence of a gain in our
experiments.

## Primary-source cross-check for the VisDrone/P2 pivot

Read the author [QueryDet CVPR 2022 paper](https://arxiv.org/pdf/2103.09136)
through its method, VisDrone experiment and failure discussion. QueryDet
predicts coarse small-object locations on lower-resolution features, then
computes the high-resolution detection head sparsely there. It already covers
the broad “adaptive high-resolution computation near likely objects” idea.
Its VisDrone study trains on four image quarters independently, uses 50k
iterations and RetinaNet/ResNet-50, and Table 2 reports 26.21→28.32 AP and
2.63→2.75 FPS for raw RetinaNet versus the full sparse method; hence its
numbers are not directly comparable to our whole-image YOLOv8n internal AP.
The paper also reports that simply adding P2 to RetinaNet initially reduced
COCO AP until feature-level loss balancing was added. It is a precedent for
testing P2 as a **control**, not assuming that extra resolution automatically
helps. QueryDet's failure analysis notes coarse-query localization misses and
false activation of large-object regions. Our 2×2/3×3 GT sector-coverage
audit rejects only a *naive fixed-grid tile-skipping* proposal; it does not
refute finer-grained feature-sparse schemes such as QueryDet.

Read the author [UGS paper (ICCV 2025)](https://arxiv.org/pdf/2303.01803)
through its VisDrone protocol/results as another small-object prior.
It frames tiny-object localization as gradient-instability/uncertainty,
not merely insufficient resolution; its Table 2 reports TPH-YOLOv5-x
AP 26.5→29.0 at 640 and 39.2→41.7 at 1536 after UGS. These are
author-reported numbers for a different YOLO variant, 80-epoch setting,
different batch sizes, and max 1500 predictions/image; they are **not**
comparable to our YOLOv8n/125-epoch, max1000 converted-label metric.
They establish that generic scale-aware localization stabilization is
also prior art. Our P2 diagnostic alone cannot distinguish weak object
features, false positives/confidence ranking, and localization curvature.

## 06:02 CST — additional Desktop PDFs screened for the next research pivot

Read the methods, experimental protocols, principal result tables, and
conclusions of three additional PDFs in `C:\Users\Lenovo\Desktop\图像增强`.
The author-reported results below are **not** measured on our data or
protocol; no venue status beyond what the PDF itself states is inferred.

| Paper | What the original PDF actually supports | Implication for this project |
|---|---|---|
| Guo et al., *Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection* (2025 arXiv-formatted PDF, 11 pages; pp.3–8, Tables 1–5) | A task-trained, content-adaptive linear RAW-to-RGB mapping plus nonlinear tone mapping and self-boost regularization precede a **ImageNet-pretrained RetinaNet**, trained 15 epochs at 400×600. Table 2: NOD Sony default-ISP 28.3 mAP versus Dark-ISP 31.5; Nikon 27.4 versus 29.9. Table 3: synthetic dark-RAW COCO 21.5 default ISP versus 23.1 Dark-ISP. These comparisons are within the authors' RAW/low-light protocol; LOD uses VOC-style mAP, NOD/SynCOCO COCO-style metrics. | Strong evidence that image formation and input format can be a real bottleneck in a *specific* domain; another close prior for task-driven front ends. It neither validates Gaussian background blur on normal RGB images nor transfers numerically to scratch YOLOv8n/VisDrone. Their pretrained detector also differs from our random-initialization rule. |
| Sasagawa and Nagahara, *YOLO in the Dark — Domain Adaptation Method for Merging Multiple Models* (2020 local PDF, 15 pages; pp.4–14, Figs.7–11, Tables 1–3) | Connects an existing See-in-the-Dark RAW encoder to YOLO through a trained glue layer, using feature alignment and COCO. The paper reports SID-set mAP@IoU.50 0.26 (original enhanced RGB) versus 0.55 (merged method) and 0.56 for separately chained SID+YOLO; Table 3 reports 113.90 versus 270.79 GMAC at 832×832. Crucially, its SID evaluation “ground truth” boxes are **predictions of the original YOLO on long-exposure reference images**, not independently hand-annotated objects. | The large AP change is a different RAW domain-transfer task and weaker pseudo-label evaluation, not proof of a general blur gain. If we explore low-light data, use authentic held-out annotations and disclose pretraining, input modality and total cost. |
| Xu et al., *SNR-Aware Low-light Image Enhancement* (CVPR 2022 local PDF, 11 pages; pp.3–8, Eqs.2–6, Tables 1–5) | Estimates an approximate SNR map from a denoised grayscale image, then mixes short-range convolution and long-range transformer features spatially while excluding low-SNR tokens from attention. Main evaluations are **image restoration PSNR/SSIM** on LOL, SID, SMID and SDSD, plus a human preference study; the paper does not supply detection AP. Table 5 has a small text/table inconsistency for SDSD-outdoor (main Table 4 says 28.66/0.866, ablation full row 28.37/0.862), so avoid collapsing these to one quoted figure. | Spatially varying treatment according to image quality is established, but restoration quality is not a surrogate for object detection. A future noise-aware detection method needs a measured noise/localization failure and detection-specific controls, not merely an SNR map. |

Combined judgment: these sources strengthen the case for choosing a *measured
domain-specific failure* (e.g., tiny-target localization or true low-light
sensor noise) before adding an adaptive front end. They do not overturn the
current P2/960 result or justify claiming that background blur is novel.

## 06:57 CST — remaining Desktop folder triage and REGDet deep read

The four user-indicated folders (`目标增强`, `图像增强`, `YOLO增强`,
`自监督预训练`) contain 20 PDF files. Earlier sections summarize the
papers most directly related to foveation, task-driven enhancement,
small-object detection, and pretraining. This pass read the three
previously unsummarized `图像增强` PDFs to distinguish restoration
evidence from detection evidence. Sources remain unchanged.

| Local PDF | What its original experiments establish | Relevance and boundary |
|---|---|---|
| Chen et al., *Learning to See in the Dark* (2018, 10 pp.; methods pp.4–5, Tables 2–3 pp.6–7) | End-to-end **RAW** short-exposure to long-exposure sRGB restoration, using camera-specific U-Nets. Table 3 reports Sony/Fuji PSNR-SSIM 28.88/0.787 and 26.61/0.680 for the default pipeline; applying their model after a traditional RAW→sRGB pipeline reduces these to 17.40/0.554 and 25.11/0.648. | A genuine example of input formation and SNR as bottlenecks, but no detection AP; it cannot support a claim that blurring clean RGB backgrounds improves detection. The table measures reconstruction, not localization. |
| Liang et al., *Recurrent Exposure Generation for Low-Light Face Detection* (2020 arXiv PDF, 12 pp.; method pp.4–6, protocol p.7, Fig.4 p.7, Tables I–III p.10) | End-to-end pseudo-exposure generation and early fusion into pretrained face detectors on DARK FACE. Authors randomly set aside **1,000 of 6,000** images because the original test split is withheld, initialize DSFD/PyramidBox/S3FD from WIDER FACE, and evaluate their own held-out split. Table I reports finetuned baseline→REG mAP 71.42→76.94 (DSFD), 72.48→77.69 (PyramidBox), 54.99→57.95 (S3FD); other exposure generators are weaker and CEG/SEG even decrease S3FD AP. Table III: REG without/with pseudo-exposure pretraining 76.36/77.69. | Strong precedent for detection-guided multi-view preprocessing and **non-monotonicity**: extra processing can hurt a smaller detector. The authors explicitly attribute some low-light methods' failure to over-smoothing face regions. Their face-only, low-light, pretrained, test-split-specific mAP cannot be compared numerically to our scratch YOLOv8n/VOC or VisDrone AP. For our blur question, examine whether small/low-confidence targets are suppressed instead of assuming background suppression helps. |
| Feijoo et al., *DarkIR: Robust Low-Light Image Restoration* (local filename 2024; embedded arXiv v3 date 2025-10-14; 17 pp.; Tables 1–8 pp.5–8) | Joint low-light deblurring/denoising/restoration. LOLBlur Table 1 gives DarkIR-m 27.00 PSNR, 0.883 SSIM, 0.162 LPIPS versus LEDNet 25.74/0.850/0.224; model cost 3.31M parameters and 7.25 GMAC at 256 px. The paper explicitly notes that depthwise operators' lower MACs do **not** reduce actual GPU inference time proportionally. | Restoration/perceptual measures do not establish detector AP. Its speed caveat applies to our proposed front-end too: report actual full-pipeline latency, not just parameter or operation counts. The PDF has no controlled object-detection benchmark for our hypothesis. |

These papers sharpen, but do not change, the locked VisDrone candidate
comparison. REGDet supports a targeted *failure analysis* of tiny
objects and confidence ranking, not a license to claim the current
Oracle blur is novel or beneficial. No new method is selected from
these PDFs before the ongoing formal 960-pixel comparison returns.

## 07:22 CST — primary-source context check beyond the Desktop collection

The [CVPR 2021 Scale-Aware AutoAug paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf)
reports a learned box-level augmentation policy whose selected area ratio
often exceeds 1 for small and medium objects; the authors interpret this
as useful surrounding visual context. The [ECCV 2018 context-aware
augmentation paper](https://arxiv.org/abs/1807.07428) likewise reports
that copying objects without suitable context can hurt detector training.
Neither directly tests our Gaussian-blur rule or VisDrone. Together with
the already reviewed [CVPR 2025 SET paper](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html),
they frame a real tension: background high-frequency clutter can obscure
tiny objects, yet context can be necessary for classifying/localizing
them. This is a **hypothesis generator**, not an explanation for our
candidate before its independent raw-val result. Any future proposal
should measure both false-positive suppression and context-dependent
misses, with an explicit strong-resolution control.

More precisely, the author-hosted [Scale-Aware AutoAug text](https://arxiv.org/abs/2103.17220)
reports a COCO **test-time all-context-removal** diagnostic in which
small-object AP falls from 25.2 to 18.0 while large-object AP rises
from 53.0 to 56.1; this is not our *train-only soft blur* setting.
Its method also uses spatially Gaussian-weighted blending to soften
box-level augmentation boundaries, so a smooth spatial transition
alone cannot be claimed novel. The author-hosted [context-augmentation
text](https://arxiv.org/abs/1807.07428) reports that placing objects
in random context hurts performance in its few-label VOC ablation,
while matching plausible context helps. Its copy-paste setup likewise
does not isolate background blur. A useful future causal experiment
would separate **background texture reduction** from **loss of
object-adjacent semantic context**; neither can be inferred from
visual appeal or an AP aggregate alone.
