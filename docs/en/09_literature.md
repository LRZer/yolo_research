# 9. Literature and research implications

[Contents](../../README.md) · [中文](../zh-CN/09_literature.md)

## Four papers emphasized by the user

| Work | What informed this project | What cannot be transferred as a claim |
|---|---|---|
| Itti,Koch,Niebur1998, *A Model of Saliency-Based Visual Attention for Rapid Scene Analysis* | Bottom-up saliency and spatial attention; inhibition-of-return distinction | A saliency map is not all-object detector accuracy or a trained safe-background mask |
| Kea et al.2023,*Accelerating YOLO-based Real-time Object Detection via Foveated Rendering* | Practical foveated input experiment | Small demonstration/runtime evidence does not establish controlled full-dataset AP or complete-pipeline speed |
| Luzio et al.2025,*Human Scanpath Prediction in Target-Present Visual Search with Semantic-Foveal Bayesian Attention* | Semantics and foveal observation in sequential attention | Known-target, target-present scanpath prediction is not detecting every object without target/location cues |
| Luzio et al.2026,*Cost-Efficient Multi-Scale Fovea for Semantic-Based Visual Search Attention* | Multiscale field of view/downsampling as actual information-budget intervention | Reduced input pixels do not alone prove lower total search latency or complete-scene detection |

The [detailed four-paper reading record](../../archives/local/voc_pgb_yolov8n/FOVEATED_ATTENTION_FOUR_PAPERS_REVIEW_2026-09-24.md) preserves task definitions, method/cost qualifications and experimental implications. No paper PDF is redistributed. Primary links: [2025 authors' arXiv record](https://arxiv.org/abs/2507.18503), [2026 authors' arXiv record](https://arxiv.org/abs/2604.03836), [1998 DOI](https://doi.org/10.1109/34.730558). The 2023 user-supplied paper is identified by title in the reading record; unverified bibliographic details are not invented.

## Other relevant prior art and adoption boundaries

- [SET, CVPR2025 official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf) and [author code](https://github.com/HuixinSun/SET): this project ported **HBS only** into YOLOv8n/VisDrone and ran an identity extra-loss control. No API/full-method reproduction or original spectral-enhancement contribution is claimed.
- [SAHI](https://arxiv.org/abs/2202.06934), [AutoFocus](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html), [ClusDet](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html): sliced, selective and clustered multiscale detection predate the current grid prototype. Fixed-central4 is not a sufficient substitute for a faithful strong SAHI/selected-region baseline.
- GOIS and DAHI were reviewed as guided/adaptive inference neighbors in the historical notes. The project did not implement or faithfully reproduce them; their numerical results are not project results.
- ATSS, ObjectBox, LapNet/PONO, quality-aware losses, ranking losses, OTA/POTO/O2F and label-reassignment work limit broad claims about “first” fair per-object gradients/coverage. [IGB novelty boundary](../../archives/local/yolo26_research/IGB_NOVELTY_UPDATE_2026-09-19.md), [coverage literature audit](../../archives/local/yolo26_research/COVERAGE_ROUTE_NOVELTY_DEEPCHECK_2026-09-19.md).
- Fine-grained global/local attention and domain-specific normalization are established areas. The current shared-vs-split BN experiment is a specific causal test, not an invention of two-view classification or domain-specific BN.
- [CUB official page](https://www.vision.caltech.edu/datasets/cub_200_2011/) defines the dataset; [PyTorch BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) describes training/evaluation statistics. The actual archived implementation controls this experiment's semantics.

## Synthesis from completed evidence

1. **Localization matters, but GT can exaggerate deployable value.** Direct full-image Oracle blur improved frozen detectors while predicted protection failed; CUB GT target-preserving representations and enlarged local crops show positive signals. The next causal question is the gap between known and predicted locations, not whether a visually plausible heatmap looks good.
2. **Blur is not a universal denoising solution.** Learned input blur, training-only Oracle blur, HBS-only and cropped background blur all failed their tested controls. CUB D barely improved C at substantially greater cost. Preserving details and target area is often more important than choosing a complex low-pass bank.
3. **Foreground detail and color remain useful.** Strong full-image blur, grayscale and pureScharr lost substantial CUB accuracy. This does not forbid a residual/auxiliary branch but rejects these stand-alone representations under the tested regime.
4. **Context and normalization must be separated.** G5 exceeded G4, but a zero branch changes shared BN. G6 was stopped; neither the BN explanation nor the proposition “background always harms” is established.
5. **Cost must be real.** Fixed-resolution Gaussian preprocessing adds work; five 960 crops add calls. The project has not reached an accuracy-versus-compute Pareto improvement.
6. **Reliable adaptation is an acceptable route.** The user's preference was to apply established methods faithfully, then make limited attributable adaptations rather than demand a wholly new paradigm. A direct faithful baseline and a compute/information-matched control are prerequisites, not obstacles to novelty.

## Prioritized next questions, not executed results

| Priority | Falsifiable question | Necessary controls before a claim |
|---|---|---|
|1|How much of G5's signal is simply GT crop magnification?|True single-local model,same view budget and normalization; GT vspredicted localization; multiple seeds |
|2|Does mixed BN explain G4 degradation?|Complete/shared-vs-split pair,BN sample-count controls,unchanged affine/backbone and healthy hardware |
|3|Can real peripheral sampling retain accuracy at lower compute?|Uniform low-resolution vsnonuniform views,equal processed pixels/model calls,preprocessing+model latency |
|4|Is background context useful conditionally?|Matched image/box geometry,blank/low-res/background-only controls,per-class/size diagnostics without test tuning |
|5|Can an established attention method transfer faithfully?|Reproduce source method and strong baseline first,then one limited adaptation with error-driven go/no-go |

Before confirmation, define a separate development/confirmation split or a new dataset, freeze primary endpoint and protocol, restore healthy hardware and repeat seeds. No new GPU experiment was started during this publication. These questions document the remaining research path rather than inflate existing artifacts into a publication-ready contribution.
