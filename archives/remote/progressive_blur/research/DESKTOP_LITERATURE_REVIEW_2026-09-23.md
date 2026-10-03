# Desktop literature review: blur, foveation, and detection-driven processing

Reviewed 2026-09-23 from user-provided Desktop PDF folders. These are
precedents and cautions, not interchangeable evidence for the current VOC P1
result. The PDFs remain in their original Desktop locations; no sources were
modified.

| Source read | What was actually studied | Relevance and limit for our project |
|---|---|---|
| *Accelerating YOLO-based Real-time Object Detection via Foveated Rendering* (2023), `目标增强` | YOLOv3/v5/v8 on five selected images, with foveated pre-rendering and runtime/detection-count comparisons | Suggests peripheral detail reduction is worth testing, but has no benchmark mAP comparison and cannot establish our VOC AP or complete deployment speed. |
| *Cost-Efficient Multi-Scale Fovea for Semantic-Based Visual Search Attention* (2026), `目标增强` | Multi-scale crops around fixation; visual search/scanpaths on 586 COCO-Search18 target-present samples | It reduces processed pixels and addresses a different task. Its YOLOv11 per-iteration time is 0.144→0.115 s; not proof that full-image pre-blur speeds up YOLO. Oracle GT stopping also limits deployment interpretation. |
| *Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions* (AAAI 2022), `图像增强` | A small predictor controls differentiable image processing, trained jointly with YOLOv3 from detection boxes for low-light/fog | Strong prior art for a trainable image-processing front end. Our original general claim “small learned preprocessor before YOLO” is not novel by itself. The degradation setting differs from clean VOC background blur. |
| *DENet: Detection-driven Enhancement Network for Object Detection Under Adverse Weather Conditions* (ACCV 2022), `图像增强` | Laplacian low/high-frequency enhancement learned through detector loss; 45K-parameter front end; evaluates low-light and fog domains | Especially relevant to task-driven rather than visual-quality-driven preprocessing. Its ExDark mAP50 table reports YOLOv3 normal-trained 43.02, hybrid-trained 45.58, IA-YOLO 49.43, DE-YOLO 51.51; **not comparable** to our YOLOv8n/VOC mAP50–95. |
| *Low-Light Enhancement Effect on Classification and Detection: An Empirical Study* (2024), `图像增强` | Tests many enhancement methods before classifiers and detectors. On its ExDark/YOLOv7 table, unprocessed mAP is 76.09 and all listed preprocessing variants are below it. | Useful negative control: images that look clearer need not be better for machine detection. It is low-light test-time enhancement, not our training-only blur; do not generalize its numeric effect to VOC. |

## Working synthesis

1. Our strong/weak train-only results on raw VOC images are not rescued by
foveation papers: most change the evaluation task, processed pixel count, or
input degradation. A full-resolution Gaussian bank adds compute and has not
yet delivered AP.
2. Task-driven learned processing before YOLO is established prior art (at
least IA-YOLO and DENet). A publication would need a specific, measured
failure mechanism, a genuinely distinct remedy, proper controls, and gains
across seeds/domains—not merely “lightweight adaptive preprocessing.”
3. The immediate, falsifiable question is whether preserving a proportion
of clean training views closes the distribution gap. If not, pivot from
generic VOC background manipulation to a diagnosed domain problem (e.g.
tiny objects in VisDrone or low-light/thermal transfer), after checking
strong conventional baselines and prior art there.

## Additional primary-source cross-check (not a Desktop PDF)

The official CVPR 2025 paper page for [SET: Spectral Enhancement for Tiny
Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html)
explicitly describes *hierarchical background smoothing* to suppress
high-frequency background noise for tiny objects. Its
[official code repository](https://github.com/HuixinSun/SET) says the
HBS/API operations apply during training without additional inference load.
This is a particularly close precedent to a training-only background-
smoothing pitch. We have not run or reproduced SET, so its reported AI-TOD
gain is not a comparator for our VOC numbers. It narrows the novelty claim:
changing the smoothing strength or moving it to YOLOv8n is insufficient on
its own. A distinct measured failure and mechanism would be needed.

Read scope: the above papers were read at least through methods, core tables
and conclusions; the two foveation PDFs were read in full. Other Desktop
PDFs have only been inventoried or abstract-screened, not cited for detailed
results here.
