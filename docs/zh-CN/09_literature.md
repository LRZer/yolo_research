# 9. 文献与研究启示

[目录](../../README.zh-CN.md) · [English](../en/09_literature.md)

## 用户重点指定的四篇

| 工作 | 本项目借鉴 | 不能直接迁移的主张 |
|---|---|---|
|Itti/Koch/Niebur1998显著性注意模型|bottom-up saliency、空间注意、inhibition of return|显著图不等于全目标AP或学好的安全背景mask|
|Kea等2023 YOLO foveated rendering|凹视输入的实际尝试|小规模演示/时间不证明完整数据AP或端到端提速|
|Luzio等2025语义中央凹Bayesian注意|语义与顺序注视观测|已知目标、target-present scanpath不等于无目标提示的全目标检测|
|Luzio等2026多尺度低成本fovea|多尺度视场、下采样真正控制信息预算|输入像素减少不直接等于总搜索更快或整场景检出|

[四篇细读记录](../../archives/local/voc_pgb_yolov8n/FOVEATED_ATTENTION_FOUR_PAPERS_REVIEW_2026-09-24.md)保留任务、方法/成本边界和实验启示，不分发PDF。一手链接：[2025作者arXiv](https://arxiv.org/abs/2507.18503)、[2026作者arXiv](https://arxiv.org/abs/2604.03836)、[1998 DOI](https://doi.org/10.1109/34.730558)。2023按用户提供论文标题在记录中识别，不补编未核实书目信息。

## 其他近邻与移植边界

- [SET CVPR2025官方论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf)、[作者代码](https://github.com/HuixinSun/SET)：本项目仅HBS移植YOLOv8n/VisDrone并做identity-extra-loss对照，非API/完整复现，不冒称原创频谱增强。
- [SAHI](https://arxiv.org/abs/2202.06934)、[AutoFocus](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html)、[ClusDet](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html)已经研究切片、选择性、多尺度集群检测。固定中央4格不是强SAHI/区域选择基线的忠实替代。
- GOIS、DAHI在历史笔记中审查为guided/adaptive inference近邻，未实际移植/忠实复现，不把原文数字算项目结果。
- ATSS、ObjectBox、LapNet/PONO、质量感知/排序损失、OTA/POTO/O2F及label reassignment限制“首次公平逐目标梯度/coverage”等宽泛主张。[IGB排重](../../archives/local/yolo26_research/IGB_NOVELTY_UPDATE_2026-09-19.md)、[coverage文献审查](../../archives/local/yolo26_research/COVERAGE_ROUTE_NOVELTY_DEEPCHECK_2026-09-19.md)。
- 细粒度全局/局部注意及domain-specific normalization已有研究。当前BN实验是具体因果验证，不是发明双视图或分域BN。
- [CUB官方](https://www.vision.caltech.edu/datasets/cub_200_2011/)、[PyTorch BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)给数据与统计语义，本次实际实验以归档源码为准。

## 已完成证据的综合判断

1. **定位重要，但GT可能夸大实际收益。**直接整图GT模糊提高冻结检测器、预测保护失败；CUB GT保护与局部放大正向。下一问题是已知/预测位置的差距，不是注意图好不好看。
2. **模糊不是普遍去干扰解。**学习式、训练Oracle、HBS、局部背景模糊都未通过所测对照；CUB D只略好C且更贵。保护细节/目标面积可能比复杂滤波bank更关键。
3. **目标细节和颜色有价值。**全图强模糊、黑白、独立Scharr大幅损失CUB；不否定辅助残差分支，但否决该制度下独立表示。
4. **上下文与归一化需分开。**G5高G4，但零分支影响BN；G6未完成，不能确认BN或“背景总有害”。
5. **成本必须真实。**固定尺寸高斯增加工作，四裁剪加调用，未建立准确率/计算Pareto收益。
6. **可靠适配可以是研究路线。**用户不要求全新范式，允许忠实复现已有方法，再做有限可归因适配。直接移植基线、同成本/信息预算对照是前提。

## 后续优先问题，非已实施结果

| 优先 | 可证伪问题 | 必要控制 |
|---|---|---|
|1|G5有多少收益只来自GT裁剪放大？|真正单局部、同视图预算/归一化、GT vs预测定位、多种子|
|2|混合BN是否导致G4退化？|完成shared/split配对、BN样本数控制、相同affine/backbone、健康硬件|
|3|外围非均匀采样能否真省算力？|统一低分辨率、非均匀视图、同像素/调用数、预处理+模型时延|
|4|背景是否有条件地有益？|同几何、blank/低清/背景单独控制、类别/尺寸诊断，不看test调参|
|5|已有注意方法能否忠实迁移？|先复现原方法与强基线，再单个失效驱动适配/门槛|

确认前需要新dev/确认集或另一数据集，冻结主指标/协议，健康硬件和多种子。本次公开整理不新开GPU实验；这些是剩余研究问题，不能包装为已经可发表的贡献。
