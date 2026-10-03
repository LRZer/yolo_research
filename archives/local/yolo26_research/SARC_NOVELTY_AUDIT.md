# SARC 论文新颖性审计（截至 2026-09-19）

## 0. 结论先行

**裁决：CONDITIONAL / 窄新颖性可防守，模块级或宽概念新颖性不可防守。**

截至 2026-09-19，在本次检索覆盖的 CVF/ECVA/OpenReview/arXiv/IEEE/PMLR 等论文原文或出版社页面中，**没有找到与下面这条完整因果链直接相同的公开工作**：

> 从 detector 的真实 O2M/TAL 正样本中，按 FPN level 汇总 `log2(sqrt(wh)/stride)` 软直方图，经 EMA 形成停止梯度目标，再用 KL 直接约束一组输入无关、最终可折叠进单个大核的 dilation-branch mixing coefficients。

但这并不支持把 SARC 描述成“新型可重参数化大核”“首个 assignment-aware receptive field”或“首个动态 FPN kernel”。其各个组成部分已有很强先例：

- **可重参数化 dilation branches → 单一大核**：UniRepLKNet 的 Dilated Reparam Block 已在 CVPR 2024 正式发表，结构相似度很高。
- **对象尺度与不同 dilation/receptive-field branches 对应**：TridentNet 在 ICCV 2019 已以尺度区间把对象分配到 dilation 1/2/3 的共享权重分支。
- **receptive field 与 label assignment 联结**：RFLA（ECCV 2022）以及 RFAssigner（2026 预印本）明确用 receptive-field distance 改变正样本分配，虽与 SARC 的因果方向相反。
- **FPN 中的 dynamic receptive-field assignment + task alignment**：IEEE TII 2024 的同名近邻工作已正式发表。当前可访问的一手出版社正文只有摘要，不能据摘要断言它是否包含更细的 assignment→kernel 反馈；这是投稿前必须人工取得全文核对的阻断项。
- **训练统计闭环调整 detector**：Dynamic R-CNN（ECCV 2020）已根据训练中 proposal 统计动态更新 label-assignment threshold 与 regression loss。
- **同一检测器同时使用动态模块和动态 assignment**：Dynamic Sparse R-CNN（CVPR 2022）以及 PaQ-DETR（CVPR 2026）都已有先例，只是公开方法中两者不是 SARC 这种显式统计反馈链。

因此，SARC 能尝试防守的贡献只应是：

> **assignment-distribution-calibrated static receptive-field allocation**：用训练期、按层的 detector positive-assignment 分布，显式校准部署可折叠的 dilation 分支系数；校准本身不增加相对于同部署块的推理结构。

而且该表述仍有两个必须用实验闭合的因果缺口：

1. R3 优于 R2 只证明“多了一个 KL 正则可能有用”，并不证明正确的 assignment 语义有用；
2. 当前 KL 约束的是名义 softmax 系数，不等价于实际分支贡献或实际有效感受野。

若下文的语义 placebo、GT-only、有效分支贡献和在线/静态先验对照不能通过，应把论文定位降为“一个 detector-specific regularization heuristic”，而不是 assignment-aware RF 方法。

---

## 1. 本次实际审计的 SARC 定义

本审计以当前冻结文档和 staging 实现为准，而不是以一个宽泛的“SARC”名称为准。已阅读：

- `outputs/yolo26_research/BROAD_DIRECTION_SCREEN.md`
- `outputs/yolo26_research/BROAD_PROBE_RESULTS_2026-09-19.md`
- `work/sarc_impl/SARC_METHOD_AND_INTEGRATION.md`
- `work/sarc_impl/sarc.py`、测试与配置
- `outputs/yolo26_research/SARC_STAGING_REPORT.md`

每个 P3/P4/P5 在 Detect 前插入独立残差块：

```text
R_s(x) = BN7(DW7(x))
       + sum_r softmax(a_s)[r] * BNr(DW3_d[r](x)), d=[1,2,3]
y_s    = x + gamma_s * SiLU(BN_mix(R_s(x)))
```

其中 `a_s` 是每层 3 个**输入无关的静态可训练标量**，不是 per-image/per-location 动态注意力。R3 使用 O2M/TAL 正样本形成：

```text
u = log2(sqrt(w_gt * h_gt) / stride_level)
h_s = EMA(soft histogram of u at FPN level s)
L_cal = 0.01 * sum_s KL(stopgrad(h_s) || softmax(a_s))
```

中心为 `[2.25, 3.00, 3.75]`，温度 `0.50`，EMA decay `0.99`，前 16 次有效更新只收集统计。DDP 先全局求和 soft counts 和 mass，再更新相同 EMA。

R1 是单 DW7，R2 是相同多分支可重参数化结构但无 `L_cal`，R3 才加入校准。分支 Conv-BN、softmax 常量和外层 BN 可折叠到一个 DW7；但 `gamma * SiLU(z)` 中 gamma 不能穿过 SiLU，所以部署态仍保留一次 frozen scalar multiply、SiLU 和 residual add。相对 R0，部署新增约 22,400 参数、0.07025 GFLOPs；相对 R2，R3 的**校准机制**没有额外推理算子。

### 已有 probe 能证明与不能证明的内容

当前 probe 显示 positive-anchor pooled JS：P3/P4 `0.01662`、P3/P5 `0.12339`、P4/P5 `0.06871`；unique-GT pooled JS：`0.02050/0.03500/0.02160`；最大相邻 epoch JS `0.007918`。这足以说明训练集上按层统计有差异且较稳定，允许进入 R1/R2/R3 实验。

它**不能**证明：

- 该差异对 AP 有因果作用；
- 差异来自 TAL 语义而非 GT 尺度先验、stride 定义或每个 GT 的 positive-anchor multiplicity；
- branch coefficient 真正改变了模型的有效感受野；
- 在线 EMA 优于预计算静态数据集先验。

尤其是 pooled positive-anchor 的 P3/P5 JS 为 `0.12339`，而 unique-GT 最大只为 `0.03500`，提示显著的一部分分离度可能来自每个 GT 被分到多少正锚，而不是纯粹的“对象尺度语义”。这应成为正式实验中的显式问题，而不是被隐藏在 histogram 定义里。

---

## 2. 检索范围、状态口径与否定结论边界

### 2.1 检索范围

重点检索 2024–2026，并回溯不可绕开的早期先例。组合关键词覆盖：

- `label assignment / task-aligned assignment / TAL / positive samples`
- `receptive field / dilation / dilated branch / scale-aware`
- `dynamic kernel / adaptive kernel / reparameterization / RepConv`
- `FPN / feature pyramid / neck / per-level histogram / EMA / statistics`
- `object detection / dense detector / anchor-free`

优先核对论文正文、CVF/ECVA/PMLR/OpenReview/arXiv 原文和 IEEE 出版社页面；不以二手博客、聚合摘要或代码 README 作为新颖性证据。

### 2.2 状态口径

- **已正式发表、顶会**：CVPR/ICCV/ECCV/ICML 等正式 proceedings。
- **已正式发表、其他同行评审 venue**：IEEE TII、WACV 等，单列而不混称顶会。
- **预印本**：只有 arXiv/未见正式 venue 的工作，不能与正式发表混写。

### 2.3 否定结论的严格措辞

本报告只能说“在上述范围内未找到 exact match”，不能说“世界上不存在”。特别限制包括：未穷尽专利、非英文文献、尚未公开/未被索引的稿件；IEEE TII 2024 的 DRFA 全文当前未能从出版社公开访问。任何论文中的 `first` 声称都应再做一次投稿日前更新，并补齐 DRFA 全文。

---

## 3. 对三个核心检索问题的回答

| 核心问题 | 审计答案 | 论文定位后果 |
|---|---|---|
| 是否已有 detector/TAL/label assignment 的 **per-FPN-level object-scale histogram/EMA** 直接校准 **可重参数化 dilation/kernel branch weights**？ | **未找到直接相同方法。** Dynamic R-CNN 用训练 proposal 统计改 label threshold/loss；SARC 的 exact statistics→reparam-branch path 仍未见。 | 这是目前唯一可尝试主张的窄贡献，但必须以 assignment-semantic controls 证明。 |
| 是否已有 **assignment-aware receptive field**？ | **宽义上明确已有。** TridentNet 用对象尺度决定 dilation branch；RFLA/RFAssigner 用 RF distance决定 assignment；TII 2024 直接使用“dynamic receptive field assignment”命名。 | 禁止声称“首个 assignment-aware RF”或“首次联结 assignment 与 RF”。只能强调因果方向和统计接口不同。 |
| 是否已有 **dynamic/reparameterized FPN kernels driven by training assignments**？ | 未找到与 SARC 一样的显式聚合反馈链；但 FPN dynamic RF、动态核、可重参数化核，以及 assignment+dynamic module 同系统均已有。 | 可以说“我们未发现显式用按层 assignment distribution 校准可折叠 branch mixture 的先例”，不能泛称首个 dynamic/assignment-driven FPN。 |

需要额外澄清：“driven by training assignments”若只指 detection loss 的梯度，几乎所有 detector architecture 都受 assignment 驱动。SARC 必须把贡献限定为**assignment statistics 的显式旁路监督**，否则该说法没有区分度。

---

## 4. 最接近工作：重叠、差异与风险

### 4.1 一级近邻：决定主张边界的工作

| 工作与状态 | 与 SARC 的实质重叠 | 可防守差异 | 风险 |
|---|---|---|---|
| [UniRepLKNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html)；**正式顶会** | Dilated Reparam Block 使用大核主分支和多个小/膨胀核分支，训练后将 BN 与稀疏膨胀核嵌入并合并为单一大核；也在 COCO detection 上验证。对 K=7 的官方构造已经包含覆盖 d=1/2/3 的近似同族设计。 | UniRepLKNet 不从 detector assignment histogram 产生分支目标，也没有 SARC 的 per-FPN EMA-KL calibration。 | **最高结构碰撞。** SARC 模块本体、dilated reparameterization、零额外 branch inference 都不能作为新颖点；R2 应被明确描述为 UniRepLKNet/RepLK 思路的 detector-neck adaptation。 |
| [Scale-Aware Trident Networks, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.html)；**正式顶会** | 权重共享的 dilation 1/2/3 平行分支；训练时用对象尺度有效区间把样本交给不同 RF 分支，并给出快速推理版本。它已直接建立 object scale ↔ dilation branch specialization。 | TridentNet 是对象/分支路由和手工 scale ranges，不是按 FPN level 汇总 TAL positives 后用 EMA-KL 校准一个可折叠静态 mixture。 | **最高概念碰撞。** 不能声称首个 scale-aware RF branch 或首个用对象尺度训练 dilation specialization。 |
| [An Efficient Anchor-Free Defect Detector With Dynamic Receptive Field and Task Alignment, IEEE TII 2024](https://ieeexplore.ieee.org/document/10472785/)；**正式期刊** | 出版社摘要明确写到带 DRFA 的 feature pyramid、灵活调整 RF，以及 anchor-free deep supervision with task alignment；题目和术语与 SARC 极近。 | 摘要将 DRFA、task-decoupling、task-aligned supervision列为不同组件；公开摘要没有证据表明它用 label-assignment histogram/EMA 去更新 kernel branches，更没有证据表明可重参数化。 | **高且尚未解除。** 只能基于摘要写“未显示 exact feedback”；投稿前必须获取原文逐式核对。 |
| [RFLA, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)；**正式顶会** | 以 Gaussian receptive-field distance 衡量位置与 GT，设计 hierarchical label assignment，明确把 RF 与正样本分配联结。 | 因果方向为 RF prior → assignment；不学习或重参数化 kernel。SARC 是 assignment distribution → branch coefficient。 | 不能声称 RF/assignment 联结首次出现；可以强调“反向反馈方向”和部署校准不同。 |
| [RFAssigner, arXiv 2601.01240](https://arxiv.org/abs/2601.01240)；**2026 预印本，未见正式 venue** | 用多尺度 Gaussian RF distance 为小目标补充 positives，是最新的通用 dense-detector RF-aware assignment。 | 同样是 RF → assignment；只改变训练 label assignment，不改变推理结构或 kernel。 | 最新术语碰撞；需明确标注预印本身份，但 related work 不能漏。 |
| [Dynamic R-CNN, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2335_ECCV_2020_paper.php)；**正式顶会** | 根据训练中 proposal IoU/回归标签分布统计，动态更新 label-assignment threshold 和 SmoothL1 参数；已经证明“训练状态统计→检测器设置”的闭环。 | 它不更新 feature extractor 的 RF/kernel，更无 per-level reparameterized branches。 | 禁止声称“首次用训练统计自适应校准 detector”。可声称被校准对象和统计对象不同。 |
| [Dynamic Sparse R-CNN, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Dynamic_Sparse_R-CNN_CVPR_2022_paper.html)；**正式顶会** | 同一检测器中同时有 dynamic label assignment 和由多个 proposal experts 动态组装的 proposal generation。 | 两条机制并列：assignment 没有汇总后直接生成/正则化 dynamic experts；对象也不是 FPN kernel。 | 禁止宽泛声称“首次联合 dynamic component 与 dynamic assignment”。 |
| [PaQ-DETR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html)；**正式顶会** | 最新工作同时学习 content-conditioned dynamic queries，并使用 quality-aware O2M assignment 平衡监督。 | 动态 query 由输入内容生成；assignment 是监督策略，并未作为聚合统计去校准 reparam kernels。 | 2026 年最新同系统近邻，进一步要求 SARC 只讲 direct statistics path，不讲“动态组件+assignment”的组合新颖性。 |

### 4.2 二级近邻：动态/自适应 RF 与 kernel 已很拥挤

| 工作与状态 | 已占据的空间 | 与 SARC 的关键区别 |
|---|---|---|
| [DetectoRS, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.pdf)；正式顶会 | Switchable Atrous Convolution 以空间相关开关融合不同 atrous rates，并结合 Recursive FPN。 | feature/context-conditioned、推理时动态；非 assignment statistics，非折叠静态核。 |
| [LSKNet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)；正式顶会 | 在 detection 中动态选择大空间 RF 来适配不同对象所需上下文。 | 由图像特征注意力选择，不用 TAL/EMA，不折叠成 SARC 式静态核。 |
| [FADC, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html)；正式顶会 | 按局部频率动态改变空间 dilation，AdaKern 调整低/高频 kernel 比例；也报告 object detection。 | per-location/per-channel frequency-conditioned dynamic operation，不由 assignment 驱动，也非训练后静态折叠。 |
| [PKINet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)；正式顶会 | detection backbone 的多尺寸 depthwise kernels 与长程 context，用于对象尺度变化。 | 固定多核结构与上下文注意力，无 assignment histogram/EMA/reparameterization。 |
| [Kernel Adaptive Convolution for Scene Text Detection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Kernel_Adaptive_Convolution_for_Scene_Text_Detection_via_Distance_Map_CVPR_2024_paper.html)；正式顶会 | 以不同 kernel sizes 进行位置自适应组合，并有 distance-map supervision。 | scene-text segmentation式检测；监督对象和动态机制均不是 detector label assignment aggregate。 |
| [KernelWarehouse, ICML 2024](https://proceedings.mlr.press/v235/li24cg.html)；正式顶会 | 大规模 kernel warehouse 与 input-dependent attention 组装动态卷积，并在 COCO 验证。 | 输入相关动态核，不是 FPN assignment statistics 或可部署静态 dilation mixture。 |
| [OverLoCK, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lou_OverLoCK_An_Overview-first-Look-Closely-next_ConvNet_with_Context-Mixing_Dynamic_Kernels_CVPR_2025_paper.html)；正式顶会 | 以全局上下文产生 token-wise dynamic kernels，并使用 Dilated RepConv。 | context-conditioned 推理动态核；RepConv 是局部构件，未见 assignment→branch feedback。 |
| [Frequency Dynamic Convolution, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.pdf)；正式顶会 | 以多频率 basis 形成动态 kernel，覆盖 detection/segmentation。 | feature/frequency-conditioned，非 assignment-conditioned；推理时仍动态。 |
| [ShiftwiseConv, CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Li_ShiftwiseConv_Small_Convolutional_Kernel_with_Large_Kernel_Effect_CVPR_2025_paper.pdf)；正式顶会 | 小核、shift、group 与 reparameterization 获得大核效果，并用于 detection。 | 没有 label assignment calibration；只说明大 RF 与重参数化本身已拥挤。 |
| [UniConvNet, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_UniConvNet_Expanding_Effective_Receptive_Field_while_Maintaining_Asymptotically_Gaussian_Distribution_ICCV_2025_paper.html)；正式顶会 | 从有效感受野分布设计多层 receptive-field aggregator，并在 COCO 验证。 | 架构先验，不使用 detector assignment。 |
| [AKCMamba-YOLO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_AKCMamba-YOLO_Selective_State_Space_Models_For_Real-Time_Object_Detection_CVPR_2026_paper.pdf)；正式顶会 | YOLO backbone/neck 使用 adaptive kernel convolution，按输入特征调整 RF。 | 输入特征驱动、非 assignment histogram、无 SARC 式静态 reparameterization target。 |

### 4.3 FPN/scale-aware 邻近工作

- [SEPC, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Scale-Equalizing_Pyramid_Convolution_for_Object_Detection_CVPR_2020_paper.html) 已在 pyramid scale 维进行卷积并处理层间失配。它不使用 assignment 统计，但排除了“首个 scale-aware FPN convolution”一类宽声明。
- [Rethinking Features-Fused-Pyramid-Neck, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8386_ECCV_2024_paper.php) 重新设计 hierarchy pyramid 和上下采样来减轻层级错配。它与 SARC 的直接机制较远，但说明 FPN level specialization 不是新问题。
- [Dynamic Coarse-to-Fine Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.pdf) 同时包含动态 label assignment 和带 dilation/DCN 的 prior capturing block，但论文没有把 assignment distribution 反馈给该 block。它是“同系统并存但无直接耦合”的又一重要反例。

---

## 5. 可以声称、需要条件化声称、不能声称

### 5.1 若因果实验通过，可防守的表述

推荐主贡献表述：

> We introduce a training-only assignment-distribution calibration for deploy-equivalent receptive-field allocation. Per-level O2M positive assignments are summarized as an EMA relative-scale distribution, which directly regularizes static mixing coefficients of dilated branches that are folded into a single depthwise kernel for deployment.

中文要点：

- 新意在 **assignment distribution → explicit branch-coefficient target**；
- 强调是 **training-only calibration** 和 **static deploy allocation**；
- 相对 R2 的 calibration 无额外推理开销，而不是相对 R0 零开销；
- 明确因果方向不同于 RFLA/RFAssigner 的 RF→assignment；
- 明确 conditioning source 不同于 DetectoRS/FADC/OverLoCK 等 input-feature-driven dynamic kernels。

只有在 GT-only 和 online/frozen controls 通过后，才可加入“model assignment carries information beyond raw object-scale priors”或“online assignment feedback”。

### 5.2 只能附条件写的表述

- “assignment-aware receptive-field allocation”：必须由正确目标胜过 shuffled/permuted placebo，并有 actual-RF diagnostics。
- “TAL-driven”：当前 target 实际只用 `fg_mask`、matched GT、level 和正锚 multiplicity，没有直接用 alignment metric/score。若不做 assignment-specific controls，宜写 “positive-assignment-derived” 而非 “task-alignment-aware”。
- “adaptive”：只能指随训练 assignment distribution 校准一次/逐步校准，不能让读者理解为 per-image adaptive inference。
- “reparameterized into one kernel”：卷积分支可折叠成一个 DW7，但完整 block 仍有 SiLU、frozen gamma multiply 和 residual add。

### 5.3 不能声称

1. **不能**声称首个 reparameterized dilated/large-kernel block；UniRepLKNet 已直接覆盖。
2. **不能**声称首个 scale-aware 或 assignment-aware RF；TridentNet、RFLA/RFAssigner 与 TII DRFA 已覆盖宽概念。
3. **不能**声称首个 dynamic FPN kernel；DetectoRS、FADC、OverLoCK、AKCMamba-YOLO 等已有不同形式。
4. **不能**把 SARC 称为 input-dynamic kernel/attention；其 `a_s` 是每层输入无关参数。
5. **不能**声称 calibration “零推理成本”而不加比较对象。正确说法是“相对同一 deploy block（R2）无额外推理成本”；相对 R0 明确增加约 0.94% 参数和 1.33% convolution FLOPs。
6. **不能**声称首次用 detector 训练统计闭环适配；Dynamic R-CNN 已存在。
7. **不能**把 histogram difference 当成性能或机制有效性的证据；当前 probe 只提供进入实验的先验。
8. **不能**在未测 effective branch contribution/ERF 前说 softmax 与实际 RF 已对齐。
9. **不能**在未胜过 GT-only/frozen prior 前说收益来自 TAL 语义或在线反馈。
10. **不能**在未取得 TII DRFA 全文前使用无限定的 “first”。

---

## 6. R3 vs R2 的核心因果缺口

R2 与 R3 唯一设计差别是 `L_cal`。若 R3 > R2，逻辑上只能推出：

> 对三个 branch logits 加这个 KL 项，在该优化设置下有益。

它不能区分以下四个机制：

1. 正确的 per-level assignment-scale 语义有益；
2. 把 logits 拉离随机初始化或打破分支对称性有益；
3. 任意稳定目标/熵约束有益；
4. 该目标只是训练集 GT 尺度先验，与模型产生的 assignment 无关。

### 6.1 若只能增加一个控制，最低要求是什么？

**最低要求是等强度的 semantic placebo，而不是 uniform target。** 推荐预注册一个固定 cyclic level permutation：

```text
P3 branch logits <- h_P4
P4 branch logits <- h_P5
P5 branch logits <- h_P3
```

保持与 R3 相同的 gain、EMA、warmup、soft centers、更新次数、DDP 聚合、训练结构与部署结构。它使用完全相同的三组 histogram，只破坏“哪一层应该接收哪一个尺度分布”的对应关系，因此比 uniform 更接近等强度反事实。还可增加每层 dilation-bin reversal，保持每个目标的 entropy 精确不变，专门破坏小/大尺度到 dilation 顺序的语义。

**为什么 uniform target 不足以单独承担该任务：** uniform 同时改变 target entropy、每层差异、与初始化距离以及是否鼓励 branch balance。即使 R3 胜过 uniform，也不能排除只是非均匀/破对称 target 更合适；即使 uniform 胜出，也不能定位 assignment 语义。

### 6.2 若论文要声称“模型 assignment 特有信息”，第二个不可省控制是什么？

**GT-only / heuristic-level control 必须加入。** 不读取 TAL `fg_mask`、matched level 或 positive multiplicity，只用 raw GT size，并用冻结的、训练前注册的 FPN 尺度规则将每个 GT 分配到 P3/P4/P5；之后使用相同 `u`、soft centers、EMA、KL gain 和 warmup。

原因是当前 target 的主要输入仍是 GT `w,h` 和 stride，且 probe 表明统计随 epoch 很稳定。若 GT-only 与 R3 持平，则可得出的结论是“dataset-scale prior regularizes branches”，而不是“TAL assignment calibrates RF”。

因此最小因果集分两层：

- 只要证明“正确语义优于任意 KL”：至少加 **cyclic-permuted/shuffled-hist placebo**；
- 要支撑论文标题或主贡献中的 “assignment-driven/TAL-aware”：还必须加 **GT-only heuristic-level**。这两个不是互相替代关系。

---

## 7. 另一个更隐蔽的缺口：softmax 系数不等于实际 RF 贡献

当前目标直接约束 `q_r = softmax(a_s)_r`，但 branch output 为：

```text
q_r * BN_r(DW3_d[r](x))
```

各分支有独立卷积权重和 BN affine。优化器可以通过放大/缩小 folded kernel 或 BN gamma 抵消 `q_r` 的变化；`q_r` 因而不是可识别的 contribution measure。进一步地：

- d=3 只规定采样支持最大可到 7×7 外围，不保证外圈权重非零；
- 固定的 DW7 主分支始终无 softmax 权重，可能主导总 RF；
- 外层 `BN_mix`、SiLU 和 residual 还会改变 feature response；
- 所以 `KL(h || q)` 至多校准“名义 branch gates”，不能直接证明“实际 RF distribution 被校准”。

至少应记录并报告：

1. 每层每分支 `q_r * ||K_r^folded||_F` 及归一化比例；
2. 每分支输出 RMS/variance 与 `q_r` 加权后的比例；
3. 合并后 7×7 kernel 按半径/坐标环的能量分布；
4. gradient-based ERF 或 impulse-response radius，比较 R2/R3 与正确/错误 target；
5. `h_s` 与上述实际贡献之间的相关性，而不只画 `h_s` 与 `q_s` 的重合图。

若 `q` 对齐而 folded-kernel energy/ERF 不对齐，论文只能声称 coefficient regularization，不能声称 receptive-field calibration。更强的后续方法可考虑对 folded branch kernel 做 norm normalization，或直接把 target 施加在可识别的 effective contribution 上；但这会改变当前冻结方法，应另立实验，不应事后悄悄修改 R3。

---

## 8. 必要消融与实验

下表按“论文能否成立”的优先级排序。前 6 项为最低推荐集；不是只做四个后任选最好结果。

| 优先级 | 实验 | 隔离的问题 | 通过标准/解释 |
|---:|---|---|---|
| 1 | **R0 / R1 / R2 / R3 固定成本因果矩阵**，相同 seed/schedule/augmentation | 大核本身、多分支 reparam、assignment calibration 各自贡献 | R3 必须稳定优于 R2；R2/R3 deploy topology 与成本完全一致。 |
| 2 | **R3-Permute**：固定 P3→P4→P5 cyclic target permutation；最好再加 bin reversal | 正确 assignment-scale 语义 vs 任意同强度 KL/破对称 | R3 应显著优于 placebo；否则只能说 KL regularization 有用。 |
| 3 | **R3-GTOnly**：raw GT size + 冻结 heuristic FPN level，不读 TAL | 模型 assignment 信息 vs 数据集尺度先验 | R3 应优于 GTOnly；持平则删去 “TAL/assignment-driven” 主张。 |
| 4 | **Uniform-target KL**，完全相同 gain/update schedule | generic branch equalization/entropy regularization | 只作补充，不可替代 Permute。若 uniform≈R3，assignment 语义不成立。 |
| 5 | **positive-anchor-weighted vs unique-GT-per-level histogram** | TAL positive multiplicity/easiness权重 vs 对象尺度分布 | 必须说明收益来自哪一个。unique-GT 持平意味着 anchor multiplicity 不是必要机制。 |
| 6 | **online EMA vs frozen train-only histogram**（如预计算或早期冻结） | 在线 assignment feedback vs 静态 dataset prior | 若 frozen≈online，不得称 online/dynamic calibration；可保留 training-only prior regularization。 |
| 7 | **effective contribution/ERF diagnostics** | 名义 gate 是否真正改变 RF | `h_s` 不仅与 `q_s`，还要与 folded-kernel energy/output RMS/ERF radius同向；否则降级主张。 |
| 8 | **不同 scale distribution 的数据/干预**：至少 COCO + VOC，或训练期预注册的 scale-resampling intervention | 方法是否真的对 assignment distribution 有响应，而非 VOC 特例 | 分布变化应带来可预测的 histogram、branch contribution 与 AP_s/m/l 变化。 |
| 9 | **多 seed 与置信区间**，报告 AP、AP50/75、AP_s/m/l | 小数据集噪声和偶然优化收益 | 不能只以单 seed `+0.002/+0.003` 作新颖性证据；至少 3 seeds，报告 mean/std 或 bootstrap CI。 |
| 10 | **deploy exactness、真实 latency/peak memory** | reparameterization 工程声明 | train-eval、folded PyTorch、实际 exporter 输出一致；同时报告相对 R0 和相对 R2 两种成本口径。 |

### 建议的硬判据

若以下任一成立，应主动降级论文主张：

- `R3 <= R3-Permute`：正确 assignment-level 语义没有证据；
- `R3 ≈ R3-GTOnly`：TAL 不提供超出 raw GT scale prior 的可检测增益；
- `R3 ≈ Uniform`：大概率只是 branch equalization/regularization；
- `R3-online ≈ frozen-hist`：不是在线 assignment feedback；
- `q` 对齐但 actual branch contribution/ERF 不对齐：不是 RF calibration；
- 仅 positive-anchor histogram 有效而 unique-GT 无效：贡献可能是 positive multiplicity/easiness prior，需按此重写机制解释；
- 只在 VOC 单 seed 有边缘收益：不足以支撑新方法论文。

---

## 9. 推荐的论文定位与 related-work 叙事

### 9.1 推荐定位

SARC 最合适被定位为**一个 detector-training feedback interface**，而不是新的 convolution primitive：

```text
O2M assignments
    -> per-level relative-scale distribution
    -> temporally smoothed training target
    -> static reparam-branch allocation
    -> one deploy kernel per level
```

related work 应清楚分三条线：

1. Dilated/large-kernel structural reparameterization：承认 UniRepLKNet 等已解决“怎么折叠”。
2. Scale/RF-aware detection and label assignment：承认 TridentNet、RFLA/RFAssigner、DRFA 已解决宽义关联。
3. Feature-conditioned dynamic kernels：承认 DetectoRS/FADC/KernelWarehouse/OverLoCK/FDConv/AKCMamba-YOLO 已解决输入相关动态性。

然后把 gap 写成：这些线尚未在所检索文献中形成“**aggregated detector assignment distribution explicitly supervises a deploy-static reparameterized RF mixture**”这一特定接口。

### 9.2 推荐避免的术语

- 少用或不用 “dynamic kernel”；SARC 部署时和单次 forward 内均不是 input-dynamic。
- 少用无修饰的 “assignment-aware receptive field”；该概念已有先例，且当前 actual RF 尚未证明。
- 可用 “assignment-distribution-calibrated branch allocation” 或 “training-assignment-conditioned static RF allocation”。
- 若 frozen-hist 与 online EMA 持平，进一步改为 “assignment-derived scale-prior regularization”。

---

## 10. 投稿前阻断项与最终建议

### 阻断项

1. 获取并逐式阅读 IEEE TII 2024 DRFA 全文，核查其 RF branch/gating source、是否使用 task-aligned positives、是否有统计反馈或 reparameterization。
2. 预注册并运行至少 `R3-Permute` 与 `R3-GTOnly`；uniform 不能替代二者。
3. 用 folded-kernel energy/output RMS/ERF 证明 softmax logits 对应实际 RF allocation。
4. 在至少一个与 VOC 尺度分布不同的数据集或 scale-shift intervention 上复现，使用多 seed。
5. related work 明确承认 UniRepLKNet 和 TridentNet，不把结构组合包装成首次提出。

### 最终建议

**继续做，但以窄贡献推进，不要把 SARC 当成已确认的新模块。** 当前检索结果对 exact mechanism 是有利的：未找到 per-level TAL histogram/EMA 直接校准可折叠 dilation mixing 的同构工作；同时，它也对宽主张非常不利：结构、尺度/RF 联结、训练统计反馈和动态 FPN 均已有强先例。

最有价值的下一步不是再增加结构复杂度，而是做强因果反证。若正确 target 能稳定击败同强度 permutation、GT-only 和 uniform，并且 actual RF contribution 随 target 改变，SARC 才具有可投稿的机制证据。否则应尽早止损或把贡献降级为一个轻量正则化技巧。

### 是否值得投入 R1/R2/R3 正式训练？

**值得投入一轮严格冻结、预算有上限的 R1/R2/R3 验伪训练；不值得在只有 R0–R3、没有语义反事实预算的情况下直接进入大规模扫参。** 理由是 exact mechanism 尚有可防守空位，部署增量也在当前预算内，足以支持一次受控验证；但结构本身不新，R3 即使胜过 R2 也尚未证明 assignment 机制。

建议资源决策顺序：

1. 先完成真实集成、DDP/导出等既定阻断检查，再按冻结协议运行 R1→R2→R3，不根据中途 test 结果改 centers、temperature、gain 或 gamma。
2. 若 R3 不能稳定胜过 R2，立即停止 SARC 主线；无需用更大 sweep 挽救新颖性。
3. 若 R3 胜过 R2，只把它视为进入机制验证的资格，随后必须运行 R3-Permute 和 R3-GTOnly；计算预算应在启动 R1–R3 前一并预留。
4. 只有 R3 同时胜过 R2、same-strength permutation 和 GT-only，并在 actual RF diagnostics 上成立，才值得扩展到多 seed、COCO/scale-shift 和论文级完整实验。

---

## 11. 一手来源索引（按发表状态）

### 2024–2026 正式顶会/正式 proceedings

- [UniRepLKNet — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html)
- [Frequency-Adaptive Dilated Convolution — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html)
- [Poly Kernel Inception Network — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)
- [Kernel Adaptive Convolution for Scene Text Detection — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Kernel_Adaptive_Convolution_for_Scene_Text_Detection_via_Distance_Map_CVPR_2024_paper.html)
- [KernelWarehouse — ICML 2024 / PMLR](https://proceedings.mlr.press/v235/li24cg.html)
- [Rethinking Features-Fused-Pyramid-Neck — ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8386_ECCV_2024_paper.php)
- [OverLoCK — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lou_OverLoCK_An_Overview-first-Look-Closely-next_ConvNet_with_Context-Mixing_Dynamic_Kernels_CVPR_2025_paper.html)
- [Frequency Dynamic Convolution — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.pdf)
- [ShiftwiseConv — CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Li_ShiftwiseConv_Small_Convolutional_Kernel_with_Large_Kernel_Effect_CVPR_2025_paper.pdf)
- [UniConvNet — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_UniConvNet_Expanding_Effective_Receptive_Field_while_Maintaining_Asymptotically_Gaussian_Distribution_ICCV_2025_paper.html)
- [PaQ-DETR — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html)
- [AKCMamba-YOLO — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_AKCMamba-YOLO_Selective_State_Space_Models_For_Real-Time_Object_Detection_CVPR_2026_paper.pdf)

### 2024–2026 正式期刊/其他正式 venue

- [An Efficient Anchor-Free Defect Detector With Dynamic Receptive Field and Task Alignment — IEEE TII 2024](https://ieeexplore.ieee.org/document/10472785/)
- [ECF-YOLOv7-Tiny — WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Bacea_ECF-YOLOv7-Tiny_Improving_Feature_Fusion_and_the_Receptive_Field_for_Lightweight_WACV_2025_paper.html)
- [SPAR-Det — WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Kwon_SPAR-Det_Segmentation-guided_and_Prior-Aided_Routing_for_Small_Object_Detection_WACV_2026_paper.pdf)

### 2024–2026 预印本（未与正式发表混写）

- [RFAssigner — arXiv 2601.01240](https://arxiv.org/abs/2601.01240)
- [GlottisNet — arXiv 2604.27383](https://arxiv.org/abs/2604.27383)：多 RF 与 advanced label assignment 并列出现，但摘要未显示 assignment→kernel 直接反馈，仅作低优先级邻近工作。
- [SCAResNet — arXiv 2404.04179](https://arxiv.org/abs/2404.04179)：使用 RFLA 体系并改 detector architecture，未见 direct coupling。

### 不可绕开的早期正式先例

- [Scale-Aware Trident Networks — ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.html)
- [SEPC — CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Scale-Equalizing_Pyramid_Convolution_for_Object_Detection_CVPR_2020_paper.html)
- [Dynamic R-CNN — ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2335_ECCV_2020_paper.php)
- [DetectoRS — CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.pdf)
- [RFLA — ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)
- [Dynamic Sparse R-CNN — CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Dynamic_Sparse_R-CNN_CVPR_2022_paper.html)
- [Dynamic Coarse-to-Fine Learning — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.pdf)
- [Large Selective Kernel Network — ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)
