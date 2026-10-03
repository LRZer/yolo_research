# YOLO26n/VOC 广域创新筛选：从 Head 之外寻找下一条可证伪主线

日期：2026-09-19  
定位：TQGB 若 no-go 后的预注册方向筛选；不是结果报告，也不是同时开跑三个方向的授权。  
范围：只读审计本地研究代码、现有实验记录、远端基线结构快照与一手论文；未修改正在运行的远端源码。

## 0. 结论先行

建议把下一条主线定为：

> **PBGA：Pyramid-Boundary Gradient Arbitration，金字塔边界梯度仲裁。** 训练时暂时解锁 O2O 部署头对 P3/P4/P5 共享特征的梯度，但只允许与 O2M 表征梯度不冲突的分量进入 Backbone/Neck；两个检测头仍各自接受原生监督，推理图完全不变。

这不是把 PCGrad 生搬到 YOLO，也不是继续做预测蒸馏。它针对当前代码中一个明确的不对称：O2O 是真正部署的分支，却在训练时接收 `detach()` 后的 Neck 特征，因此 Backbone/Neck 的更新由 O2M 分配主导。已有 QGAT/S-QGAT 结果说明“固定让 O2M 预测教 O2O”没有净收益，但不能推出“O2O 的任务信号不该影响共享特征”。PBGA 研究的是后一件事，并在三个金字塔边界上做非冲突的一阶梯度路由。

三个进入候选池的方向按优先级为：

1. **PBGA：分配非对称的金字塔边界梯度仲裁**（优化/共享表征，首选）；
2. **SARC：Assignment-Calibrated Re-parameterized Context，分配校准的可重参数化感受野**（Backbone/Neck/卷积）；
3. **ASAC：Assignment-Stability Augmentation Curriculum，分配稳定性驱动的增强课程**（增强/优化）。

三者都不是“堆 CBAM、换 IoU loss、加 P2”式组合。每个候选都有一个单独的机制假设、训练侧先验或探针、匹配成本的控制组，以及固定第 50 轮的 no-go 门槛。

## 1. 当前问题画像：哪些事实已经确定

### 1.1 模型与训练协议

- 基线是 YOLO26n，`2,378,736` 参数、`5.3 GFLOPs`；Backbone 为 C3k2 + SPPF + C2PSA，Neck 为标准 top-down/bottom-up FPN-PAN，最终 P3/P4/P5 通道为 `64/128/256`。
- 检测头为原生 O2M/O2O 双头；O2O 是 NMS-free 部署路径。
- 当前源码在训练时对送入 O2O head 的 P3/P4/P5 特征执行 `detach()`，所以 O2O 原生损失只更新自己的 head；共享 Backbone/Neck 由 O2M 原生损失更新。
- ProgLoss 只按 epoch 把权重从 O2M 偏重逐渐转到 O2O 偏重：初值约 `0.8/0.2`，末期约 `0.1/0.9`。但后期 O2O 权重再高也不能直接训练共享特征，因为特征已被 detach。
- YOLO26n 的 `reg_max=1`，是 direct box regression，没有 DFL 分布回归。
- 当前统一短训为随机初始化、VOC07+12 trainval、50 epochs、batch 64、GPU 4–7、workers=4、pinned memory、固定 seed 42、固定末轮 O2O NMS-free 比较。

上述双头、DFL-free、ProgLoss/STAL 设计与官方论文一致；`detach()` 和具体权重更新则由当前锁定源码直接核验。[YOLO26 官方论文（2026）](https://arxiv.org/abs/2606.03748)

### 1.2 已有实验把哪些直觉排除了

- B50 固定第 50 轮 O2O mAP50-95 为 `0.51276`；150 轮基线 O2O/O2M 为 `0.57689/0.58705`，部署头长期仍落后辅助头约 1.0 AP。
- 原 D2 的 `+0.00308` 在梯度强度匹配后变为 `+0.00001`；S-QGAT 为 `-0.00101`。因此不能继续把“可靠 O2M 分类预测传给 O2O”当成已成立机制。
- VOC07+12 trainval 在 640 静态缩放后，COCO-small 只占 `2.724%`，STAL `<16 px` 触发只占 `0.7339%`；静态 P2、小目标专用分配不适合作为当前 VOC 主线。
- VOC 目标以大目标为主（`72.922%`），但这不支持粗暴加大所有卷积核；更合理的是先测量不同检测层实际分配到的目标尺度/上下文需求，再决定感受野如何分配。
- 当前训练有全局 `max_norm=10` 梯度裁剪，既往所有探针批次均触发。任何新增梯度都必须记录裁剪前范数、裁剪比例和分组更新，否则很容易把统一缩放产生的间接耦合误认成方法效果。

## 2. 广域筛选：哪些方向保留，哪些暂时排除

| 范畴 | 当前真实接口或问题 | 最近先例与拥挤度 | 本轮裁决 |
|---|---|---|---|
| Backbone | n 模型每个 C3k2 stage 实际重复很浅；容量有限，但没有证据表明缺某一种成熟模块 | C3/C2f/Rep/大核/轻量注意力替换已极度拥挤 | 不做普通模块替换；只保留“由实际分配尺度校准、可部署重参数化”的 SARC |
| Neck / 多尺度融合 | nearest upsample + pointwise concat 可能产生层级/空间失配；FPN-PAN 始终静态融合 | ECCV 2024 已直接研究 features-fused pyramid 的层级失配；DySample 已覆盖轻量动态上采样 | 不直接换 DySample/CARAFE/AFPN；避免只复现已有 neck。SARC 从“每层需要多大上下文”切入，另设 fixed-cost 控制 |
| 卷积 / 注意力 | P5 有 C2PSA，P3/P4 主要是局部 C3k2；不同层的上下文预算没有由数据校准 | UniRepLKNet、PKINet 已覆盖大核、膨胀重参数化与多核卷积 | 简单加大核/EMA/CBAM/ECA 为 no-go；SARC 必须证明 assignment calibration 超过同成本 7×7 DWConv |
| Label assignment | O2M top-10、O2O top-7→top-1，STAL 对本 VOC 的特殊小目标覆盖很低 | One-to-Few、consistent dual assignment、Dual Weighting/TAL 已覆盖软正样本与质量分配 | 暂不手调 top-k、alpha/beta 或 STAL 阈值；这些既拥挤又容易被 VOC test 反复调参 |
| Loss / 排序 | AP 依赖分类分数与定位质量对齐，direct regression 也缺少分布不确定性 | GFL、aLRP、Rank-DETR、Varifocal/Dual Weighting 已形成成熟路线；恢复 DFL/普通 IoU loss 新意不足 | 不把“换一个 loss”列为前三；可作为最终方法的对照，不独立包装创新 |
| Augmentation | 当前 `mosaic=1.0`，最后 10 轮固定关闭；HSV/scale/translate 同样固定，不观察模型是否还能保持分配 | MADAug 已做模型自适应课程，ReCon 等已做检测专用生成增强 | 保留 ASAC，但必须用“GT 对应下的 assignment stability”而不是验证集驱动策略网；优先级第三 |
| Optimization / 双分配 | 部署 O2O head 的特征被 detach；共享表征只由 O2M 直接训练，ProgLoss 的 O2O 后期高权重不能改变这一点 | PCGrad/ConsMTL 处理一般多任务冲突，但没有针对原生 O2M/O2O、部署头被隔离、P3/P4/P5 边界的非对称仲裁 | **保留 PBGA，优先级第一** |

文献边界很重要：Rethinking FPN 已在 VOC/COCO 上提出层级失配与二次对齐，[DySample](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html) 已把动态上采样做得很轻；[UniRepLKNet](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large_Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html) 和 [PKINet](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html) 已分别覆盖膨胀重参数化和多核上下文。因此单独换这些成熟组件不足以成为论文主创新。

## 3. 方向一（首选）：PBGA，金字塔边界梯度仲裁

### 3.1 问题诊断

当前双头训练有一个容易被 ProgLoss 权重掩盖的结构事实：

```text
Backbone/Neck ── P3/P4/P5 ── O2M head ── L_o2m
                    └─ detach ─ O2O head ── L_o2o  (部署头)
```

所以共享特征被优化成“适合 one-to-many 分配”的表示，部署用 O2O head 只能适配一份由别的分配规则塑造且不断变化的特征。One-to-Few 论文已经指出，纯 one-to-one 的正样本过少会削弱表征学习；YOLO26 用 O2M 辅助头解决了覆盖问题，但完全 detach 又把部署目标从共享表征优化中隔离。[One-to-Few，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_One-to-Few_Label_Assignment_for_End-to-End_Dense_Detection_CVPR_2023_paper.html)

已有 TQGB 是“预测目标怎么在两个 head 之间传”；PBGA 是“两个原生监督对共享特征的梯度怎么合成”。两者研究对象不同，TQGB 失败并不否定 PBGA。

### 3.2 核心数学

对每个尺度 `s ∈ {P3,P4,P5}`，令共享特征为 `F_s`，带 ProgLoss 原生权重的两个分支损失为：

```text
L_m = w_m(e) · L_O2M
L_o = w_o(e) · L_O2O
g_m^s = ∂L_m / ∂F_s
g_o^s = ∂L_o / ∂F_s
```

O2M 继续作为表征锚点；只投影 O2O 的冲突分量：

```text
if <g_o^s, g_m^s> < 0:
    g_o,⊥^s = g_o^s - <g_o^s,g_m^s> / (||g_m^s||² + ε) · g_m^s
else:
    g_o,⊥^s = g_o^s

c_s = min(1, ||g_m^s|| / (||g_o,⊥^s|| + ε))
g_shared^s = g_m^s + c_s · g_o,⊥^s
```

最后用一阶 surrogate 把 `g_shared^s` 注入原始共享图：

```text
L_sur = Σ_s <stopgrad(g_shared^s), F_s>
```

两个 head 自身仍由 `L_m`、`L_o` 的原生损失更新；只在共享特征边界替换梯度。投影后有 `⟨g_{o,⊥}^s, g_m^s⟩ ≥ 0`（数值误差除外），范数帽防止训练后期 `w_o` 很大时 O2O 梯度压倒 O2M 表征梯度。这里没有通过 VOC test 扫 gain；ProgLoss 权重仍保留，强度由同 batch 梯度范数自动界定。

### 3.3 与最近工作的差异

- [PCGrad，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 是一般多任务参数梯度投影；PBGA 是同一检测任务的两种 label assignment，在三个 FPN 边界做**非对称、部署分支解锁**，不对所有参数做昂贵 surgery。
- [ConsMTL，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Ta[REDACTED_CREDENTIAL].html) 同时优化 shared/task-specific parameters 的一致性，是通用多任务双层优化；PBGA 不引入双层训练，也不改变 head 专属参数的原生梯度。
- [YOLOv10，NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf) 和 YOLO26 使用双分配解决 NMS-free 训练，但没有把“被 detach 的部署头如何安全参与 shared pyramid 优化”作为显式梯度问题。
- 因此安全表述应是“assignment-asymmetric gradient arbitration at pyramid boundaries”，不能声称首次梯度冲突处理或首次双头协同。

### 3.4 实现路径与风险

最稳妥的实现不是直接删除 `detach()` 后让两项损失一起 backward，而是：

1. 在 Detect 训练 forward 中保留原始 `F_s`，为两个 head 分别创建数值相同但叶节点独立的 `F_m^s/F_o^s = detach(F_s).requires_grad_(True)`；
2. 用原生 head 得到两套预测与损失；
3. `autograd.grad` 只求三张边界特征的 `g_m/g_o`，不对全模型逐参数求两遍梯度；
4. 返回原生 head loss（更新 head 参数）加 `L_sur`（更新共享 Backbone/Neck）；
5. DDP 前先做单卡数值等价/路由测试，再做 4 卡梯度等价和 BF16 empty-target smoke。

推理参数、FLOPs、延迟均为 **0 增量**。训练代价主要是两组边界梯度：预计显存增加约 10%–30%，墙钟增加约 25%–70%；正式实现后必须实测，不能把估计写成结果。最大工程风险是 autograd/DDP 的双计算图与 AMP；最大科学风险是 O2M/O2O 在共享特征上其实大多同向，此时复杂仲裁没有必要。

### 3.5 先做零更新探针

用 B50 的 epoch 11/21/31/41/50 checkpoints、固定 8 个 train batches：

- 分 P3/P4/P5 记录 cosine、`||g_o||/||g_m||`、冲突率和投影删除比例；
- 按 cls/box 分开，再记录合并后的总分支梯度；
- 模拟投影前后 global clip scale，防止收益只来自裁剪系数变化；
- 不更新权重、不读 test 指标调方法。

预先 no-go：五个 checkpoint 的三个尺度上，O2O 总梯度冲突率均 `<5%`，或 `||g_o||/||g_m||` 的中位数均 `<0.05`。若至少一个尺度跨两个以上 checkpoint 冲突率 `≥10%` 且 O2O 梯度非退化，进入训练。

### 3.6 最小 50 轮因果矩阵

| ID | 共享特征梯度 | 回答的问题 |
|---|---|---|
| G0 | 官方 detach | 现有 B50；新代码 gain=0 必须逐值等价 |
| G1 | naive reattach：`g_m + g_o` | 单纯让部署头训练共享特征是否已足够 |
| G2 | norm-cap only：`g_m + c·g_o` | 收益是否只来自控制 O2O 强度 |
| G3 | **PBGA：投影 + norm-cap** | 非冲突方向本身是否贡献 |

先跑 G1；若严重发散仍完整保留日志，但可直接进入 G2/G3。G3 只有同时满足以下条件才进入多 seed：

- 固定 epoch 50 相对 G0 `ΔmAP50-95 ≥ +0.003`；
- last-5 均值相对 G0 `≥ +0.002`；
- 相对 G2 至少 `+0.002`，否则只能说“范数控制有效”，不能说梯度仲裁有效；
- O2O AP 提升时 O2M 不得下降超过 `0.003`，且没有新增 NaN/SIGBUS；
- 若仅单个末轮点为正、last-5 不稳，判 no-go，不扫投影阈值。

## 4. 方向二：SARC，分配校准的可重参数化感受野

### 4.1 问题诊断

YOLO26n 的最终 P3/P4/P5 只有 `64/128/256` 通道，n 缩放使多数 C3k2 实际只重复一次；P5 有注意力，P3/P4 主要依赖局部卷积。直接把全部模块换成大核或 PKI 是成熟模块移植，而且不能解释哪一层需要哪种上下文。

SARC 的问题定义是：**能否用训练时真实 label assignment 给出的“目标尺寸相对 stride 的上下文需求”，校准各检测层的可重参数化卷积分支，并在部署时折叠为固定的小预算卷积？**

### 4.2 核心结构

在最终 Neck 的 P3/P4/P5、进入 Detect 之前各加一个残差深度卷积块。训练态包含一个 7×7 DWConv 主分支及 `3×3,d={1,2,3}` 三个线性 DWConv+BN 分支：

```text
R_s(F) = DW7_s(F) + Σ_r softmax(a_s)_r · DW3_{s,r,d_r}(F)
F'_s = F_s + γ_s · SiLU(BN(R_s(F_s)))
```

对每个正样本，计算相对尺度：

```text
u_j = log2(sqrt(w_j h_j) / stride_s)
```

把 `u_j` 投到与 dilation 分支对应的三个软区间，得到该层训练期 assignment histogram `h_s`，用 EMA 汇总；加入很小的校准项：

```text
L_cal = Σ_s KL(stopgrad(h_s) || softmax(a_s))
```

训练结束后，标量、BN、膨胀 3×3 核全部等价嵌入并相加到一个 7×7 depthwise kernel。最终推理没有并行分支和动态控制流。

### 4.3 与最近工作的差异

- [UniRepLKNet，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large_Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html) 用膨胀小核重参数化大核，证明了等价折叠机制；SARC 不能把“可重参数化大核”当新贡献。
- [PKINet，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html) 用多核 inception 和上下文注意力覆盖尺度变化；SARC 不建立通用新 backbone，而把真实检测分配统计变成各 FPN 层的分支先验。
- [Rethinking Features-Fused-Pyramid-Neck，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8386_ECCV_2024_paper.php) 直接处理层级融合失配，并在 VOC/COCO 验证；SARC 不声称首次 FPN 对齐，而是研究 assignment-calibrated receptive-field allocation。

### 4.4 参数、FLOPs 与风险

按 `P3=80²×64, P4=40²×128, P5=20²×256` 估算，部署态三层 7×7 depthwise kernel新增：

- 卷积参数约 `(64+128+256)×49 = 21,952`，约为基线参数的 `0.92%`；加 BN/标量仍约 `<1.0%`；
- 约 `35.1M MACs`，若按乘加两次计约 `0.070 GFLOPs`，相对 5.3 GFLOPs 约 `1.3%`；
- 训练态并行小核会增加约 12k 可折叠参数和额外计算，但部署时消失。

风险是 `h_s` 可能只是数据集尺度分布，不能真正指导 feature context；另一个风险是 7×7 DWConv 本身就解释全部收益。因此同成本控制不是可选项，而是核心因果消融。

### 4.5 探针与 50 轮矩阵

先在 B50 checkpoints 上统计 P3/P4/P5 的 assigned GT 数、`u_j` 直方图、跨 epoch Jensen-Shannon 距离。若三个层的直方图几乎相同（任意两层 JS `<0.02`）或随 epoch 完全不稳定（连续 checkpoint JS `>0.3`），assignment calibration 缺少可利用结构，直接 no-go。

| ID | 改动 | 因果问题 |
|---|---|---|
| R0 | 官方 B50 | 基线 |
| R1 | 每层单个 7×7 DWConv，匹配部署参数/FLOPs | 普通大一点的感受野是否足够 |
| R2 | 多分支可重参数化，但无 `L_cal` | 训练期 over-parameterization 是否解释收益 |
| R3 | **SARC 完整版** | assignment calibration 是否额外有效 |

R3 继续标准：epoch 50 相对 R0 `≥+0.003`、last-5 `≥+0.002`，并且相对 R2 `≥+0.002`；部署参数增幅 `<1.2%`、GFLOPs 增幅 `<2%`。若 R1/R2 与 R3 相当，则只能保留为工程改进，不能以“分配校准”投稿。

## 5. 方向三：ASAC，分配稳定性驱动的增强课程

### 5.1 问题诊断

当前从头训练采用固定增强：`mosaic=1.0`、`scale=0.5`、`translate=0.1`，仅在最后 10 轮硬关闭 mosaic。固定日程不知道当前模型在强几何变化下是否仍能为同一 GT 产生稳定候选；过强会把 assignment 噪声当监督，过弱又浪费小数据上的正则化机会。

已有 [MADAug，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hou_When_to_Learn_What_Model-Adaptive_Data_Augmentation_Curriculum_ICCV_2023_paper.html) 用验证集双层优化学习分类增强策略；[ReCon，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c83f1f1290e80236587e4c89fa24f4e-Abstract-Conference.html) 研究生成式检测增强的区域语义/位置对齐。ASAC 不学习一个大型 policy network，也不生成图像；它只用训练 GT 对应下的 detection assignment stability 控制现有几何/mosaic 强度。

### 5.2 核心机制

每个 epoch 在固定的 train-only panel（建议 256 图）上，以 no-grad 生成 weak/strong 两个有可逆几何记录的视图。对同一 GT `j`，把 TAL 的 top-k alignment metrics 归一为候选分布 `π_j^w, π_j^s`，映射回 GT 归一化坐标后计算：

```text
S_e = mean_j exp(-JS(π_j^w || π_j^s))
```

用单个课程状态 `q_e∈[0,1]` 同时控制 mosaic probability、scale range 和 translate range：

```text
q_{e+1} = clip(q_e · exp(η(S_e - S_ref)), q_min, q_max)
```

`S_ref` 只在 warmup 结束时由 train panel 的中等强度视图冻结；之后不读 VOC07 test，不重设阈值。稳定度高说明当前模型还能承受更强增强，稳定度低则降低扰动。最后 5 轮统一线性到零，保证所有方法以干净分布收尾；这个 terminal cooldown 在所有 ASAC 控制组中保持一致。

### 5.3 成本与风险

- 推理参数/FLOPs：0；训练参数：0。
- 只在 epoch 末做 256 图双视图 no-grad panel，预估训练墙钟增加 `<10%`，需实测。
- 主要工程风险是 Mosaic 的四图坐标对应和 DDP 各 rank 状态一致；最小版本可以先只控制 scale/translate，证明反馈有效后再纳入 mosaic。
- 主要科学风险是 stability 仅反映模型置信度/训练进度，最终退化成普通 epoch schedule。必须用“平均强度匹配的固定 schedule”控制拆开反馈价值。

### 5.4 探针与 50 轮矩阵

先用 B50 五个 checkpoints、三个预定强度 `q={0.25,0.5,0.75}` 做无更新 panel。只有当 `S` 随强度大体单调下降，且 object-level stability 与对应监督损失/quality 有稳定相关（至少两个 checkpoint 的 `|Spearman ρ|≥0.2`）才实施控制器。

| ID | 增强策略 | 因果问题 |
|---|---|---|
| A0 | 官方固定增强 + close_mosaic=10 | 基线 |
| A1 | 固定 schedule，逐 epoch 平均强度匹配 A2 | “用了不同强度曲线”本身是否解释收益 |
| A2 | **ASAC 反馈控制** | assignment stability 反馈是否贡献 |
| A3（仅 A2 为正时） | 打乱 epoch stability 信号，但强度边际分布匹配 | 真实时序反馈还是随机正则化 |

A2 继续标准：epoch 50 相对 A0 `≥+0.003`，相对 A1 `≥+0.002`，last-5 同号，训练开销 `<12%`；若只比 A0 好但不优于 A1，论文机制不成立。

## 6. 三方向综合评分

| 方向 | 机制针对性 | 新颖性空间 | 首轮实现成本 | 推理开销 | VOC 50 轮可证伪性 | 主要风险 |
|---|---:|---:|---:|---:|---:|---|
| **PBGA** | 5/5 | 4/5（组合层面） | 中高 | 0 | 5/5 | autograd/DDP；冲突可能稀少 |
| **SARC** | 4/5 | 3.5/5 | 中 | 约 +1.3% FLOPs | 4/5 | 普通 7×7 就解释收益；校准无增益 |
| **ASAC** | 3.5/5 | 3.5/5 | 中 | 0 | 4/5 | 反馈退化成普通 epoch schedule |

没有把定位分布辅助头列入前三：YOLO26 去 DFL 是明确的效率设计，而训练期 distribution teacher / DLD 已有强先例；若只是恢复 DFL 再蒸馏 direct head，容易成为 YOLOv6 DLD 风格的移植。没有把 score-IoU ranking loss 列入前三：aLRP、GFL、Dual Weighting、Rank-DETR 等已经把质量排序/对齐研究得很充分，除非先找到 YOLO26 特有的新失效模式，否则论文边界弱。[aLRP，NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b2eeb7362ef83deff5c7813a67e14f0a-Abstract.html)；[Rank-DETR，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)

## 7. 若 TQGB no-go，冻结的执行顺序

1. **先做 PBGA 零更新冲突探针**；这是最便宜、信息量最大的裁决。若 conflict/gradient coverage 过低，当天停止，不写训练代码。
2. 探针通过则实现 PBGA 单卡合成测试、真模型单卡测试、4 卡 BF16 3-epoch smoke；必须证明 PBGA-off 与官方 loss/参数梯度等价。
3. 顺序跑 G1→G2→G3，固定 50 轮；不根据 VOC test 曲线扫阈值、尺度或 gain。
4. G3 通过门槛后做 seeds `{0,1}` 的 B50/G3 成对补齐；至少 2/3 paired deltas 为正、均值 `≥+0.003`，再进入健康硬件 150 轮与 COCO。
5. PBGA 探针或训练 no-go 后，立即转 SARC，不在 PBGA 上做微小参数搜索。
6. SARC 若 full 不胜同成本 R2，则停止“assignment calibration”叙事；只有单纯 7×7 有效时把它记为工程基线，不包装论文创新。
7. ASAC 为第三条独立路线；只有前两条 no-go 或需要数据侧补充时投入。

所有探索继续标记为 `faulty-host exploratory`；GPU 仅用 4–7、workers=4、pinned memory、NUMA1。任何新增 MCE/SIGBUS 都使对应运行无效，必须从头重跑；论文结论仍需健康硬件复现。

## 8. 可直接落地的首个工作包（PBGA-P0）

为了避免“先写复杂代码再发现没有信号”，首个工作包只包含：

1. 从 B50 checkpoint 取固定 8 个 train batches；
2. 在 Detect 外部或研究副本中重算 non-detached O2O feature gradients，不改远端当前训练源码；
3. 输出 `checkpoint × P3/P4/P5 × {cls,box,total}` 的 cosine、norm ratio、negative-dot rate、projection removal、模拟 clip scale；
4. 固化 JSON/CSV/Markdown 与 checkpoint/hash；
5. 按 3.5 节门槛自动给出 go/no-go。

若 P0 为 go，PBGA 是目前最值得优先实现的广域方向；若 P0 为 no-go，直接进入 SARC 的 assignment histogram probe。这个顺序让每一步都能用训练侧证据推翻自己，而不是靠反复跑 VOC test 猜模块。

## 9. 核验的一手来源

1. [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models（官方，2026）](https://arxiv.org/abs/2606.03748)
2. [YOLOv10: Real-Time End-to-End Object Detection（NeurIPS 2024）](https://papers.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)
3. [One-to-Few Label Assignment for End-to-End Dense Detection（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Li_One-to-Few_Label_Assignment_for_End-to-End_Dense_Detection_CVPR_2023_paper.html)
4. [Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
5. [Towards Consistent Multi-Task Learning（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Ta[REDACTED_CREDENTIAL].html)
6. [Rethinking Features-Fused-Pyramid-Neck for Object Detection（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8386_ECCV_2024_paper.php)
7. [Learning to Upsample by Learning to Sample / DySample（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html)
8. [UniRepLKNet（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large_Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html)
9. [Poly Kernel Inception Network（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)
10. [When to Learn What: Model-Adaptive Data Augmentation Curriculum（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Hou_When_to_Learn_What_Model-Adaptive_Data_Augmentation_Curriculum_ICCV_2023_paper.html)
11. [ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c83f1f1290e80236587e4c89fa24f4e-Abstract-Conference.html)
12. [Rank-DETR for High Quality Object Detection（NeurIPS 2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)
13. [A Ranking-based, Balanced Loss Function Unifying Classification and Localisation（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b2eeb7362ef83deff5c7813a67e14f0a-Abstract.html)
