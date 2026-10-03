# YOLO26n/VOC 下一波大胆创新筛选

日期：2026-09-19  
状态：**候选机制预注册；未修改远端、未使用 GPU、未授权正式训练**

## 0. 结论先行

本轮不再延伸 TQGB/SARC/ASAC，也不把 CBAM、动态卷积、换 IoU loss 或普通 FPN 模块包装成新方向。保留五个彼此独立、可在 10 epoch 内证伪的候选：

1. **IGB：Instance-Quota Gradient Budgeting，实例配额梯度预算**（loss/optimization）
2. **SPAR：Subpixel Phase-Alias Recovery，亚像素相位细节恢复**（backbone/downsampling）
3. **CSPC：Cross-Scale Predictive Coding，跨尺度预测编码**（neck/cross-scale）
4. **BECM：Box-Evidence Counterfactual Masking，盒证据反事实遮蔽**（augmentation）
5. **FORC：Forgetting-Organized Replay Curriculum，遗忘事件组织的重放课程**（curriculum）

按“新颖性 × 可实施性 × 预期信号”排序后，首选两条是：

> **首选 A：IGB。** 它先问一个很具体的问题：原生 O2M/TAL 是否让少数拥有较大 positive target mass 的 GT 支配 shared feature gradient？若零更新探针成立，只需改正样本归一化，不改 assignment、不跨 head 传知识、推理零开销。

> **首选 B：SPAR。** 它不增加推理模块，而用训练期 Haar 相位细节的可解码性监督现有 stride-2 downsamplers，专门检验小目标/边界信息是否在 P3/P4/P5 形成前被不可逆混叠。

CSPC 排第三：概念更大胆且可精确折叠回原 neck，但“可预测冗余”尚无本地证据，fold 约束也比前两项更容易踩工程坑。BECM 和 FORC 保留为数据侧候选，不与 ASAC 混合运行。

## 1. 当前研究边界与排重

### 1.1 已知本地事实

- YOLO26n backbone 通过 stride-2 convolutions 逐级降采样；neck 主要是 nearest upsample、pointwise concat 和 C3k2，没有证据说明某个普通注意力块是瓶颈。
- 原生 O2M loss 以全 batch 的 `target_scores_sum` 归一化；每个 GT 的 positive 数量与 target-score mass 并不相等。
- SARC 探针中 positive-anchor pooled 的层间分离明显大于 unique-GT 版本，说明 positive multiplicity 会实质改变训练统计，但这还没有证明它造成 per-GT gradient domination。
- PBGA 已因 total-gradient conflict coverage 为零而 no-go；ANOR 研究 O2O shared-feature reattachment 的幅值控制。因此本轮不再做 O2O/O2M gradient surgery。

### 1.2 与既有路线的硬排重

| 本轮候选 | 不是什么 | 与已有路线的分界 |
|---|---|---|
| IGB | 不是 KD、不是新 assigner、不是 PBGA/ANOR | 只重分配原生 O2M 内不同 GT 的正样本预算；O2O 与 shared routing 不改 |
| SPAR | 不是 SARC、大核、注意力或动态卷积 | 约束现有 stride-2 表征保留目标相位细节；训练辅助头删除，推理图不变 |
| CSPC | 不是 AFPN/BiFPN/动态上采样堆叠 | 把相邻层分成“可预测 coarse content + innovation residual”，线性变换最终折入原 C3k2 首层 |
| BECM | 不是 ASAC 的 assignment-stability feedback | 使用固定 box counterfactual，监督“物体仍在/物体被移除”的证据顺序；不调 augmentation 强度 |
| FORC | 不是 ASAC、OHEM 或 validation-driven scheduler | 根据同一 GT 的时间遗忘事件改变未来曝光概率；augmentation policy 不变 |

任何实现若需要同时打开 TQGB、SARC、ASAC、PBGA/ANOR 或双头 KD，均视为偏离本筛选定义。

## 2. 所有 10-epoch screen 的共同协议

### 2.1 数据与评价隔离

1. 从 VOC trainval 按图像 ID 的固定 SHA256 分桶冻结一个 `screen-val-v1`；类别和 box-area 四分位分层，之后所有想法共享这一份划分。
2. screen 不读取 VOC07 test；只有候选通过 screen、机制控制和第二 seed 后，才允许进入现有标准 50-epoch VOC 协议。
3. 指标只读固定 epoch 10 与 epoch 9–10 均值，不挑 best epoch。

### 2.2 最小因果三元组

每个想法先用 seed 42 跑：

- `S0`：当前真实 R0，全部研究开关关闭；
- `S1`：同计算量/同权重分布但破坏语义对应的 placebo；
- `S2`：完整机制。

只有 S2 同时越过 S0 与 S1 的预注册门槛，才补 seed 0 的 `S0/S2` 配对确认。seed 0 必须同号，两 seed 平均增益至少 `+0.0025` absolute mAP50-95，才进入 50 epoch。

### 2.3 禁止小型 sweep

- 任何辅助 loss gain 只在固定 train calibration batches 上按 gradient RMS 匹配：初始 auxiliary/shared gradient RMS 为 base detection gradient RMS 的 `5%`，随后冻结。
- 每个候选只允许一个定义、一个 gain、一个 clip 范围；screen 失败后不读 screen-val 反调参数。
- 所有 off 模式必须逐张量证明 graph、loss、gradient 与 R0 等价。
- 10 epoch 结果必须同时报告 overall AP、AP50、AP75、box-area 四分位 recall、20 类最差五类 AP、训练 wall-time/peak memory 和候选自身机制统计。

## 3. 候选一：IGB——实例配额梯度预算

### 3.1 机制假设

TAL 为 GT `g` 分配的正锚集合记为 `A_g`，assigned-class target quality 为 `q_a`。当前每个 GT 的正监督质量总量

```text
m_g = sum_{a in A_g} q_a
```

会随对象尺寸、可匹配位置和当前预测质量变化。全 batch 只除以 `sum_g m_g`，因此拥有较大 `m_g` 的少数 GT 可能支配 O2M shared-feature gradient；这与“难例更重要”不是同一件事，因为大 mass 可以来自重复正锚而非更高信息量。

IGB 不改 TAL 选择，只给每个 GT 一个 detached、保守的预算因子：

```text
m_bar = mean_g(m_g)
w_g_raw = clip(sqrt(m_bar / (m_g + eps)), 0.5, 2.0)
w_g = w_g_raw * sum_g(m_g) / sum_g(w_g_raw * m_g)
```

它保持本 batch 总 positive target mass 不变。`w_g` 只乘：

- 正锚的 assigned-class BCE 项；
- 同一正锚的 box/L1 项。

背景 anchor BCE、正锚上的非 assigned classes、O2O loss、assignment 和全部预测 target 均保持原样。这样测试的是“每个真实对象得到接近相等的正监督预算”，不是另一种质量标签。

### 3.2 最小实现

- 在 `v8DetectionLoss.get_assigned_targets_and_loss()` 内利用已有 `target_gt_idx/target_scores/fg_mask` 做 `scatter_add`。
- 仅 O2M criterion 启用；O2O 原样。
- 无跨 batch state、无新 parameter、无 DDP collective；记录 per-GT mass、weight、gradient proxy 的 quantiles/Gini。
- 默认 `igb_gain=0` 或 `igb_mode=off`，off 路径在任何额外 reduction 前直接返回原损失。

### 3.3 廉价零更新探针 IGB-P0

在 B50 同一轨迹的 epoch 1/10/25/50 checkpoint 上，用 24 个锁定 train batches、最多 128 个 GT：

1. 对每个 GT 单独保留它的 O2M positive cls+box loss；
2. 求 P3/P4/P5 shared features 的 per-GT gradient norm；
3. 统计 gradient-norm Gini、top-20% GT gradient share、effective number，以及 gradient norm 与 `m_g`、positive count、area、quality 的 Spearman；
4. 全程 eval BN、no optimizer step、no checkpoint write。

**P0 GO 条件必须全部成立：** median-batch Gini `>=0.35`；top-20% GT 占 `>=45%` gradient norm；`rho(grad,m_g)>=0.40`，且至少三个 checkpoint 同号。任一失败就停止 IGB，不写训练实现。

### 3.4 10-epoch screen

| 组 | 定义 |
|---|---|
| I0 | R0 原生归一化 |
| I1 | IGB-PERM：保持同一批 `w_g` 数值和总 mass，但在 GT 间固定循环置换 |
| I2 | 正确 GT→`w_g` 的 IGB |

**继续线：** epoch-10 I2 相对 I0 `>=+0.0030`、相对 I1 `>=+0.0020`；epoch 9–10 均值同号；per-GT gradient Gini 相对 I0 降 `>=20%`；最低 `m_g` 四分位 recall `>=+0.015`，最高四分位 recall 下降不超过 `0.005`；clip 到 0.5/2.0 的 GT 比例均 `<20%`。

### 3.5 推理与训练成本

- 推理：**严格 0**；图、参数、FLOPs、export 均不变。
- 训练：若干 `scatter_add/gather`，预期 `<1%` wall-time；无额外 forward。

### 3.6 2024–2026 最近一手工作与碰撞

- [Bucketed Ranking-based Losses，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7634_ECCV_2024_paper.php) 处理正负不平衡并高效近似 ranking losses；它不在原生 O2M 内按 GT 守恒 positive gradient budget。
- [DEIM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html) 用 Dense O2O 增加监督并以 Matchability-Aware Loss 改善低质量匹配；它改变 supervision density/quality，不均衡同一 O2M 中各 GT 的 mass。
- [SA-Matching DETR，CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Yang_SA-Matching_DETR_A_Lightweight_Transformer_Detector_with_Enhanced_Scale_Adaptive_CVPRF_2026_paper.html) 按 object scale 扩展匹配目标；IGB 不改变匹配数量，只重预算已匹配正样本。

因此不能声称“首个平衡 detection loss”；若成立，只能主张 native O2M 的 **per-instance positive-mass conservation**。正式投稿前仍需更完整查重。

### 3.7 Falsification / stop rule

- P0 没有 mass→gradient domination：当天停止。
- I2 不胜 I1：即便 I2 胜 I0，也只能说明某种重权有 regularization 效果，GT 对应机制不成立。
- 只提高低-mass recall但明显伤害高-mass或整体 AP：停止，不调大 clip。
- gain 来自 cls/box 总 gradient scale 漂移超过 10%：先做 exact norm-matched control；控制后无增益即停止。

## 4. 候选二：SPAR——亚像素相位细节恢复

### 4.1 机制假设

YOLO26n 的 stride-2 convolutions 只观察一个固定采样相位。小物体边界可能在进入 P3/P4/P5 前因相位混叠而丢失；之后再加大核或 neck 模块无法恢复已经不可解码的信息。

对一个 stride-2 层的输入 `x_s` 做固定 Haar 分解，取 `LH/HL/HH` 三个 detail bands：

```text
d_s = Haar_detail(x_s)
t_s = R_s d_s
L_spar = SmoothL1(P_s(y_s), stopgrad(t_s))
```

其中 `y_s` 是现有 stride-2 输出，`R_s` 是冻结的 Rademacher channel sketch，`P_s` 是训练期小 1×1 predictor。监督集中在 GT box 内部和一格 boundary ring；等面积背景区域只约束 predictor 输出趋近零，避免把纹理噪声当成目标细节。训练结束删除 `P_s/R_s`，原 downsampler 不变。

### 4.2 最小实现

- 只挂在形成 P3/P4/P5 的三个 backbone stride-2 Conv，不改 stem、不改 neck。
- Haar filters 固定、无学习；每层 sketch/predictor 输出 16 channels，避免重建全部输入 channels。
- target、box mask 与 sketch 全 detached；梯度只通过 `P_s(y_s)` 进入现有 downsampler/backbone。
- gain 以一次 train-only gradient RMS calibration 锁到 base gradient 的 5%；不按 AP sweep。
- checkpoint 可保存 predictor 以 resume；export 前必须删除并断言无 `spar_*` key。

### 4.3 廉价零更新探针 SPAR-P0

对锁定 B50 checkpoint 的 32 个 train images：

1. 收集每个 stride-2 层的 pre/post features；
2. 在一半 images 上拟合 frozen ridge 1×1 probe，由 post feature 预测 pre feature 的 Haar detail sketch；另一半报告 object-ring/background 的 held-out `R²`；
3. 对输入做 `(+1,0)/(0,+1)` pixel translation，映射回坐标后测 P3/P4/P5 feature equivariance error；
4. 按 GT area quartile 检查 high shift-error 是否对应低 matched quality/最终 miss。

**P0 GO：** 至少两个层的 object-ring `R²` 比 matched background/LL control 低 `>=0.10`，且 shift-error top quartile 的 GT miss rate 是 bottom quartile 的 `>=1.5×`。否则说明没有待恢复的特异信号，停止。

### 4.4 10-epoch screen

| 组 | 定义 |
|---|---|
| P0 | R0 |
| P1 | SPAR-SHUFFLE：同 mask、同 detail 能量、同 loss strength，但 batch 内循环置换 detail targets |
| P2 | 正确 image/position phase-detail target |

**继续线：** P2 相对 P0 `>=+0.0030`、相对 P1 `>=+0.0020`；bottom-area quartile recall `>=+0.010`；一像素 shift equivariance error 降 `>=15%`；top-area quartile AP 下降不超过 `0.003`；predictor 删除后的模型与普通推理图完全一致。

### 4.5 推理与训练成本

- 推理：**严格 0**；没有 wavelet op、predictor 或额外 feature branch。
- 训练：三个固定 Haar+小 1×1 heads，估计 `3–6%` wall-time、少量 activation memory。

### 4.6 2024–2026 最近一手工作与碰撞

- [Wavelet Convolutions for Large Receptive Fields，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7137_ECCV_2024_paper.php) 把 wavelet convolution 作为可部署层扩大 RF 并形成多频响应；SPAR 不引入 wavelet inference layer，也不声称新的 wavelet convolution。
- [Frequency-aware Feature Fusion，TPAMI 2024](https://arxiv.org/abs/2408.12879) 在 feature fusion 中用 adaptive low/high-pass filters 和 resampling 改善 intra-class consistency/boundary；SPAR 发生在 backbone downsampling，且只是 training-only decodability objective。
- [SET: Spectral Enhancement for Tiny Object Detection，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html) 已用频域归因、背景平滑和 adversarial perturbation 改善 tiny-object detection。它也是对 SPAR 最大的反证提醒：高频可能主要是背景噪声，所以 SPAR 必须 box-gated 并胜过 phase-shuffled control。

可防守差异只是：**training-only, object-gated subpixel detail recoverability at existing strided downsamplers**；不能写“首个频域/小波检测器”。

### 4.7 Falsification / stop rule

- P0 没有 object-specific information-loss gap：停止。
- P2 与 P1 持平：提升只是 auxiliary reconstruction/deep supervision，不是正确相位细节。
- shift consistency改善但 AP 不升：记录为 robustness trick，不扩 50 epoch。
- background predicted-detail energy上升、false positives增加或大目标 AP 明显下降：停止，不用更强 gain 挽救。

## 5. 候选三：CSPC——可折叠跨尺度预测编码

### 5.1 机制假设

nearest-upsampled coarse feature `u` 与同分辨率 lateral feature `l` 直接 concat，会把大量可预测的 coarse semantic content 重复交给随后的 C3k2；真正需要保留的是 `l` 中不能由 `u` 预测的 innovation。

训练时学习一个无 bias 的 1×1 线性 predictor `A`：

```text
r = l - A(u)
z_train = concat(u, r)
L_pred = ||A(stopgrad(u)) - stopgrad(l)||_Huber
```

随后 C3k2 首层权重分为 `[W_u,W_r]`。部署时可精确改写：

```text
W_u' = W_u - W_r A
W_l' = W_r
W_u u + W_r(l - Au) == W_u' u + W_l' l
```

因此删除 `A` 后仍直接输入原 `concat(u,l)`，推理图与 R0 完全相同。这不是“加一个 neck block”，而是用可解释的可逆输入基变化塑造优化。

### 5.2 最小实现

- 仅两个 top-down concat（P5→P4、P4→P3），不碰 bottom-up path。
- `A` 为纯 1×1 linear、无 BN/act/bias；否则无法按上述公式安全折叠。
- prediction loss 只训练 `A`，target features detached；detection loss 仍通过 `r` 更新两层特征。
- 编写 float64 block fusion、整模型 fusion、export graph 计数与 R0-off 测试。

### 5.3 廉价零更新探针 CSPC-P0

冻结 B50，在 32 batches 上离线拟合 `u→l` ridge predictor；held-out 报告：

- GT interior、box boundary ring、matched background 的 channel-whitened `R²`；
- residual energy 在 boundary 相对 interior/background 的 enrichment；
- 原 concat 两半的 canonical correlation 与 effective rank。

**GO：** 两个 fusion 中至少一个 GT-interior `R²>=0.45`，且 residual boundary/interior energy ratio `>=1.5`；否则不存在足够“可预测冗余 + 有用创新”的结构，停止。

### 5.4 10-epoch screen

| 组 | 定义 |
|---|---|
| C0 | 原 concat |
| C1 | 同一可折叠参数化，但 `A` 只受 detection loss，不加正确 prediction target |
| C2 | 完整 CSPC |

**继续线：** C2 相对 C0 `>=+0.0030`、相对 C1 `>=+0.0015`；AP75 `>=+0.004`；FP32 整模型 fold 前后 max error `<1e-5`；fold 后 parameter/FLOPs/operator count 与 C0 完全一致。

### 5.5 成本

- 推理：正确 fold 后 **严格 0**。
- 训练：两个 1×1 predictors 与 Huber loss，预计 `<3%` wall-time；实现/fusion 风险中等。

### 5.6 2024–2026 最近一手工作与碰撞

- [Rethinking Features-Fused-Pyramid-Neck，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8386_ECCV_2024_paper.php) 已明确研究跨 hierarchy point-to-point fusion misalignment，并提出 unfused pyramid、soft interpolation 与 adaptive downsampling；CSPC 不能声称首次发现 FPN 融合问题。
- [Frequency-aware Feature Fusion，TPAMI 2024](https://arxiv.org/abs/2408.12879) 用 adaptive filters/resampling 解决融合一致性和边界问题；CSPC 的窄差异是 static linear predictive residual 可以完全折回原 post-concat convolution。
- [RHCNet，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RHCNet_Residual-Guided_Hierarchical_Calibration_Network_for_Robust_Underwater_Object_Detection_CVPR_2026_paper.pdf) 也以 residual/hierarchical calibration 处理 cross-scale semantic drift，但保留专用推理模块且面向 underwater detection。

### 5.7 Falsification / stop rule

- P0 `R²` 或 boundary enrichment 不达线：停止。
- C2 不胜 C1：只能说明可逆参数化有影响，predictive-coding 语义不成立。
- A 的 condition number 爆炸、fold error 超线或 exporter 残留 A：停止。
- 只提升训练收敛而 epoch10/last2 AP 不升：不进入长跑。

## 6. 候选四：BECM——盒证据反事实遮蔽

### 6.1 机制假设

VOC 样本量小且类别与场景共现强，检测器可能在物体被移除后仍凭背景对 GT 类别给高分。普通 Cutout 只增加噪声，不能区分“保留物体/移除物体”两种因果干预。

对无明显 overlap 的一个 GT 构造两个 box-aware views：

- `object-present`：保留 box 加窄 context ring，外部替换为 matched low-frequency background；
- `object-absent`：保留上下文，但以 matched-frequency patch 填掉 box 加 ring。

clean forward 的原生 assignment 只用于锁定该 GT 的 anchor set；不比较 O2O/O2M、不产生 teacher logits。直接施加：

```text
L_evidence = softplus(margin + max_score_absent - max_score_present)
```

`present` view 另接原 GT 的 cls/box supervision；`absent` view 只把 assigned-class 作为 counterfactual negative，不做 box regression。

### 6.2 最小实现

- 只在 `12.5%` images 生成一对附加 views，约 `+25%` batch-equivalent compute。
- 仅选 box 占图像 `1–40%` 且与其他 GT IoU `<0.1` 的对象；其余跳过。
- ring width 按 `0.15*sqrt(box_area)` 固定；fill patch 保持 per-channel mean/std 和低频能量。
- 不改主图数据增强、不学 policy、不读 validation。

### 6.3 廉价零更新探针 BECM-P0

在 B50 上对 128 个 eligible GT 做三次 forward，计算：

- `context leakage = score_absent / (score_clean+eps)`；
- `object sufficiency = score_present / (score_clean+eps)`；
- leakage 与类别、box area、训练集 class-background PMI 的关系。

**GO：** 至少 `15%` GT 的 context leakage `>0.5`，且这些 GT 的最终 false-positive/miss odds 是其余 GT 的 `>=1.5×`。否则不存在足够 context shortcut，停止。

### 6.4 10-epoch screen

| 组 | 定义 |
|---|---|
| E0 | R0 |
| E1 | 与 boxes 无关、面积/频谱/compute 匹配的 random rectangles |
| E2 | 正确 object-present/object-absent intervention |

**继续线：** E2 相对 E0 `>=+0.0025`、相对 E1 `>=+0.0015`；median context leakage 降 `>=20%`；clean screen-val AP 不下降；任何单类 AP 下降不得超过 `0.010`；训练 wall-time增量 `<30%`。

### 6.5 成本

- 推理：**0**。
- 训练：预注册上限约 `+25%` forward compute；无新增参数。

### 6.6 2024–2026 最近一手工作与碰撞

- [Masking meets Supervision / MaskSub，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Heo_Masking_meets_Supervision_A_Strong_Learning_Alliance_CVPR_2025_paper.html) 用 clean main branch 指导强 masking sub-branch；BECM 不以 clean logits蒸馏 masked view，而用 box-aware present/absent 的有向证据排序。
- [Background Invariance Testing According to Semantic Proximity，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liao_Background_Invariance_Testing_According_to_Semantic_Proximity_ICCV_2025_paper.html) 系统选择语义距离不同的背景来测试 invariance；它是测试框架，不是用 detection boxes 构造训练期正/负反事实。
- [ReCon，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c83f1f1290e80236587e4c89fa24f4e-Abstract-Conference.html) 用 diffusion、region rectification 和 region-aligned attention 生成检测增强；BECM 不依赖生成模型，成本和主张都更窄。

### 6.7 Falsification / stop rule

- P0 leakage 不集中或与错误无关：停止。
- E2 不胜面积/频谱匹配的 E1：收益只是遮蔽 regularization。
- 提高 counterfactual robustness 但不提高 clean AP：不进入主线，可只保留 robustness 附录。
- paste edge 成为捷径（能从边缘轻易区分 present/absent）：停止，不能靠进一步美化掩盖定义问题。

## 7. 候选五：FORC——遗忘事件组织的重放课程

### 7.1 机制假设

当前 uniform image sampling 不区分“始终不会”“刚学会又忘记”“始终稳定”的 GT。瞬时高 loss 容易选择噪声，真正有训练价值的可能是跨多次出现发生 learned→forgotten 转换的对象。

为每个稳定 GT ID 记录部署 O2O head 的

```text
q_g = max_a sqrt(p_assigned_class * IoU(pred_box, GT))
```

并更新 detached EMA。若 `q_g` 曾越过固定阈值、随后跌回阈值下，计一次 forgetting event。下一 epoch 的 image sampling weight 由 capped forgetting counts 决定，但在 class 和 box-area strata 内重分配，保持全局类别/面积 marginals；最后 2 epochs 统一恢复 uniform sampling。

它不把 O2O prediction教给 O2M，不改变任何 loss，也不控制增强强度。

### 7.2 最小实现

- dataloader 为每个 GT 携带 `(source_image_id, local_gt_id)`，mosaic 后仍可追踪来源。
- 在已有 assignment forward 中无梯度记录 `q_g`；epoch 末只汇总 scalar state。
- sampler 的总 epoch length、类别 histogram 和 area-quartile histogram固定；每个 image weight clip `[0.5,2.0]`。
- state 随 checkpoint 保存，resume 后继续；off 模式仍是原 RandomSampler。

### 7.3 廉价零更新探针 FORC-P0

利用同一 B50 轨迹的多个 checkpoint，在固定 512-GT clean train panel 上离线恢复 quality trajectory：

- 每个 GT 的 crossings、最大回落、最终 miss；
- top-forgetting 20% 对最终 misses 的覆盖率；
- forgetting 与瞬时 loss、class frequency、area 的偏相关。

**GO：** top-forgetting 20% 覆盖 `>=50%` 最终 persistent misses，且控制 class/area 后仍有 odds ratio `>=2.0`。否则 forgetting 只是类别/尺寸代理，停止。

### 7.4 10-epoch screen

| 组 | 定义 |
|---|---|
| F0 | uniform sampling |
| F1 | epoch 1 后冻结的 static-hard sampling，marginals/clip相同 |
| F2 | online forgetting curriculum |

**继续线：** F2 相对 F0 `>=+0.0025`、相对 F1 `>=+0.0015`；top-forgetting stratum recall `>=+0.015`；sampling class/area JS 均 `<0.02`；最后两轮 uniform 后增益仍保持；wall-time增量 `<5%`。

### 7.5 成本

- 推理：**0**。
- 训练：scalar bookkeeping 与 sampler 更新，目标 `<2%`；checkpoint 增加小型 GT-state table。

### 7.6 2024–2026 最近一手工作与碰撞

- [Look Around and Learn，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7259_ECCV_2024_paper.php) 在 embodied self-training 中通过 exploration/disagreement 挖 hard samples；FORC 是 fully supervised 固定数据集内的 temporal forgetting replay。
- [Efficient NeRF Optimization — Not All Samples Remain Equally Hard，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5300_ECCV_2024_paper.php) 在线筛选仍难的 rays 以节省 backward；它使用当前 hardness，不跟踪 object-level learned→forgotten events，也不是 detection。
- [Uncertainty Meets Diversity，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Uncertainty_Meets_Diversity_A_Comprehensive_Active_Learning_Framework_for_Indoor_CVPR_2025_paper.html) 用 uncertainty/diversity 选待标注 3D scenes；FORC 不做 active learning，且保持 label/stratum marginals。

### 7.7 Falsification / stop rule

- P0 forgetting 不集中或完全由 class/area 解释：停止。
- F2 不胜 F1：online temporal signal没有超过普通 hard sampling。
- sampler 反复播放 annotation noise、训练 loss下降但 clean AP下降：停止，不放宽 clip。
- state 在 DDP/resume 下不能 bitwise/数值同步：不得进入正式训练。

## 8. 综合排序

评分为 1–5；总分是三项乘积，目的只是强迫显式权衡，不把主观印象写成“综合最好”。

| 排名 | 候选 | 新颖性 | 可实施性 | 预期信号 | 乘积 / 125 | 推理增量 | 最大风险 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **IGB** | 4.0 | 4.8 | 4.3 | **82.6** | 0 | gradient domination 可能根本不存在；均衡会抹掉有益 hard-example emphasis |
| 2 | **SPAR** | 4.4 | 4.1 | 4.0 | **72.2** | 0 | 高频主要是背景噪声；auxiliary reconstruction 而非正确 phase 解释收益 |
| 3 | **CSPC** | 4.6 | 3.3 | 3.7 | **56.2** | 0 after fold | predictor 非可识别；fold/export fragile；相邻层并不冗余 |
| 4 | **BECM** | 4.0 | 3.6 | 3.8 | **54.7** | 0 | box/background intervention 有伪影；训练成本较高 |
| 5 | **FORC** | 3.9 | 3.9 | 3.5 | **53.2** | 0 | forgetting 追逐噪声；稳定 GT identity 与 DDP state复杂 |

### 为什么选 IGB + SPAR

- 两者分别触及 **supervision allocation** 和 **early representation loss**，因果对象正交；不会再把所有赌注押在双头交互。
- 都有便宜、无更新、可当天 no-go 的 P0；不用先写完整训练系统。
- 都是训练期机制，部署图零增加，适合 YOLO26n 的轻量约束。
- IGB 有本地 multiplicity 迹象但尚缺 gradient 证据；SPAR 有强烈工程直觉但没有本地频率证据。两者都被迫先测关键中介量，不依赖 AP 猜测。

### 为什么暂不先做 CSPC

CSPC 的论文形态最漂亮：prediction residual + exact folding。但它可能只是一个不可识别的线性换基，且 C3k2 首层 fold、BN/export/DDP 都需要更严谨实现。只有离线 ridge probe 同时证明 `R²` 和 boundary innovation 后，才值得越过 SPAR。

## 9. 冻结执行顺序

1. **IGB-P0**：已有 checkpoints 上做 per-GT gradient ledger；未达三项 GO 门槛即归档。
2. **SPAR-P0**：同一 source/checkpoints 做 feature capture、ridge decodability、1-pixel shift probe。
3. 两者都 GO 时，先实现 IGB 的 off-equivalence 与合成 permutation control；再实现 SPAR fixed-Haar heads。
4. 每个想法独立跑 `S0/S1/S2` 10 epoch；不得组合 IGB+SPAR。
5. 只有通过 seed42 三元组才补 seed0；只有双 seed 通过才进入 50 epoch。
6. IGB no-go 时按排名转 CSPC-P0；SPAR no-go 时不以普通 anti-alias/downsample module 替代。
7. BECM/FORC 只在架构/损失两条均 no-go，或需要独立数据侧路线时启动；二者也不得与 ASAC 同时打开。

## 10. 新颖性措辞边界

本次是定向 screen，不是系统综述，不能据此写“首次”。截至本次核验，未在上述 2024–2026 一手近邻中看到与五个精确定义完全相同的方案，但各自宽概念均已有强先例：frequency-aware detection、FPN alignment、loss reweighting、mask-based supervised learning、hard-sample curriculum 都不新。

若实验成立，安全措辞只能落在窄机制：

- IGB：`per-instance positive-mass conservation within native O2M assignment`；
- SPAR：`training-only object-gated phase-detail recoverability at existing strided downsamplers`；
- CSPC：`exactly foldable cross-scale predictive residual parameterization`；
- BECM：`box-supervised present/absent evidence ordering without a generative model`；
- FORC：`stratum-preserving replay driven by object-level forgetting events`。

任何方向在 semantic placebo 上失败，都必须降级为普通 regularization/optimization trick，而不能继续保留机制名称。
