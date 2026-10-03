# YOLO26 双分配协同研究：后续方向与执行路线

> 2026-09-18 执行覆盖：用户选择优先快速产出，现阶段继续用完整 VOC07+12 trainval，并按固定末轮公平比较。每轮 VOC07 test 仅作探索监控，禁止用最佳轮次挑结果。内部 dev 拆分保留为论文候选锁定后的干净复现方案，而不是当前短训的前置条件。

日期：2026-09-18  
当前证据：B50/D1/D2 单种子 50-epoch 探索，D2 固定末轮相对 B50 为 +0.00308 mAP50-95

## 1. 总体判断

D2 不是一个已经完成的论文方法，而是一个值得继续的机制信号。下一阶段不应立即跑 150 epochs，也不应同时加入注意力、卷积或多种 loss。当前最重要的研究问题只有三个：

1. +0.00308 是否超过随机种子波动；
2. 收益是否真的来自“选择可靠 O2M 知识”，而不是 KD 数值强度、稀疏正则化或全局梯度裁剪；
3. O2M 教师随训练变弱时，能否用训练状态自动调节知识迁移，而不是依赖人工指定某个截止 epoch。

如果这三点成立，主线升级为 **S-QGAT：状态感知的质量路由双分配蒸馏**。更复杂的双向 TQGB 暂时作为候选 D3，不抢在因果控制之前实现。

## 2. 从现有曲线得到的新证据

| 阶段 | D2-B50 平均 mAP50-95 | 平均 KD schedule scale | O2M 整体更优率 | 共同位置平均质量差 | KD/分类损失 |
|---|---:|---:|---:|---:|---:|
| epochs 6–10 | +0.00350 | 0.30 | 67.81% | 0.02744 | 0.42% |
| epochs 11–20 | -0.01096 | 0.90 | 61.62% | 0.01826 | 1.26% |
| epochs 21–30 | -0.00768 | 1.00 | 57.01% | 0.01502 | 1.65% |
| epochs 31–40 | -0.00674 | 1.00 | 53.55% | 0.01240 | 1.85% |
| epochs 41–50 | +0.00116 | 0.45 | 43.44% | 0.00542 | 0.76% |

这是探索性相关而非因果证明，但模式很清楚：低强度时为正，高强度平台期为负，fade 后又转正；同时 O2M 的整体优势持续衰减并最终低于 50%。

源码审计还发现一个关键点：当前 D2 先把 advantage 权重归一化到有效候选数量，再除以有效候选数量。这样做保证不同候选数的 batch 尺度相近，却同时消除了“可靠位置变少、质量优势变小”对总 KD 强度的自然衰减。D2 只选约 18.24% 的位置，但其平均 KD loss 反而约为 D1 的两倍。因此当前 D2 混合了“质量选择”和“KD 强度变化”，必须拆开。

## 3. 立即执行的阶段 A：零更新因果审计

本阶段不训练模型，只对保存的 B50/D1/D2 checkpoints 和固定训练 batch 做 forward/backward 探针。

### A1. Reduction 与强度审计

新增并分别记录：

- `KD sum`；
- 以 active candidates 为分母的 mean；
- 以全部 common same-GT positions 为分母的 mean；
- KD 梯度范数、基础监督梯度范数及二者余弦；
- 按 P3/P4/P5、目标大小和类别拆分的门控覆盖率。

目的：明确 D2 的效果来自可靠样本选择、稀疏化还是单纯更大的有效梯度。

### A2. Gradient-clipping 审计

当前训练器对全模型执行 `max_norm=10` 的 global clipping。固定同一权重和 batch，分别计算基础目标、D1、D2 的：

- clip 前总梯度范数；
- clip hit rate；
- 实际缩放因子 `alpha=min(1, 10/norm)`；
- O2O head、O2M head、Backbone/Neck 的分组梯度范数。

若 D2 相对基线的 clip hit rate 增量低于约 1–2%，且 `alpha` 中位数接近 1，可把裁剪降为次要因素；若 hit rate 超过 10% 或 `alpha` 中位数变化超过 5%，必须安排 baseline/D2 成对的 clipping 控制，不能直接声称 gradient bridge 或纯 O2O 改进。

### A3. TQGB go/no-go 探针

对固定 checkpoints 统计分类和定位两项任务中的 O2M→O2O、O2O→O2M 有效覆盖、方向不一致率以及监督投影后的非零信号。

- 反向有效覆盖长期 <1%：停止双向方案，专注 S-QGAT 单向状态控制；
- 1%–5%：保留但暂不训练；
- 在至少两个中后期 checkpoint >5%，且分类/定位路由差异明显：才进入 D3 实现。

## 4. 阶段 B：建立不污染 test 的研发协议

目前 VOC07 test 已被每轮读取，已有 B50/D1/D2 只能作为探索结果。从下一项新方法开始冻结 test：

- 内部研发训练集：VOC2012 trainval + VOC2007 train；
- 内部开发集：VOC2007 val；
- 所有结构、归一化、门槛和调度只根据训练统计与该开发集决定；
- 最终方法锁定后，再用完整 VOC07+12 trainval 从头训练，并只在固定最终轮评测 VOC07 test。

这样既保留官方 VOC07+12→VOC07 test 协议，又避免继续用 test 调方法。

## 5. 阶段 C：最小 50-epoch 因果矩阵

先只使用一个固定 seed，在内部开发协议上按顺序运行。每个实验都增加 clipping 与梯度统计，不改变 GPU4–7、batch 64、workers 4、pinned memory 和其它训练参数。

| ID | 变化 | 回答的问题 | 继续标准 |
|---|---|---|---|
| C0 | 新 train/dev 协议的无 KD baseline | 建立未污染基准 | 必做 |
| C1 | 当前 D2 active-mean | 当前信号能否在开发集复现 | 相对 C0 不为明显负值 |
| C2 | D2 strength/count-normalized | 消除 active-mean 放大，固定 gate 只改归一化 | 与 C1 比较强度混杂 |
| C3 | Random-Matched | 与 C2 匹配候选数和梯度量级，但随机选 common pairs | C2 应至少高约 0.002–0.003 |
| C4 | S-QGAT state controller | 局部门控不变，按全局教师可靠性自动衰减 | 相对 C0 ≥0.003，且优于 C2、曲线更稳 |

强度匹配只允许使用训练 batch 上的 loss/gradient calibration，不能根据开发集精度扫 gain。Random-Matched 使用固定随机规则，并匹配候选数量和总梯度量级，用来区分“质量选择”与“随机稀疏正则化”。

### S-QGAT 的候选控制器

令 `rho_t` 为训练期 common same-GT 位置中 `q_o2m > q_o2o` 的比例，以 ramp 结束时的全局 EMA 作为 `rho_ref`。候选控制器为：

```text
state_scale = clip((EMA(rho_t) - 0.5) / (rho_ref - 0.5 + eps), 0, 1)
L_KD = lambda * ramp_scale * state_scale * L_local_quality_gate
```

`0.5` 是两头无整体优劣的自然边界，而不是从 test 精度挑出的 epoch。统计按 epoch 在 DDP ranks 间汇总，用上一 epoch 的状态控制下一 epoch，避免不同 rank 使用不同尺度。正式实现前还要与基于连续质量差的无阈值版本做离线稳定性比较，只保留一个定义，避免堆超参数。

## 6. 阶段 D：配对多种子复现

只把阶段 C 中唯一胜出的版本送入多种子，不让多个变体同时进入昂贵验证。

- seeds 预先固定为 `{0, 1, 42}`；
- 每个 seed 成对运行 baseline 与方法，保证初始化和数据顺序配对；
- 主指标为固定 epoch 50 的 paired ΔmAP50-95；last-5 均值和曲线 AUC 只作稳定性补充；
- 继续标准：至少 2/3 seed 为正，平均提升 ≥0.003，且标准差没有明显大于均值；
- 停止标准：平均差 ≤0，或提升主要由单一 seed 驱动。

三种子原始值、paired differences、均值和标准差全部报告，不只报告最好 seed。

## 7. 阶段 E：150 epochs 正式 VOC

只有阶段 D 通过后才投入正式长训：

- 健康硬件上完整 VOC07+12 trainval；
- baseline 与最终方法各 3 个配对 seeds；
- 不利用 VOC07 test 选轮次，固定 `last.pt/epoch 150` 为主结果；
- 成功下限：平均 +0.003 mAP50-95 且至少 2/3 seed 为正；更有论文说服力的目标是 +0.005 或以上；
- 若 50 epochs 有效但 150 epochs 增益消失，不能继续声称提高最终精度。只有 time-to-AP 和训练曲线 AUC 在多种子下稳定改善时，才可转为“训练效率”故事；否则停止该方向。

还需做 ProgLoss 解耦：baseline/方法在官方 ProgLoss 下成对比较，再在固定或等权双分支权重下成对比较。若角色变化只在 ProgLoss 下出现，论文只能主张“与 ProgLoss 协同”，不能主张通用的自发角色反转。

## 8. 阶段 F：通用数据集与模型尺度

VOC 用于形成和验证机制，最终“通用目标检测改进”必须进入 COCO2017：

1. 先跑 YOLO26n 的官方 COCO schedule：baseline vs 最终方法；
2. 报告 AP、AP50、AP75、APS/APM/APL，以及参数量、FLOPs、训练时间和推理延迟；
3. n 规模成立后再扩展 YOLO26s，证明不是只适用于极小模型；
4. 方法不改变推理图，理论上参数/FLOPs/延迟应与 baseline 相同，但仍需实测；
5. 若 COCO 不成立，则不能称通用方法，只能重新判断是否有“小数据/快速训练”专项价值。

不使用 COCO128 或任意小子集作为论文性能证据；它们只能做工程 smoke test。

## 9. 论文创新边界

### 若 S-QGAT 成立，最稳妥的论文故事

1. 诊断双分配检测器中 O2M/O2O 的教师可靠性会随训练衰减甚至反转；
2. 用 exact same-anchor/same-GT 的局部质量路由避免无条件 O2M→O2O 迁移；
3. 用训练期全局可靠性状态控制迁移强度，避免候选归一化掩盖教师退化；
4. 不改变部署网络，保持零额外推理成本。

### 只有 TQGB 消融全部成立时才能升级的故事

必须同时看到：Dynamic > forward-only、Dynamic > reverse-only、Dynamic > no-trunk bridge，并优于最接近的对称 L1 一致性，才能主张 task-wise bidirectional routing 和 sparse-to-dense gradient bridge。

### 明确不能主张

- 首个双头自蒸馏；
- 首个动态 teacher/student 切换；
- 首个质量门控检测 KD；
- 首个 O2M/O2O 一致性训练；
- 仅凭 VOC 单种子 +0.308 AP 就声称通用、显著或达到 SOTA。

## 10. 预算与停止纪律

按当前约 31 分钟/个 50-epoch 四卡实验估计：

- 阶段 A：固定 batch 探针，远低于一个完整 R50；
- 阶段 C：最多 5 个 R50，约 2.5–3 小时串行墙钟；
- 阶段 D：3 对共 6 个 R50，约 3–3.5 小时；
- 阶段 E：3 对共 6 个 R150，按现有约 1.54 小时/次约 9–10 小时；
- COCO 预算先用一次 baseline 实测后再登记，不能用 VOC 时间直接外推。

任何阶段不满足 go/no-go 标准即停止扩张，避免用大量实验从噪声中挑最好结果。当前服务器的所有运行继续使用 GPU4–7、NUMA 1、workers 4 和 pinned memory；每次运行前后检查新增 MCE/SIGBUS，发生 SIGBUS 后不从该 checkpoint 续训。论文最终结果必须在健康硬件从头复现。

## 11. 最近一步

截至 2026-09-19，reduction/gradient-clipping/TQGB 三项无更新探针均已完成。D2-GM 强度匹配后固定末轮相对 B50 仅 +0.00001，证明原 D2 的 +0.00308 不能归因于质量选择本身。S-QGAT v1 50 轮固定末轮为 0.51175，较 B50 -0.00101，last-5 也为 -0.00088；controller 虽在 epoch 40 正确退休，但没有产生净精度增益，因此按预注册 no-go，不调 test 驱动阈值、不扩多 seed。下一步正式转入已通过覆盖探针的 TQGB：先实现并测试投影后的分类单向 T1/双向 T2，只有 T2 显示增益才加入定位 T3。

更新：分类 T1 已完成并通过单 seed 筛选。固定 epoch 50 mAP50-95=`0.51946`，较 B50 `+0.00670`；last-5 `+0.00525`，独立 O2O/O2M `last.pt` 复评分别提升 `+0.00648/+0.00519`。因此当前执行顺序改为：完成已经冻结的 T2 双向因果对照；运行同增益 T1-NP 无投影控制；补齐 seeds 0/1 的 B50–T1 配对复现。T1 不因单 seed 正值直接进入论文结论。与此同时将搜索空间扩展到 head 之外：PBGA 零更新金字塔梯度冲突探针已列为并行准备项，SARC/ASAC 作为后续独立路线；详细门槛见 `TQGB_DECISION_PROTOCOL.md` 与 `BROAD_DIRECTION_SCREEN.md`。

再次更新：T2 固定末轮虽比 B50 高 `+0.00409`，但比 T1 低 `-0.00261`，独立复评也显示反向路由没有带来 O2M 净增益，因此双向路线停止扩张。PBGA 探针因所有 total-gradient 冲突率为 0 而 `NO_GO`；SARC 探针因显著且稳定的层级 assignment-scale 分布而 `GO`。当前串行正式训练顺序为：T0-current → T1-NP → seeds 0/1 的 B50–T1 配对；并行工程顺序为：T1-BRC 真模型测试、SARC R1/R2/R3 staging、ASAC-P0 审查。T0-current 已启动。若 T0 与旧 B50 不能复现，先解释源码漂移，暂停所有 T1 性能归因；若复现，则继续冻结矩阵，不在 seed42 上扫参数。

FCA-P0 正式更新（2026-09-19）：按冻结协议完成 VOC07+12 trainval、六个 checkpoint 的全量零训练审计。O2O 理论可避免 coverage gap 为 `31/240348=0.0129%`，image-cluster bootstrap 95% CI 为 `[0.0087%,0.0176%]`，显著低于预注册 `0.25%` 门槛；epoch20/30/40 均为 0，dense-tiny 定向触发也未成立。正式裁决为 `NO-GO_ON_CLEAN_VOC`：关闭 clean-VOC coverage-first matcher，不进入 Phase 1、不据此启动训练。该结果不否定遥感密集小目标场景，但若转向领域数据，须独立预注册并重新做只读审计。下一步先与用户讨论并冻结新研究路线，再进行任何新训练。

ASAC-P0 正式更新（2026-09-19）：按预注册协议完成 B50 epochs 11/21/31/41/50、固定 train-only 256 图、q=0.25/0.50/0.75 的零训练审计。O2M/O2O 的强度单调 checkpoint 均为 5/5，平均 weak→strong 稳定度下降为 0.04537/0.03655，最强视图质量正相关 checkpoint 也均为 5/5，最低 assignment coverage 99.84%，自动 verdict=`GO`。该 GO 只批准 scale/translate-ASAC 原型；下一步在实现前冻结 q 更新公式、上一 epoch/EMA 因果状态以及重放规则，然后仅执行 A0 固定增强、A1 匹配强度调度、A2 真实反馈、A3 打乱反馈时序的最小矩阵。A2 必须同时胜过 A0、A1、A3，才可把增益归因于 assignment-stability feedback；Mosaic 暂不纳入。
