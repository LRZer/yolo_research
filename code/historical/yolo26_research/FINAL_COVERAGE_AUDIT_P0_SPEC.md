# FCA-P0：YOLO26 最终 GT 覆盖丢失审计预注册草案

日期：2026-09-19  
状态：**APPROVED / 用户于 2026-09-19 明确指令“执行 FCA-P0：按预注册先做零训练全量审计”**

## 1. 要回答的问题

当前真实 `TaskAlignedAssigner` 先为每个 GT 选出 task-aligned top-k 候选，再让共享 anchor 归属于 overlap 最大的 GT，最后 O2O 只保留每个 GT 的 top-1。合成反例已经证明：即使初始候选图存在覆盖所有 GT 的合法一对一匹配，当前贪心顺序仍可能让某个 GT 最终没有正样本。

FCA-P0 只回答三个事实问题：

1. 这种覆盖丢失在真实 VOC trainval 上是否出现、出现多少；
2. 丢失主要发生在几何/候选选择、共享 anchor 冲突，还是最终 `topk2` 稀疏化；
3. 在**不增加候选 anchor**的前提下，候选图中是否仍存在质量可接受的更高覆盖匹配。

P0 不回答“AP 是否提高”，也不允许根据结果修改阈值、训练模型或在 VOC07 test 上调参。

## 2. 冻结研究对象与 provenance

- Ultralytics：`8.4.146`。
- Git 基线：`9d45ee4613e79d7bceef7a39e42c15c278b0d4e8`。
- `head.py` SHA256：`1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8`。
- `tal.py` SHA256：`9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861`。
- criterion 必须严格为 `ultralytics.utils.loss.E2ELoss`。
- O2M assigner 必须为 `topk/topk2=10/10`；O2O 必须为 `7/1`。
- 两支 stride 必须严格为 `[8,16,32]`。
- `Detect.forward` 的 O2O 输入必须来自 detach 后的特征；不得把结果解释成共享 backbone 上的双头直接梯度冲突。

冻结 baseline run：

`b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1`

该 run 已通过同配置 baseline 逐数值精确复现。使用六个固定 checkpoint：

| 文件 | 实际训练状态 |
|---|---:|
| `epoch0.pt` | 第 1 个 epoch 完成后 |
| `epoch10.pt` | 第 11 个 epoch 完成后 |
| `epoch20.pt` | 第 21 个 epoch 完成后 |
| `epoch30.pt` | 第 31 个 epoch 完成后 |
| `epoch40.pt` | 第 41 个 epoch 完成后 |
| `last.pt` | 第 50 个 epoch 完成后 |

六个 checkpoint 必须来自同一个 run，并逐文件固定 SHA256；禁止用 `best.pt`，禁止混合 seed 或训练轨迹。

## 3. 数据与只读执行协议

- 数据：VOC2007 trainval + VOC2012 trainval，即现有 `voc_server.yaml` 的 `train` split。
- 全量顺序扫描：`16,551` 张图、原始标注 `40,058` 个 GT；若 loader 的有效计数与该清单不一致，必须解释并生成逐图差异，不能静默继续。
- 输入：`imgsz=640`，`augment=False`，`rect=False`，`shuffle=False`，`fraction=1.0`。
- 六个 checkpoint 必须使用完全相同的有序图像、letterbox 后 GT 几何和 batch 划分；分别生成 SHA256 指纹并要求一致。
- 不使用 VOC2007 test，不计算 AP，不选择 best epoch。
- 无 optimizer、无 backward、无参数更新、无 BN running-stat 更新；前后模型参数和 buffer 哈希必须一致。
- 执行时仅允许物理 GPU 4，`workers=4`，pinned memory 开启，NUMA node 1，BF16 autocast，inference mode，BN eval。
- BF16 是与 baseline 一致的主口径；`last.pt` 另做一次 FP32 决策敏感性复核。两种精度若落在 GO 阈值两侧，结果必须标为 `PRECISION_SENSITIVE / INCONCLUSIVE`，不能挑选有利精度。
- 当前主机有已知 RAM MCE/SIGBUS 历史。正式执行前后检查内核日志；发生任一新 MCE/SIGBUS/EDAC 事件则本次结果无效，不能 resume 拼接。

已知但仅作背景的数据特征：VOC trainval 平均 `2.4203` 个目标/图，`12.08%` 图像至少 5 个目标，`44.81%` 图像存在任意 GT 框相交。这说明做全量审计有现实样本基础，但框相交不是 candidate competition 的证据。

## 4. 精确阶段定义

对每个有效 GT `g` 和 anchor `a`，在锁定的原生 assigner 的精确源码边界处 clone 以下掩码；探针不得在函数返回后重建阶段或自行重写候选生成规则。仪器化 assigner 必须在固定 smoke panel 上逐位复现未仪器化 criterion 的最终 `fg_mask`、`target_gt_idx`、target boxes 和 target scores。

### S0：STAL 几何可用边 `E_geom`

`select_candidates_in_gts()` 返回的边。它包含当前代码对短边小于 `stride_val=16` 像素的 STAL 扩张。

### S1：task-aligned top-k 候选边 `E_topk`

`get_pos_mask()` 返回的 `mask_pos`，即 `E_geom` 再乘 task-aligned top-k 掩码。O2M 的 k=10，O2O 的 k=7。

### S2：冲突消解后的边 `E_conflict`

只执行 `select_highest_overlaps()` 中共享 anchor 按最大 overlap 归属的步骤，停在 `topk2` 之前。每个 anchor 至多归属一个 GT，但一个 GT 仍可能拥有多个 anchor。

### S3：原生最终边 `E_final`

继续执行 `topk2`。O2O 每个 GT 最多保留 1 个 anchor；O2M 因 `topk2=topk=10`，应满足 `E_final == E_conflict`，否则硬失败。

对每个阶段定义 GT coverage：

`covered_s(g) = 1` 当且仅当 GT `g` 在该阶段至少有一条边。

必须分别报告：

- `valid -> S0`：没有 STAL 几何 anchor；
- `S0 -> S1`：有几何 anchor，但 task-aligned top-k 后无候选；
- `S1 -> S2`：候选全部在共享 anchor 冲突中被其他 GT 获得；
- `S2 -> S3`：冲突后仍有 anchor，但 `topk2` 后归零；
- `S1 -> S3`：本路线关心的总最终覆盖丢失。

“点竞争导致丢失”只允许指 `S1 -> S2`；不得把 S0/S1 的几何或质量候选不足混入该术语。

## 5. 候选图与可恢复上界

每幅图、每个 head 独立构造二分图：左侧为有效 GT，右侧为 anchor，边集合严格为 `E_topk`。

1. 在每个冲突连通分量内求精确最大基数匹配 `M*`。
2. `C_final` 为原生 `E_final` 覆盖的 GT 数。
3. 理论可恢复缺口定义为 `gap = |M*| - C_final`。
4. `gap > 0` 才能证明原生顺序在同一候选图上留下了可避免的 coverage 损失。

不能把“某个丢失 GT 有候选边”等同于它一定可在不牺牲其他 GT 的情况下恢复；主统计必须使用整幅图/连通分量的匹配基数差。

### 5.1 质量约束上界

原生 alignment metric 记为 `A(g,a)`。为跨 GT、跨 checkpoint 可比，定义：

`r(g,a) = A(g,a) / max_{a' in E_topk(g)} A(g,a')`。

若分母为 0，单独标记为 zero-quality GT，不能伪装成 `r=1`。固定报告阈值 `r >= {0, 0.25, 0.50, 0.75, 0.90}` 下的最大匹配覆盖曲线。禁止看结果后新增阈值。

同时求字典序最优匹配：先最大化覆盖数，再最大化匹配边的总 `r`。报告：

- 新覆盖 GT 的 `r` 与 IoU 分布；
- 原生和字典序最优匹配在共同覆盖 GT 上的逐 GT 质量变化；
- 被改写的原生 anchor-GT pair 比例；
- 每个 gap-positive 分量的 GT 数、anchor 数、边数和最大度数；
- 精确匹配求解时间与原生 assignment 时间。

总体 rate 另外给出以“图像”为 cluster 的 10,000 次固定 seed bootstrap 95% 区间。每次重采样同一图像时必须连同其六个 checkpoint 的全部 GT 观测一起重采样，以保留图内 GT 和跨 checkpoint 的相关性。六个 checkpoint 是同一训练轨迹的时间切片，不是六次独立重复；时间一致性只能说明问题持续存在，不能替代独立 seed。

## 6. 预注册分组

主结果始终是所有有效 GT；分组只能解释机制，不能替换总体结果。

- 尺寸：letterbox 后 `min(w,h) <16`、`[16,32)`、`[32,64)`、`>=64` 像素。
- candidate competition degree：与该 GT 共享至少一个 `E_topk` anchor 的其他 GT 数，分为 `0/1/2-3/>=4`。
- 图像目标数：`1/2-4/5-9/>=10`。
- 与其他 GT 的最大框 IoU：`0/(0,0.1)/[0.1,0.3)/>=0.3`。
- 最佳 alignment anchor 所在层：P3/P4/P5；另报所有候选边的层级分布。
- 类别：20 类逐类计数；样本不足 100 的类别只描述，不参与单类裁决。

禁止在看到结果后按新尺寸阈值、类别组合或拥挤度定义寻找“显著子集”。

O2M `10/10` 使用同一 forward 做 sidecar control，但不参与 O2O 主 verdict。若 O2M 出现相近或更强的 gap，必须撤回“O2O 特有失效”表述，将现象改称为通用 TAL conflict-order 问题。

## 7. P0 自动裁决

这里的 GO 仅表示“值得进入不训练的 Phase 1 离线对照”，不表示方法有效或可发表。

### 7.1 有效性硬门槛

以下任一项失败则为 `INVALID`，不得给 GO/NO-GO：

- 任一 provenance、源码哈希、criterion、top-k/topk2、stride 或 checkpoint 来源不符；
- 六个 checkpoint 的图像/GT 几何指纹不一致；
- 数据不是全量、无增强、无 shuffle 的冻结 train loader；
- 主扫描的 workers/pinned memory/BF16/inference/BN eval 或物理 GPU 约束不符；唯一例外是预先声明的 `last.pt` FP32 敏感性复核；
- 参数或 BN buffer 前后哈希改变；
- O2M 的 `E_conflict != E_final`；
- maximum matching 小图单元测试、合成反例或 brute-force 交叉验证失败；
- 仪器化与未仪器化 criterion 在 smoke panel 上不是逐位一致；
- 新 MCE/SIGBUS/EDAC 事件出现。

### 7.2 主 GO 条件

O2O 同时满足以下条件才进入 Phase 1：

1. 六个 checkpoint×全部 GT 的 pooled avoidable-gap rate，其 image-cluster bootstrap 95% 下界 `>=0.25%`；
2. 六个 checkpoint 中至少 4 个的点估计 `gap / 有效GT >=0.25%`，且各自 `gap >=100`；
3. 在上述至少 4 个 checkpoint 中，使用 `r>=0.50` 边得到的质量约束 gap 至少保留 unrestricted gap 的 50%；
4. 通过者中至少一个必须是 `epoch40.pt` 或 `last.pt`，防止仅由早期瞬态现象推动路线；
5. 结果不是由 zero-quality GT 主导：zero-quality GT 在理论可恢复 gap 中占比必须 `<25%`；
6. `last.pt` 的 BF16 与 FP32 复核落在 `0.25%` 阈值同侧。

`0.25% + 100` 不是“统计显著性”阈值，而是进入下一项便宜离线工作的最低工程效应量：在当前 40,058 个 GT 上两者约等价，可防止用几十个边角事件设计复杂方法。

### 7.3 预注册的独立压力数据触发条件

若主条件不满足，仅在同一个预先定义的尺寸/competition-degree 二维 cell 中同时满足以下条件，标记 `DENSE_TINY_AUDIT_TRIGGER`：

- 六个 checkpoint 中至少 4 个的 gap rate `>=1.0%`；
- 每个通过 checkpoint 的 gap `>=50`；
- `r>=0.50` 的质量约束 gap 保留率 `>=50%`；
- 该 cell 在所有 checkpoint 中定义和成员规则完全一致。

这**不能**把 VOC 上的 proposed method 送入 Phase 1，更不能启动训练；它只允许另行提议并预注册一个 AI-TOD-v2/VisDrone 等 dense-tiny 数据集的同类只读审计。VOC 子组不得替代主 verdict，也不能直接外推到遥感数据。

### 7.4 NO-GO 与边界结果

- pooled avoidable-gap rate 的 95% 上界 `<0.25%`，或六次全量扫描总 gap `<30`：`NO-GO_ON_CLEAN_VOC`。
- 未达到 GO、但区间跨越 `0.25%` 且事件数不低于 30：`INCONCLUSIVE`；不得换 panel、修改阈值或新增事后子组来翻转裁决。
- 只在最早 checkpoint 达标：`EARLY_ONLY / NO-GO`；可记录新假设，但不能用本轮结果直接启动训练。
- gap 数足够但高质量 gap 保留不足：`LOW_QUALITY / NO-GO`。
- 仅 O2M 达标、O2O 不达标：不支持当前 O2O 主线，记录为另一问题，不自动改题。

`NO-GO_ON_CLEAN_VOC` 只否定当前 clean-VOC 路线；它不能证明训练增强视图或密集微小目标数据中不存在该失效。若要检验这些不同问题，必须另写新预注册，不能事后把它们并入本轮。

## 8. 若 P0 GO：Phase 1 离线对照

仍使用同一冻结候选图，不更新模型，至少比较：

1. 原生 conflict -> topk2；
2. 直接 `topk=1/topk2=1`；
3. 先 topk2、后 conflict；
4. 原生后按 alignment 最高的空闲候选做一次贪心回配；
5. 不同固定 GT 顺序的增广路径匹配；
6. 每个冲突分量的字典序最优 matching/Hungarian 上界。

拟议复杂方法只有在以下条件全部满足时，才允许请求 Phase 2 训练批准：

- 恢复 unrestricted gap 的 `>=80%`；
- 在 `r>=0.50` 条件下仍恢复 `>=60%`；
- 最简单贪心回配若已达到拟议方法覆盖率的 95%，则复杂方法 NO-GO；
- 与最优上界相比的覆盖差 `<=5` 个百分点；
- 新覆盖边的 `r` 中位数 `>=0.75`，第 10 百分位 `>=0.25`；
- 全量离线 assignment 额外 wall-time `<10%`，且无大分量导致的不可控尾延迟。

上述门槛同样只授权最小因果训练矩阵；不授权大规模超参数搜索。

## 9. Phase 2 的最小训练矩阵边界

只有用户在看完 P0/P1 后明确同意才设计并启动。最低需要：

- B0：冻结 baseline；
- B1：最简单且性能最接近的 greedy repair；
- B2：拟议局部 coverage-first matching；
- B3：等 coverage、质量打乱 placebo，用于检验“覆盖数量”与“匹配质量”。

先做固定短预算、固定末轮的因果筛选，再做多个 seed。VOC 只能用于机制发现；若结论主张 tiny/dense 或遥感通用性，必须在独立压力数据与健康硬件上复现。不得使用 VOC07 test 反调 matching 阈值。

## 10. 必须落盘的输出

- `protocol.json`：全部冻结常量、CLI 偏差、软件/硬件信息；
- `provenance.json`：checkpoint、源码、数据清单和 GT 几何 SHA256；
- `objects.csv`：逐 GT、逐 checkpoint、逐 head 的 S0-S3 coverage、anchor count 与分组字段；
- `components.csv`：逐冲突分量的图规模、原生覆盖、最大匹配覆盖和质量曲线；
- `summary.csv`：总体与全部预注册分组统计；
- `decision.json/csv`：机器可读 GO/NO-GO 及每条门槛；
- `report.md`：人读报告，必须同时呈现支持和反对路线的证据；
- `runtime.log` 与执行前后内核故障检查；
- 全部产物的 `SHA256SUMS.txt`。

## 11. 当前决策边界

用户已批准执行 FCA-P0，因此允许实现隔离探针、CPU 单测、单批 bitwise smoke 与通过门禁后的全量只读扫描。批准范围仍严格限于本协议：

- 不访问 VOC2007 test 指标；
- 不修改服务器原生 assigner 源码；
- 不启动训练、反向传播或参数更新；
- P0 即使 GO，也只授权提交 Phase 1 离线方案供用户决定，不自动授权 Phase 1 或任何训练。

## 12. 协议修正 A1：原生 conflict 是非单调 rewiring（全量结果揭盲前冻结）

第一次正式尝试在第 2 个 batch、尚未完成 `epoch0.pt` 扫描时被保护断言终止。终止原因不是 GPU、数据或 trace 错误，而是第 4 节隐含的 `S2 ⊆ S1` 假设不符合冻结源码：对共享 anchor，原生 `select_highest_overlaps()` 使用**所有 GT**的 `overlaps.argmax(1)`，未先以 S1 屏蔽。因此它既会删除 S1 边，也可能把该 anchor 改派给不在 S1、甚至不在 S0 的 GT。失败目录必须保留，不得拼接或 resume。

本修正于任何 checkpoint 的全量统计、bootstrap 或 GO/NO-GO 结果产生前冻结；数据、checkpoint、精度、质量阈值、bootstrap、效应量门槛和主候选图 `E_topk=S1` 均不改变。修正内容如下：

1. S2 保持原生 post-conflict 掩码，不再要求它是 S1 子集；禁止把 S2/S3 静默投影回 S1，也禁止修改原生 TAL。
2. 继续硬验证：`S1 ⊆ S0`、`S3 ⊆ S2`、S2/S3 每个 anchor 至多属于一个 GT；S1 与 S2 的差异只能发生在 S1 已有多 GT 冲突的 anchor 上，且 S2 winner 必须等于原生全 GT overlap argmax。
3. 逐 GT、逐图报告 S1→S2 coverage lost/gained、removed/introduced edge、`S2\S1`、`S3\S1` 以及 introduced edge 中位于 S0 外的数量。阶段覆盖不再解释为单调漏斗。
4. 第 5 节的主 matching 图仍严格等于 S1；原生 off-graph pair 不得加入主图。`C_final` 仍按 raw S3 覆盖的 GT identity 计数。
5. 为建立“不牺牲原生已覆盖 GT”的严格 witness，先在 S1 对原生已覆盖 GT identity 集 N 求匹配；若能覆盖全部 N，再以该 surrogate seed 增广至最大基数。该 seed 只保持 coverage identity，不冒充原生 anchor pair。
6. 若任一图的 N 无法在 S1 中同时匹配，扫描仍继续并完整保留 raw 证据，但 additive recoverable-GT、zero-quality 归因和分量归因不作强结论；自动 GO 被硬阻断，裁决为 `INCONCLUSIVE_NATIVE_COVERAGE_NOT_REMATCHABLE`。
7. `r` 的定义域仍是 S1。原生 final edge 不在 S1 时，注册字段 `native_r` 必须为空；其 raw alignment、clipped-CIoU 和独立 standard IoU 只作诊断。含 off-graph final edge 的图不生成 component-local gap，避免把跨分量的原生覆盖错误归入某个 S1 分量；image-level 主比较仍保留。
8. 正式运行前的回归 smoke 从 1 batch 提升到至少 2 batches，并加入正 overlap 改派、全零 tie 改派到 S0 外、identity-rematch 可行与 Hall deficit 不可行等单元测试。
9. dense-tiny cell 的 `r>=0.50` 归因必须使用同一个 unrestricted surrogate witness 中的新覆盖边，确保 quality-retained GT 是 recoverable GT 的严格子集；禁止把两个独立最大匹配的 GT identity 直接相除。全局主质量门仍使用预注册的精确 `|M*(G_r)|-C_final`。

该修正只使审计与已冻结原生语义一致，不改变模型行为，也不构成 Phase 1 干预。
