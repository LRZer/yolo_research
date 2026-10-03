# 新研究材料复核与下一阶段路线选项

日期：2026-09-19  
状态：**供讨论决定；未启动任何新训练、未下载新数据集**  
输入材料：`research_notes_and_bibliography.md` 与用户粘贴的第三轮总结。材料中的建议仅作为待验证假设，不作为执行指令或既定事实。

## 1. 总体评价

这份材料的研究方法是对的：不再把“YOLO26 + 一个已有模块”当成创新，而是先核对源码、构造反例、定义可测失效和停止条件。它比按论文标题堆候选模块更有价值。

但 62 篇条目的数量不能等同于 62 篇全文核验或顶刊顶会证据。工作库混合了顶会论文、正式期刊、arXiv、SSRN、作者页接收声明和仅摘要核验条目。材料本身已标注多数证据深度，这是优点；实际选题仍必须回到一手论文和锁定源码逐项复核。本次只审计会改变路线决策的核心前提和最近邻工作，没有把全部 62 篇都假装成全文精读。

## 2. 与实际研究版本的源码核对

锁定版本：

- Ultralytics `8.4.146`；
- Git 基线 `9d45ee4613e79d7bceef7a39e42c15c278b0d4e8`；
- `head.py` SHA256 `1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8`；
- `tal.py` SHA256 `9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861`；
- 两个文件均相对 Git 基线未修改。

| 材料中的关键判断 | 复核结果 | 对选题的影响 |
|---|---|---|
| 训练时 O2O 接收 `detach()` 特征 | **确认**。`Detect.forward` 明确把 detach 后的 FPN 特征送入 O2O 头 | 撤回“两个头在共享骨干直接梯度冲突”的表述 |
| O2M+NMS 是默认验证/预测路径，`nms=False` 才是 O2O | **确认** | AP、延迟和分支必须成套报告，不能混搭 |
| O2M/O2O 分别为 topk=10 与 topk=7、topk2=1 | **确认** | O2O 不是简单全局 Hungarian；存在局部贪心冲突问题的检查空间 |
| STAL 候选覆盖不保证冲突和 topk2 后仍有最终正样本 | **逻辑确认，发生率未知** | 可作为新路线的 Phase-0 诊断，但不能直接宣称真实缺陷 |
| DFL-free 可先验认定为定位缺陷 | **不成立**。官方受控消融在 YOLO26s 上报告去掉 DFL 于 640/1280 均不降总 AP | “把 DFL 加回来”不应成为主线 |

官方一手依据：

- [Detect.forward 源码文档](https://docs.ultralytics.com/reference/nn/modules/head/) 显示训练期 `x_detach`；
- [TaskAlignedAssigner 源码文档](https://docs.ultralytics.com/reference/utils/tal/) 显示先按最大 overlap 解决共享候选，再进行 `topk2` 过滤；
- [YOLO26 模型文档](https://docs.ultralytics.com/models/yolo26/) 明确 O2M 默认、`nms=False` 选择 O2O；
- [YOLO26 官方论文](https://arxiv.org/abs/2606.03748) 的 DFL 消融在 640 下为 46.0→46.3 AP、1280 下为 49.8→51.1 AP。

## 3. 最小反例已在真实代码上通过

用当前 `TaskAlignedAssigner(topk=7, topk2=1)` 构造 2 个 GT、7 个共享候选：每个候选对 GT-A 的 overlap 都高于 GT-B，但合法的 A→候选0、B→候选1 双目标分配存在。

真实函数输出：

```text
initial_gt_coverage = [true, true]
final_gt_coverage   = [true, false]
final_anchor_counts = [1, 0]
foreground_count    = 1
```

脚本：`work/yolo26_research/verify_assignment_counterexample.py`。

它只证明以下命题：**当前处理顺序在逻辑上可能把“已有合法候选”的 GT 变成零正样本。**它没有证明 VOC、DIOR、AI-TOD-v2 或任一真实数据集里出现得足够频繁，也没有证明修复后 AP 会提高。

## 4. 新增查重后，四条主候选的真实位置

### 4.1 候选竞争后的覆盖修复：保留为优先诊断，但创新边界明显变窄

该问题不是空白：

- [OTA（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/html/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.html) 已从全局最优传输角度处理 GT—候选分配与歧义；
- [DDQ（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.html) 研究密集且互异的端到端 queries；
- [DCFL（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.html) 与 [RFLA（ECCV 2022）](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf) 已研究微小目标分配失配；
- 更直接的是 [Solve Point Competition with Label Reassignment in UAV Object Detector（IEICE 2025）](https://globals.ieice.org/en_transactions/information/10.1587/transinf.2024EDL8088/_f)，它已明确讨论多个目标竞争同一点导致小目标失去正样本，并用扩展点做重分配。

因此不能把贡献写成“首次发现点竞争”或“首次给失配 GT 补正样本”。还可能防守的窄边界是：

> 针对 YOLO26/STAL 的 O2O 候选图，在每个小型冲突连通分量内做覆盖优先、质量约束的增广路径匹配；不创建新候选、不改变推理图，并形式化保证：只要候选子图存在合法匹配，就尽量不让 GT 在贪心冲突消解后失去全部监督。

这和 IEICE 工作的“扩展额外点 + 分类处理竞争”不同，也和全局 OTA/Hungarian 的成本与作用范围不同。但只有真实发生率和严格对照成立，这个差异才有论文价值。

### 4.2 像素/标注误差感知定位：问题真实，但主线碰撞强

[NWD/AI-TOD-v2](https://arxiv.org/abs/2206.13996) 已系统讨论微小框对位置偏差的敏感性与重标数据；[TOLF](https://arxiv.org/abs/2601.00617) 更直接用 normalizing flow 和不确定性调制处理 tiny-object noisy localization。该方向适合做诊断和必要对照，不适合直接以“训练期误差范围/不确定性头”作为新贡献。

### 4.3 采样相位稳定性：适合便宜诊断，不适合直接立为创新

已有工作比材料列出的基础抗混叠更接近本题：

- [Shift Equivariance in Object Detection](https://arxiv.org/abs/2008.05787) 已证明一像素平移可显著改变检测输出并提出检测级度量；
- [The Importance of Anti-Aliasing in Tiny Object Detection](https://proceedings.mlr.press/v222/ning24a.html) 已直接在 TinyPerson、WiderFace、DOTA 研究 tiny-object anti-aliasing；
- [TIPS（WACV 2025）](https://openaccess.thecvf.com/content/WACV2025/html/Saha_Improving_Shift_Invariance_in_Convolutional_Neural_Networks_with_Translation_Invariant_WACV_2025_paper.html) 已在 RetinaNet/Faster R-CNN 上同时报告 AP 与 shift consistency。

因此“发现一像素平移不稳定”或“增加抗混叠下采样”不新。仍可做的价值是把波动分解成 **特征采样、assignment 离散跳变、O2O top-k 排序** 三个来源，并仅在其中一个来源形成新的可测失效时继续。

### 4.4 双头监督调度：继续降级

源码 detach 修正使“共享骨干直接冲突”解释失效；YOLO26 官方已经有 Progressive Loss，而 H-DETR、Co-DETR、Group DETR、One-to-Few 等已覆盖训练期密集辅助监督和混合分配。再加上本项目 TQGB 两 seed 反号，这条线当前不值得优先追加 150 轮或控制器搜索。

## 5. 对旧路线排序的修正

新材料和补充查重使先前的 `RS-IGB > RS-SPAR` 排序需要更新：

- IGB 的“按实例质量动态分配监督预算”与 2026 年 [CQDA](https://doi.org/10.3390/rs18091406) 的 cumulative-quality dynamic assignment 和 balanced supervision 发生直接叙事碰撞；即使我们的 batch-conserving soft-mass correction 不完全相同，新颖性风险也明显升高。
- SPAR/phase 路线又与 detection shift-equivariance、tiny anti-aliasing、SET 等工作重合，不能作为默认第二主线。
- 当前更值得先测的是 **STAL/O2O 候选图的最终覆盖损失**，因为它对应精确代码顺序、可形成形式化性质，而且尚未在本项目真实数据上测过。

修正后的优先级：

1. **A：最终覆盖审计 → 覆盖约束局部匹配（推荐先讨论）**；
2. **B：shift/noise 两类鲁棒性诊断，只作问题筛选**；
3. **C：跨切片边界一致性，作为遥感系统路线备选**；
4. **暂停：TQGB、普通双头调度、IGB 直接训练、SPAR 直接训练、DFL 恢复、普通注意力/P2/大核组合**。

## 6. 如果选择路线 A，建议的最小执行协议

### Phase 0：只统计，不训练

在冻结 baseline checkpoints 和 train split 上分别记录：

1. STAL/几何候选阶段的 GT coverage；
2. 共享候选冲突消解后的 coverage；
3. `topk2` 后的最终 coverage；
4. 丢失 GT 的目标像素尺寸、局部密度、重叠度、层级、类别；
5. O2M 与 O2O 分开统计，并核对 `max_det=300` 是否对目标密集图形成独立召回上限。

预先约定停止条件：若总体丢失极低、tiny/dense 子组也无集中现象，或事件数不足以支持稳定分组统计，路线 A 立即停止。VOC 上低发生率不等于直接证明遥感无问题，但不得在没有压力数据证据时先写复杂方法。

### Phase 1：离线分配对照，不更新模型

在完全相同候选图上比较：

- 原始贪心流程；
- 简单 `topk2` 改动；
- 先 topk2 后冲突；
- 最近空闲候选贪心回配；
- 冲突连通分量内的 coverage-first augmenting path；
- 小分量 Hungarian/最优传输上界。

必须同时报告恢复的 GT 数、assignment quality 损失、额外时间和候选噪声，不能只报 coverage。若简单 top-k 或普通回配已达到同等效果，复杂方法停止。

### Phase 2：用户确认后才训练

只允许最小因果矩阵：baseline、简单修补、拟议局部匹配、等覆盖但打乱质量的 placebo。先在小预算/固定末轮筛选，再做多个 seed。若目标是遥感微小目标，压力数据应优先 AI-TOD-v2 或 VisDrone 类密集小目标数据；DIOR-HBB 可作为较广泛遥感外部验证。暂不从 DOTA OBB 起步，以免混入切片、角度和评测工程变量。

## 7. 三个可供用户决定的选项

| 选项 | 做什么 | 优点 | 主要风险 |
|---|---|---|---|
| **A（推荐）** | 先做最终覆盖审计；信号成立才研究局部覆盖约束匹配 | 源码和反例最具体，可快速证伪，推理零开销 | 已有点竞争/重分配前作；必须证明窄差异和真实发生率 |
| B | 先做一像素平移与标签扰动诊断，再选 assignment/feature/localization 分支 | 成本低，适合遥感 tiny failure 分层 | 相关工作很多，诊断本身不是论文贡献 |
| C | 转向跨切片边界一致性和整图计算预算 | 遥感问题真实，系统价值明确 | 工程量最大，公平计时和数据划分要求高 |

我的建议是讨论并选择 A 的 **Phase 0**，而不是直接承诺整条方法路线。Phase 0 的价值在于它可能在几小时内明确告诉我们：这个逻辑反例在真实目标上到底是主要失效，还是一个漂亮但罕见的边角情况。

## 8. 当前执行边界

- TQGB 已完成并停止；
- 没有启动 IGB、SPAR、SARC、coverage repair 或新数据集训练；
- 没有因为材料中的“推荐顺序”自动修改服务器；
- 等用户讨论决定后，再冻结下一阶段指标、停止条件和计算预算。
