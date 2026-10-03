# 最终覆盖路线的近邻工作深查与创新边界

日期：2026-09-19  
状态：**只读文献/源码复核；未启动探针、训练或数据下载**

## 1. 裁决先行

路线 A 暂时**不应被杀死**，但它只能保留为一个低成本、强可证伪的诊断假设，不能再把以下内容单独当作创新：

- 发现密集小目标会发生 point competition；
- 给丢失 GT 重新分配 point/anchor；
- 使用 bipartite/Hungarian/optimal-transport matching；
- 训练期改变 assignment、推理零开销。

这些元素均有明确前作。路线仍可能成立的窄问题是：

> YOLO26/STAL 保证的是几何候选 coverage，但原生 O2O 随后执行共享 anchor 冲突仲裁和 `topk2=1`。候选 coverage 是否在这两个阶段被再次破坏？若真实发生，能否只在原 top-7 候选图的小型冲突分量中，以 coverage-first、quality-second 的局部匹配保证“只要存在可行匹配就不留下可避免的零监督 GT”，同时不扩展候选、不改变推理图？

这不是“匹配算法创新”，而是一个**特定实现语义缺口 + 可测真实失效 + 受约束训练机制**的组合命题。四部分缺一，论文性都会明显不足。

## 2. 为什么官方 STAL 留下了可检查的语义缝隙

[YOLO26 官方论文](https://arxiv.org/abs/2606.03748)把 STAL 表述为给微小目标保证 positive coverage，并明确写出 O2O 先取 `topk=7`、再做 `topk2=1`。

锁定源码却显示：

1. `select_candidates_in_gts()` 只扩张用于候选筛选的微小框；
2. `get_pos_mask()` 得到每 GT 的 top-k 候选；
3. `select_highest_overlaps()` 先把共享 anchor 交给 overlap 最大的 GT；
4. O2O 随后才执行 `topk2=1`。

因此，源码能够保证或改善的是**候选阶段**，并没有形式化保证**最终监督阶段**仍覆盖每个 GT。当前真实函数上的合成反例已证明：初始两个 GT 都有候选且存在合法双覆盖匹配时，原生顺序仍可输出 `[1,0]` 个最终 anchor。

但逻辑反例不等于真实问题。FCA-P0 的唯一作用，就是测出这种“candidate-to-final coverage gap”在真实数据上的发生率、质量和持续性。

## 3. 最近邻工作的直接碰撞

| 工作 | 已经覆盖的核心内容 | 与拟议路线的碰撞 | 尚可防守的差异 |
|---|---|---|---|
| [Solve Point Competition with Label Reassignment（IEICE 2025）](https://globals.ieice.org/en_transactions/information/10.1587/transinf.2024EDL8088/_f) | 明确识别 intra-group/invasion competition；其示例甚至指出“大目标已有其他 point，却仍抢走小目标的唯一 point”；用 label reassignment 和 extended points 修复 | **最高**。不能声称首次发现点竞争、首次发现可让点、首次给失配目标重分配 | 该工作扩展候选并依赖 IoU 阈值；我们的假设是不创建新候选，在原 STAL top-7 图上求覆盖—质量词典序解，并给出存在可行匹配时的性质 |
| [OTA（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/html/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.html) | 把检测 label assignment 视为全局 optimal transport | 高。不能把“从局部贪心改成全局优化”当作创新 | 只处理原 STAL 候选图中的小型冲突分量、首先最大化 GT coverage、再约束质量；目标是低开销替换一段特定 O2O 顺序，而不是重新定义完整 assignment |
| [POTO/DeFCN（CVPR 2021）](https://arxiv.org/abs/2012.03544) | 用 prediction-aware one-to-one permutation/Hungarian matching 实现全卷积检测器的端到端预测 | **高**。one-to-one bipartite matching 和 conflict-free assignment 本身早已存在 | 当前问题不是重新构造全量 O2O，而是检验 STAL 候选 coverage 在现有快速 TAL 冲突顺序中的丢失，并只修复可避免缺口 |
| [What Makes for End-to-End Object Detection?（ICML 2021）](https://proceedings.mlr.press/v139/sun21b.html) | 系统分析 O2O positive assignment 与匹配成本，核心目标包括避免样本冲突和冗余预测 | 高。不能把“用 O2O 匹配避免两个 GT 共享正样本”作为新原理 | 只剩锁定 YOLO26 的候选→最终覆盖语义缺口，以及不支付全局 matching 成本的稀疏局部修复 |
| [One-to-Few（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Li_One-to-Few_Label_Assignment_for_End-to-End_Dense_Detection_CVPR_2023_paper.html) | 研究端到端 dense detector 的 O2O 稀疏监督，用 soft anchors 增加有效监督 | 中。不能笼统声称首次改善 O2O 稀疏监督 | 不增加正样本 cardinality；只保证原候选图内“每个可覆盖 GT 至少一个” |
| [DDQ（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.html) | 同时处理 query 密度和 distinctness，改善端到端 O2O | 中 | 我们关注的是 GT-anchor candidate graph 的 Hall/coverage 缺口，不是 query 去重或密度设计 |
| [YOLOv10（2024）](https://arxiv.org/abs/2405.14458) | consistent dual assignments，形成 YOLO 系 O2M/O2O 训练框架 | 中高；双头、一致匹配、零推理开销都不是新点 | 若真实失效跨 YOLO26/YOLOv10 存在，可把问题提升为 dual-assignment dense detector 的 final-coverage preservation |
| [DEIM（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf) | Dense O2O 和 matchability-aware loss 改善稀疏监督与低质量匹配 | 中。不能笼统声称首次同时考虑 O2O coverage 与 match quality | 不复制 targets、不增加 Dense O2O；固定候选图内以 coverage 为硬优先级、quality 为第二目标 |
| [RFAssigner（2026）](https://arxiv.org/abs/2601.01240) | 从 point priors 中按质量区间/排序补入原本未分配样本；训练期生效、推理零开销 | 中。与“未分配样本 + 质量约束 + 零推理开销”叙事重合 | RFAssigner 补充 O2M positives；当前假设不增加 positive cardinality，只在固定 O2O 候选边上用 alternating path 保持一一约束 |

## 4. 允许与禁止的论文表述

### 禁止

- “首次发现 point competition”；
- “首次通过 label reassignment 解决小目标零正样本”；
- “首次把 bipartite/Hungarian matching 用于检测 assignment”；
- “STAL 没有作用”或“YOLO26 没有保证小目标候选”；
- 在 P0 前称其为真实 VOC/遥感缺陷；
- 用单 seed、单 VOC 增益称为通用 assignment 方法。

### 只有证据成立后才允许

- “区分 candidate coverage 与 final supervision coverage，并在锁定的 STAL/O2O 实现中测得两者存在可重复缺口”；
- “在不扩展候选的相同图上，给出 coverage-first matching 的明确可行性性质”；
- “相对原生顺序、简单 greedy repair、先 top1 后冲突、全局 Hungarian/OTA 上界，在质量—覆盖—训练开销上形成非支配结果”；
- “训练期改动、推理图和延迟完全不变”；
- 在多 seed、多数据和至少另一个同类 detector 验证后，才讨论一般性。

## 5. 论文性必须跨过的门槛

即便 FCA-P0 为 GO，下面任一失败都足以停止把它当主论文路线：

1. **真实失效**：avoidable final-coverage gap 必须达到预注册工程效应量，并跨训练阶段存在；
2. **质量可用**：可恢复边不能主要是 zero/极低 alignment 候选；
3. **非简单替代**：一次 greedy 空闲候选回配若已达到 proposed method 的 95%，复杂匹配没有方法贡献；
4. **训练因果**：正式训练不仅要提高 AP，还应优先改善受影响 tiny/dense/recall 指标；等 coverage 但质量打乱的 placebo 必须更差；
5. **跨设置**：至少多 seed、健康硬件、dense-tiny 压力数据和另一种相近 assignment 实现；
6. **叙事诚实**：IEICE LR、OTA/POTO 等必须作为直接最近邻，而不是藏在泛泛 related work 中。

## 6. 当前推荐

保留并只执行 **FCA-P0 全量只读审计**仍然合理，原因不是它已经“很新”，而是：

- 它直接检验官方 candidate-coverage 表述和最终源码行为之间的语义缝隙；
- 全量 VOC、六个现有 checkpoint，无训练即可快速得到否定或支持证据；
- NO-GO 也能避免后续数十次盲训；
- 若 GO，再决定是否值得在独立 dense-tiny 数据上复核，而不是立即实现复杂方法。

当前不建议因为这个文献深查而直接进入训练。最理性的顺序仍是：用户确认 FCA-P0 → 全量只读审计 → 依据预注册结果继续或停止。
