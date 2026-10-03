# IGB 近期工作排重与主张边界更新

日期：2026-09-19  
状态：针对性检索；不构成系统综述，正式投稿前仍需扩大数据库检索。

## 结论

IGB 不能声称“首次公平对待每个目标”“首次 per-object normalization”“首次平衡检测梯度”。这些宽泛主张已有明确先例。仍可能防守的窄机制是：

> **在不改变原生动态 O2M/TAL assignment 的前提下，对每个 GT 已获得的 soft positive target mass 做 batch-conserving quota correction，同时共同作用于该 GT 的 assigned-class positive classification 与 localization，推理零开销。**

是否值得继续完全取决于 IGB-P0 是否证明 native O2M 中存在 `target-score mass → shared-feature gradient domination`，以及后续正确 GT 映射能否战胜 mass/value/compute 匹配的 permutation placebo。

## 主要近邻与碰撞

### ATSS（CVPR 2020）

[Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html) 把核心问题定位为正负训练样本定义，并从每个 GT/每层 top-k 候选中自适应确定正样本。它通过 assignment 设计降低对象尺度与正样本定义的偏差。

与 IGB 的分界：ATSS 改变哪些 anchor 是正样本；IGB 冻结 YOLO26 的原生 TAL 选择，只在已分配 soft positives 上守恒 per-GT mass。论文不能把“不同 GT 正样本不公平”本身写成新发现。

### ObjectBox（ECCV 2022 Oral）

[ObjectBox: From Centers to Boxes for Anchor-Free Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700385.pdf) 明确提出以 object center 作为正样本，并让不同尺寸/形状对象在多个尺度上被同等对待。

与 IGB 的分界：ObjectBox 重新定义正样本与回归目标；IGB 不强制相同 anchor 数或相同尺度覆盖，而只校正动态 TAL 产生的总 soft mass。不能写“首个 equal treatment of objects”。

### LapNet / PONO（2019）

[LapNet: Automatic Balanced Loss and Optimal Assignment for Real-Time Dense Object Detection](https://arxiv.org/abs/1911.01149) 已提出 Per-Object Normalized Overlap，并通过学习权重处理 assignment ambiguity、class 与 object-size imbalance。这是“per-object normalized”措辞上最直接的先例。

与 IGB 的分界：PONO 参与 soft assignment/overlap normalization，并学习额外权重；IGB 的 quota 是由当前 TAL target mass 解析计算、detached、总 mass 守恒，且不改变 assignment。方法命名与摘要必须避免泛称 per-object normalized loss。

### Generalized Focal Loss（NeurIPS 2020）

[Generalized Focal Loss](https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) 把连续 localization quality 融入 dense classification target；它是 soft quality target 的重要直接先例。

与 IGB 的分界：IGB 不提出 quality-aware label，而把现有 TAL quality 当作需要按 GT 汇总的 supervision mass。不能把 soft target 或 quality-weighted positives 当作贡献。

### Bucketed Ranking Loss（ECCV 2024）

[Bucketed Ranking-based Losses for Efficient Training of Object Detectors](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7634_ECCV_2024_paper.php) 处理 positive/negative ranking imbalance 并降低训练复杂度。

与 IGB 的分界：其对象是排序损失的正负对与计算复杂度，不是原生 O2M 内 GT 间的 mass/gradient allocation。

### UGS（ICCV 2025）

[Uncertainty-Aware Gradient Stabilization for Small Object Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html) 从小目标 localization loss 曲率解释梯度不稳定，并以分布式分类、uncertainty loss/refinement 稳定梯度。

与 IGB 的分界：UGS 处理单个小目标的回归曲率与不确定性；IGB 假设少数 GT 因正样本 target mass 较大而支配 aggregate shared gradient。若 IGB-P0 最终只发现 area→gradient 而非 mass allocation，UGS 会显著削弱 IGB 的新颖性解释。

## 对实验设计的新增要求

1. P0 除预注册四个 GO gates 外，必须同时展示 gradient norm 与 positive count、area、mean target quality、matched prediction quality 的相关性；不能只画 mass 散点图。
2. 若 mass 与 count/area 高度共线，报告分层或多变量分析，但不在看到结果后改变 GO 门槛。
3. I1 permutation control 必须保持同一批 quota 数值、多重集和总 target mass，只破坏 GT→quota 对应；否则无法区分 semantic quota 与普通 reweighting。
4. 若正确 IGB 不胜 permutation，即使胜 baseline，也只能降级为一般优化正则，不能保留 per-instance mass-conservation 机制主张。
5. 至少报告 small/medium/large 或 area quartile 的分组变化，避免“均衡”只是把性能从大目标转移给小目标。

## 当前裁决

**仍可进入 P0，但新颖性为 conditional。** 最强风险不是实现成本，而是 assignment-balancing / per-object normalization 的先例已经很多；只有“固定 native dynamic assignment + analytic batch-conserving per-GT soft-mass correction + semantic permutation control”这一组合可能形成可防守边界。
