# Classification-only TQGB：严格新颖性检索更新

日期：2026-09-19  
适用版本：当前已经实现的 classification-only TQGB（T1/T2）；不涵盖尚未实现的定位 T3/T4。  
检索边界：重点检查 2022–2026 年检测/KD 一手论文，同时向前追溯一个会直接限制“GT 约束蒸馏”表述的 CVPR 2021 先例。本文是投稿前的阶段性新颖性审计，不等价于穷尽所有论文、专利和同期未公开工作。

## 1. 当前方法的最小技术定义

当前实现不是“完整 TQGB”，而是分类子方法：

1. 在同一原生 O2M/O2O 双分配检测器内部，只取两头同时为前景、空间 anchor 相同、分配到同一 GT 且类别相同的位置；
2. 根据两头对该 GT 类别的当前置信度和预注册阈值选择 O2M→O2O 或 O2O→O2M 路由；
3. 教师概率停止梯度；
4. 对接收头，令当前概率为 `p_s`、本头原生 soft TAL 目标为 `t_s`、教师概率为 `p_t`，构造

   ```text
   p_proj = clamp(p_t, min(p_s, t_s), max(p_s, t_s))
   ```

5. 用 `KL_Bernoulli(p_proj || p_s)` 作为附加分类损失，并按所有 common anchors 而不是 active anchors 归一化；
6. 不修改推理图。当前 T1 是单向 O2M→O2O，T2 才是双向分类路由。

对 Bernoulli KL，接收 logit `z_s` 的梯度为 `p_s-p_proj`；原生 BCE 类分类项在同一 logit 上的梯度方向为 `p_s-t_s`。由于 `p_proj` 位于二者闭区间内，两者乘积非负。因此能够严格说的是：**在当前 forward point、逐 anchor、逐接收 GT 类 logit 上，附加分类梯度与该 soft target 的原生 BCE 梯度同号或为零。**这不是全参数、全 batch 或经过 global clipping 后的“全局无冲突”证明。

## 2. 最接近工作的逐项对照

| 工作 | 已有内容 | 与当前 TQGB 相同处 | 关键不同处及新颖性约束 |
|---|---|---|---|
| YOLOv10, NeurIPS 2024 | 原生 O2M/O2O 双头、共同训练、统一 matching metric；论文明确希望 O2O 朝 O2M 的优化方向靠拢，并使 O2O top-1 成为 O2M 的最佳正样本 | 同一检测器、双分配、训练期双头、部署只保留 O2O | YOLOv10 通过 assignment/监督目标的一致性实现协同，不做两头预测概率的在线蒸馏，不比较当前两头谁更可靠，也不把教师概率投影到接收头的监督线段。故“双分配协同、同一最佳匹配、O2M 帮 O2O”均非新颖点 |
| CrossKD, CVPR 2024 | 把学生 head 的中间特征送入冻结教师 head，进行 prediction mimicking，并以结构隔离缓解标注目标与教师目标冲突 | 都关注检测器跨 head 知识传递以及直接 prediction KD 的目标冲突 | CrossKD 是预训练外部教师→学生；其 cross-head 是学生特征经过教师 head 生成蒸馏输出。它不是同一原生 O2M/O2O 双头从头联合训练，不用 same-GT common-anchor 路由，也不修改教师目标使其落入学生—原生目标线段 |
| Label Assignment Distillation / Co-learning LAD, WACV 2022 | 教师输出参与学生 label assignment；Co-learning LAD 可让两个从头训练的网络动态交换教师/学生角色 | 都不必依赖固定的大型教师，并涉及训练中角色变化 | LAD 迁移的是分配决策/硬标签，主体是两个网络；当前方法在一个检测器的原生双头间传递分类概率，角色按 common anchor 的任务质量决定，并做接收头特定的 target projection。故“在线共同学习、动态换教师”本身不能主张新颖 |
| Dual-head self-distillation via an auxiliary lightweight neck, Neurocomputing 2026 | YOLO11 单代双头自蒸馏；main head 向带轻量 neck 的 auxiliary head 传递分类与定位知识；无外部教师且辅助支路推理时删除 | YOLO、单次训练、双头内部知识迁移、分类 KD、零额外推理开销 | 它采用新增 main/auxiliary 结构和固定主辅关系，不是原生 O2M/O2O assignment heads。公开的论文摘要和方法说明没有给出 same-GT common-anchor 双向路由或监督线段投影。故“YOLO 双头、单代自蒸馏、无外部教师、零推理开销”均不能作为主贡献 |
| Positive-Unlabeled Data Purification, CVPR 2021（超出重点年份，但构成必要先例） | 提出 ground-truth bounded KD：分类仅在教师正确时蒸馏；回归使用 GT 约束的 teacher-bounded regression | 都用 GT/原监督信号阻止错误教师知识伤害学生 | 其分类机制是正确性二值门控，不把教师概率投影到由学生当前概率和 soft target 定义的闭区间；场景是预训练 teacher/student 和同一 proposal，不是原生 O2M/O2O 头。故不能声称“首次 GT-bounded/GT-aware KD”，但仍可检验更窄的 supervision-segment projection |

补充判断：KD-DETR/CrossKD 等工作已经说明检测蒸馏中的预测对应关系和监督冲突是核心问题；TGeo-KD 等分类 KD 工作也已用 student–teacher–GT 三方几何来调节蒸馏权重。因此，“考虑三者关系”“consistent points”“选择可靠教师”都只能作为背景。当前尚未在所核验的一手来源中发现与下列完整组合相同的方法：**原生 O2M/O2O heads + exact common anchor/same GT + 在线可逆方向 + 将教师的 Bernoulli 分类目标投影到接收头当前预测与其原生 soft assignment target 的闭区间。**

## 3. 可以安全使用的论文表述

以下表述必须带有“据我们检索/在所比较工作中”之类的范围限定：

- “我们研究原生双分配检测器内部的 cross-assignment prediction transfer，而不是外部大教师到小学生的压缩。”
- “我们只在 O2M/O2O 对同一空间位置分配到同一 GT 和同一类别时建立分类知识通路，避免另做 query/box 匹配。”
- “我们提出 receiver-native supervision-segment projection：教师概率先投影到接收头当前概率与该头原生 soft assignment target 的闭区间，再用于 Bernoulli KL。”
- “该投影保证逐接收 GT 类 logit 的附加梯度在当前 forward point 不反向于原生 BCE 梯度；这是一项局部方向性质，不是全局优化保证。”
- “方向可以按两头的当前任务质量在 O2M→O2O 与 O2O→O2M 间选择；反向路线允许部署头信息通过 O2M 路径影响 shared trunk。”
- “方法只改变训练损失，不增加推理参数、FLOPs 或后处理。”
- 若 T2 和后续严格消融为正，可把贡献描述为“assignment-native, receiver-bounded cross-head self-distillation”。在获得实验支持前，应称为“候选机制”而非“有效方法”。

## 4. 不可安全声称的内容

- 不可声称首次 YOLO 自蒸馏、首次双头自蒸馏、首次 O2M/O2O 协同或首次零推理开销辅助训练；
- 不可声称首次跨 head prediction KD、首次冲突感知 KD、首次 GT-aware/GT-bounded KD；
- 不可把 exact common-anchor/same-GT 说成一般意义上的首次对应关系对齐；YOLOv10 和 KD-DETR 已分别覆盖双分配匹配一致性和 consistent distillation points；
- 不可声称双向/在线互教本身新颖；在线 KD、mutual learning、Co-learning LAD 均已有先例；
- 不可声称“梯度无冲突”“理论保证训练一定改进”或“不会伤害原任务”。局部 logit 同号并不排除共享参数 Jacobian、跨样本求和、其他类别、global norm clipping 造成的参数级相互作用；
- 不可把当前结果称为完整 TQGB。定位路线和相应 no-projection/bridge 消融尚未实现；
- 不可凭 VOC 单 seed 或故障主机探索结果声称通用、显著或可发表。新颖性只回答“与谁不同”，不替代多 seed、COCO、另一尺度和健康硬件复现。

## 5. 阶段性结论与投稿风险

新颖性判断为“**组合层面有可防守空间，单个组件层面多数已有先例**”。最有辨识度的不是双头、自蒸馏、质量门控或无推理开销，而是：

> 对原生双分配 heads 的同 anchor/同 GT 分类预测进行可逆知识路由，并把每个教师建议变成接收头原生监督线段上的有界目标，从而获得可证明的逐 logit 局部方向兼容性。

这一表述能否成为论文贡献，取决于因果实验而不是名称：至少需要 forward-only、reverse-only/bidirectional、无投影、简单对称一致性、强度匹配和多 seed 基线。若 T1/T2 没有稳定收益，现有新颖性不足以单独支撑论文；若双向版本稳定优于这些控制，再扩展定位并在 COCO/YOLO26s 复现，才具备较强的完整故事。

## 6. 五条可直接引用的一手来源

1. [YOLOv10: Real-Time End-to-End Object Detection — NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)
2. [CrossKD: Cross-Head Knowledge Distillation for Object Detection — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)
3. [Improving Object Detection by Label Assignment Distillation — WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf)
4. [Dual-head self-distillation via an auxiliary lightweight neck for YOLO-based object detection — Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.133750)
5. [Positive-Unlabeled Data Purification in the Wild for Object Detection — CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Positive-Unlabeled_Data_Purification_in_the_Wild_for_Object_Detection_CVPR_2021_paper.html)
