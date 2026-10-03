# 研究战略调整：从“改 YOLO26 刷分”转向“借 YOLO26 发现通用机制”

日期：2026-09-19  
触发：用户提出“YOLO26 已针对通用检测充分优化，继续直接拼通用 AP 的边际空间可能很小”；TQGB seed0 负复现提供了直接证据。

## 1. 判断

这个判断基本正确，但需作一个精确区分：官方主要围绕 COCO 风格的通用检测和产品约束联合优化，并不等于专门把 VOC 调到极限；因此 VOC 上仍可能有局部增益。然而，VOC 的小规模和较大随机波动会让“单次提点”尤其容易成为偶然，且这种增益很可能不能迁移回 COCO。

真正困难的不是在某次 VOC 运行中超过官方，而是同时满足：

1. 多种子稳定；
2. 固定末轮而非挑 checkpoint；
3. 胜过匹配成本/强度的 placebo；
4. 在 COCO 与另一个模型尺度或双分配检测器上仍成立；
5. 参数、FLOPs、延迟或训练开销合理；
6. 新颖性不是已有注意力、卷积、IoU loss 或 FPN 的重新组合。

## 2. 当前证据已经说明什么

- projected T1 seed42：固定 e50 相对 B50 `+0.00670` mAP50-95；
- projected T1 seed0：固定 e50 相对旧 B50 `-0.00215`，last-10 0/10 获胜；
- 两 seed paired mean 仅 `+0.002275`，样本标准差约 `0.00626`；
- 因而“+0.67 AP”目前更像值得解释的随机轨迹敏感信号，而不是已经稳定的模型改进。

这不是白做：它暴露了一个更有论文价值的问题——训练期辅助梯度、全局裁剪和双分配路径之间的耦合可能让单 seed 增益极不稳定。当前 T1-NP、同开机 B50 与后续必要的 BRC 正是用来分离这些因素，而不是继续扫 gain。

## 3. 新的论文对象

不再把论文问题写成：

> 如何改进 YOLO26n 在 VOC 上的 AP？

而写成可迁移的机制问题，例如：

> 双分配检测器中，如何在不增加推理开销的情况下控制实例、分配或相位信息在训练中的丢失与失衡？

YOLO26n 是快速发现与证伪平台；最终方法至少要能解释为何也适用于 YOLOv10 或另一种 O2M/O2O 检测器，或适用于不同尺度的同族模型。

## 4. 重排后的方向优先级

### A. 完成 TQGB 因果闭环，但停止“救结果式”调参

- 完成当前 T1-NP seed0 与同开机 B50 seed0；
- 再按冻结协议决定是否需要 seed1/BRC；
- 不根据 VOC07 test 改阈值、gain 或挑 epoch；
- 若同开机证据仍不稳定，TQGB 作为负结果/机制发现归档，主线转移。

### B. IGB：优先级最高的独立机制探针

问题不是“给 YOLO 加 loss”，而是检验原生 TAL/O2M 是否让少数 GT 的 target mass 支配 shared-feature gradient。先运行已经实现并审计的 exact per-GT VJP ledger；只有 Gini、top-20% share 和 mass-gradient 相关性同时过线，才实现 batch-conserving 配额修正。它推理零开销，且问题可迁移到其他 dense/dual-assignment detectors。

### C. SPAR：优先级最高的表示层训练机制

检验 stride-2 下采样是否不可逆丢失目标相关的亚像素相位细节。先做 held-out ridge decodability + 一像素平移等变误差的零更新 P0；只有出现 object-specific gap 才加入训练期 Haar-detail predictor。推理时删除全部辅助结构，避免与官方部署图直接拼模块。

### D. SARC：保留为小预算结构候选

SARC 已有 assignment-level separation 的 GO 探针与可重参数化实现，但它仍增加约 `1.3%` GFLOPs，且容易被“普通 7×7 DWConv 就有效”解释。只有完整 R3 明显超过同成本 R2，才保留 assignment-calibration 论文叙事。

## 5. 分阶段证据标准

```text
train-only 零更新探针
        ↓ GO
内部 train/dev 10-epoch 因果三元组（baseline / placebo / semantic method）
        ↓ GO
VOC fixed-e50 多种子
        ↓ GO
VOC 150-epoch + 健康硬件复现
        ↓ GO
COCO2017 + 另一模型尺度/双分配架构
```

任何阶段失败就归档并换机制，不在 VOC07 test 上小步扫参。这样即使最终结果是否定的，每次试验也会留下关于 YOLO26/双分配检测器内部机制的可复用证据，而不是一堆无法解释的模块组合。

## 6. 当前执行结论

当前 T1-NP 继续运行是合理的，因为它回答 projection 因果，而不是追求把负 seed 调正。其后优先完成同开机 B50；GPU 空闲窗口先运行 IGB-P0。SPAR-P0 随后实现为外部零更新探针，不修改正式训练源码。TQGB 若不能达到冻结的多种子门槛，正式从主线退出。
