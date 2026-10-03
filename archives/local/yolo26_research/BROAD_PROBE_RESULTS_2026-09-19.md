# Backbone/Neck 广域候选零更新探针结果

日期：2026-09-19  
数据：固定 512 张 VOC trainval 图像，B50 epoch 11/21/31/41/50 checkpoints  
运行约束：physical GPU 4、actual workers 4、pinned memory、NUMA 1、BF16；无 optimizer、无权重更新；无新增 MCE/SIGBUS

## 1. PBGA-P0：NO-GO

PBGA 假设 O2M 与反事实解锁的 O2O 监督在 P3/P4/P5 共享边界上存在可投影的冲突梯度。结果为：

- 5 个检查点 × 3 个尺度的 **total-gradient negative-dot rate 全部为 0/8**；
- total-gradient cosine 中位数始终为正：P3 约 `0.241–0.287`、P4 `0.087–0.139`、P5 `0.068–0.091`；
- O2O/O2M norm ratio 并不小，且随 ProgLoss 后期权重迁移升高，epoch 50 的 P3/P4/P5 中位数为 `24.17/75.34/77.03`；
- 因为没有负点积，PBGA 的 projection 与单纯 norm-cap 完全相同，projected norm removed 始终为 0。

预注册门槛要求至少一个尺度在至少两个 checkpoint 同时达到 negative-dot rate `>=0.10` 且 norm ratio `>=0.05`。没有任何尺度满足，自动裁决为 `NO_GO`。

附带的重要工程结论：直接解除 O2O feature detach 会让后期 O2O 梯度主导 shared trunk。epoch 50 naive reattach 的完整梯度 norm 均值约 `662.75`，官方约 `56.42`，global-clip 相对尺度均值仅 `0.0928`；即使做 norm-cap，仍只是在控制幅值，并没有“冲突仲裁”的机制基础。因此不浪费一个正式 50 轮任务验证 PBGA。

## 2. SARC assignment histogram：GO

SARC 检查分配给 P3/P4/P5 的正 anchor 是否具有层级特异且时间稳定的

```text
u = log2(sqrt(w*h) / stride)
```

分布。固定面板 fingerprint 为 `0586e541229571cbe4849dc1efab4008f6e3a051a262dba6892f902893818472`。

### 层间结构

五个 checkpoint pooled positive-anchor JS：

| 对比 | JS |
|---|---:|
| P3 vs P4 | 0.01662 |
| P3 vs P5 | 0.12339 |
| P4 vs P5 | 0.06871 |

去重 `(image, GT, level)` 后仍为 `0.02050/0.03500/0.02160`，说明差异不只是 TAL top-k anchor multiplicity 造成。

### 时间稳定性

- 同层相邻 checkpoint 最大 JS 仅 `0.007918`，远低于不稳定 no-go 阈值 `0.30`；
- pooled positive counts 为 P3/P4/P5=`10697/18959/32991`，远高于每层 100 的样本下限；
- epoch 50 的 mean `u` 为 P3/P4/P5=`2.675/2.746/3.146`；分布差异随训练后期保持稳定。

因此满足“存在层间结构、时间稳定、样本充分”的全部预注册条件，自动裁决为 `GO`。

## 3. 科学含义与正式矩阵

SARC 的 GO 只证明 assignment-calibrated receptive-field allocation 有可利用的统计底座，不证明 AP 会提升。下一步必须保留三层因果控制：

| ID | 结构 | 回答的问题 |
|---|---|---|
| R1 | 每层单个 7×7 DWConv | 固定大感受野是否足够 |
| R2 | 可重参数化多分支，无 calibration loss | 训练期 over-parameterization 是否解释收益 |
| R3 | 完整 SARC | assignment calibration 是否提供额外收益 |

只有 R3 固定 epoch 50 相对同源码 R0 `>=+0.003`、last-5 `>=+0.002`，且相对 R2 `>=+0.002`，才可把 assignment calibration 作为论文贡献。若 R1/R2 与 R3 相当，只能报告为结构工程改进。

## 4. 产物与哈希

- PBGA：`diagnostics/pbga_p0_b50_train/`，JSON SHA-256 `37aaeec55564e5a1a70ca0fcf9860f8d9974ce6ad7fa61bedc6d584786cb7da9`；
- SARC：`diagnostics/sarc_assignment_b50_train/`，JSON SHA-256 `c0be277c260684ecd1d4b073cfc5f457514d53b27c38200022e7c393b8b58912`；
- 两个探针均强制检查 actual workers 4、pinned memory、固定面板一致性和 zero-update 条件。
