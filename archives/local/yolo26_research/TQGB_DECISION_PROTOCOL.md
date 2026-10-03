# TQGB 结果前冻结的决策协议

日期：2026-09-19  
用途：在 T1 固定第 50 轮结果尚未知时，预先规定后续动作，减少根据 VOC07 test 曲线追参数的自由度。

## 1. 固定比较

- 主指标：固定 epoch 50 的 O2O NMS-free mAP50-95；
- 同协议基线：B50，seed 42，随机初始化，完整 VOC07+12 trainval；
- 辅助诊断：last-5/last-10、独立 `last.pt` O2O/O2M 复评、路由覆盖、投影保留率和 MCE/SIGBUS；
- 辅助诊断不得替代固定末轮，也不得用于挑 best epoch；
- 当前服务器结果统一标记为 faulty-host exploratory，论文结论必须在健康硬件从头复现。

## 2. 第一层：分类路线 T1/T2

按顺序完整运行，不中途早停：

1. T1：仅 O2M→O2O 分类路由，projection on，增益 0.08；
2. T2：双向分类路由，projection on，增益 0.08/0.015。

增益、阈值与 margin 均已由训练数据探针冻结，不根据 T1/T2 test 结果重调。

## 3. 结果触发规则

### A. T1 固定末轮相对 B50 ≥ +0.003

把“单向有界迁移”升级为候选最小方法，随后运行：

- T0-current：在与 T1 完全相同的最终研究源码快照下，以 gains=0/route=none 完整重跑 seed 42，排除源码或环境漂移；
- T1-NP：同增益、同路由但关闭 projection，检验有界目标是否真正贡献；
- seeds 0/1 的 B50–T1 成对复现，与现有 seed 42 合成三对；
- 只有 T1 多 seed 平均为正、至少 2/3 seed 为正，且 T1 优于 T1-NP，才继续长训与 COCO。

### B. T2 相对 B50 ≥ +0.003，且比 T1 至少高 +0.002

把“deployment-to-trunk bridge”升级为候选贡献，随后运行：

- T2-R：只保留 O2O→O2M 反向分类路由，隔离 bridge；
- T2 的 no-projection 控制；
- global clipping/分组 clipping 控制，排除统一裁剪系数带来的间接耦合；
- seeds 0/1 的 B50–T2 成对复现。

只有 bidirectional 稳定优于 forward-only、reverse-only 和 no-projection，才可主张可逆路由组合有效。

### C. 固定末轮为小幅正值但低于 +0.003

不扫 gain、threshold 或 margin。先看预先规定的三对 seed；若平均差不为正或由单一 seed 驱动，停止该分类方向。若三对一致为正，再把它视为小效应候选，而不是在 seed 42 上继续调参放大。

### D. T1、T2 均不为正

停止分类 TQGB，不围绕 test 调阈值。切换到独立的定位假设：只实现并校准 **O2O→O2M localization bridge**，原因来自结果前已有训练探针——中后期反向定位覆盖仍为 7.11%–18.26%，监督残差投影存活率为 100%，且分类/定位赢家约 48%–54% 不一致。先用训练 checkpoint 做梯度校准，再只跑一个 localization-only 固定配置。

若 localization-only 也失败，则 TQGB 整条路线停止，转入新的机制诊断，不把分类和定位负结果叠加成复杂模型。

## 4. 论文升级门槛

VOC seed 42 的单次 +0.003 只代表进入复现，不代表可发表。升级顺序固定为：

1. 三组配对 seeds；
2. 健康硬件从头复现；
3. 固定 150 epochs；
4. projection、方向与 clipping 因果消融；
5. COCO2017 YOLO26n；
6. YOLO26s 或另一通用模型尺度；
7. 参数、FLOPs、训练时间与推理延迟报告。

投稿叙事只使用严格查重后可防守的范围：assignment-native、receiver-bounded cross-head self-distillation。不得声称首次双头、自蒸馏、在线互教、GT-aware KD 或全局无梯度冲突。
