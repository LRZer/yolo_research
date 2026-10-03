# T1-NP 结果前分支决策

日期：2026-09-19  
状态：T1-NP 正在运行，尚未读取固定第 50 轮结果

本文件在 T1-NP 固定末轮未知时冻结后续分支，允许快速推进，同时避免看完 seed 42 后临时重写机制故事。

## 共同规则

- 主指标仍为固定 epoch 50 的 O2O NMS-free mAP50-95；last-5/last-10 只用于稳定性诊断。
- 比较基准使用已证明与最终源码逐数值、逐张量完全一致的 T0/B50。
- 不按 best epoch 选结果，不在 seed 42 上继续扫 gain、margin 或 score threshold。
- 所有当前主机结果仍是 faulty-host exploratory；任一 SIGBUS 使对应运行失效。

## 分支 A：Projected T1 明确优于 T1-NP

若 T1 固定末轮高于 T1-NP，且 T1 仍高于 T0/B50，则保留“receiver-bounded projection”作为候选机制，按 seed 0、1 运行配对 B50→T1。BRC clipping control 仍必须完成，只有多 seed 与 BRC 均支持时才升级论文主张。

## 分支 B：T1-NP 与 T1 持平或更高

这将否定“projection 是 T1 增益来源”的因果表述，但不自动否定同 GT 的 O2M→O2O 路由。若 T1-NP 固定末轮相对 T0/B50 仍达到 `+0.003`，则把 T1-NP 透明标记为由预注册阴性控制产生的新候选，不再称 receiver-bounded 方法；随后运行 seed 0、1 的 B50→T1-NP 配对，而不是继续为 projected T1 扩种子。

为避免结果后改配置，conditional seed-0/1 YAML 已在结果未知时创建；只有本分支被触发才同步和运行。

## 分支 C：两者均无稳定正收益

停止分类 TQGB，不扫参数挽救。保留 T1 为单 seed 偶然信号，资源转向已通过训练侧探针且具有独立结构假设的 SARC，以及通过加固审计后的 ASAC 稳定性课程。

## 额外判断

- 若 T1 与 NP 都为正但差值小于 VOC 单 seed 噪声，只能说明“projection 尚无因果证据”；后续优先复制更简单的 NP。
- 若 NP 仅固定末轮高、而 last-5/last-10 明显不支持，先将其记为边界证据；不能挑末轮后立刻包装为成功。
- 任一进入多 seed 的候选必须至少 2/3 paired deltas 为正，并报告三对原始值、均值和标准差。
