# ASAC A0 前缀等价审计（epoch 1–24）

审计快照：2026-09-19 23:12:38+08:00。对象为正在运行的 A0 `asac_a0_instrumented_e50_g4567_s42_w4_pin_numa1` 与旧官方 B50 seed42。审计只读，未读取大 checkpoint、未修改训练进程。

## 结果

A0 与旧 B50 的 bitwise/numeric equivalence 门禁 **FAIL**：

- A0 前缀 24 行与 B50 对应 epochs 完全对齐。
- 两者有 75 个共同非 wall-clock 字段，共比较 1,800 个数值格；1,276 格非零差异，58/75 字段出现差异。
- epoch1 的 75/75 字段完全一致；从 epoch2 开始即有 53 个字段分叉，之后每轮约 54–57 字段不同。
- epoch24 的 mAP50-95 为 A0 `0.41409`、B50 `0.41975`，差 `-0.00566`；该前缀数值只用于定位分叉，不能用于正式末轮裁决。
- A0 自身协议状态正常：train_q=q_next=1、固定 panel/population/RNG/source/config hash、population=624、最低 coverage=623/624、无 assigner OOM fallback。

## 解释边界与处置

epoch1 完全一致、epoch2 起分叉，与 ASAC 基础设施在每轮边界重置 persistent DataLoader 以清除旧 q 预取的时机一致，是当前首要因果候选；精确代码级诊断另行记录。无论具体原因，A0 不再称为“旧官方 B50 的逐位复现”，而应称为“ASAC 基础设施下固定 q=1 的配对仪器化控制”。

A2 与 A0 使用同一 ASAC trainer、逐轮 loader reset、固定 panel 和完整性协议，只有 q 策略不同，因此完成 A0 仍是评估动态 q 相对固定 q=1 的必要配对对照。旧 B50 保留为外部官方训练参考；论文级结论必须同时区分 `A2-vs-paired-A0` 和 `A2-vs-official-B50`，不能混称。

发现发生在 A2 完成、A0 epoch24 时；A0 配置已经事先冻结。继续 A0、不修改任何超参数，也不根据已见前缀 AP 调整方法。
