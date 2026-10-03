# YOLO26 O2M→O2O 监督冲突轨迹

> 固定 checkpoint、无权重更新、无训练增强；所有比例由跨 batch 原始计数汇总。

| epoch | split | common | quality candidates | candidate rate | grad agree | teacher closer | overshoot | p_t | p_s | y |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | val | 11901 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0000 |

- `grad agree`：KD 与 O2O soft-target BCE 对类别 logit 的更新方向一致。
- `teacher closer`：教师概率比学生概率更接近 O2O TAL soft target；这是更严格的安全条件。
- `overshoot`：教师概率高于 O2O soft target；它不必然有害，但提示原始教师概率可能过强。
- 本报告只诊断机制，不包含精度改进结论。
