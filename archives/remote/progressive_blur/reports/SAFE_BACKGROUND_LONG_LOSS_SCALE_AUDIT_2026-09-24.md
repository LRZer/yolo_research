# 安全背景模糊 125 轮：联合损失缩放审计

记录：2026-09-24 约 02:45 CST。此审计在长训运行中以只读方式核对本地与服务器代码；没有修改正在运行的模型、配置或进程。

## 实际计算图

- 配置字段 `pgb_map_loss_weight=0.25`。`safe_gate_loss` 在目标保护区域使用权重 4.0，在参考图强度至少 0.8 的远背景使用权重 0.25；中间区域不施加辅助监督。`train/map_loss` 记录的是 `0.25 * safe_gate_loss`，不是未加权值。
- 当前服务器的 `v8DetectionLoss.loss()` 返回三元素张量 `batch_size * [box_loss, cls_loss, dfl_loss]`，并返回三项已加检测权重的记录值。
- `SafeBackgroundDetectionModel.loss()` 用 `detector_total + weighted_map_loss * batch_size`。标量辅助损失广播到上述三个元素。训练器随后执行 `loss.sum()`，因此联合阶段实际优化的是

  `batch_size * (box_loss + cls_loss + dfl_loss + 3 * 0.25 * safe_gate_loss)`。

  有效辅助系数为 **0.75**，不是配置字面的 0.25。DDP 对整项损失的共同缩放不改变这个 3 倍相对系数。此问题也存在于复用的通用 PGB 联合损失写法；`map-only` 分支返回标量，不受三元素广播影响。
- 第 56 轮训练日志：box=1.08546、cls=1.42500、dfl=1.45814，三项和约 3.96860；显示的 map=0.06817，实际相加约 `3 * 0.06817=0.20451`，约为三项检测损失和的 5.2%。这是损失值比例，不是门控参数梯度比例；不能据此断言门控监督过弱或过强。检测损失也通过可微的图像处理链更新门控网络。

## 对实验的影响与处置

- 预注册文档对辅助损失权重的表述与运行时实际效果不一致。不得把本轮写作 `lambda=0.25` 的严格预注册验证；应标注为“配置 0.25、实际有效 0.75 的已审计运行”。
- 原图基线没有门控辅助损失，仍按同一数据、检测器起点、125 轮和训练设置排队。完成后的配对 AP 比较仍可描述此**实际实现**的效果，但不能解释为原计划的辅助系数实验。
- 不在运行中改代码或配置，也不从中途 checkpoint 以新系数续跑拼接训练史。如要检验真正的 0.25，应先修复广播问题、加标量总损失单测，然后从共同起点重新做完整配对训练。
- 是否需要调整辅助权重，应依据门控保护率、远背景选择率、检测 AP、分层漏检及最好在同一 batch 上测量各项对门控参数的梯度范数/方向，而不能只比较 `0.07` 与 `1.x` 的显示数值。

证据路径：远端 `/home/ubuntu/rz/progressive_blur_detection_research/pgb/safe_background_integration.py` 的 `SafeBackgroundDetectionModel.loss()`；`/home/ubuntu/rz/yolo_study/ultralytics/ultralytics/utils/loss.py` 的 `v8DetectionLoss.loss()`；`/home/ubuntu/rz/yolo_study/ultralytics/ultralytics/engine/trainer.py` 的 `self.loss = loss.sum()`；运行目录 `runs/sbg_long_e125/sbg_joint_voc_train90_e125_s2027_g0123_numa0/results.csv`。
