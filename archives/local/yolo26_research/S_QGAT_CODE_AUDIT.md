# S-QGAT v1 独立代码审计

日期：2026-09-19  
对象：正在运行的 50-epoch S-QGAT v1 正式筛选配置  
结论：在**从头启动、不中断、无 NaN recovery**的前提下，未发现足以令当前 run 作废的致命实现错误。epoch 时序、DDP 汇总、KD 尺度和冻结配置一致。

## 1. 当前 run 的有效性检查

- 运行参数明确为 `resume=False`；训练日志中没有 NaN、recovery、Traceback、SIGBUS 或续跑记录。
- epoch 1–5 的 KD 为 0；epoch 5 结束冻结 `rho_ref`；epoch 6–15 按 0.1→1.0 ramp；epoch `t` 使用 epoch `t-1` 的完整训练统计。
- DDP 状态计数在每轮训练结束后统一 `all_reduce`，位于验证之前；验证不会回写 live controller。
- 当前配置显式覆盖 `gain=1/3`、`fade=0`、`state=teacher_better`，不受安全默认值 `gain=0/state=none` 影响。

## 2. P1：禁止 resume 或自动恢复

controller 的 `kd_state_scale`、`kd_reliability`、`kd_reliability_ref` 和本轮计数是 criterion 的普通 Python 状态，目前没有写入 checkpoint。手工 resume 只恢复 epoch/ProgLoss 进度；已经冻结的 reference 和单调退休状态会重置。自动 NaN recovery 则可能在回滚模型权重时不回滚 controller，造成 schedule 和状态提前一轮。

因此 S-QGAT v1 的硬规则是：

1. 任何 SIGBUS、NaN recovery、OOM restart 或人工中断都使当前 run 不再符合预注册；
2. 不从故障 checkpoint 续训，必须修复原因后从头运行；
3. 下一版应显式保存/恢复完整 controller 状态，或在 state controller 开启时检测到 resume/recovery 直接失败。

## 3. P2：实现和文档需要统一的边界

代码实际使用：

```text
denominator = max(rho_ref - 0.5, 1e-6)
evidence = clip((rho_t - 0.5) / denominator, 0, 1)
state_next = min(state_current, evidence)
```

这与旧规范中分母写作 `rho_ref - 0.5 + eps` 不完全相同。当 `rho_ref <= 0.5` 时，当前实现会 fail-closed、迅速退休教师，符合已有 smoke 的预期；因此修正文档，不改正在运行的代码。

DDP 正常路径安全，但 `_kd_epoch_common is None` 时会在 collective 前提前返回。当前每个 rank 每批都会建立零或正计数，因此不会分叉；更通用的实现应在所有 rank 无条件用 `int64` 零计数参与同一个 collective。

## 4. P2：验证指标的解释边界

EMA validation 的 criterion 不继承 live controller 的 `updates/rho_ref/state_scale`。这不影响 AP 推理，也不影响训练优化，但验证日志中的 `val/kd_*` 和 validation loss 不能被解释为训练 controller 的同期状态。论文机制图和控制器分析只使用 `train/kd_*`；验证侧只读取 P/R/AP。

## 5. P2/P3：测试覆盖仍需补齐

当前已实际通过单卡真模型、两进程 CPU DDP 合成测试和四卡真实 DDP smoke；但版本化回归脚本本身尚未完整覆盖：

- checkpoint/resume round-trip；
- NaN recovery rollback；
- EMA validation 状态复制/禁写；
- rank 间空候选不均衡；
- warmup reference 无 common evidence；
- BF16 下的完整 controller trainer 集成。

这些不推翻当前 run，但在健康硬件正式复现前必须变成可重复的自动测试。

## 6. P3：active reduction 的准确含义

active 模式先把 advantage 权重在每张图内归一化到有效候选数，再除以有效候选数。因此：

- 非空图的总 KD 强度不会随候选数自然下降；
- advantage 绝对幅度不会控制总强度，只保留候选间相对权重；
- 空候选图仍以零项参与 local-batch mean。

这不是当前实验错误：S-QGAT v1 故意保留 D2 的 active reduction，只改变全局状态控制。但论文不能把它表述为“质量差越小，总蒸馏强度自然越弱”。后续 common-reduction 消融仍是必要控制。

## 7. 日志的一轮语义差

CSV 同一行中：

- `kd_state_scale/kd_reliability/ref` 是该轮前向时使用的、来自上一完整 epoch 的状态；
- `teacher_better` 等普通批统计来自当前 epoch；
- epoch-end update 在保存附近发生，但不会回填本轮 `tloss`。

后续分析把 controller 列明确解释为 `prev-state used in this epoch`，不能把同一行的 reliability 和 teacher_better 当作同一统计窗口。

## 8. 正式复现前的修复清单

1. checkpoint 保存并恢复完整 controller 状态；recovery 事务化。
2. 所有 rank 无条件进行 epoch-end collective，计数改为 `int64`。
3. validation criterion 只读同步 controller，或完全关闭 validation 的 controller 诊断累积。
4. 对所有超参数做 `isfinite` 检查；无 reference 时 fail-closed。
5. masked reduction 改用 `torch.where` 或索引，避免潜在 `NaN * False`。
6. 将日志字段重命名为 `*_prev`，并单独记录刚完成 epoch 的全局 `rho_t`。

