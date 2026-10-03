# S-QGAT v1：方法定义与实验预注册

日期：2026-09-18  
状态：实现与 CPU/DDP 合成测试已通过；完整训练尚未读取。本文档冻结首轮定义，避免看到 test 结果后改规则。

## 1. 目标

现有 D2 只在 O2M 比 O2O 更优的同 anchor/同 GT 位置做分类蒸馏，但 active-count normalization 会把少数候选重新放大为满强度；同时，O2M 的总体教师可靠性会随训练下降并最终反转。S-QGAT v1 的目标是保留早期高质量指导，同时依据训练期证据单调退休教师，不用 test 指标手工选择停止 epoch。

## 2. 局部质量门控

对两头都分配到同一 GT 的 anchor，取对应 GT 类别的概率 `p_m`、`p_o`，以及两头预测框对该 GT 的 IoU `u_m`、`u_o`：

```text
q_m = sqrt(p_m * u_m)
q_o = sqrt(p_o * u_o)
```

O2M→O2O 候选需同时满足：

```text
q_m >= 0.25
q_m - q_o >= 0.05
p_m > p_o
```

教师 logits 完全 stop-gradient。分类迁移使用 Bernoulli KL；候选按质量优势加权。主实验保留 active reduction，以便相对 D2-GM 只改变状态控制；common reduction 作为后续归一化消融。

## 3. 全局教师状态

在第 `t` 个训练 epoch 内，四个 DDP ranks 对所有 common same-GT 位置累计：

```text
rho_t = count(q_m > q_o) / count(common)
```

只在 epoch 结束时做一次跨 rank 求和。第 5 轮 warmup 完成后的 `rho_5` 冻结为 `rho_ref`。下一轮使用：

```text
evidence_t = clip((rho_t - 0.5) / max(rho_ref - 0.5, eps), 0, 1)
state_{t+1} = min(state_t, evidence_t)
```

其中 0.5 表示两头在全局上不存在明确教师优势。`min` 使教师退休单调，避免胜率在 0.5 附近的随机抖动重新激活已经失效的知识流。

完整 KD 系数为：

```text
lambda_t = (1/3) * ramp_t * state_t
```

- epochs 1–5：warmup，KD=0；
- epochs 6–15：线性 ramp；
- 不使用手工 terminal fade；
- `state_t` 只来自上一完整 train epoch，严格因果；
- val/test 从不参与控制器。

实现边界：当 `rho_ref <= 0.5` 时使用 `max(..., eps)` 使控制器 fail-closed；S-QGAT v1 的 controller 状态尚未进入 checkpoint，因此任何 resume、自动 NaN recovery 或 epoch restart 均使该 run 不符合本预注册，必须从头重跑。CSV 中的 controller 列表示本轮实际使用的上一轮状态，而普通 `teacher_better` 批统计属于当前轮。

## 4. 为什么 gain=1/3

固定训练批次梯度探针显示，在相同 gain=0.5 下，D2 质量门控 KD 梯度比 D1 无门控高 32%–53%。按训练数据上 `0.5 × ||g_D1|| / ||g_D2||` 校准，中后期匹配值约 0.33–0.34。因此首轮固定为 1/3，不基于 VOC07 test 扫描。

## 5. 冻结训练协议

| 项目 | 设置 |
|---|---|
| 模型 | 官方结构 YOLO26n，随机初始化 |
| 数据 | VOC07+12 trainval；VOC07 test 每轮仅作探索监控 |
| epochs | 50 |
| batch | 64，全局 |
| GPU | 仅物理 GPU 4–7 |
| workers | 每 rank 4 |
| pinned memory | 开启（框架默认） |
| NUMA | CPU 与内存绑定 node 1 |
| seed | 42 |
| precision | BF16 |
| 主指标 | 固定第 50 轮 O2O NMS-free mAP50-95 |
| run name | `sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1` |

## 6. 首轮判定规则

首轮是 faulty-host、单 seed 方向筛选，不是论文结论：

- `ΔmAP50-95 vs B50 >= +0.003`：进入配对多 seed；
- `0 < Δ < +0.003`：保留机制但不立即扩种子，先比较 D2-GM 和归一化控制；
- `Δ <= 0`：停止 S-QGAT v1，不围绕 test 曲线继续调阈值；转向 TQGB 的任务级双向路由；
- 无论结果如何，报告固定末轮、last-5/last-10 均值、完整曲线、O2M 旁路和机制统计；不以 best.pt 替代主结果。

## 7. 工程验收

完整训练前必须满足：

1. 默认 `state=none` 与现有实现数值兼容；
2. active/common reduction 的合成比例测试通过；
3. 两进程 DDP 汇总后得到完全相同的状态尺度；
4. 教师一旦退休，不会因后续波动重新激活；
5. 增益为 0 时官方基础损失不变；
6. 真模型反向传播与短时 DDP smoke 无 NaN、无死锁；
7. 开跑前后检查新增 MCE/SIGBUS，故障后不续接 checkpoint。
