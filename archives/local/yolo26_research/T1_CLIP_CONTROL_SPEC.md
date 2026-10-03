# TQGB-T1 全局梯度裁剪因果控制规范

日期：2026-09-19  
状态：设计冻结；本地 staging 已实现并通过 CPU 合成测试，**尚未同步远端、尚未完成 YOLO/NCCL smoke、尚未训练**  
适用对象：classification-only TQGB-T1（O2M→O2O，`gain=0.08`）

## 1. 要排除的混杂

T1 的路由项单独反传时，直接梯度只进入 O2O classification head；O2M head、box head 和 shared trunk 的路由梯度均为零。但是当前训练器把基础损失和路由损失相加，再对全模型执行一次 `clip_grad_norm_(..., 10)`。因此附加的 O2O-classification 梯度仍会改变统一裁剪系数，间接缩放其余参数。

令基础目标和 T1 路由目标分别为

\[
L_B,\qquad L_R,
\]

其梯度为

\[
g_B=\nabla_\theta L_B,\qquad g_R=\nabla_\theta L_R.
\]

当前标准 T1 实际使用

\[
g_{\mathrm{std}}=\alpha_T(g_B+g_R),\qquad
\alpha_T=\min\left(1,\frac{C}{\lVert g_B+g_R\rVert_2}\right),\quad C=10.
\]

虽然对所有非 O2O-classification 参数都有 `g_R=0`，这些参数仍从 `alpha_B g_B` 变成了 `alpha_T g_B`。这就是需要隔离的 shared-scale coupling；它不是 stop-gradient 失效。

现有固定-checkpoint 校准给出的 T1 `alpha_T/alpha_B` **batch mean** 为：

| reported epoch | `alpha_T/alpha_B` | 平均额外缩放 |
|---:|---:|---:|
| 11 | 0.999730 | 0.027% |
| 21 | 0.998774 | 0.123% |
| 31 | 0.998704 | 0.130% |
| 41 | 0.994641 | 0.536% |
| 50 | 0.996128 | 0.387% |

这些数值表明混杂很小，但它们只是每个 checkpoint 的 8-batch 均值，不是逐 batch 上界，不能据此直接宣布混杂已被排除。

## 2. 优先推荐：Base-Referenced Clipping（T1-BRC）

### 2.1 定义

保持 T1 的 forward、same-GT/same-class route、route count、common-anchor normalization、监督线段投影、`gain=0.08` 和完整附加 O2O-head 梯度全部不变。唯一变化是：裁剪系数由**移除 TQGB 路由后的基础梯度**计算，再施加到完整梯度上：

\[
\alpha_B=\min\left(1,\frac{C}{\lVert g_B\rVert_2}\right),
\]

\[
g_{\mathrm{BRC}}=\alpha_B(g_B+g_R).
\]

于是对任意非 O2O-classification 参数 `theta_j`：

\[
g_{R,j}=0\quad\Longrightarrow\quad
g_{\mathrm{BRC},j}=\alpha_B g_{B,j},
\]

即它在当前状态下与“没有路由时的官方全局裁剪”完全相同；T1 仍以原方向、原幅度相对于 `alpha_B` 更新 O2O classification head。总梯度范数可能略高于 10，但只高出 `||g_B+g_R||/||g_B||` 的小量；不能再做第二次全局裁剪，否则会把混杂重新引入。

### 2.2 为什么优先做这个控制

1. 它只改变一个明确的优化器操作，不改变 T1 的数据选择、教师概率、投影或损失强度；
2. route count、common normalization、路由 loss 和额外 O2O-classification 梯度均与标准 T1 **完全相同**，不是近似匹配；
3. 对无路由基线，`g_R=0`，BRC 数学上退化为现有官方 global clipping，因此无需为了这个控制重新定义基线优化目标；
4. 相比“关闭 clipping”或把参数粗分成两个各自 `max_norm=10` 的组，BRC 不会大幅改变基础训练动力学；
5. 它直接回答审计问题：T1 的收益是否依赖附加头部梯度对 shared/O2M 参数造成的统一缩放。

不优先采用简单的 two-group `10/10` clipping。该做法会把 O2O classification head 和其余参数各自裁到 10，后期可显著放大两个组的总更新，形成一个新的、大于原混杂的优化器变化。

## 3. 最小实现接口

### 3.1 配置项

在 `ultralytics/cfg/default.yaml` 增加：

```yaml
tqgb_clip_reference: combined  # combined（当前行为）或 base（BRC）
```

并在配置校验中把它登记为字符串枚举。默认必须为 `combined`，保证官方行为和已完成实验不变。

`base` 首版只允许以下组合：

- `detection_research: true`；
- `tqgb_direction: m2o`；
- `tqgb_cls_m2o_gain > 0` 且 `tqgb_cls_o2m_gain == 0`；
- `o2o_kd_gain == 0`；
- O2O head 存在 `one2one_cv3`；
- 每次 optimizer step 只有一个 micro-batch（本协议 `batch=64, nbs=64`，应为 `accumulate=1`）；
- 首版禁用 `compile`。

任一条件不满足都应 fail fast，而不是静默回落到普通 global clipping。

### 3.2 路由梯度捕获

现有 `E2EDetectLoss.__call__` 返回的 loss vector 最后一项是已经乘以 local batch size 的 TQGB loss。训练器在标准总反传前，对 `one2one_cv3` 参数执行一次保留计算图的 `autograd.grad(route_loss, params, allow_unused=True)`，得到未缩放的 local `g_R`。

DDP 下必须使它与现有总梯度口径一致：当前训练循环把总 loss 乘 `world_size`，DDP 再平均，因此最终 `.grad` 等于各 rank 原始 local 梯度之和。手工捕获的 `g_R` 应在 ranks 间执行 `SUM all_reduce`，不能再除以 `world_size`。这个等式必须由两进程数值测试确认，不能只凭推理上线。

首版只需捕获 `one2one_cv3`，因为 T1 的真模型梯度隔离测试已证明其它组的 `g_R=0`。不要为了泛化 T2 而对全模型执行第二遍 `autograd.grad`；那会增加显存和实现风险。T2/O2O→O2M 若以后需要 BRC，应另行扩展到 shared/O2M 参数并重新验证。

### 3.3 optimizer step

在 `scaler.unscale_(optimizer)` 之后、任何裁剪之前：

1. 将 DDP 聚合后的 route gradient 从 `one2one_cv3` 的 combined `.grad` 中减去，得到这些参数的 `g_B`；其它参数的 `.grad` 本身就是 `g_B`；
2. 用所有可训练参数的 `g_B` 计算 FP32 全局 L2 norm；
3. 计算 `alpha_B=min(1, 10/base_norm)`；
4. 不改梯度方向与组成，直接把所有 combined `.grad` 乘 `alpha_B`；
5. 正常执行 scaler/optimizer/EMA 后续流程；
6. 每个 step 清空缓存的 route gradients，禁止跨 step 复用。

等价伪代码：

```python
route_grads = capture_and_ddp_sum(loss_vector[-1], one2one_cv3_params)
backward(loss_vector.sum())

scaler.unscale_(optimizer)
base_sq = 0.0
for p in all_trainable_params:
    g_route = route_grads.get(id(p))
    g_base = p.grad if g_route is None else p.grad - g_route
    base_sq += g_base.float().square().sum()
alpha_base = min(1.0, 10.0 / sqrt(base_sq))
for p in all_trainable_params:
    if p.grad is not None:
        p.grad.mul_(alpha_base)
optimizer_step_without_second_clip()
```

实现时不要依赖裸的“loss vector 第 5 项”而没有断言。至少同时断言 criterion 是研究版 `E2EDetectLoss`、loss vector 长度和 T1 配置；更稳妥的后续接口是 criterion 显式暴露 `tqgb_loss_index`。

### 3.4 必须落日志的量

每 step 记录并按 epoch 汇总：

- `grad/base_reference_norm`；
- `grad/combined_norm`；
- `grad/tqgb_route_o2o_cls_norm`；
- `grad/clip_scale_base_applied`；
- `grad/clip_scale_combined_counterfactual`；
- `grad/shared_scale_ratio_counterfactual = alpha_T/alpha_B`；
- `grad/postscale_combined_norm`；
- route-gradient capture 是否覆盖全部 `one2one_cv3` 参数。

DDP 报告必须是全局口径；不可把 rank-0 local norm 标成四卡全局 norm。

## 4. 测试门槛

在正式训练前必须依次通过：

### 4.1 单卡解析/数值测试（不更新权重）

1. **route off 恒等性**：`gain=0` 时 `alpha_B == alpha_T`，BRC 后每个参数梯度与现有 global clipping 一致；FP32 最大相对误差 `<1e-6`；
2. **base 重建**：在同一 forward 上分别直接求 `grad(L_B)`，并与 `grad(L_B+L_R)-grad(L_R)` 比较；FP32 `<1e-6`，BF16 `<5e-4`；
3. **组隔离**：捕获的 `g_R` 在 `one2one_cv3` 非零，在 shared、O2M cls、两个 box groups 精确为零；
4. **T1 保真**：BRC 与标准 T1 的 route mask/count、common count、projection keep、raw/scaled route loss、`g_R` 张量完全一致；
5. **后裁剪关系**：非 O2O-classification 参数的 BRC 梯度等于 `alpha_B*g_B`；O2O-classification 等于 `alpha_B*(g_B+g_R)`。

### 4.2 两进程 DDP 等价测试

构造固定的两份 local batch，比较：

- 两进程捕获并 `SUM all_reduce` 的 `g_R`；
- 单进程把两份 local loss 相加后直接得到的 route gradient；
- 两进程 optimizer-step 前重建的 `g_B`；
- 单进程总 batch 的 `g_B`。

FP32 相对误差目标 `<1e-5`。测试还必须确认 `autograd.grad` 不会破坏 DDP reducer 状态、不会重复触发同步或产生 unfinished-reduction 错误。

### 4.3 BF16/四卡 smoke

GPU 释放后才运行；仍只用物理 GPU 4–7、workers=4、pinned memory、NUMA1。3 epochs、`fraction=0.02`：

- 无 NaN/Inf、deadlock、unused-gradient 或 DDP reducer 错误；
- route gradient 非零且只在 O2O cls；
- `alpha_B`、`alpha_T`、postscale norm 全部有限；
- BRC 的 route 诊断与普通 T1 同输入 smoke 一致；
- 无新增 MCE/SIGBUS。若出现硬件故障，本次 smoke 作废并从头重跑，不能 resume。

## 5. 冻结正式配置

建议文件名：

```text
configs/research/tqgb_t1_brc_cls_m2o_proj_g008_e50_w4_pin_numa1.yaml
```

核心配置：

```yaml
id: TQGB-T1-BRC-CLS-M2O-PROJ-G008-Y26N-VOC-E50-S42-W4-PIN-NUMA1
hypothesis: "T1 should retain its fixed-epoch gain when global clipping is referenced to the route-free base gradient, excluding indirect shared-scale coupling."
train:
  model: /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/models/yolo26n_baseline.yaml
  data: /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml
  epochs: 50
  patience: 50
  batch: 64
  nbs: 64
  imgsz: 640
  device: "4,5,6,7"
  workers: 4
  pretrained: false
  cls_remap: true
  optimizer: auto
  seed: 42
  deterministic: true
  amp: bf16
  fraction: 1.0
  val: true
  nms: false
  plots: false
  save: true
  save_period: 10
  project: /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs
  name: tqgb_t1_brc_cls_m2o_proj_g008_e50_g4567_s42_w4_pin_numa1
  exist_ok: false
  o2o_kd_gain: 0.0
  tqgb_cls_m2o_gain: 0.08
  tqgb_cls_o2m_gain: 0.0
  tqgb_min_score: 0.25
  tqgb_cls_margin: 0.05
  tqgb_direction: m2o
  tqgb_projection: true
  tqgb_clip_reference: base
```

除 `nbs:64` 的显式锁定和 `tqgb_clip_reference:base` 外，其余训练参数必须与 T1 完全一致。随机初始化，不能从 T1 checkpoint 继续。

## 6. 结果判据（结果前冻结）

主比较仍为固定 epoch 50 O2O NMS-free mAP50-95；辅助证据为 last-5/last-10 曲线和独立 `last.pt` O2O/O2M 复评。不得选择 best epoch。

以 B50 固定 epoch 50 `0.51276` 为主参考，标准 T1 为 `0.51946`（`+0.00670`）：

- **通过 clipping 因果控制**：T1-BRC 相对 B50 `>= +0.003`，last-5 平均差为正，独立 `last.pt` O2O 复评为正，且无技术/硬件异常。此时可以说“单 seed 下，T1 收益不需要 global-clip shared-scale coupling”；仍不能说多 seed 显著或已经可发表。
- **边界/不确定**：固定末轮在 `+0.001` 到 `+0.003`，或末轮为正但 last-5/独立复评不支持，或与标准 T1 的差超过 `0.003`。保留候选，但必须把标准 clipping/BRC 纳入 seeds 0/1 的成对复现，不能只挑更好的一条。
- **未通过**：相对 B50 `<= +0.001`，或曲线/独立复评为负。此时 T1 的正结果对裁剪实现高度敏感，不能把 `+0.0067` 归因于跨头语义迁移；应先做 clip-scale-only/语义阴性对照，或降低优先级转向其它路线。

T1-BRC 通过只能排除 clipping 混杂，**不能**单独证明 O2M teacher semantics 有效。

## 7. 下一层语义阴性对照（BRC 通过后）

为区分“真正的逐 anchor 教师信息”与“稀疏附加正类监督”，建议再做 Count-matched Random-GT control（T1-CRG）：

1. 仍在每图 common same-GT/same-class 集合 `M_i` 中工作；
2. 先按标准 T1 计算该图 route 数 `k_i`，但丢弃哪些 anchor 被路由以及教师概率；
3. 用与预测无关、可复现的整数 hash，从 `M_i` 中选出恰好 `k_i` 个 anchor；
4. 在这些随机 anchor 上以接收头自身 TAL target `t_o2o` 为 target：

\[
L_{\mathrm{CRG}}=\frac{\lambda_C}{B}\sum_i\frac{1}{|M_i|}
\sum_{j\in C_i}\mathrm{KL}_{\mathrm{Ber}}
(t^{o2o}_{ij}\,\|\,\sigma(z^{o2o}_{ij})).
\]

5. 在原 B50 的固定 train checkpoints 上、读取 CRG test 结果前，搜索并冻结 `lambda_C`，使 O2O-classification route-gradient norm、combined norm 和 `alpha_T/alpha_B` 尽量匹配 T1；报告 mean/P95/max，不只报告 checkpoint mean；
6. CRG 必须与 T1 使用同一 common normalization、逐图 route count 和 zero-inference-overhead 结构。

CRG 只保留每图“路由数量”这一汇总量，不保留 O2M 教师的逐 anchor 位置或概率语义。如果 T1 明显优于 CRG，才支持跨头 teacher semantics；若两者相当，应把方法重新表述为 O2M-count-conditioned sparse supervised reweighting，而不能包装成知识迁移。CRG 是第二阶段，不应与 BRC 同时堆进一个 run，否则两个因果问题无法区分。

## 8. 执行顺序

1. 等当前正式训练完全退出并确认 GPU 4–7 空闲；
2. 只在本地研究副本实现 BRC 和测试，禁止修改正在运行进程所用源码；
3. 单卡数值测试 → 两进程 DDP 等价 → 四卡 3-epoch smoke；
4. 固定配置和源码 hash 后运行 T1-BRC seed 42；
5. 固定末轮分析、独立评估、文件 hash、MCE/SIGBUS 核验并落盘；
6. BRC 通过后再实现/校准 T1-CRG；未通过则先解释 clipping 敏感性，不继续堆叠 TQGB 复杂度。

本规范不授权占用 GPU，也不修改远端源码；它只冻结下一步因果控制设计。
