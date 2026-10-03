# ASAC Phase 1：闭环增强课程预注册

日期：2026-09-19  
状态：**在首次 ASAC 训练前冻结；实现与冒烟测试尚未构成性能实验**  
上游证据：`ASAC-P0-v2` 在 B50 epochs 11/21/31/41/50 上自动裁决为 `GO for affine-ASAC prototype`。

## 1. 研究问题与边界

ASAC（Assignment-Stability Adaptive Curriculum）检验的问题是：

> 使用上一轮模型在固定 train-only panel 上的 O2M/O2O 分配稳定性，闭环调节下一轮 scale/translate 增强强度，是否优于官方固定增强以及具有相同强度边际分布、但不保留真实状态—增强对应关系的开环课程？

本阶段只控制 Ultralytics 原生 `RandomPerspective` 的 `scale` 与 `translate`，不控制 Mosaic、颜色、翻转、MixUp、CutMix 或其它训练项。方法不改变推理图、参数量、FLOPs 或部署延迟。VOC07 test 仍是探索性固定末轮读数；本阶段不能单独支撑论文结论，最终证据必须在健康硬件、多随机种子和 COCO 上复现。

## 2. 固定训练条件

- 模型：YOLO26n 官方结构 YAML，严格沿用 B50 的 `pretrained=False` 随机初始化；
- 数据：完整 VOC2007+2012 trainval 训练，VOC2007 test 仅按现有协议监控；
- 训练：50 epochs、seed 42、batch 64、imgsz 640、BF16；
- 设备：仅物理 GPU 4–7，NUMA node 1；
- dataloader：`workers=4`，`pin_memory=True`；
- 其它超参数必须与已冻结 B50 完全相同；
- 官方 `close_mosaic=10` 保持不变；关闭 Mosaic 后仍继续控制原生 scale/translate；
- 不从 SIGBUS/MCE 中断点续训；发生新的硬件内存错误则该次运行作废并从头重跑；
- 当前主机结果统一标记为 `faulty-host exploratory`。

## 3. 强度定义

令 `q_e` 为第 `e` 轮训练使用的几何增强强度：

```text
effective_scale_e    = 0.50 * q_e
effective_translate_e = 0.10 * q_e
```

其它 `RandomPerspective` 参数保持 B50 原值。自适应与匹配对照均限制：

```text
q_min = 0.25
q_init = 0.50
q_max = 1.00
```

官方固定基线 A0 对应 `q=1.00`，即原生 `scale=0.50, translate=0.10`。P0 的最强正式探针为 `q=0.75`；允许控制器达到 `q=1.00` 是为了覆盖而不弱于官方增强，但属于训练动作范围，不把 `q>0.75` 外推成新的 P0 机制证据。

实现必须包装而非改写 `RandomPerspective` 的采样顺序：一次变换仍以完全相同顺序抽取 perspective、rotation、scale、shear、translation 随机数。`q=1` 时，相同 RNG 状态下矩阵与官方实现逐值一致。

## 4. 固定反馈 panel 与信号

- split：train；无增强、`rect=False`、`shuffle=False`；
- panel：顺序前 256 张图，batch 64，共 4 batch；
- panel 内容 SHA256 必须精确等于 ASAC-P0 正式运行的冻结内容哈希：`8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`；
- 观测模型：每轮优化结束后的 rank-0 EMA model；这与 P0 所读 checkpoint 的 EMA 权重一致，也避免 DDP `broadcast_buffers=False` 下把 rank-0 局部 BN 统计混入控制器；
- forward：`torch.inference_mode()`、CUDA BF16；模型保留 training-format 双头输出，但全部 BatchNorm 与 Dropout 临时切为 eval，之后逐模块恢复原始 training flag；
- panel 测量前后保存并恢复 Python、NumPy、CPU Torch 和该 DDP 进程当前 CUDA device 的 RNG 状态；参数、全部注册 buffer、BN buffer 以及 Detect 的非注册 tensor 属性 `stride/anchors/strides` 必须逐哈希不变；不读取其它可见 GPU 的 RNG，以免每个 rank 创建跨卡 CUDA context；
- clean 与固定 `q_probe=0.50` 的受控 affine view使用 ASAC-P0 的固定路径哈希方向、GT 过滤、原生 O2M/O2O assigner、13×13 Gaussian occupancy 和 `exp(-JS)` 定义；探针强度与当轮训练动作 `q_e` 完全独立，不使用 AP、val/test 指标或训练 loss 作为反馈；
- 对象母体固定为 P0 clean/q=.25/.50/.75 全视图均存活的 624 个 GT，身份哈希为 `f57cffdeb8bc29ef66a21c384ac0c3cd7ceff9552c48245263cb6c7c51b14926`；不能随 epoch 或训练 q 重选对象，probe 中缺失 assignment 仍记 `S=0`；
- 对每个 head，`S_h(e)` 是共同保留 GT 上的 mean stability，未在一侧获得前景分配者计为 0；
- 联合信号：`S_e = 0.5 * (S_O2M(e) + S_O2O(e))`，两支 head 等权；两项原始值始终单独记录；
- 每轮每支 head 要求：共同 GT `>=100`、双侧分配 GT `>=100`、assignment coverage `>=0.95`。任一条件失败立即中止该运行，不允许用无效信号继续调节。
- 若原生 `TaskAlignedAssigner` 因 CUDA OOM 进入逐图 fallback，即使最终算出结果也立即将该次 panel 与整个 run 判为无效；不得把显存争用导致的不同执行路径混入反馈。

panel 每轮都以固定 `q_probe=0.50` 测，包括 A0/A1/A2/A3/A4；这样信号跨轮可比、各组承担相同诊断开销，且不会把 P0 已证明的“当前增强越强，稳定度越低”直接混进反馈。

## 5. A2 闭环控制器

### 5.1 暖身与参考值

- epochs 1–5：`q_e=0.50`；
- 每轮记录 `S_e`；
- epoch 5 后冻结 `S_ref = median(S_3,S_4,S_5)`；
- 令状态 EMA `m_5=S_ref`，因此 epoch 6 仍为 `q_6=0.50`。

### 5.2 因果更新

对 `e=6,...,49`，只用已经完成的 epoch `e` 观测控制下一轮：

```text
m_e          = 0.80 * m_(e-1) + 0.20 * S_e
log_q_target = log(0.50) + log(2) * (m_e - S_ref) / 0.02
delta_e      = clip(log_q_target - log(q_e), -0.05, +0.05)
q_(e+1)      = clip(exp(log(q_e) + delta_e), 0.25, 1.00)
```

若固定探针稳定性高于暖身参考，下一轮增强变强；低于参考则变弱。`0.02` 来自 P0 在 `q_probe=.50` 上观察到的训练期信号变化量级（O2M 约 +.014、O2O 约 +.023），而不是 AP 调参。状态到动作是有界映射，并限制每轮 `|Δlog q|<=.05`，避免积分 windup 和突变。`beta=.80`、尺度 `.02`、速率 `.05` 在首次训练前冻结。

### 5.3 无额外 affine 冷却

不再加入独立的 epochs 46–50 scale/translate 冷却；只保留所有组共有的官方 `close_mosaic=10`。A2 按同一闭环规则控制到 epoch 50，避免额外末段课程成为混杂。

## 6. 最小因果矩阵

| ID | q 策略 | 回答的问题 |
|---|---|---|
| A0 | 官方 `q=1.0` | 是否优于当前产品基线 |
| A2 | 第 5 节闭环反馈 | 待检验方法 |
| A1 | 三 seed 的 A2 日程做冻结的跨 seed 无固定点置换：`seed0←q42, seed1←q0, seed42←q1` | 保持整组真实强度日程集合，但切断模型/seed 与其反馈日程的配对 |
| A3 | 每个 seed 自己的 A2 epochs 6–50 日程循环移位 10 epochs，epochs 1–5 不变 | 保持完全相同的 per-seed q 多重集合及大部分局部平滑性，但破坏正确时序对齐 |
| A4 | 同 seed 精确重放 A2 q 日程 | 只作工程等价性检查；预期 A4≈A2，不设“A2 胜 A4”性能门槛 |

重要修正：同 seed 精确重放与 A2 是同一干预，不能作为反馈性能对照。A1 因而升级为跨 seed 日程置换；A3 使用平滑循环移位，而不是完全随机打乱造成不公平的增强冲击。

A1/A3/A4 日程由独立 freezer 只读取 A2 的 ASAC state JSONL，不读取 `results.csv`、checkpoint fitness、AP 或任何 val/test 指标。freezer 代码哈希、输入状态哈希和输出日程哈希须在启动控制组前落盘。正式多 seed 因果矩阵中，应先运行三个 A2 并冻结日程，之后才解盲性能并启动 A1/A3；A4 可只在一个 seed 上验证实现等价性。

## 7. 记录、完整性与禁止项

每轮 JSONL 至少记录：epoch、variant、训练 q、有效 scale/translate、固定 probe q、O2M/O2O stability、固定对象数、双侧分配数、coverage、`S_e`、`S_ref`、状态 EMA `m_e`、target q、下一轮 q、测量耗时、panel/object-population/source/config hash。所有 rank 在下一轮前广播并断言 q 一致。

硬失败条件包括：

- 非 4–7 号物理 GPU 可见；
- `workers!=4` 或 pinned memory 未启用；
- panel 哈希不符、panel 不满 256 张或不是顺序 sampler；
- 反馈读取了 val/test/AP；
- q 超界、DDP rank q 不一致、关闭 Mosaic 后 wrapper 丢失；
- 测量改变参数、任一注册/额外 Detect tensor 状态、模块 training flags 或任一 RNG 状态；
- panel 的任一 O2M/O2O assigner 触发 CUDA OOM fallback；
- A1/A3/A4 对应的强度多重集合与来源 A2 不完全相同；
- A1/A3 freezer 在读取 A2 性能指标后被改写。

不允许根据 seed42 的 test AP 调 `q_min/q_max/q_init/beta/eta/S_ref`、panel、冷却、排序方向或 permutation key。失败后应按预注册判定 no-go，而不是扫参翻转结论。

## 8. Phase-1 裁决

Phase 1a 先做 seed42 的 A0/A2 快速筛选，只回答“是否有继续投入价值”，不主张反馈因果。继续标准为 `A2-A0 >= +0.003`、last-5 同号、q 动态范围非退化、开销 `<12%`。

若 Phase 1a 通过，再进入 Phase 1b：先完成 seeds `{0,1,42}` 的 live A2 并盲态冻结全部日程，再运行配对 A0/A1/A3。主读数固定为 epoch 50 VOC mAP50-95；last-5 mean 和训练 AUC 为补充。Phase 1b 同时要求：

1. 三 seed 平均 `A2-A0 >= +0.003`；
2. 三 seed 平均 `A2-A1 >= +0.002`；
3. 三 seed 平均 `A2-A3 >= +0.002`；
4. A2 相对 A0/A1/A3 的 last-5 mean 均同号；
5. 相对 A0 的训练墙钟开销 `<12%`，推理参数/FLOPs 不变；
6. 全部完整性门禁通过，运行窗口无新增 MCE/SIGBUS。

若不同 seed 的 A2 日程几乎相同，或 q 动态范围极小，则该方法应降级解释为普通 epoch schedule，停止反馈因果主张。三 seed 仍只算发现性证据；当前故障主机上的任何结果都必须在健康硬件和未污染数据集上从头复现。

## 9. 实施门禁

正式 50 轮运行前必须依次通过：

1. 纯函数单元测试：控制器、freezer、多重集合与哈希；
2. `q=1` RandomPerspective RNG/矩阵逐值等价；
3. 4-worker persistent loader 的旧预取可被清空：每轮先设置 sampler epoch，再 `reset()` 重建 iterator/workers，下一轮第一批至末批都只读到新 q；
4. `close_mosaic` 重建 transforms 后 wrapper 仍存在；
5. 单卡黄金 panel：必须在固定 B50 epoch50 checkpoint 上复现 P0 的 O2M/O2O `q=.50` 稳定度（绝对误差 `<=1e-12`），同时证明原生 E2ELoss、O2M `10/10`、O2O `7/1`、stride `8/16/32`、BF16、624 GT、模型/RNG 零副作用；
6. 4 卡 DDP 2-epoch smoke：各 rank q 一致、实际 workers=4、pinned memory、只见物理 GPU4–7；
7. smoke 前后检查服务器时间、NVML 实时状态与新增 MCE/SIGBUS。

此外，正式 launcher 必须在导入自定义训练模块前验证完整隔离源码树、自定义 trainer/runtime/freezer、模型 YAML、数据 YAML、B50 args 与全 VOC trainval manifest；必须消费并校验与当前源码树绑定的黄金 PASS artifact。任何输出目录已存在、隐式 `name2` 或 `exist_ok=true` 均硬失败。

只有七项及 provenance/golden 门禁全部通过，才允许启动 A2/A0 正式训练。

## 10. 实施门禁执行记录（2026-09-19）

七项门禁及 provenance/golden 校验均已通过。当前冻结 golden 为 `panel_golden_epoch50_v3.json`，SHA-256=`aabeb03b02da7e8e5c7d8f4b26efe8314b933816beb4364fc08c8e20ef1d0d41`；O2M/O2O 相对 P0 的差值均为 0。四卡两轮联调成功，实际物理 GPU4–7、每 rank workers4、pinned memory、BF16，固定 624 对象 coverage=1.0，未触发 OOM fallback，运行窗口无新增 MCE/SIGBUS。

因此允许启动 Phase-1a，但本记录不改变第 8 节的性能裁决门槛，也不构成 AP 改进证据。完整工程证据见 `outputs/yolo26_research/ASAC_PHASE1_ENGINEERING_GATE_2026-09-19/REPORT.md`。
