# TQGB-T1 独立结果审计

日期：2026-09-19  
审计范围：`TQGB_T1_FINAL_REPORT.md`、`TQGB_METHOD_SPEC.md`、`TQGB_CODE_AUDIT.md`、B50/T1 原始 CSV、四份 `last.pt` 独立复评 JSON、逐类比较与本地 checkpoint。  
审计方式：只读本地归档；未修改远端、未占用 GPU。  

## 结论

**未发现使 T1 作废的数值错误、末轮错取、best/last 混用或评测路径混用。** 报告中的固定末轮、last-5/last-10、独立 O2O/O2M 和逐类数字均可由原始文件重新得到。T1 因此是一个可信的、明显强于此前候选的 **single-seed exploratory screening signal**。

但现有证据只支持“这一整套 T1 训练配置在 seed 42 上优于旧 B50”，尚不支持把收益因果归给 supervision-segment projection，也不支持显著性、跨 seed 稳定性、广泛类别一致性或论文级泛化结论。最关键的缺口是：同一当前源码的 T0、T1-NP、配对多 seed，以及干净 test/健康硬件复现。

## 1. 原始数字复核

### 1.1 固定末轮

B50 与 T1 CSV 都有且只有 50 行，epoch 连续覆盖 1–50。固定 epoch 50 重算如下：

| run | P | R | mAP50 | mAP50-95 | ΔmAP50-95 |
|---|---:|---:|---:|---:|---:|
| B50 | 0.74053 | 0.65166 | 0.72216 | 0.51276 | — |
| T1 | 0.74619 | 0.65231 | 0.72862 | 0.51946 | **+0.00670** |

与主报告完全一致。T1 CSV 总时间为 1887 s，也与报告一致。

训练日志在 epoch 50 后自动评测了 `best.pt`，其显示约 0.520，但主报告没有使用这个更高值；主表来自固定 epoch 50，独立复评也显式读取 `last.pt`。因此没有 best-epoch 替代 fixed-final 的问题。

### 1.2 last-5/last-10 与阶段均值

重新逐行计算得到：

| 窗口 | B50 均值 | T1 均值 | 平均差 | T1 胜出轮数 |
|---|---:|---:|---:|---:|
| epochs 41–50 | 0.509639 | 0.514690 | **+0.005051** | 10/10 |
| epochs 46–50 | 0.512832 | 0.518078 | **+0.005246** | 5/5 |

epochs 21–30、31–40 的平均差也分别为 `+0.003828`、`+0.002856`。epoch 49/50 为 `0.51955/0.51946`。因此“末轮不是孤立尖峰”和“后半程形成持续正窗口”是准确的描述。

需要收紧的统计口径是：这些轮次来自同一个训练轨迹、在同一 VOC07 test 上反复评测，相邻 checkpoint 高度相关。它们只能证明曲线平滑/末段方向一致，不能当作 5 次或 10 次独立复现实验，也不能据此计算普通显著性。

### 1.3 独立 `last.pt` 复评

四份 JSON 重算得到：

| path | B50 | T1 | Δ |
|---|---:|---:|---:|
| O2O NMS-free | 0.5129008908 | 0.5193847312 | **+0.0064838404** |
| O2M+NMS | 0.5282397018 | 0.5334306164 | **+0.0051909147** |

独立 O2O 与训练 CSV 固定末轮只相差约 `1.4e-4`（B50）和 `7.5e-5`（T1），属于评测实现/设备数值层面的极小差异，没有结论冲突。独立 O2O 的 P/R/mAP50 也与报告一致。

这里的“独立复评”只表示重新载入 checkpoint、单独走锁定推理路径；它仍使用同一个 VOC07 test 和同一模型，不是独立样本或独立复现。O2M 与 O2O 又来自同一 checkpoint，不能把两条路径同时为正当作两次独立统计证据。

## 2. 逐类别复核

O2O 逐类宏平均差为 `+0.006483839`，20 类中 12 类上升、8 类下降；按实例数加权后为 `+0.002840139`。O2M 对应为 12/8、宏平均 `+0.005190915`、实例加权 `+0.003477897`。所有类别排序、最大正负类和 person 近零变化均与报告一致。

需要避免两种过度解释：

1. 12/20 并不是“多数类别稳定受益”的有力统计证据。即使暂时忽略类别相关性，最朴素的双侧符号检验也只有 `p≈0.503`；而类别 AP 本身也不是独立同分布样本，所以不应把该 p 值当正式推断。
2. O2O 实例加权提升只有宏平均的约 44%，说明较大收益更多落在相对低频类别。报告说“不是由高频 person 主导”是正确的，但不能进一步外推为频次鲁棒或长尾改进；需要专门的频次分层与多 seed 才能支持这种叙事。

## 3. checkpoint 与哈希

本地重新计算：

- T1 `last.pt` SHA-256：`fdba78326273242130e3ef243b4164ff174a708b4d19d2ca406595d768afe68a`；
- B50 `last.pt` SHA-256：`49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb`。

它们分别与对应 O2O/O2M JSON 内嵌的 `checkpoint_sha256` 完全一致，因此四份复评确实针对归档中的正确 checkpoint，且 T1 报告给出的 `last.pt` 哈希正确。

本地 `results.csv`、`best.pt`、`last.pt` 和评测 JSON 都能正常读取。不过，当前最终归档没有单独保存“远端路径—远端 hash—本地路径—本地 hash”的完整 manifest，也没有把 MCE 查询输出作为不可变附件保存。因此“所有远端/本地哈希一致”和“无新增 MCE/SIGBUS”在总账中有操作记录，但仅凭当前本地交付物不能由第三方完全重放验证。以后每个正式 run 应保存 `ARTIFACT_MANIFEST.sha256`、源代码 commit/diff hash、启动命令/环境和前后内核事件摘录。

## 4. 协议公平性

### 已核实

- B50 与 T1 的共同训练参数一致：模型/data、50 epochs、batch 64、imgsz 640、GPU 4–7、workers 4、随机初始化、seed 42、deterministic、BF16、增强、优化器与学习率参数均匹配；
- T1 方法参数与冻结规范一致：M→O、`0.08/0`、`min_score=0.25`、`margin=0.05`、projection on；
- 训练日志明确显示四张 RTX 3090、4-rank DDP、workers=4、BF16；
- 增益/阈值和 T1/T2 决策协议文件时间早于 T1 结束，支持“结果前冻结”；
- T1 的路由在 epoch 2 后非零，epoch 50 rank-0 统计为 route rate `0.11752`、projection keep `0.96314`、scaled loss `0.00119`，不存在 gain 配置但机制未激活的问题；
- 当前代码审计对 exact same-GT/same-class、teacher detach、supervision-segment projection、common normalization 和方向隔离的描述与实现相符。

### 尚未完全闭合

1. **旧 B50 不是在最终 TQGB 源码快照下重新训练的 T0。** B50 `args.yaml` 与 T1 的所有旧字段匹配，T1 epoch 1 的十项训练/评测值又与 B50 完全一致，这两点很强地支持公平性；但 T1 日志显示研究源码已有 484 行增改，单 batch gain=0 等价测试也不能替代完整 50 epoch end-to-end T0。论文级因果比较应在同一源码 commit 下重跑零路由 T0。
2. **NUMA1 与 pinned-memory 的归档证据主要来自配置名/研究总账和框架默认值。** 日志能直接证明 GPU/workers/BF16，但未把实际 `numactl` 命令和 `ULTRALYTICS_PIN_MEMORY` 环境值打印出来。以后应在日志头显式写入启动命令、CPU/NUMA 亲和性和 pin-memory effective value。
3. **VOC07 test 已在整个研究过程中充当开发集。** 固定末轮、结果前冻结能减少 T1 内部挑点偏差，但此前的方向选择、停止/继续和多个候选都看过同一个 test。因此当前 `+0.0065～0.0067` 只能作为探索性信号，不能视为无偏的最终泛化估计。
4. **服务器既往不可纠正内存故障仍是外部有效性风险。** 本轮没有已记录的新故障并不等于硬件健康；健康机器从随机初始化复现仍是硬门槛。

## 5. 不能由 T1 单独支持的因果归因

T1 相对 B50 同时改变了 exact correspondence、质量阈值/margin、Bernoulli KL、common-anchor normalization、projection 和新增梯度强度。单个 T1 vs B50 只能说明复合配置有效，不能分别证明：

- supervision-segment projection 是收益来源；
- quality route 优于相同强度的普通/随机 route；
- exact same-GT 筛选本身贡献精度；
- O2M→O2O 的知识语义而不是附加正则化或 global clipping 轨迹变化导致收益。

尤其是 T1 路由项直接只进入 O2O classification head；O2M mAP 同时提高不能被解释为“知识成功传入 shared trunk”。更合理的当前解释是全模型 global clipping 及后续优化轨迹分叉的间接结果。O2M 正差是有趣诊断，不是 T1 核心机制的独立验证。

## 6. 最小后续实验优先级

已经按结果前协议启动的 T2 可以继续完成；它回答 reverse bridge 是否带来额外收益，但 **无论 T2 正负，都不能替代以下 T1 控制**。

### P0：同一最终源码的 T0（最高优先级）

在当前 commit、相同 launcher/NUMA/pin/DDP 环境下，以 `tqgb gains=0` 重跑 seed 42。若 deterministic 协议成立，它应复现旧 B50；否则先量化环境/源码漂移，再解释 T1。该 run 同时给后续 seeds 提供真正同源码 baseline。

### P1：T1-NP（projection off，其余完全相同）

这是当前方法名中“receiver-bounded / supervision-segment projection”能否成为论文贡献的最小因果对照。若 T1 不稳定优于 T1-NP，只能保留“质量路由正则化”结果，不能把投影包装为已验证贡献。

### P2：配对 seeds 0/1，加上 seed 42 形成三对

每个 seed 都应在同一最终源码下成对训练 T0/T1，固定末轮报告逐 seed 差值、均值、标准差和 95% CI；不要把 50 个 epoch 或 20 个类别当独立重复。最低继续条件仍应是平均为正且至少 2/3 配对为正。若资源允许，T1-NP 至少再补一个非 42 seed，避免 projection 结论完全由单 seed 决定。

### P3：强度/裁剪匹配的 route control

加入相同 common normalization、相近 route 数和梯度范数的 random-matched 或 all-common control，并记录 per-step global clipping scale。此前 D2 的表面收益在梯度强度匹配后基本消失，这是本项目已经观察到的直接警告；T1 也必须排除“附加梯度量级/裁剪正则化”解释。

### P4：干净验证链与健康硬件

锁定方法后，停止用 VOC07 test 调方向；先在内部 dev 做选择，再在健康硬件从头跑固定 150 epochs，最后一次评测 VOC07 test，并扩展 COCO2017/YOLO26s。只有这一步之后，当前候选才可升级为论文性能证据。

## 7. 推荐表述

当前最稳妥的结论是：

> 在 faulty-host、VOC07 test 可见的 50-epoch 探索协议中，classification-only TQGB-T1 在 seed 42 的固定末轮将 YOLO26n O2O mAP50-95 从 0.51276 提升到 0.51946（+0.00670）；独立 `last.pt` 复评为 +0.00648，末 10 轮均高于对应 B50。该结果通过单 seed 机制筛选，但 projection 的因果贡献、随机种子稳定性与无偏泛化仍待 T0/T1-NP、配对多 seed 和健康硬件干净协议验证。

不建议使用“已显著提升”“多数类别稳定受益”“O2M 复评独立验证机制”“projection 已被证明有效”或“已达到可发表结果”等表述。
