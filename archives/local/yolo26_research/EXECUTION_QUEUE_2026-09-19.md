# YOLO26n/VOC 快速研究执行队列

日期：2026-09-19  
目标：在完整 VOC07+12 trainval 快速协议上，尽快把“单次提点”推进成可复现、可解释、可写论文的候选；最终结论仍需健康硬件与干净测试协议复现。

## 当前最强候选

TQGB-T1（O2M→O2O 分类路由 + receiver supervision-segment projection）在 seed 42 的固定第 50 轮相对精确基线提升 `+0.00670` mAP50-95；关闭 projection 后只提升 `+0.00198`，projected T1 在最后 10 轮对 T1-NP 逐轮 10/10 胜，均值优势 `+0.00393`。因此目前不是“已经发表级”，但已经从普通模块试错进入了值得扩种子的机制候选。

## GPU 串行队列

同一时刻只允许一项正式四卡训练，固定 physical GPU 4–7、NUMA1、batch64、BF16、workers=4、pinned memory。

1. `B50 seed0`：已完成并逐哈希归档；固定 epoch 50 mAP50-95=`0.51359`，但运行末段出现未向训练进程发送 SIGBUS 的 deferred EDAC UE，故仅作 faulty-host exploratory 配对参考。
2. `projected T1 seed0`：已完成并归档；固定 e50=`0.51144`，相对旧 B50 seed0 `-0.00215`，last-5/10=`-0.00220/-0.00306`，独立 O2O/O2M 也分别为 `-0.00201/-0.00295`。负复现，TQGB 已降级为跨种子不稳定候选。
3. `T1-NP seed0`：已通过配置/源码/硬件/GPU 空闲门禁，于 projected T1 归档后从头启动，当前运行中；用于检验 projection 是否至少在同一开机窗口优于无投影 route。
4. `B50 seed0 same-boot repeat`：在 seed0 T1/T1-NP 后，用完全相同的零路由配置补跑当前开机窗口的基线，量化跨重启及 faulty-host 的 run-to-run 漂移；若它与旧 B50 的差异已接近方法差值，seed0 的基线归因必须以同期开机重复为主并降级证据强度。冻结配置为 `b50_no_kd_e50_s0_reboot_r1_w4_pin_numa1.yaml`，SHA256=`1f3588848bca74bf716455bca4a8207fefe939557dc2357a0d5ed4efac9a8632`，远端目标目录已确认不存在。
5. `B50 seed1 → projected T1 seed1`：补齐 `{42,0,1}` 三组 baseline/T1 配对。
6. 根据前三对结果分支：
   - 至少 2/3 paired delta 为正，且三对平均增益有实际量级：运行 T1-BRC clipping control；
   - projection 在 seed0 也胜 NP：保留 receiver-bounded 主张；否则把 projection 因果表述降级；
   - 多种子不稳定：停止继续扩 TQGB 参数，不用 VOC07 test 做 gain sweep，资源转给 IGB/SARC。
7. T1-BRC 若支持：完整 trainval 150 epoch 的 B50/T1 固定末轮验证；随后准备 COCO 子集 smoke 与 COCO2017 正式协议。

### 2026-09-19 05:03 运行快照（非裁决）

`projected T1 seed0` 已连续完成 20 个 epoch，epoch 20 单轮相对同 seed B50 为 `-0.00667`；last-5 paired mean 为 `-0.00145`，last-10 paired mean 为 `+0.00046`。此前 epoch 18 单轮为 `+0.00345`，说明中途差值明显波动，不能挑选某个有利 checkpoint 作为结果。当前路由与辅助损失保持非零，GPU4–7 正常工作；从本次训练启动到该快照未出现 `Memory failure`、`Sending SIGBUS`、EDAC UE 或训练异常。正式裁决仍严格等待固定 epoch 50。

## 与主线并行但不抢正式 GPU 的候选

### IGB

先做 exact per-GT gradient ledger。只有 gradient-norm Gini、top-20% share 和 mass→gradient 相关性同时达线才实现训练；不达线当天停止。若达线，先跑 baseline / mass-matched permutation / semantic IGB 三元组，防止把普通重权正则包装成实例配额机制。

### SARC

实现与部署折叠测试已经通过，assignment 层级分布探针也 GO。正式最小矩阵仍是 R0 / R1 / R2 / R3；只有 R3 超过 R2，才有 calibration 机制证据。进一步必须加入同强度层级置换 target，排除普通 KL/熵正则。

### ASAC

v3 的远端真实 provenance preflight 因 stripped `last.pt` 缺少 `save_dir` 正确停止。只有 v4 对“终态 strip 的唯一允许缺失”建立内容哈希 + `epoch=-1` + 固定路径 + 独立 `args.yaml` 联合证明并重新通过恶意轨迹测试，才恢复 GPU 探针。

## 论文级最低证据

- 至少 3 个配对 seeds，报告每对原始值、paired mean/std，不只报最好的 seed。
- 固定最终轮，不用 `best.pt` 选择结论。
- 方法开关关闭时 exact equivalence；语义 placebo、强度控制和全局 clipping 控制齐全。
- 推理参数/FLOPs/延迟及训练开销；TQGB/IGB 应为推理零开销。
- VOC 只是首个基准；主张通用性前至少补 COCO，并视方向补一个独立数据集。
- 故障主机所有结果标为 exploratory；最终表格需在健康硬件从头复现。

## 停止规则

- 不因一次负结果在 VOC07 test 上连续扫 gain、阈值或结构。
- 不把多个尚未独立成立的方法叠加后声称创新。
- 任一训练进程 SIGBUS/Bus error 命中该运行，对应运行立即作废且不 resume；仅有未向训练进程送达信号的 deferred EDAC/MCE 时允许完成，但必须标为 hardware-contaminated exploratory，不能进入最终论文表格，且需健康硬件从头复现。
- 任何 probe provenance、off-equivalence 或 placebo 失败，先修复证据链，不启动长训。
