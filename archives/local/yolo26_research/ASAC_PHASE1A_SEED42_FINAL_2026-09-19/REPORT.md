# ASAC Phase-1a seed42 最终报告

日期：2026-09-19  
主结论：**NO-GO；停止 ASAC Phase-1b，不进行结果驱动调参**  
实验身份：faulty-host exploratory；本轮窗口未观察到新增硬件错误

## 1. 研究问题

ASAC（Assignment-Stability Adaptive Curriculum）尝试用固定训练面板测得的 O2M/O2O assignment stability，因果地决定下一轮 `scale/translate` 增强强度。它只改变训练课程，不改变推理网络、参数量或 FLOPs。

Phase-1a 在 seed42 上预注册两组：

- **A0**：ASAC 基础设施下固定 `q=1` 的配对仪器化控制；
- **A2**：真实闭环 ASAC，前 5 轮 `q=.5`，之后只使用上一轮固定 panel 信号更新下一轮 `q`。

旧官方 B50 只作为外部参考。A0 因逐轮 DataLoader reset 从 epoch 2 起与旧 B50 分叉，不能称为官方基线的数值复现；详见 `ASAC_A0_DATALOADER_DIVERGENCE_DIAGNOSIS_2026-09-19.md`。

## 2. 固定第 50 轮主结果

| 运行 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| A0 paired fixed-q=1 | 0.74356 | 0.64914 | 0.72061 | **0.51357** |
| A2 live ASAC | 0.74450 | 0.64107 | 0.71908 | **0.50847** |
| official B50（外部参考） | 0.74053 | 0.65166 | 0.72216 | **0.51276** |

A2 相对配对 A0：

- Precision `+0.00094`；
- Recall `-0.00807`；
- mAP50 `-0.00153`；
- **mAP50-95 `-0.00510`（-0.51 AP）**。

A2 相对旧 official B50 的 mAP50-95 为 `-0.00429`。A0 相对旧 B50 的固定末轮仅 `+0.00081`，但两者训练数据流不相同，因此该差值不作方法因果解释。

## 3. 曲线稳健性

| 读数 | A0 | A2 | A2-A0 |
|---|---:|---:|---:|
| epoch 50 mAP50-95 | 0.51357 | 0.50847 | **-0.00510** |
| last-5 mean | 0.512220 | 0.507656 | **-0.004564** |
| last-10 mean | 0.507982 | 0.503261 | **-0.004721** |
| 50-epoch curve mean | 0.365385 | 0.361775 | **-0.003610** |

A2 在 50 个 epoch 中相对 A0 为 7 胜、1 平、42 负。即使查看非主读数的单次最高点，A0 在 epoch49 为 `0.51438`，A2 的最高点仍是 epoch50 的 `0.50847`；不存在被固定末轮掩盖的正向尖峰。

分阶段的 A2-A0 mAP50-95 均值为：

| 阶段 | 平均差值 | A2 获胜轮数 |
|---|---:|---:|
| epochs 1–5 | -0.004852 | 1/5 |
| epochs 6–15 | -0.004661 | 2/10 |
| epochs 16–30 | -0.001528 | 4/15 |
| epochs 31–50 | **-0.004336** | **0/20** |

epochs31–50 的 A2 与 A0 都使用 `q=1`，但 A2 在这 20 轮一次也未超过 A0。这表明早期不同增强课程形成了持续的路径依赖，后期回到固定官方强度并未恢复差距；负结果不是末轮噪声，也不是简单的“控制器太晚才回到 q=1”。

## 4. 控制器行为

- A2 `q_min=0.489525`、`q_max=1.0`、均值 `0.809132`，动态范围 `0.510475`；
- epoch31 首次达到上限，epochs31–50 共 20 个训练 epoch 使用 `q=1`；
- A0 的 50 个 epoch 均为 `q=1`；
- 两组最低 assignment coverage 均为 `623/624=0.998397`；
- 两组均未触发 assigner OOM fallback。

因此控制器并非全程常数，早中期确实实施了不同增强课程；但后 40% 训练饱和为固定官方强度。该机制真实改变了训练轨迹，却没有改善最终性能，反而稳定落后于配对控制。

结合 P0 与本轮行为，固定 panel stability 更像随训练成熟度上升的状态指标，而不一定是“下一轮应该增强多少”的充分控制误差信号。当前单调映射把稳定度上升转成更强增强，最终必然靠近上限；这解释了为什么控制器会饱和，但不是继续事后修改映射的授权。

## 5. 预注册门禁

| 门禁 | 结果 | 判定 |
|---|---:|---|
| `A2-A0 epoch50 >= +0.003` | `-0.00510` | **FAIL** |
| last-5 同号为正 | `-0.004564` | **FAIL** |
| q 动态范围非退化 | `0.510475` | PASS，但后期饱和 |
| A2 相对 A0 墙钟开销 `<12%` | `+0.0177%` | PASS |
| 50 状态行、coverage≥.95、无 OOM fallback | 全部满足 | PASS |
| A0 全程固定 q=1 | 50/50 | PASS |

综合裁决：**Phase-1a NO-GO**。不运行 seeds 0/1，不运行 A1/A3/A4，不进入 150 轮或 COCO，也不根据该结果调整 `beta/q_min/q_max/q_init/eta/S_ref`。ASAC-P0 的“增强强度与 assignment stability 有稳定关系”仍然成立；被否定的是当前闭环映射能够带来检测精度收益。

## 6. 运行完整性与硬件

- A0/A2 均为 50/50 epochs、退出码 0；CSV 各 51 行（header+50），state JSONL 各 50 行；
- A0/A2 墙钟分别为 `1973.60 s` 与 `1973.95 s`；
- A0 正式窗口自 `2026-09-19 22:56:12 CST` 起未发现新增 MCE、SIGBUS、Memory failure 或 qualifying EDAC error；A2 retry 窗口同样无新增事件；
- A0 结束后 direct NVML 显示物理 GPU4–7 均为 0% utilization、约 452 MiB 驱动上下文、0 个 compute process；
- 两个 run 的远端 12 个原始文件均完整归档，本地与远端共同文件逐 SHA-256 比较：A0 `12/12` 一致、A2 `12/12` 一致，mismatch=0；
- run 内外两份 launch record 对每组均逐字节相同。
- 独立冷复核重新从原始 CSV/JSONL 计算 epoch50、last-5/10、50 轮均值、分阶段差值、墙钟、q 轨迹、coverage、OOM 与门禁，全部与 `comparison.json` 一致，未发现实质数值或解释错误。

服务器历史上存在不可纠正内存/CPU MCE；首次 A2 formal 也曾因整机重启在 epoch1 前作废。本次无事件不等于服务器健康。因此本报告只形成探索性 NO-GO；若未来有新的独立机制取得正结果，仍须在健康硬件上从头复现。

## 7. 关键哈希

| 产物 | SHA-256 |
|---|---|
| A0 results.csv | `551d034240ce0fa71336c48b51f7d2de9ad7362d28ecf43da261e2a1603f3938` |
| A0 state JSONL | `9de578ff9c5a8f1beabb90919826f243f12224c1be21a1ae2904b5078c92e448` |
| A0 launch record | `0cfa229792656417a8610906f1a4cf0de77a964017c00dbca31a69202d144f15` |
| A0 best.pt | `de27b8134ba7c08f3468e0a5a957aa846899bda80f20773442e1de5e36916a26` |
| A0 last.pt | `e38affd539e8bca8cf597fd33e64c419e6b770276290f592f8a7b099064dea32` |
| A2 results.csv | `b64135f336e37948874532afdba88a5d4b2c6899d279423d867f9c1603ca2491` |
| A2 state JSONL | `c6ef9f96cf999abfd86ec580613fd522841c453941034746d2cbcfb110dea9ec` |
| A2 launch record | `ed634dd7394eb7b3751fbf23bf8fa920874b18e846888224d0f53937961371dc` |
| A2 best.pt | `08726d1c95809bd6f70ddfd5e047f10c0852b223cd944a64503c46fa60181891` |
| A2 last.pt | `58d4a12fb8e8fa60b0d154fdcab52ebf6d5d4286f791f10ee941707b4d4cdd43` |

机器可读裁决见本目录 `comparison.json`；自动分析脚本为 `work/yolo26_research/analyze_asac_phase1a.py`。

## 8. 后续边界

本结果不支持继续“救”ASAC，也再次说明在 YOLO26n/VOC 上用单一训练技巧稳定提点很困难。下一步不立即开跑；先向用户提交独立候选路线，讨论后只批准一个低成本 Phase-0 诊断。
