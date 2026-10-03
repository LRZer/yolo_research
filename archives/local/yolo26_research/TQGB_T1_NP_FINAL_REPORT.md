# TQGB-T1-NP 无投影因果对照最终报告

日期：2026-09-19  
状态：50/50 完成；固定末轮与末段窗口审计完成；**按预注册 seed 42 逻辑，supervision-segment projection 获得方向性支持**

## 1. 这轮只回答什么问题

T1-NP 保留 projected T1 的 O2M→O2O 分类路由、gain、阈值、训练协议和随机种子，只关闭 `tqgb_projection`。归档 `args.yaml` 的逐键比较确认，两次训练除 `name`、`save_dir` 外，唯一方法差异为：

```text
T1:    tqgb_projection=true
T1-NP: tqgb_projection=false
```

因此它是当前“接收头监督线段投影是否有贡献”的最小单 seed 因果对照，而不是一次新的超参数搜索。

- 模型：随机初始化 YOLO26n；
- 数据：完整 VOC07+12 trainval 训练，VOC2007 test 每轮仅作快速探索评估；
- 训练：50 epochs、batch 64、imgsz 640、BF16、seed 42；
- 硬件：physical GPU 4–7、workers 4、pinned memory、NUMA 1；
- 路由：O2M→O2O classification，gain `0.08`，`min_score=0.25`，`margin=0.05`；
- 主指标：固定 epoch 50 O2O NMS-free mAP50-95，不使用 best epoch。

本次使用的 `loss.py`、`default.yaml` 和训练入口哈希与 T0-current 完全相同，排除了核心源码快照漂移。

## 2. 完成性与硬件审计

- `results.csv` 连续覆盖 epochs 1–50，共 50 行；
- trainer 正常报告 `50 epochs completed in 0.524 hours`，CSV 训练时间为 `1887.76 s`；
- `last.pt`、`best.pt` 及 `epoch0/10/20/30/40.pt` 阶段 checkpoint 均完整落盘；
- trainer 与四个 DDP rank 正常退出；退出后的 NVML 快照显示 physical GPU 4–7 均无计算进程；
- 日志未命中 Traceback、RuntimeError、SIGBUS、Bus error、CUDA error、NCCL error/warn、Killed、NaN 或 Inf；
- 运行窗口 `2026-09-19 03:38:30–04:11:00 +08:00` 的内核日志未出现新的 `Uncorrected hardware memory error`、`Sending SIGBUS` 或 `Memory failure`；
- 本地归档的 9 个原始 run 文件与服务器逐文件 SHA-256 manifest 完全一致；另存的日志和 source config 也与服务器哈希一致。

服务器已有不可纠正内存故障历史，所以该结果仍是 **faulty-host exploratory**；“本轮未见故障事件”不能替代健康硬件复现。

## 3. 固定第 50 轮主结果

T0-current 与旧 B50 的 50×75 个共同非时间 CSV 数值已逐值完全相同，因此下表把二者合并为精确基线。

| 实验 | P | R | mAP50 | mAP50-95 | Δ vs T0/B50 | Δ vs projected T1 |
|---|---:|---:|---:|---:|---:|---:|
| T0/B50 | 0.74053 | 0.65166 | 0.72216 | 0.51276 | — | -0.00670 |
| T1 projection on | 0.74619 | 0.65231 | 0.72862 | 0.51946 | **+0.00670** | — |
| **T1-NP projection off** | **0.74658** | **0.64181** | **0.72337** | **0.51474** | **+0.00198** | **-0.00472** |

关闭投影后仍比精确基线高 `0.00198`，说明“质量路由本身”可能含有弱正信号；但它低于 projected T1 `0.00472`，也没有达到此前用于升级单 seed 候选的 `+0.003` 筛选线。

T1 相对 T1-NP 的变化为 P `-0.00039`、R `+0.01050`、mAP50 `+0.00525`、mAP50-95 `+0.00472`。两者 precision 几乎相同，主要区别是 projected T1 保住了约 1.05 个百分点的 recall，并转化为两个 AP 指标的明确优势。

## 4. 末段稳定性

| 窗口 | T0/B50 | projected T1 | T1-NP | T1-NP−T0 | T1-NP−T1 |
|---|---:|---:|---:|---:|---:|
| epochs 41–50 | 0.509639 | 0.514690 | 0.510762 | +0.001123 | **-0.003928** |
| epochs 46–50 | 0.512832 | 0.518078 | 0.513780 | +0.000948 | **-0.004298** |

projected T1 在最后 10 个 epoch 中 **10/10** 高于 T1-NP；逐轮差值范围为 `+0.00275` 到 `+0.00519`，均值 `+0.00393`。因此 epoch 50 的 `+0.00472` 投影优势不是单个末轮尖峰。另一方面，T1-NP 在最后 10 轮只有 7/10 高于 T0，且平均优势仅 `+0.00112`，其无投影正信号较弱。

## 5. 路由确实工作，且对照强度没有失控

epoch 50 的 rank-0 训练侧局部均值：

| 统计 | projected T1 | T1-NP |
|---|---:|---:|
| common anchors | 36.9073 | 36.7722 |
| M→O route count | 4.4286 | 4.5405 |
| M→O route rate | 0.11752 | 0.12404 |
| recorded projection keep | 0.96314 | 0.97683 |
| raw route loss | 0.01484 | 0.01702 |
| scaled route loss | 0.00119 | 0.00136 |

两组都在 49/50 个 epoch 中出现非零路由。末 10 轮 route rate 为 `0.13415`（T1）与 `0.13823`（T1-NP），覆盖接近；无投影版本的末 10 轮 scaled loss 为 `0.00147`，投影版本为 `0.00138`。因此 T1-NP 不是因为路由没打开而落后，也不存在新增损失突然大一个数量级的问题。

需要正确解释 T1-NP 的 `projection_keep=0.97683`：当 projection 关闭时，代码直接使用教师概率，任一实际路由候选都会保留。该 CSV 字段是 rank-0 的逐 batch 比率均值，而无候选 batch 以 0 计入，所以它可以低于 1；它不表示 T1-NP 仍偷偷执行了投影。`source_config.yaml` 和训练生成的 `args.yaml` 均明确记录 `tqgb_projection: false`。

## 6. 预注册裁决

**裁决：projection 在 seed 42 上得到支持。**

预注册逻辑要求使用固定 epoch 50，并要求 projected T1 优于同 gain 的 T1-NP。实际差值为：

```text
T1 projected 0.51946 - T1-NP 0.51474 = +0.00472 mAP50-95
```

这个方向同时覆盖固定末轮、last-5、last-10，并且 projected T1 在末 10 轮逐轮全胜。因此可以保留“receiver-bounded / supervision-segment projection”作为 TQGB 的候选组成，而不能再把 T1 的全部收益解释成任意附加 KL 或单纯路由覆盖。

边界同样明确：这仍只有一个 seed，且 trainval→VOC07 test 的快速探索协议、故障主机以及 global clipping 混杂仍存在。它不证明统计显著性，也不授权写成最终可发表性能。最低后续证据仍包括：

1. T0–T1 的 seeds 0/1 配对复现，与 seed 42 合成三对；
2. 至少再补一个非 42 seed 的 T1-NP，避免 projection 结论只依赖单 seed；
3. 完成 T1-BRC，隔离 combined global clipping 的间接耦合；
4. 在健康硬件、干净开发/测试协议下从头复现，再进入 150 epochs 与 COCO。

按本次任务约束，没有另起独立 GPU 评估；本报告只使用预注册的固定训练 CSV 指标回答 projection 因果对照，不使用 `best.pt` 选轮。

## 7. 归档与校验和

完整本地归档：`outputs/yolo26_research/runs/tqgb_t1_np_cls_m2o_g008_e50_g4567_s42_w4_pin_numa1/`

| 产物 | SHA-256 |
|---|---|
| `results.csv` | `c700e1c7d8ab4d575b52ddacac13838b3be7bae16ce9a432b6a873a70f977043` |
| `weights/last.pt` | `80605dfc9926508c0073c4e3a4d9b43daeb719b62e576d02032210fdd2d80117` |
| `weights/best.pt` | `60ed776ef2aa6aa4e04d9d18d438875d1af98a31ab03c5f9a006b9ca5d23cd6e` |
| `train.log` | `af06cddfd9ad3c8aa4b60bbb866affbd75f396bb22505fdf2af16e92ca2dab94` |
| `source_config.yaml` | `155eaee7211d690e3c863dcffdb133215e724a8636b0963fe5fa2035772d6591` |

运行时核心源码：

| 文件 | SHA-256 |
|---|---|
| `ultralytics/utils/loss.py` | `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9` |
| `ultralytics/cfg/default.yaml` | `1294dd04cdb28c7ecdbaf15dc01c9684b8365ba81be59f6ac47a5939bbd0e63a` |
| `scripts/train_research.py` | `d8f39b19c11434f4f322bc4630b02af795528c8618c9bc20e48a58433022f7cd` |
