# TQGB-T0-current 最终源码零路由复现报告

日期：2026-09-19  
状态：50/50 完成；CSV 与 `last.pt` 状态审计完成；**精确复现旧 B50**

## 1. 目的与固定协议

T0-current 用最终 TQGB 源码快照运行，但把所有 KD/TQGB gain 设为 0、路由方向设为 `none`。它只回答一个因果问题：最终研究源码在关闭新增训练信号时，能否精确复现旧 B50，而不是重新评估一个新方法。

- 随机初始化 YOLO26n；
- VOC07+12 trainval 训练，VOC2007 test 每轮评估；
- 50 epochs、batch 64、BF16、seed 42；
- physical GPU 4–7、workers 4、pinned memory、NUMA 1；
- 固定 epoch 50，不使用 best epoch 选择结论；
- run：`tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1`。

运行源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `ultralytics/utils/loss.py` | `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9` |
| `ultralytics/cfg/default.yaml` | `1294dd04cdb28c7ecdbaf15dc01c9684b8365ba81be59f6ac47a5939bbd0e63a` |
| `scripts/train_research.py` | `d8f39b19c11434f4f322bc4630b02af795528c8618c9bc20e48a58433022f7cd` |
| T0 source config | `b04ec96864bfb48fda7f04bd680c21876a461bf258b5ab17ae42d4a6f31d13f4` |

## 2. 完成性与硬件审计

- `results.csv` 完整包含 50 行，末行 epoch=`50`；
- 固定末轮 P/R/mAP50/mAP50-95=`0.74053/0.65166/0.72216/0.51276`；
- 训练进程已退出，日志完成 epoch 50、最终 checkpoint stripping 和 `best.pt` 验证；日志未出现 traceback、SIGBUS、CUDA/NCCL error 或 killed 标记；
- 覆盖本次约 31 分钟运行的最近 45 分钟内核日志中，匹配 `Uncorrected hardware memory error`、`Sending SIGBUS` 或 `Memory failure` 的事件数为 0。

因此本次作业完整落盘且没有观察到显式硬件故障。不过服务器已有不可纠正内存错误历史，T0 及其对照仍保留 **faulty-host exploratory** 标签，不能替代健康硬件最终复现。

## 3. CSV 精确复现审计

以旧 B50 的 50 行 CSV 为 reference，排除唯一允许变化的 wall-clock `time` 后：

| 审计项 | 结果 |
|---|---:|
| baseline/candidate 行数 | 50 / 50 |
| 共同比较字段 | 75 |
| 比较的数值单元 | 3,750 |
| 非零差异 | **0** |
| 最大绝对差 | **0** |
| T0-only 字段 | 32 |
| B50-only 字段 | 0 |

也就是说，epoch、六项训练 loss、验证指标、学习率以及旧 B50 已存在的全部诊断字段都在 CSV 保存精度下逐轮完全相同。T0 比 B50 多出的 32 列只是最终源码新增的 KD/TQGB 诊断 schema；其中 TQGB 路由 count/rate/loss 全程关闭，不构成旧字段轨迹差异。由于 schema 和 wall-clock metadata 不同，两份 CSV 的文件级哈希本来就不应相同；正确的结论是 **全部共同非时间字段精确相等**。

机器可读明细：`runs/tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1/exact_repro_vs_b50.json`。

## 4. `last.pt` 模型状态精确复现

两份归档 checkpoint 均已 stripping，`ema=None`，有效权重保存在 `model`。按现有 scope 比较全部状态张量：

| 范围 | 张量数 | 元素数 | 差异张量 | 差异元素 | max abs | relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| shared + O2M | 588 | 2,403,960 | **0** | **0** | 0 | 0 |
| O2O box | 42 | 72,018 | **0** | **0** | 0 | 0 |
| O2O cls | 78 | 54,728 | **0** | **0** | 0 | 0 |
| **合计** | **708** | **2,530,706** | **0** | **0** | **0** | **0** |

因此 T0 与旧 B50 的固定末轮模型状态逐元素完全相同。两份 `.pt` 的整文件哈希不同，只反映日期、训练参数、扩展结果列等 checkpoint metadata 不同，不代表模型权重不同。既然 `last.pt` 模型状态相同，另起 GPU 作业重复 O2O/O2M 推理不会增加因果证据，本报告按任务要求没有启动额外评测。

机器可读明细：`runs/tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1/checkpoint_scope_vs_b50.json`。

## 5. 因果结论与下一步

T0 关闭新增信号后同时精确复现了 B50 的完整 50 轮共同 CSV 轨迹和全部 708 个末轮状态张量。这排除了“最终 TQGB 源码本身或新增诊断代码在 gain=0 时改变基线”这一解释，也关闭了当前源码快照相对旧 B50 的基线漂移混杂。

因此 T1 相对 B50/T0 的固定末轮 `+0.00670` mAP50-95 不能再归因于 gain=0 源码漂移；但它仍只是单 seed 信号，projection 是否真正贡献、global clipping 是否造成间接耦合，仍必须由已预注册的 T1-NP 与 T1-BRC 控制回答。T0 不授权跳过这些控制或直接写成论文性能结论。

## 6. 归档与校验和

完整本地归档：`outputs/yolo26_research/runs/tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1/`

| 产物 | SHA-256 |
|---|---|
| `results.csv` | `9437fdf5cd77a96e151b35a9177bd7bfa3e47c0cedf9ea90e8272abef19768e1` |
| `weights/last.pt` | `17bfffdbf9634810d5758eeb9f1cdd89bc997d4337c17a4f6358e8ef795eb178` |
| `weights/best.pt` | `e928a8c330fc1ec0264026d115b46b86a4a0a54836d261593ce521eb2849d8b7` |
| `train.log` | `3ac99442836fec5fa4daf50ee54627bd92b017968d09958bdbde8a0d86ea1425` |
| `source_config.yaml` | `b04ec96864bfb48fda7f04bd680c21876a461bf258b5ab17ae42d4a6f31d13f4` |
| `exact_repro_vs_b50.json` | `f8cb05ec047707e5db903b38f719a93373d14ddcd0d0f00137be029c2f6f0591` |
| `checkpoint_scope_vs_b50.json` | `6854662295c29b52fefffe0341516f08a0a178520b6f3603b657adfdbe191f4d` |

远端与本地的 `results.csv`、`last.pt`、`best.pt`、日志和 source config 哈希逐项一致。
