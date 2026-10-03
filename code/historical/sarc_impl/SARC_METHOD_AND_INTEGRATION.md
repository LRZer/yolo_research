# SARC staging：方法契约、折叠条件与集成清单

日期：2026-09-19  
状态：独立 CPU staging；未修改 `work/remote_research_source`，未同步远端，未使用 GPU。

## 1. 探针裁决

预注册探针真实通过 GO：positive-anchor pooled JS 为 P3/P5 `0.1234`、P4/P5 `0.0687`，最大相邻 epoch JS 仅 `0.0079`；unique-GT pooled 三对也均大于 `0.02`。因此各层 assignment-relative scale 分布既有可利用差异，又没有跨 epoch 剧烈漂移，允许进入 SARC 实现与 R1/R2/R3 因果矩阵。

## 2. 冻结结构

每个 P3/P4/P5 在 Detect 前插入独立残差块：

```text
R(x) = BN7(DW7(x))
     + sum_r softmax(a)[r] * BNr(DW3_d[r](x)), d=[1,2,3]
y    = x + gamma * SiLU(BN_mix(R(x)))
```

- 所有卷积 stride=1、depthwise、无 bias；7×7 padding=3，3×3 padding=dilation；
- `softmax(a)` 是每层三个输入无关的可训练标量，不是动态注意力；
- `gamma` 是每层一个标量，初始化为 `0.01`；
- R1 只有 7×7 主分支；R2 使用全部分支但不加校准；R3 使用全部分支和校准；
- 三者部署结构一致为 `DW7(bias=True) → SiLU → frozen-gamma multiply → residual add`。训练得到的 gamma 在导出时冻结为 buffer，不再是 deploy parameter；R1/R2/R3 的部署参数/FLOPs完全匹配。

## 3. 严格折叠审查

### 3.1 dilation kernel 嵌入

3×3 dilation `d` 的坐标偏移是 `{-d,0,d}`，在 7×7 中映射到索引 `{3-d,3,3+d}`：

| dilation | 7×7 索引 |
|---:|---|
| 1 | `{2,3,4}` × `{2,3,4}` |
| 2 | `{1,3,5}` × `{1,3,5}` |
| 3 | `{0,3,6}` × `{0,3,6}` |

因此只有 d=1/2/3 能在当前 7×7 中无损嵌入。该等价要求 stride=1、输入/输出空间一致，源分支 padding=d、部署 padding=3。

### 3.2 BN 与 softmax

每个分支先用 frozen running mean/variance 融合 Conv-BN；softmax 权重同时乘该分支 kernel 和 bias，再将膨胀小核嵌入并求和。`BN_mix` 最后再融合到求和后的 kernel/bias。严格条件：

1. 导出前必须 `eval()`，不可用训练 batch statistics；
2. softmax 权重必须是输入无关参数，并在导出时冻结；
3. 所有分支必须有相同 channel、stride、group 和输出网格；
4. 不允许在分支相加前放非线性。

### 3.3 residual gamma

方法定义是 `gamma * SiLU(z)`。由于 SiLU 不是正齐次函数，一般有：

```text
gamma * SiLU(z) != SiLU(gamma * z)
```

所以 gamma **不能**穿过 SiLU 融入 7×7 kernel。staging 在部署态把训练所得 gamma 冻结成 buffer，保留一次无可训练参数的常数标量乘法，严格等价且开销可忽略。只有把结构改成 `SiLU(gamma*z)`、删除 SiLU，或使用满足相应齐次条件的激活，才可能进一步折叠；这些都会改变已冻结的方法，当前不采用。

## 4. assignment EMA / calibration 接口

每个层对 O2M 正样本计算：

```text
u = log2(sqrt(w_gt * h_gt) / stride_level)
```

三个固定中心 `[2.25, 3.00, 3.75]` 通过温度 `0.50` 的高斯距离 softmax 得到每个样本对 dilation 1/2/3 的软成员度。它们先累加为 soft counts，再更新 decay `0.99` 的每层 EMA。前 16 次有效更新只收集统计，不施加校准。

```text
L_cal = 0.01 * sum_level KL(stopgrad(h_ema[level]) || softmax(a[level]))
```

DDP 契约是：各 rank 先对 soft counts 和有效样本 mass 做 `all_reduce(SUM)`，再在所有 rank 用相同全局值更新 EMA。不可直接平均各 rank 的归一化直方图，否则不同正样本数会产生偏差。EMA 是 buffer，不进 optimizer；校准目标完全 detached。

中心和 gain 是 staging 预注册值，正式 R3 前写入研究日志，不读取 VOC07 test 调整。若需改变，只能由 train-only histogram 的预定义统计规则产生，并形成新的独立实验协议。

## 5. R1/R2/R3 集成 patch 清单

不要直接复制 staging 到正在运行的源树；待 BRC 占用结束后，在新的 SARC 源副本按以下最小 patch 集成：

1. `ultralytics/nn/modules/conv.py`：加入 `SARCStagingBlock`、`SingleDW7StagingBlock` 和 deploy fusion；
2. `ultralytics/nn/modules/__init__.py`：显式导出两个模块；
3. 模型 YAML：在最终 Neck 的 P3/P4/P5 输出与 Detect 之间各插一块，并让 Detect 接收处理后的三尺度特征；
4. loss/assigner 接口：复用 O2M TAL 的 `fg_mask`、matched GT index 与 level block boundaries，按层收集 assigned GT `w/h`；不可从预测框反推；
5. DDP：all-reduce 三层 soft counts 和 mass 后更新 EMA；每个 rank 的 EMA 必须 bitwise 同步；
6. trainer：将 `model.sarc_calibration_loss()` 以单独 loss item 加入；R1/R2 恒为精确 0，R3 gain=`0.01`；
7. logger：保存每层 EMA histogram、branch softmax、有效 mass、KL 和 gamma；跨 rank 日志只写全局量；
8. exporter/fuse：模型 `eval()` 后替换为 deploy block；导出前后在同一输入做严格误差测试；
9. checkpoint：EMA buffers、branch logits、gamma 必须随 checkpoint 保存；resume 时不可重置 EMA；
10. 默认功能关闭，R0 必须保持官方模型 state/loss/梯度等价。

三个配置已冻结在 `configs/r1_fixed_dw7.yaml`、`r2_reparam_no_cal.yaml`、`r3_sarc_full.yaml`。

## 6. 预算

P3/P4/P5 为 `80²×64 / 40²×128 / 20²×256`：

| 项目 | 数值 |
|---|---:|
| deploy 7×7 DW weights | 21,952 |
| deploy bias | 448 |
| deploy gamma 常数（buffer，不是参数） | 3 |
| deploy 总新增参数 | 22,400 |
| deploy 总存储标量（含 gamma buffers） | 22,403 |
| 参数相对融合推理模型 2,378,736 | 0.942% |
| deploy MACs | 35,123,200 |
| FLOPs（2/MAC） | 0.07025 G |
| 相对融合推理基线 5.3 GFLOPs | 1.325% |
| R1 训练参数 | 23,747 |
| R2/R3 训练参数 | 38,540 |
| R2/R3 训练卷积 MACs | 54,476,800 |

参数门槛 `<1.2%`、FLOPs 门槛 `<2%` 均满足。这里不把 BN、SiLU、残差和标量乘法的逐元素操作计入卷积 FLOPs；正式论文应统一使用同一 profiler 对 R0/R1/R2/R3 重新报告。

上述 `2,378,736 / 5.3 GFLOPs` 是当前 VOC 模型完成融合后的推理口径；训练日志中的未融合模型为 `2,511,600 / 5.9 GFLOPs`。部署增量使用前者比较，训练显存/参数开销另行报告，禁止跨口径计算百分比。

## 7. 正式进入训练前的阻断项

- 在真实 YOLO26 特征上验证 P3/P4/P5 channel/order，而不是只依赖文档；
- 校验 level block index 与 stride 8/16/32 一致，错位立即中止；
- 补单卡真模型 forward/backward、R0-off 等价和梯度有限性测试；
- 补 4 卡 DDP soft-count/EMA 同步测试及 BF16 3-epoch smoke；
- 导出路径必须验证 train-eval、fused-PyTorch、实际 exporter 三者；
- 正式 R1→R2→R3 顺序运行，不按 test 结果调整 centers、temperature、gain 或 gamma init。
