# ASAC-P0 全量零训练审计终报

日期：2026-09-19  
正式目录：`outputs/yolo26_research/ASAC_P0_B50_V2/`  
裁决：**GO（仅批准 scale/translate-ASAC 原型，不代表已经提升 AP）**

## 1. 本轮回答的问题

本轮不训练、不看 VOC07 test AP，只回答一个先验机制问题：在 YOLO26n 的原生 O2M/O2O 双分配中，确定性的缩放和平移扰动是否会产生随强度变化、且与监督质量保持相关的 assignment-stability 信号。只有该信号在两个原生 head 上都存在，才值得实现训练期自适应增强控制。

自动门槛全部通过：

| 证据 | O2M | O2O | 预注册要求 |
|---|---:|---:|---:|
| 稳定度随强度单调下降的 checkpoint | 5/5 | 5/5 | 至少 3/5 |
| q=0.25 到 q=0.75 的平均稳定度下降 | 0.04537 | 0.03655 | 至少 0.02 |
| 最强扰动下质量相关 `rho>=0.2` 的 checkpoint | 5/5 | 5/5 | 至少 2/5 |
| 最低 clean/view assignment coverage | 99.84% | 100.00% | 至少 50% |

因此自动 verdict 为 `GO`：两个原生 head 均表现出强度敏感、质量相关的 affine assignment stability。

## 2. 冻结协议与完整性

- checkpoint：同一 B50 轨迹的 epochs 11、21、31、41、50；五个文件逐 SHA256 固定。
- 数据：train-only、顺序固定的前 256 张图，batch 64、4 batches、imgsz 640、workers 4、pinned memory 开启。
- 扰动强度：`q={0.25,0.50,0.75}`，只做预注册的 scale+translate；不含 Mosaic。
- 运行：物理 GPU4、NUMA1、BF16 inference mode；GPU5–7 未使用。
- 原生 criterion：O2M `topk/topk2=10/10`，O2O `7/1`，stride `8/16/32`。
- 无 optimizer、无 backward、无参数更新；每个 checkpoint 的参数 SHA 与 BN-buffer SHA 在前后完全一致，114/114 个 BN 均保持 eval。
- 五次读取的 panel、GT geometry 和 image+label 内容指纹完全一致。

关键来源指纹：

- 探针源码：`be350ae3b7fcdf033916ac1aa797d5c3b9fb38103a94b6e911cac0fe0666ea05`
- 训练轨迹：`0e6b09538be74edefb21185dc1edca18541fc454d5d87e365f2b6e0abdde9292`
- panel 路径/张量：`5bc7260afc2919b071948b2e496f017cdc72d9f5d424bb62cd7f02705dea8266`
- 256 张图及对应标签字节内容：`8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`
- GT geometry：`baeae282f7f3810370b5633eb8f0fc46f1461afd5d35bacb553213d2c95ddaec`

正式运行从 20:11:42 到约 20:12:12。启动前直接 NVML 确认 GPU4–7 均无计算进程；运行期间只使用 GPU4；运行窗口内核日志无新增 MCE、SIGBUS、Memory failure 或 EDAC error；结束后 GPU4–7 均回到 0% 且无计算进程。本机仍有历史平台级内存故障，因此该结果标记为 **faulty-host exploratory**，最终论文证据仍须健康硬件复现。

## 3. 样本与几何边界

256 张图共有 657 个 clean GT；在三种强度的全部视图中共同存活 624 个，占 94.98%。逐强度存活率为：

| q | 存活 GT | 比例 |
|---:|---:|---:|
| 0.25 | 647 | 98.48% |
| 0.50 | 634 | 96.50% |
| 0.75 | 624 | 94.98% |

被排除的 33 个 GT 明显偏小且靠边：其面积比例中位数为 0.01972，而 retained 组为 0.08039；box-boundary margin 中位数为 0.002，而 retained 组为 0.127；center-edge distance 中位数为 0.089，而 retained 组为 0.3275。因此 GO 严格适用于“最强扰动下仍可见”的对象，不能声称已经证明边缘小目标同样适合该控制器。

每个 checkpoint/head/q cell 有 624 个对象；只有四个 O2M cell 少 1 个双视图共同 assignment，最低 coverage 仍为 623/624=99.84%，不存在样本不足裁决。

## 4. 主要数值与解释

稳定度 `S=exp(-JS)` 随 q 增大在全部 checkpoint、两个 head 中下降。O2M 的 q=0.25→0.75 下降依次为 0.05283、0.04856、0.04179、0.04339、0.04027；O2O 为 0.03056、0.03591、0.03777、0.03736、0.04116，均高于 0.02 门槛。

最强扰动 q=0.75 下，`rho(S,Q-retain)`：

| epoch | O2M | O2O |
|---:|---:|---:|
| 11 | 0.2220 | 0.3139 |
| 21 | 0.3163 | 0.3372 |
| 31 | 0.2865 | 0.2947 |
| 41 | 0.3309 | 0.2964 |
| 50 | 0.3114 | 0.3948 |

全部 10 个值均为正且超过 0.2。也就是说，assignment 在扰动下越稳定的对象，监督质量通常保持得越好；该关联在训练早、中、晚期均可重复观察。

同时，强扰动会提高主特征层翻转率。O2M 的 q=0.75 flip rate 从 epoch11 的 16.21% 降至 epoch50 的 9.94%；O2O 从 11.06% 到 10.58%。这提示稳定度反馈确实捕捉到了 assignment/尺度层级变化，而不只是输出质量的同义改写。

## 5. GO 能说明与不能说明的内容

能说明：

1. scale/translate 强度是一个可测、方向正确的 assignment-stability 干预量；
2. 该信号同时存在于原生 O2M 和 O2O，而非单一 head 偶发现象；
3. 信号与监督质量保持具有对象级正相关，足以进入训练期控制器因果验证；
4. 控制器可以只作用于训练增强，理论上不改变推理网络、参数量和延迟。

不能说明：

1. 尚未证明 ASAC 能提高 mAP；本轮完全没有训练或 test AP；
2. 五个 checkpoint 是同一条训练轨迹的时间切片，不是五个独立随机种子；
3. 固定前 256 图不是整个 trainval 的随机代表样本；
4. 只分析了 scale/translate，不能外推到 Mosaic；
5. dropped GT 显著偏向小而靠边对象，后续必须单独报告 small/edge subgroup，不能隐藏该选择偏差；
6. 当前服务器有历史硬件故障，正式论文数值必须在健康硬件从头复现。

## 6. 下一阶段最小因果矩阵

P0 只批准以下四组，不批准直接叠加 Mosaic、注意力、损失或结构模块：

| ID | 训练策略 | 因果问题 |
|---|---|---|
| A0 | 官方固定 `scale=.5, translate=.1, close_mosaic=10` | 基准；现有 B50 可作探索参考 |
| A1 | 固定 epoch schedule，匹配 A2 实际 q 的均值/边际 | 仅改变强度曲线是否足够 |
| A2 | train-only assignment-stability 反馈控制 q | 真实反馈是否优于固定调度 |
| A3 | 打乱 A2 的 epoch stability 序列，同时保持 q 边际分布 | 时序反馈是否真正有因果价值 |

执行纪律：先在独立 staging 实现控制器、确定 train-only 统计与 DDP 聚合、补齐 synthetic/单 batch/多 rank 测试；之后才允许正式训练。单 seed 50-epoch 初筛的预注册继续线为：A2 相对 A0 固定末轮 `>=+0.003` mAP50-95、相对 A1 `>=+0.002`、last-5 差值同号、训练耗时增加 `<12%`。只有 A2 相对 A1 和 A3 均成立，才把结果解释为 assignment-stability feedback，而不是普通增强调度。

当前建议在实现前再冻结三点：每 epoch 的 q 更新公式、反馈使用上一完整 epoch 还是 EMA、以及 A1/A3 的离线重放规则。不得依据 VOC07 test 曲线调这三项。

## 7. 归档

正式 JSON、逐对象 CSV、逐 GT geometry CSV、summary、decision、Markdown、最终 smoke 以及 256 对 image+label 完整清单均已复制到 `outputs/yolo26_research/ASAC_P0_B50_V2/`；本地正式六个输出的 SHA256 与服务器逐项一致。

