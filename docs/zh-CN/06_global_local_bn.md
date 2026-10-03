# 6. 全局/局部分类与归一化

[目录](../../README.zh-CN.md) · [English](../en/06_global_local_bn.md)

## G4/G5精确双视图结构

下一假设是同时利用全局上下文和放大的目标细节。全局图保持native500画布；局部图按 **GT鸟框** 截取，无外扩，左上floor、右下ceil，裁边且保证非空；bicubic等比例将局部长边放到500，再居中补边。11,788框全量检查，训练前展示原/裁剪/放大尺寸。这是真正放大鸟，与A–F不缩放保护区不同。

G4输入**原全局＋局部图**，G5输入**归一化后严格零全局图＋相同局部图**。G5不加载原全局图、不透露其尺寸，但零分支仍经过主干和BN，不是删分支的单局部模型。

同一个ResNet50处理`[global_batch; crop_batch]`，两路2048维全局平均池化特征拼为4096，单200类线性头，参数24,327,432。共享卷积/BN不等于少处理一张图，每对仍两个500×500。scratch seed2005、SGD/增强同前、200轮LR前125项，train和test都需要GT框。

物理GPU2/3、NUMA0、DDP：每rank micro8对（16视图），两rank×累积4=有效64对。SyncBN每microstep混合32视图；workers2/rank合计4，评测16对/rank。RGB增强由seed、epoch、ID确定，两视图对齐，翻转半开框经过测试。精确sampler完整覆盖5,994/5,794图，无重复填充/丢尾；94updates，尾42张更新按全局16/16/10微步。rank sumCE×world_size/当次全局update图片数保证正确batch均值。

AMP初始scale65,536 smoke跳过1次更新并**阻断正式启动**，保留诊断；scale1,024完整smoke零skip后才开跑。保留source/config/bbox/stats SHA及checkpoint控制。[协议](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_DDP23_E125_PROTOCOL_2026-10-01.md)、[数值/AMP控制](../../archives/remote/classification/reports)、[裁剪尺寸](../../archives/local/foveated_classification/previews/cub_global_local_gt_long500_v1_2026-10-01/dimensions.csv)。

## 完整配对结果

| 固定125轮 | train Top1/Top5 | train loss | test Top1 | test Top5 | test loss | 次要best |
|---|---|---:|---:|---:|---:|---|
|G4原图＋局部|100% /100%|0.009257|52.278219%|78.736624%|1.960690|72轮，Top1 53.883328%|
|G5空白＋局部|100% /100%|0.009684|64.687608%|86.555057%|1.456469|118轮，Top1 65.429755%|

G5比G4高 **12.409389个Top1百分点、7.818433个Top5百分点**。两组均`verified_native_e125_ddp_pair`，完整125有限行、源码/配置/stats、best/last轮次、样本和update计数合格。[独立终审](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_DDP23_FINAL_2026-10-01.json)。

G4平均131.999秒/轮，G5为69.258秒，期间共享资源负载变化，**不是受控结构提速**。G5与A49.9655%相比同时改变GT裁剪放大、输入特权、两视图、分布式归一化与环境，不能隔离背景作用；更窄的主比较是G4/G5。

## 共享BN的混杂

BatchNorm训练计算：

$$\mu_B=\frac1m\sum_jx_j,\qquad v_B=\frac1m\sum_j(x_j-\mu_B)^2,\qquad z_j=\gamma\frac{x_j-\mu_B}{\sqrt{v_B+\epsilon}}+\beta.$$

G4用同一批统计混合整图特征（小鸟＋上下文）与局部特征（放大鸟）；G5混合局部与空白图特征。两分支都会影响均值/方差及推理running统计。空白不含场景内容，却仍提供BN数值。因此G5**不等于干净单路局部基线**，G5>G4不证明自然背景总有害。输入分布、放大、相关视图、归一化是竞争解释，不是已证实原因。

## G6：分离统计、共享参数

最小G4消融：53层BN分别维护global/local running均值、方差和counter，但**共享卷积及BN gamma/beta参数对象**。仍24,327,432参数，初始权重与G4逐tensor一致。这叫split-statistics BN，不是两个独立主干或两套独立仿射BN。

每路同步microstep从混合32视图变16视图，因此也改变有效BN样本数，不能仅归因于分布混合。CPU检查有限梯度、序列化参数alias、eval统计分离；DDP归一化及170train/34test smoke完成3update零skip，两rank状态SHA一致。[协议](../../archives/remote/classification/reports/CUB_GLOBAL_LOCAL_SPLIT_BN_E125_PROTOCOL_2026-10-02.md)。

## 当前准确状态：50轮用户停止

10月2日15:24CST在GPU2/3启动，按用户停止要求于 **18:20:47CST完成50整轮后停止**，只终止并核查本任务manager/rank/子进程。last50、best46 CPU可读、有限、来源一致。**未运行、未完成125、未自动续训**。后续GPU6请求只进行资源检查，收集记录中**没有启动GPU6训练**。

| G6未完成末轮 | 值 |
|---|---:|
|完成轮数|50/125|
|train Top1/Top5|91.508175% /99.449449%|
|train loss|0.321017|
|test Top1/Top5|43.182603% /73.023818%|
|test loss|3.051473|
|已见best Top1|43.803935%，46轮|

[权威停止记录](../../archives/remote/classification/reports/G6_splitstats_DDP23_2026-10-02_USER_STOP.json)、[50轮CSV](../../archives/remote/classification/runs/resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23/results.csv)。旧“正在运行”Markdown仅是历史快照，被停止记录取代。

![G4/G5完整曲线，G6虚线未完成前缀](../../figures/cub_global_local_and_incomplete_bn.png)

50轮前缀可启发诊断，不能对比G4/G5 125轮宣判BN。单局部模型、BN样本数严格控制、多种子及盲测确认都未完成。已完成结论只有条件性的G4/G5配对，BN因果解释**仍未解决**。
