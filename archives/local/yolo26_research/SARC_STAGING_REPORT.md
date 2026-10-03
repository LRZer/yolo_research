# SARC 独立 staging 结果

日期：2026-09-19  
状态：探针 GO 后的实现阶段完成；没有同步远端或占用 GPU。

## 结论

SARC 已在隔离目录 `work/sarc_impl` 完成方法契约、可折叠模块、assignment-histogram EMA/calibration 接口、R1/R2/R3 配置、参数/FLOPs 预算和 CPU 合成验证。

- 14/14 CPU 测试通过；
- dilation 1/2/3 的 3×3 kernel 嵌入 7×7 误差均为 `0`；
- 完整 R2/R3 train-eval 到单 DW7 deploy 的 float64 最大误差 `8.882e-16`；
- R1 最大误差 `4.441e-16`；
- deploy 新增参数 `22,400`（相对当前 VOC checkpoint 的**融合推理模型** 2,378,736 参数为 `0.9417%`），另有 3 个 frozen gamma buffers；含 buffer 的总存储标量为 `22,403`；
- deploy 增量 `0.07025 GFLOPs`（相对融合推理基线 5.3 GFLOPs 为 `1.3254%`）；训练态未融合模型为 2,511,600 参数、5.9 GFLOPs，不能把两套统计口径混用；
- R1/R2/R3 部署 topology 与成本完全一致。

## 一个必须保留的实现边界

分支 BN、softmax 静态权重、dilation kernels 和外层 BN 都可融合到单个 7×7 depthwise kernel；但已定义公式中的 `gamma * SiLU(z)` 不能变成 `SiLU(gamma*z)`。因此 deploy block 把训练后的 gamma 冻结为 buffer，并保留一次无可训练参数的常数标量乘法。合成反例的最大差值为 `0.23363`，若强行把 gamma 吸收进卷积会破坏严格等价。

## 校准接口

R3 对每层 O2M 正样本计算 `u=log2(sqrt(wh)/stride)`，使用三个固定 soft centers 形成 dilation 1/2/3 的 soft counts。DDP 必须先 all-reduce counts 与 mass，再更新每层 EMA；校准项为：

```text
0.01 * sum_s KL(stopgrad(h_ema_s) || softmax(a_s))
```

前 16 个有效更新仅积累 EMA。R2 拥有同一多分支训练结构但 gain=0，R1 是同 deploy 成本的单 7×7 控制。

## 下一阶段前置条件

该 staging 还不是远端训练补丁。正式集成前仍需：真实 YOLO26 P3/P4/P5 order/channel 审计、单卡真模型 forward/backward、R0-off 等价、4 卡 DDP EMA 同步、BF16 smoke、checkpoint resume 和实际 exporter 等价。只有完成这些检查后，才顺序运行 R1→R2→R3。

详细实现与测试文件：

- `work/sarc_impl/sarc.py`
- `work/sarc_impl/SARC_METHOD_AND_INTEGRATION.md`
- `work/sarc_impl/SARC_TEST_REPORT.md`
- `work/sarc_impl/configs/`
