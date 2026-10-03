# SARC staging CPU test report

日期：2026-09-19  
结论：**PASS（staging 数学与 CPU 合成实现）**；尚未表示已完成 YOLO26 真模型、DDP、BF16 或 exporter 集成。

## 执行边界

- 运行环境：Python 3.11 / PyTorch 2.6.0；所有测试张量明确位于 CPU；
- 未连接远端、未占用 GPU、未修改 `work/remote_research_source`；
- 命令：`python -m unittest -v test_sarc.py`；
- 结果：14/14 通过；最终复核耗时约 0.07 秒。

## 核心数值

| 检查 | 实测结果 | 判定 |
|---|---:|---|
| dilation=1 的 3×3→7×7 最大绝对误差 | `0.0` | PASS |
| dilation=2 的 3×3→7×7 最大绝对误差 | `0.0` | PASS |
| dilation=3 的 3×3→7×7 最大绝对误差 | `0.0` | PASS |
| R1 eval→deploy float64 最大绝对误差 | `4.441e-16` | PASS |
| R2/R3 eval→deploy float64 最大绝对误差 | `8.882e-16` | PASS |
| 错误地把 gamma 穿过 SiLU 所造成的最大差值 | `0.23363` | 证实不可折叠 |
| R1 实际训练参数 / 解析值 | `23,747 / 23,747` | PASS |
| R2/R3 实际训练参数 / 解析值 | `38,540 / 38,540` | PASS |

## 覆盖的 14 项测试

1. dilation 1/2/3 kernel 嵌入严格等价；
2. Conv-BN eval 融合严格等价；
3. 完整 SARC 分支、softmax、两级 BN 的 deploy 等价；
4. R1 固定 7×7 控制的 deploy 等价；
5. gamma 不可穿过 SiLU 的反例；
6. gamma=0 时严格恒等残差；
7. `u=log2(sqrt(wh)/stride)` 数值；
8. soft histogram、EMA、warmup 和 detached calibration target；
9. softmax 分支权重非负且和为 1；
10. DDP global-count 接口与直接单批更新等价；
11. R1/R2/R3 forward 与校准 loss 接口；
12. 参数/MAC 预算精确值；
13. 解析参数公式与实际 module 参数计数一致。
14. deploy 实际参数计数为 22,400，且 gamma 只存在于 buffers、不在 parameters。

## 部署预算复核

- deploy 参数：`22,400`，另有 3 个 frozen gamma buffers；总存储标量 `22,403`，相对融合推理模型 2,378,736 参数为 `0.9417%`；
- deploy MACs：`35,123,200`；按 2 FLOPs/MAC 为 `0.0702464 GFLOPs`，相对融合推理基线 5.3 GFLOPs 为 `1.3254%`；训练日志中的 2,511,600 参数、5.9 GFLOPs 属于未融合口径；
- R2/R3 训练分支卷积 MAC 为 deploy SARC block 的 `1.5510×`；
- 三个实验的 deploy topology 相同，满足 fixed-cost 因果对照。

## 当前不能由该报告推出的结论

- 不能推出真实 YOLO26 YAML 中 P3/P4/P5 接口已经接对；
- 不能推出 DDP 的 all-reduce/EMA checkpoint resume 正确；
- 不能推出 BF16、ONNX/TensorRT 或真实 exporter 等价；
- 不能推出 R3 精度优于 R2，也不能推出 assignment calibration 有效；
- 不能把 CPU float64 等价误差当作 BF16 容差。正式 exporter 应分别规定 FP32/BF16 容差。

## 文件哈希（测试时版本）

| 文件 | SHA256 |
|---|---|
| `sarc.py` | `0a534ba66980e91340bbf5885b295a7838576a942692c54c845065827eb2970b` |
| `test_sarc.py` | `4ea998b85ddac5b6349cc1e5aa1eb640ecf20d00e638faaa92d4375e9e5171d6` |
| `audit_sarc.py` | `47254a7ee7db9d3c917d68b93c336e5d6ac96a356758c5e68a3cf5813255c4f6` |
| `r1_fixed_dw7.yaml` | `76f6bc24ec3c586c00963200b8750a473db98f81afc25e314604b9e4a6298e7f` |
| `r2_reparam_no_cal.yaml` | `291c297f5fe2435b6ed8094ed92e81938b2d9f39c198b2d6d733c52531da11a7` |
| `r3_sarc_full.yaml` | `d8fddd10e9c2115481604acedca7f96996752f3055b078182f15e17daeee2ed9` |
