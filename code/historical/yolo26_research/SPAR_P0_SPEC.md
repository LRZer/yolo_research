# SPAR-P0：目标相关亚像素相位信息丢失探针

状态：结果未知时冻结的实现规范；尚未写核心训练代码、尚未运行 GPU。  
目的：在尝试 SPAR 训练辅助目标前，先证明 YOLO26n 的现有 stride-2 下采样确实丢失了与目标相关、而不是一般纹理噪声的相位细节。

## 1. 假设

YOLO26n backbone 的 layer 3/5/7 是形成 P3/P4/P5 尺度的三个 3×3 stride-2 Conv。若目标边界与局部结构的 Haar detail 在这些下采样输出中难以线性解码，并且这种不可解码性与一像素平移不稳定和低 assignment quality 同时关联，则训练期恢复该信息可能有价值。

反之，若 detail 与 low-pass 同样可解码、目标区域不比匹配背景更特殊，或 shift error 与检测困难无关，则停止 SPAR；不得因为概念新颖就写训练模块。

## 2. 冻结输入

- 模型：精确 B50 seed42 `last.pt`；第一阶段只读终态 checkpoint。
- 数据：VOC07+12 trainval，`augment=false`、`shuffle=false`、`rect=false`、imgsz 640；绝不读取 VOC07 test。
- panel：按规范化图像路径 SHA256 排序后取前 32 张非空标签图像；偶数位 16 张只用于 ridge fit，奇数位 16 张只用于 held-out metrics。
- 模块：模型 graph 的 backbone layer 3/5/7；同时 hook 每层 stride-2 Conv 的输入 `x_s` 与输出 `y_s`。
- 状态：BN/eval，BF16 forward、FP32/FP64 统计；无 optimizer、无 backward、无参数或 buffer 写入。
- 若终态 P0a 为 GO，再在冻结的 epoch10/epoch30 checkpoint 上原样复核；阶段阈值不得修改。

## 3. 固定目标与 probe

对每个 pre-downsample feature `x_s` 用固定 2×2 Haar filters 计算 `LL/LH/HL/HH`。`LH/HL/HH` 拼接后通过 seed 固定的 Rademacher channel sketch 压到 16 channels；LL 也用独立固定 sketch 压到 16 channels，作为同维度低频控制。所有 target 均 detach。

分别为 detail 与 LL 拟合一个 ridge 1×1 probe：

```text
z_detail = W_detail y_s + b_detail
z_ll     = W_ll y_s + b_ll
```

- 仅 fit images 参与均值/方差标准化和闭式 ridge 求解；held-out images 不得参与任何拟合。
- ridge 系数固定为标准化设计矩阵 trace 的 `1e-3 / channels`，不根据 held-out R² 调节。
- 每张图每层从目标 mask 和背景 mask 各确定性采样最多 4096 个位置，避免像素数量大的图支配拟合。

## 4. 目标与背景 mask

GT box 映射到该层输出格点。目标 mask 是 box 内部并集；boundary ring 是 box 外扩一格减去内部；object region 为内部与 ring 并集。背景候选需距任一 box/ring 至少一格，并在同图内按位置哈希抽取与 object region 等量的位置。

所有统计分别报告 interior、ring、object-union、matched background；主门槛使用 object-union。重叠 GT 位置只计一次像素，但 GT 级 shift/quality 统计保留每个目标。

## 5. Held-out decodability

对每层、每区域报告标准 `R²=1-SSE/SST`，允许为负。为了排除“所有高频都比低频难预测”的平凡解释，定义目标特异的信息缺口：

```text
gap_obj = R2_LL_object - R2_detail_object
gap_bg  = R2_LL_background - R2_detail_background
specific_gap = gap_obj - gap_bg
```

P0 的第一项证据要求至少两个层 `specific_gap >= 0.10`，且 `R2_detail_object <= R2_LL_object - 0.10`。只出现一般 detail<LL、但 object 与 background 相同，不算 GO。

## 6. 一像素平移与 GT 难度关联

对同一 held-out image 分别做 `(dx,dy)=(1,0),(0,1)` 的零填充输入平移。按每层 stride 将 feature 坐标反向映射，并用固定 bilinear sampling 对齐到原坐标；在每个 GT 的 interior+ring 内计算归一化 feature L2 shift error。

检测困难只从同一 train-only forward 的 native O2M/TAL 取得：每个 GT 记录最大 matched target quality；未匹配记为 0。将 quality 最低四分位定义为预注册的 `low-quality`，不是按 AP 或 test 调阈值。

第二项证据同时要求：

- pooled GT 中，shift-error top quartile 的 low-quality incidence 至少是 bottom quartile 的 `1.5×`；
- shift error 与 `1-quality` 的 Spearman `rho >= 0.25`；
- 该方向在至少两个层成立。

## 7. 完整性门槛

- 32/32 图像全部完成，fit/held-out 各 16；
- held-out 至少 32 个有效 GT；
- 每层 object/background 均至少 8192 个 held-out 像素；
- 所有 R²、specific gap、quartile rate 与 rho 有限；
- 三层均成功 hook，空间尺寸严格为预期 stride；
- checkpoint、model YAML、data YAML、probe 源码和 panel 路径全部写入 SHA256 manifest；
- 所有参数与 persistent buffers 前后逐哈希相同，所有 `.grad is None`。

任一完整性条件失败时结果为 `INVALID`，不是 GO/NO-GO，也不能通过放宽阈值补救。

## 8. 自动裁决

```text
GO = decodability gate 在 >=2 层成立
     AND shift/quality gate 在 >=2 层成立
     AND 全部完整性门槛通过
```

否则为 `NO-GO`。P0a GO 后才做 epoch10/30 时间复核；至少终态和另一个 checkpoint 同方向，才允许实现训练期 SPAR。P0 不读取精度、不修改模型、不产生任何“最好参数”。

## 9. 若进入训练的必要阴性控制

- baseline：无 SPAR；
- phase-shuffle：mask、detail能量、loss强度与 predictor 完全相同，只在 batch 内循环置换 detail target；
- semantic SPAR：正确 image/position target。

只有 semantic SPAR 明显超过 phase-shuffle，才能把收益归因于正确亚像素相位信息；否则只能叫普通 auxiliary reconstruction/deep supervision。
