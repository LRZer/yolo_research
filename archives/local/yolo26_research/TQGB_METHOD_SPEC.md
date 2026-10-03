# TQGB v0：任务解耦质量路由与监督线段投影

日期：2026-09-19  
状态：完整 train 探针已通过机制覆盖门槛；本文件冻结候选数学定义与实现前校准规则。仅当 S-QGAT 首轮未达到继续线时进入实现，不在其运行中改源码。

## 1. 动机

O2M/O2O 的相对优劣并不是一个 anchor 级单一状态。B50 完整训练划分探针显示：

- 约 48%–54% common same-GT 位置的分类赢家与定位赢家不同；
- 分类 O2O→O2M 反向路由由 18.93% 增长到 39.17%；
- 定位 O2O→O2M 在第 50 轮仍有 7.11%；
- 投影后分类反向建议保留 73.67%，定位反向建议保留 100%。

因此，给整个 anchor 指定固定 O2M 教师会混合两个任务的相反信号。TQGB（Task-decoupled Quality routing with Ground-truth-segment Bridging）把分类和定位分别路由，并用原监督目标约束知识迁移方向。

## 2. 公共匹配集合

只处理同时满足以下条件的 anchor：

```text
foreground_o2m
AND foreground_o2o
AND target_gt_idx_o2m == target_gt_idx_o2o
AND target_label_o2m == target_label_o2o
```

这样无需跨 query/anchor matching，也不会把两个不同物体的预测互相蒸馏。

## 3. 分类任务路由

对离散 GT 类别 `y`：

```text
c_m = sigmoid(z_m[y])
c_o = sigmoid(z_o[y])
```

冻结首轮路由阈值：`tau_cls=0.25`、`margin_cls=0.05`。

```text
R_cls_m2o = common AND c_m >= tau_cls AND c_m - c_o >= margin_cls
R_cls_o2m = common AND c_o >= tau_cls AND c_o - c_m >= margin_cls
```

两条路线互斥。教师概率 stop-gradient；接收头的原始 soft TAL 分类目标记为 `t_s`。

## 4. 分类监督线段投影

直接模仿教师仍可能越过接收头的原监督目标。令学生当前概率为 `c_s`、教师概率为 `c_t`：

```text
c_proj = clamp(c_t, min(c_s, t_s), max(c_s, t_s))
```

`c_proj` 完全 detached。迁移损失采用 Bernoulli KL：

```text
L_cls_route = KL_Bernoulli(c_proj || sigmoid(z_s))
```

若教师位于学生到监督目标的方向上，它被保留；若超过目标，截断到目标；若位于反方向，截断到学生、产生零迁移梯度。因此知识项只能缩短监督残差，不能制造与原目标相反的更新。

## 5. 定位任务路由

使用预测框对共同 GT 的 IoU：

```text
b_m = IoU(box_m, GT)
b_o = IoU(box_o, GT)
```

冻结首轮阈值：`tau_box=0.25`、`margin_box=0.05`。

```text
R_box_m2o = common AND b_m >= tau_box AND b_m - b_o >= margin_box
R_box_o2m = common AND b_o >= tau_box AND b_o - b_m >= margin_box
```

为避免 xyxy 坐标的尺度不一致，将每个框变换为相对 GT 的四维残差：

```text
r = [dx / w_gt, dy / h_gt, log(w / w_gt), log(h / h_gt)]
```

GT 在该空间恒为零。逐坐标投影：

```text
r_proj = clamp(r_teacher, min(r_student, 0), max(r_student, 0))
L_box_route = SmoothL1(r_student, stopgrad(r_proj))
```

这同样保证迁移只把接收框拉向 GT 方向。

## 6. 梯度方向与部署桥

| 路由 | 直接接收参数 | 是否进入 shared trunk |
|---|---|---|
| cls/box O2M→O2O | O2O 对应 head | 否；YOLO26 对 O2O 输入 detach |
| cls/box O2O→O2M | O2M 对应 head | 是；O2M 路径连接 backbone/neck |

反向路线由部署头 O2O 的稀疏高质量预测指导 O2M，再把梯度送回 shared trunk，形成 deployment-to-trunk bridge。教师一侧始终 detach，避免互相追逐形成闭环。

## 7. 归一化与增益校准

不重复 D2 的 active-count 混杂。首选分母为每图所有 common anchors，使候选覆盖自然调节总强度：

```text
L_route = sum(route_point_loss) / max(count(common), 1)
```

分类、定位增益不凭 test 扫描。实现后先在 B50 的 epoch10/20/30/40/last 固定 checkpoint、训练数据固定 batches 上分别反传：

- `g_cls_m2o`、`g_cls_o2m`；
- `g_box_m2o`、`g_box_o2m`；
- base 分组梯度与全局 clip 后更新。

初始增益只允许由训练侧梯度比例校准，使每个新增任务的 clip 前范数不超过对应 base task/head 范数的预注册小比例；具体数值在看到任何新 test 精度前写入本文件。

### 7.1 分类路由冻结增益（2026-09-19，尚未运行 TQGB test）

在 B50 的 epoch 11/21/31/41/50 checkpoints 上，各取 8 个固定 train batches（共 512 图/checkpoint），无增强、BF16、无 optimizer、无参数更新。gain=1 时：

- O2M→O2O 路由对 O2O classification head 的新增梯度/基础梯度最高为 1.1855；
- O2O→O2M 路由对 O2M classification head 最高为 5.8808，shared trunk 最高为 2.1130；
- 两方向不能共享同一增益，否则反向路线会在 ProgLoss 后期因 O2M 基础权重下降而被严重放大。

因此在读取任何 TQGB 新 test 精度前冻结：

```text
lambda_cls_m2o = 0.08
lambda_cls_o2m = 0.015
```

缩放后，在五个 checkpoints 各 8 batches 的**批均值**中，最大值为：接收 O2O cls 9.48%、接收 O2M cls 8.82%、shared trunk 3.17%、全模型 3.36%；global-clip 后 shared update scale 的最小批均值为 0.99399，即平均额外缩放约 0.61% 以内。它们不是单 batch 的数学上界；后续需补 per-batch max/P95。这一校准使 T1/T2 的平均强度差异主要来自路由方向，而不是某一方向梯度量级大数倍。原始数据见 `diagnostics/tqgb_gradient_calibration*/`。

复核注：旧脚本配置写 `workers=4`，但框架 `mode="val"` loader 会把实际 workers 翻倍为 8；正式训练仍严格为 train loader workers=4。该差异不改变同一固定 batch 的梯度公式，但使旧探针的运行协议表述不准确。新版脚本直接构建固定无增强 loader，断言实际 workers=4/pinned memory，并将以逐 batch mean/P95/max/min 重跑；重跑前上述数值只按历史校准口径使用。

## 8. 首轮最小因果矩阵

若进入训练，按复杂度顺序而不是一次堆满：

| ID | 分类 | 定位 | 方向 | 投影 | 回答问题 |
|---|---|---|---|---|---|
| T0 | 无 | 无 | — | — | 同协议 B50 |
| T1 | 有 | 无 | O2M→O2O | 有 | 单向安全分类迁移是否有效 |
| T2 | 有 | 无 | 双向 | 有 | 角色反转是否带来增益 |
| T3 | 有 | 有 | 双向、逐任务 | 有 | 任务解耦定位是否再提升 |
| T4 | 有 | 有 | 双向、逐任务 | 无 | 投影是否必要；只在 T3 为正后运行 |

只有 T3 同时优于 T0、T1、T2，且 T4 证明投影有贡献，才能把完整 TQGB 作为论文主方法。否则保留最小有效子集，不能用复杂度替代证据。

T1 使用 `lambda_cls_m2o=0.08, lambda_cls_o2m=0`；T2 使用 `0.08/0.015`。两个增益只由上述训练梯度校准得到，不因 VOC07 test 结果重调。

## 9. 工程与报告约束

- 新功能默认关闭，gain=0 保持官方损失等价；
- 分别验证四条路由的梯度隔离和 stop-gradient；
- 空候选/空 GT 必须有限且为零；
- 不改变推理图、参数量、FLOPs；
- 研发继续使用完整 trainval 和固定第 50 轮，禁止 best-epoch 结论；
- 单 seed 只作筛选，最终需健康硬件、多 seed、150 epochs、COCO2017 与 YOLO26s。

当前代码仅实现分类 T1/T2；定位 T3/T4 尚未实现，因此现阶段不得把结果称为完整 TQGB。当前 CSV 的 `train/tqgb_*` 为 rank-0 局部 batch 均值，不是四卡全局计数；在 logger 侧完成跨 rank numerator/denominator 汇总前，只用于趋势诊断。
