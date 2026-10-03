# PBGA 之后的新假设：Agreement-Normalized O2O Reattachment

日期：2026-09-19  
状态：**由 PBGA-P0 结果事后生成，尚未预注册、尚未实现或训练；不能与原 PBGA 假设混为一谈**

## 观察

PBGA-P0 否定了“共享边界存在频繁负梯度冲突”：P3/P4/P5 在五个 checkpoint 的 total-gradient negative-dot rate 均为 0/8。但它同时暴露了另一种结构：反事实 O2O shared gradient 与 O2M 方向始终正相关，却因 ProgLoss 权重迁移在后期大得多。epoch 50 的 O2O/O2M norm ratio 中位数达到 P3/P4/P5=`24.17/75.34/77.03`。

因此不能直接删除官方 O2O feature detach：naive reattach 会让 O2O 梯度支配 trunk，并通过 global clipping 把官方更新缩小到原来的约 9%。但“无冲突且幅值失衡”本身形成了一个不同于 PBGA 的新问题。

## 新问题

> 能否保留官方 O2M 对 trunk 的主导作用，只把与其方向一致、按尺度归一化且不改变基础裁剪系数的少量 O2O deployment gradient 注入 P3/P4/P5，从而让部署头直接塑造共享表示？

暂名 **ANOR（Agreement-Normalized O2O Reattachment）**：

1. 在独立叶特征上计算原生 O2M/O2O 边界梯度；
2. 仅在 `dot(g_o2m, g_o2o)>0` 时保留 O2O 分量；
3. 每尺度把保留分量范数限制为 `rho * ||g_o2m||`；
4. 用 route-free official gradient norm 计算 global clip coefficient，再施加到 combined gradient，避免 O2O 通过统一裁剪间接缩放所有参数；
5. 两套 head 参数仍接收各自原生梯度，推理图不变。

公式：

```text
g_aux,s = I[<g_m,s,g_o,s> > 0] * min(1, rho*||g_m,s||/(||g_o,s||+eps)) * g_o,s
g_shared,s = g_m,s + g_aux,s
alpha = min(1, C/||g_official||)
```

## 必需控制

这个想法是结果后新假设，不能直接跑一个最优版本。若以后进入正式实验，至少需要：

- N0：官方 detach；
- N1：只做 norm cap，不做 agreement gate；
- N2：ANOR；
- N3：与 N2 匹配 shared update norm 的随机/固定方向控制；
- combined clipping 与 base-referenced clipping 对照；
- `rho` 只能由 train checkpoint 梯度预算冻结，目标是限制额外 shared norm 和裁剪扰动，不能根据 VOC test AP 搜索。

## 当前优先级

暂列为 SARC 之后的备选，不立即占用正式 GPU 队列。原因：

1. 正相关 cosine 只有约 `0.07–0.29`，是否足以转化为 AP 尚不明确；
2. 边界 `autograd.grad`、DDP 与 BF16 工程复杂度明显高于 SARC；
3. 必须先查重“dual-assignment O2O gradient reattachment / norm-balanced auxiliary gradient”相关工作；
4. 现有 T0、T1-NP、配对 seeds 和 SARC R1/R2/R3 的因果优先级更高。

这份记录的作用是保留 PBGA 负结果中产生的新方向，同时明确它是透明的 post-hoc hypothesis，不把 NO-GO 改写成成功。
