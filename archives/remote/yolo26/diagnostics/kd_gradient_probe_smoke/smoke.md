# KD 梯度与 global clipping 探针

> 固定 checkpoint、无增强、无 optimizer、无参数更新；同一权重/图像比较基础目标与两种 KD。

| epoch | mode | KD loss | base norm | KD norm | combined norm | base clip | combined clip | shared scale | cosine(cls) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | ungated | 1.72821 | 201.981 | 17.658 | 202.717 | 100.0% | 100.0% | 0.9964 | -0.0194 |
| 11 | quality | 1.61143 | 201.981 | 20.899 | 202.998 | 100.0% | 100.0% | 0.9950 | -0.0289 |

- `shared scale` 是加入 KD 后相对基础目标的 global-clip 缩放比；小于 1 表示 KD 会间接缩小共享参数更新。
- `cosine(cls)` 只在 O2O 分类头上比较基础监督与 KD 梯度方向。
- 单卡 batch 等于正式四卡 global batch，用于近似 DDP all-reduce 后的裁剪前范数。
