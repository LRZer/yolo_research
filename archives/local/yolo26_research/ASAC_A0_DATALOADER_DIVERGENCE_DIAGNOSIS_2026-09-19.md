# ASAC A0 与旧 B50 从 epoch 2 分叉的代码级诊断

日期：2026-09-19  
性质：只读静态诊断；未修改远端文件或训练进程  
结论强度：高置信机制定位，但未用单因素短训闭合各子机制占比

## 结论

A0 在 epoch 1 与旧官方 B50 的 75 个共同非耗时字段逐值相同，epoch 2 起立即大范围分叉。与这一时点精确对应的首要原因是：ASAC 为清除旧 `q` 的预取 batch，从第二轮开始在每个 epoch 边界额外重建训练 DataLoader iterator；旧 B50 通常只调用 `sampler.set_epoch(epoch)`，仅在 close-mosaic 时重建 iterator。

因此：

- A0 不能称为旧 B50 的 bitwise/numeric reproduction；
- A0 应称为“ASAC 基础设施下 fixed-q=1 的配对仪器化控制”；
- A2 与 A0 使用完全相同的 trainer、panel、逐轮 loader reset 和训练参数，主干预只有训练 `q` 策略，故 `A2 vs A0` 仍是有效主比较；
- 旧 B50 只能作为外部历史参考。

## 精确代码路径

冻结源码根：

```text
/home/ubuntu/rz/yolo_study/asac_research/snapshot/ultralytics_research
```

本地镜像根：

```text
work/asac_integration/snapshot/ultralytics_research
```

关键路径如下：

1. `ultralytics/models/yolo/detect/asac_train.py:364-375`
   - `on_train_epoch_start` 在 1-based epoch > 1 时先更新 sampler epoch，再调用 `train_loader.reset()`；
   - close-mosaic 对应轮次跳过该额外 reset，由基础 trainer 处理。
2. `ultralytics/engine/trainer.py:476-490`
   - 官方训练循环平时只更新 sampler epoch；仅在 close-mosaic 时 reset loader。
3. `ultralytics/data/build.py:70-85,97-105,108-125`
   - `InfiniteDataLoader` 长期持有 `_RepeatSampler` iterator；
   - `reset()` 关闭旧 workers 并创建新 iterator。
4. `ultralytics/data/build.py:346-385`
   - DDP 训练使用 `DistributedSampler`；本实验每 rank 4 workers，train `prefetch_factor=4`，每 rank 最多存在 16 个预取 batch。
5. PyTorch `torch/utils/data/dataloader.py:673-677` 与 `torch/utils/data/_utils/worker.py:258-265`
   - 每个新 iterator 从 loader 私有 generator 抽取新 `base_seed`；
   - 新 workers 用该 seed 重设 Python、Torch 与 NumPy RNG。
6. PyTorch `torch/utils/data/distributed.py:106-149`
   - `DistributedSampler.__iter__()` 按 `seed + epoch` 固化本次索引；`set_epoch()` 本身只更新属性。

## reset 产生的确定影响

- 丢弃旧 iterator 中最多约 16 个已预取 batch/rank；
- 重建 workers 并改变其 Python/NumPy/Torch 增强随机流；
- 按新的 sampler epoch 立即重建图像索引流；
- 旧 B50 的无限 iterator 可能已经跨越 epoch 边界预取，而 A0/A2 会丢弃该队列。

YOLO 的 Mosaic、MixUp/CutMix、RandomPerspective、HSV 和翻转均消费上述随机流，所以从 epoch 2 起训练轨迹分叉是预期结果。

## 排除的主要嫌疑

- `q=1` affine wrapper 已有逐矩阵及 RNG 消耗精确等价测试：`tests/test_asac_static.py:56-78`；
- panel 每轮保存并恢复 Python、NumPy、CPU Torch 与当前 CUDA RNG，同时哈希校验参数、buffer、BN 与 Detect 附加 tensor：`ultralytics/utils/asac.py:650-758`；
- A0/A2 的 trainer、measurement、source bundle、panel、population 哈希一致；
- A0 与 A2 都在相同轮次执行同样的 reset，A0/A2 之间不存在 loader 基础设施不配对。

## 证据边界

当前证据已把主因定位到“额外 reset 改变数据流”，但未通过反事实短训把以下三项贡献逐一拆开：丢弃预取队列、sampler 新排列立即生效、worker 重播种。若论文附录必须闭合，可另做 2-epoch 单因素实验，记录每个 rank 的图像路径序列和增强后 tensor hash；这不影响本次 Phase-1a 的 A2–A0 裁决。

