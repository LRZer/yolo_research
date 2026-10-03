# PBGA-P0 零更新探针：隔离实现说明

日期：2026-09-19  
状态：本地 staging；尚未同步远端、尚未占用 GPU、尚未产生实验结果。

静态与 CPU 审查：

- `python -m py_compile`：PASS；
- CPU-only `--self-test`：PASS（projection、norm-cap、独立 head/surrogate autograd topology）；
- 自测通过标准输入交给服务器已有 Python 环境执行，显式设置 `CUDA_VISIBLE_DEVICES=''`，未创建远端文件、未调用任何 GPU；
- 初版 staging 脚本 SHA256 为 `fd279d23ff5f87ef5d9f2cb48749caf810a2d22f031208a63871cacb23029632`；正式首跑前因审计发现框架 `mode=val` 会把 workers 翻倍，已改为直接构建固定 loader，并强制校验实际 `num_workers=4` 与 pinned memory。更新后脚本 SHA256 为 `41ddca4783bc16653512c1646be963d5588f7ac633e24a53656e343c467aada8`。

## 1. 要回答的问题

PBGA-P0 只回答一个训练侧问题：在固定 B50 checkpoint 和固定 train batches 上，带相应 ProgLoss 权重的 O2M 与 O2O 原生监督，对 P3/P4/P5 共享特征的梯度是否存在足够频繁、足够强的冲突，从而值得实现 PBGA 训练。

固定检查点为 epoch 11/21/31/41/50；每个 checkpoint 使用相同的 8 个无增强 train batches。输出按 `checkpoint × P3/P4/P5 × cls/box/total` 汇总：

- feature-gradient cosine；
- `||g_o2o|| / ||g_o2m||`；
- negative-dot rate；
- 投影删除的 O2O 梯度范数/能量比例；
- 投影后的 norm cap；
- official / naive / cap-only / PBGA 四种路由下，完整模型参数梯度的模拟 global clip norm、scale 和相对 official scale。

YOLO26n 的 `reg_max=1`，原生定位有 `box_loss` 与 `l1_loss` 两项。因此报告里的 `box` 明确定义为两者之和；`total=box+cls`。

## 2. 从锁定源码确认的关键事实

远端锁定源码：

- `/home/ubuntu/rz/yolo_study/ultralytics_research/ultralytics/nn/modules/head.py`
- `/home/ubuntu/rz/yolo_study/ultralytics_research/ultralytics/nn/tasks.py`
- `/home/ubuntu/rz/yolo_study/ultralytics_research/ultralytics/utils/loss.py`

`Detect.forward()` 的训练路径先用原始 `x` 计算 O2M，再执行：

```python
x_detach = [xi.detach() for xi in x] if self.training else x
one2one = self.forward_head(x_detach, **self.one2one)
```

所以对官方 `preds["one2one"]` 直接调用 `autograd.grad(loss_o2o, P3/P4/P5)` 不可能得到共享特征梯度。它只能证明图已断开，不能测量“如果解锁，O2O 想怎样更新共享特征”。

## 3. 隔离求梯度路径

脚本不 patch `Detect.forward()`，也不修改 loss：

1. 精确复刻 `BaseModel._predict_once()` 的路由，只运行到最终 Detect 的输入，取得仍与 Backbone/Neck 相连的 `F={P3,P4,P5}`；
2. 创建数值相同、图彼此独立的叶节点 `F_m=detach(F).requires_grad_()` 与 `F_o=detach(F).requires_grad_()`；
3. 直接调用原生 `head.forward_head(F_m, **head.one2many)` 与 `head.forward_head(F_o, **head.one2one)`；
4. 继续调用原生 `v8DetectionLoss.get_assigned_targets_and_loss()`，没有复制 TAL 或损失公式；
5. 用 `autograd.grad` 分别求 cls、box、total 对三个叶特征的梯度；不调用 `.backward()`，不创建 optimizer；
6. 对 total 梯度生成四种 first-order boundary surrogate，连回原始 `F`，从而测量完整模型参数梯度 norm 和 `max_norm=10` clip scale。

这个拓扑保证：两套 head 都保留各自原生参数梯度，而共享 Backbone/Neck 只接收所选择的边界梯度。

## 4. BatchNorm 与零更新保证

探针使用训练模式，以保留真实 train-batch normalization；但每个 batch 前快照所有 BatchNorm running mean/variance/counter，完成所有前向和 `autograd.grad` 后原样恢复。结束时还会检查：

- 所有 BN buffer 与初始 checkpoint 一致；
- 所有模型参数的 `.grad is None`；
- loader 使用 pinned memory；
- 五个 checkpoint 的 ordered batch fingerprints 完全相同；
- checkpoint SHA256 和 head/tasks/loss 源码 SHA256 写入 JSON。

这比简单设置全部 BN 为 eval 更接近训练态，同时不会让诊断污染内存中的 checkpoint 状态。

## 5. 自动裁决（预注册、不读取 VOC test）

对 total feature gradient，若至少一个尺度在两个以上 checkpoint 同时满足：

```text
negative-dot rate >= 0.10
median ||g_o2o|| / ||g_o2m|| >= 0.05
```

则 `GO`，进入 PBGA 训练实现；否则 `NO_GO`，按路线图转向 SARC assignment-histogram probe。

附加报告 Broad Direction Screen 的两个 no-go 条件：所有 checkpoint/scale 冲突率都 `<5%`，或所有 median norm ratio 都 `<0.05`。固定 8 batches 时冲突率粒度为 `1/8=0.125`，报告会明确这一限制。

## 6. 真实实现风险

1. **不能删除 detach 后直接相加损失。** 那只实现 naive reattach（G1），不能替换共享梯度，也无法保留 PBGA 的非对称投影语义。
2. **head 原生梯度与 shared surrogate 必须解耦。** 如果 head loss 仍从原始 `F` 回传，再叠加 surrogate，会把 shared 梯度计算两次。正式实现需沿用独立叶特征拓扑。
3. **global clipping 会间接耦合所有参数组。** 即便 PBGA 只改 shared feature gradient，clip scale 改变也会缩放两套 head 梯度；因此 P0 已测完整参数梯度，后续训练仍需记录每组 pre/post-clip norm。
4. **AMP/DDP 是主要工程风险。** `autograd.grad` 得到的边界梯度、GradScaler/unscale、DDP reducer 的执行顺序若错误，可能出现双归约、unused parameters 或不同 rank 路由不一致。P0 单进程结论不能替代正式实现的单卡等价、4 卡 DDP 等价和 BF16 empty-target smoke。
5. **训练成本明显增加。** 每个 batch 至少需要两套原生 head 图、两组边界梯度和 surrogate；正式版本需在不保留无用二阶图的前提下实现，并实测显存/墙钟。
6. **尺度独立投影会改变 cross-scale 总方向。** P0 和方法定义均按 P3/P4/P5 独立仲裁；不能先 concat 后统一投影，否则不再是预注册机制。
7. **探针可行不等于 AP 有效。** `GO` 只说明存在可利用冲突；G1/G2/G3 的固定 epoch-50、last-5 与 matched-control 门槛仍必须全部执行。
8. **P0 的 global clip 是单进程固定 panel 模拟。** 它使用完整模型参数梯度而非 feature norm 近似，但四卡 DDP 的每-rank BatchNorm 统计、梯度 all-reduce 顺序仍可能产生小差异；正式实现必须另做 4 卡测量，不能把 P0 scale 当作训练日志的精确替代。

## 7. 本地审查命令

```bash
python probe_pbga_p0.py --self-test
python -m py_compile probe_pbga_p0.py
```

正式运行应在 T2 完成、GPU 空闲且 MCE 检查通过后，由外层 `numactl --cpunodebind=1 --membind=1` 启动；脚本自身只允许 physical GPU 4–7、固定 workers=4，并通过直接构建 loader 强制实际 `num_workers=4` 和 pinned memory。不得改回 `trainer.get_dataloader(..., mode="val")`，因为当前框架会将验证 loader workers 翻倍。

冻结的首跑参数（同步脚本后执行，当前不执行）：

```bash
env PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research PIN_MEMORY=True \
numactl --cpunodebind=1 --membind=1 \
/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python scripts/probe_pbga_p0.py \
  --checkpoints \
    runs/b50_no_kd_e50_g4567_s42/weights/epoch10.pt \
    runs/b50_no_kd_e50_g4567_s42/weights/epoch20.pt \
    runs/b50_no_kd_e50_g4567_s42/weights/epoch30.pt \
    runs/b50_no_kd_e50_g4567_s42/weights/epoch40.pt \
    runs/b50_no_kd_e50_g4567_s42/weights/last.pt \
  --data configs/datasets/voc_server.yaml \
  --physical-device 4 --batch 64 --imgsz 640 --workers 4 --max-batches 8 \
  --clip-norm 10 --seed 0 \
  --output-dir diagnostics/pbga_p0_b50_train \
  --name pbga_p0_b50_train
```
