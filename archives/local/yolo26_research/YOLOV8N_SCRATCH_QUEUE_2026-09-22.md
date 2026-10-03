# YOLOv8n 全数据集随机初始化基线队列（2026-09-22）

## 已冻结实验协议

- 模型：Ultralytics YOLOv8n 源码结构，随机初始化；不加载任何 `.pt` 预训练权重。
- 数据：`processed_yolo` 下 11 个派生检测数据集。
- 每个数据集使用两个种子：`2027`、`2005`，共 22 次独立训练。
- 每次：125 epochs、batch 64、imgsz 640、SGD、workers 4、AMP。
- GPU：只使用物理 GPU 4、5、6、7；进程内映射为 0、1、2、3。
- NUMA：CPU 与内存均绑定到 NUMA node 1。历史不可纠正内存错误位于 node 0；GPU 4–7 的 NUMA affinity 也为 node 1。
- 验证：沿用用户指定的数据协议，每轮使用各数据集 `test` 集评估；不另划 val。

## 队列顺序

对以下每个数据集依次运行 seed 2027、seed 2005：

1. AP-10K
2. DUO
3. ExDark
4. HIT-UAV
5. HRSID-JPG
6. LLVIP-infrared
7. LLVIP-visible
8. ORCA-class-level-seen
9. SeaClear
10. VisDrone2019-DET
11. ZeroWaste-f

## 故障与重启恢复

- 队列状态原子写入 `queue_state.json`；单实例文件锁防止重复启动。
- 每 25 epochs 保存一次阶段检查点，同时保存 `last.pt` / `best.pt`。
- 服务器或训练进程中断后，先验证检查点确实可读取，再从最新有效检查点续训。
- 已完成任务以 `results.csv` 的 125 条 epoch 记录及 `last.pt`、`best.pt` 三项共同确认，不重复训练。
- 若不完整目录没有任何有效检查点，仅归档该不完整目录后重跑，不覆盖证据。
- 队列只等待 GPU 4–7，不终止、不修改任何其他用户进程。

## 结果目录

`/home/ubuntu/rz/object_detection_dataset_test/runs/yolov8n_scratch_baselines`

每个运行目录名包含数据集、epoch、batch、imgsz 和 seed，保证结果可追溯。

## 最终验收

队列结束后运行`_management/audit_yolov8n_scratch_queue.py`。只有22项全部满足以下条件才判定完成：

- `results.csv`严格包含epoch 1至125；
- `best.pt`和`last.pt`均存在、可读取且包含模型载荷；
- checkpoint内训练参数与冻结协议完全一致；
- 模型YAML的SHA256未变化；
- 逐运行指标及每个数据集双种子mean±std成功生成。
