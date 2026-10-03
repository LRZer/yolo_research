# YOLOv8n随机初始化基线方案（2026-09-22）

> 状态更新：本文件中的300轮/batch128预案尚未执行，已被用户随后冻结的125轮、batch64、双种子、22次队列协议取代。当前有效协议见`YOLOV8N_SCRATCH_QUEUE_2026-09-22.md`。

## 决策

- 不下载、不加载`yolov8n.pt`或其他预训练权重。
- 直接从YOLOv8n源码结构YAML创建网络并随机初始化。
- 主基线固定300轮；先用3轮完成单数据集冒烟训练，再正式训练。
- 所有后续改进与对应基线使用相同轮数、分辨率、batch、seed和数据版本。

从零训练比预训练微调收敛慢，150轮可以作为中间观察点，但不作为默认最终基线。若300轮末段loss和mAP仍明显改善，再预注册扩展到400轮；不得根据某个方法的结果单独增加训练轮数。

## 固定配置

| 项目 | 配置 |
|---|---|
| 模型 | YOLOv8n源码结构，随机初始化，`pretrained=False` |
| epochs | 300 |
| imgsz | 640 |
| 总batch | 128（4卡各32） |
| GPU | 物理4、5、6、7卡 |
| workers | 4/进程 |
| pinned memory | 当前Ultralytics dataloader默认开启 |
| cache | False，避免占用大量主机内存 |
| optimizer | SGD |
| lr0/lrf | 0.01/0.01 |
| momentum | 0.937 |
| weight decay | 0.0005 |
| warmup | 3轮 |
| early stopping | 关闭（`patience=0`），固定计算预算 |
| seed | 0；正式结论后补1、2两个种子 |
| val/test | 均指向test，按用户指定每轮评测test |
| checkpoints | 每25轮保存，另保留last/best |

## 轮数建议

- 3轮：只验证DDP、显存、数据读取、loss和保存流程，不报告精度结论。
- 150轮：检查是否基本收敛，属于中期结果。
- 300轮：所有数据集统一的有效scratch基线主结果。
- 400轮：只有在300轮曲线仍持续上升时，对基线和所有对比方法统一扩展。

## 训练入口

服务器脚本：`/home/ubuntu/rz/object_detection_dataset_test/_management/train_yolov8n_scratch.sh`

示例（仅说明，尚未执行）：

```bash
# 三轮冒烟
/home/ubuntu/rz/object_detection_dataset_test/_management/train_yolov8n_scratch.sh DUO 3 128 640 0

# 正式300轮
/home/ubuntu/rz/object_detection_dataset_test/_management/train_yolov8n_scratch.sh DUO 300 128 640 0
```

脚本强制使用后四张GPU，模型参数来自任务专用`models/yolov8n_scratch.yaml`，不会解析或加载`.pt`权重。
