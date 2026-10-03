# VOC + YOLOv8n：目标保护的渐变背景模糊核心原型

## 当前交付范围

包含可运行的 PyTorch 核心模块、单元测试与本次测试记录，不是已经接好 Ultralytics 的完整训练工程。
不含 VOC 图片、YOLO 权重、训练结果、检测精度或端到端速度结论。
CPU / PyTorch 2.10.0 上通过了张量形状、保护区不变、插值权重、梯度、空监督、存取和 640 输入测试。
没有执行 VOC 训练，也没有验证 CUDA、AMP、DDP、ONNX 或 TensorRT。

前端可学习参数为 **38,513**。高斯核为固定 buffer，不计入可学习参数，但仍有计算与显存开销。

## 任务定义

图像 → 预测连续模糊强度 S → 多尺度高斯图混合 → 与原图残差混合 → YOLOv8n。
S=0 表示保留原图，S=1 表示采用最大模糊尺度。训练框仅构造 S 的辅助监督，不作为推理输入。
仍检测图中全部 VOC 类别实例，不是逐目标搜索，不是实例分割，也不是低照度专用模型。
“背景”是所保护框外的区域，不等于精确像素级背景。

## 文件

- `pgb_core.py`：连续几何监督、38.5K 预测器、可微多尺度模糊、区域平衡监督损失、YOLO 框转换。
- `test_core.py`：只测试核心模块，不训练检测器。
- `test_results.json`：本次实际测试输出。

## 运行

Python 3.10+、PyTorch 2.x。装有 PyTorch 后，在本目录执行：

```bash
python test_core.py
```

## 核心模块的最小调用

```python
import torch
from pgb_core import ProgressiveBlurFrontEnd, make_targets, balanced_map_loss

images = torch.rand(2, 3, 640, 640)  # 真正使用时换为归一化 RGB batch
boxes = [torch.tensor([[120., 80., 260., 440.]]),
         torch.tensor([[300., 200., 450., 380.]])]

frontend = ProgressiveBlurFrontEnd(mix=0.5)
processed, predicted_strength = frontend(images)
targets = make_targets(images, boxes)
aux_loss = balanced_map_loss(predicted_strength, targets)
aux_loss.backward()  # 这里只演示辅助监督；完整训练还需要 YOLO 检测损失。

# 推理时只调用 frontend(images)，不得调用 make_targets 来修改推理输入。
```

输入为浮点 RGB `[B,3,H,W]`，范围 `[0,1]`。框为对应当前图像的绝对像素 `xyxy`。
图像和框必须已经完成一致的 resize / letterbox / flip / perspective 等几何变换。
空框图不直接当成确定背景；本原型忽略它的模糊监督，检测任务仍保留其原始监督。
letterbox 区域可通过 `valid_pixels` 排除；difficult 区域可通过 `ignore_boxes_by_image` 排除。

## 几何监督

每个框每侧外扩其宽/高的 10%，作为保护区。对框外像素计算水平和垂直距离，分别除以
`max(0.5*框宽, 8)` 和 `max(0.5*框高, 8)`。对所有框取最小平方距离 d²。

`S_target = 1 - exp(-d²/2)`

这保证任意保护区内的监督值为 0，而非只保护某一个目标。离所有保护区越远，值越大。
此保证属于几何监督图；预测网络不保证在每张新图上严格满足该规律，也不保证保护到所有弱目标。

## 渲染方式

固定尺度为 sigma = `[0, 1, 2, 4, 8]`，单位是当前图像像素。
预测 `sigma = 8*S`，在相邻两个尺度的高斯滤波结果之间线性插值。
实际结果是不同高斯滤波图的连续空间混合，不是逐像素精确动态高斯卷积，也不是光学景深模拟。

最终输出：`I_out = (1-rho)*I + rho*I_blur`，默认 rho=0.5。
rho 允许在训练时逐步增加，验证/测试时应固定为预先确定值。
改变输入大小时需明确是否同步缩放 sigma；第一轮建议固定 imgsz=640、关闭 multi_scale。

## 与 YOLOv8n 集成时必须完成的部分

1. 记录实际 Ultralytics/PyTorch 版本或 Git commit。不同版本的 loss 返回结构可能不同。
2. 基础检测网络使用 VOC 20 类，正确加载 `yolov8n.pt` 中尺寸匹配的预训练参数。
3. 将 `ProgressiveBlurFrontEnd` 注册为训练模型的 `nn.Module` 子模块。
4. 图像经过标准归一化后调用前端，把 processed 送入底层 DetectionModel 的张量前向。
5. 原始检测标注保持不变；将更新后的 batch xywh 转为当前像素 xyxy 生成辅助监督。
6. 组合原生 YOLOv8 box/cls/dfl 损失和 `lambda_map * L_map`，核对按 batch 求和/求均值尺度。
7. 优化器、EMA、checkpoint、resume 都必须包含前端；不能只保存 YOLO 权重。
8. Validation/Predict 必须调用同一个带前端模型。不能训练时带前端，验证时静默跳过。
9. 不得把含 `predict`/NMS 的高层推理接口放进训练损失链路。
10. 不得在学习式前端与 YOLO 之间使用 NumPy、OpenCV、uint8 转换或 detach。

Ultralytics 的 `YOLODataset.build_transforms` 可以接训练期规则增强；自定义检测 Trainer/Model
可以管理真正的学习式前端。只有训练期增强时，前端不属于推理模型；这两种实验不能混称。
本包没有假装提供一个适用于任意 Ultralytics 版本的一键训练适配器。

## 建议的首轮实验设置（待验证，不是推荐最优值）

- 固定 VOC07+12 trainval 内部开发划分；VOC2007 test 留到最终一次比较。
- imgsz=640，batch=16（显存不足则所有组统一减少），epochs=100。
- 固定优化器与全部数据增强参数；首轮关闭 Mosaic/MixUp/CopyPaste/CutMix，随后再与强基线比较。
- 所有检测器从同一 COCO 预训练 YOLOv8n 初始化；不可一组多训练、另一组少训练而不说明。
- 前 5 个 epoch 设置 rho=0，YOLO 学原图、前端学辅助监督；接着 5 个 epoch 将 rho 增至 0.5。
- 后续联合训练，初始 lambda_map=1；先检查损失尺度再在内部验证集调参。
- 所有参数仅在内部验证集确定，不用 VOC2007 test 选择最优参数或 epoch。

## 对照实验

A：原始 YOLOv8n。
B：仅训练期使用真值框引导的渐变模糊增强，推理输入为原图。
C：学习式前端 + 原图/单张固定强度模糊图的软混合。
D：学习式前端 + 本原型的多尺度连续模糊。

C/D 使用相同预测器与监督，比较多尺度是否真正有额外作用。
真值框指导的测试图只能作额外信息条件下的诊断，不是可部署结果，也不保证是理论上界。
不得从 B 的效果直接断言 D 一定有效或无效，因为它们的训练/推理路径不同。

报告 mAP50、mAP50-95、分类型 AP、小目标/密集图像错误、前端+YOLO+后处理总耗时、峰值显存。
Ultralytics 默认 AP 计算与历史 VOC2007 的 11 点 AP 并非同一实现，应注明 evaluator。

## 主要参考接口

- Ultralytics VOC 数据配置与 difficult 转换：https://docs.ultralytics.com/datasets/detect/voc/
- YOLOv8 模型与预训练权重：https://docs.ultralytics.com/models/yolov8/
- 自定义训练器：https://docs.ultralytics.com/usage/engine/
- 数据变换：https://docs.ultralytics.com/reference/data/dataset/
- 检测训练流程：https://docs.ultralytics.com/reference/models/yolo/detect/train/
- YOLOv8 损失：https://docs.ultralytics.com/reference/utils/loss/
- AP 实现：https://docs.ultralytics.com/reference/utils/metrics/

以上来源用于接口核对，不证明本原型有效或具有独创性。
