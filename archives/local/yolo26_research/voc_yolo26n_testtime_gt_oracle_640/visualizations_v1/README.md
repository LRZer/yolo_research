# YOLO26n/VOC 真实框 Oracle 背景模糊：可视化

这是 2026-09-23 VOC2007 test 的 6 张示例，使用 150 轮 YOLO26n `last.pt` 对应的验证数据加载与 LetterBox 输入，真实评估框来自当前 batch。仅为**特权 Oracle 可视化**：正常未知图像推理时没有真实框，不能把这些图当成可部署预处理效果。

每个编号目录下：

- `source_original.jpg`：数据集原始照片的副本，用来核对内容。
- `input_raw.png`：实际送入模型的、LetterBox 后的原图；图像尺寸可能因 `rect=True` 批次填充超过名义 `imgsz=640`，这是验证器的实际输入。
- `light_sigma3.png`、`medium_sigma8.png`、`heavy_sigma16.png`：同一输入图像、同一真实框保护区域，框外渐变模糊的三个强度。框内像素在保存为 PNG 前的浮点张量中经逐像素恒等校验。
- `input_with_gt_boxes.png`：绿色轮廓标出被保护的真实评估框。
- `hard_protection_mask.png`：白色代表完全保持原像素；黑色代表框外可处理区域。
- `soft_protection_mask.png`：灰度显示框外 `sigma=8` 的过渡权重；越白越接近原图。
- `comparison_2x2.jpg`：原图与轻／中／重四格总览。JPEG 仅用于直观看图，实验指标并非根据这些保存后的文件计算。

| 样例 | 真实评估框数 | 完全保护面积 | 适合观察 |
|---|---:|---:|---|
| `000002` | 1 | 3.8% | 单个小目标，背景模糊差异最直观 |
| `000004` | 7 | 9.9% | 多目标但保护区仍小 |
| `000014` | 6 | 30.5% | 中等密度 |
| `000076` | 11 | 61.0% | 较密集场景 |
| `000025` | 10 | 79.6% | 多个大框覆盖画面，背景可模糊空间有限 |
| `000001` | 2 | 85.4% | 少数大框同样会几乎保护整图 |

远端主目录：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/research/testtime_gt_oracle_640/visualizations_v1/`。本目录是其本机镜像；原始 VOC 数据及标签未被修改。`manifest.json` 记录模型/数据哈希、每张图的原图与输入尺寸、框数、保护面积和像素变化。生成过程使用与三档验证同一 `protect_and_blur` 实现，且原图重跑的 O2O mAP50–95 为 `0.5770064130`，与先前本轮 raw 结果完全一致。
