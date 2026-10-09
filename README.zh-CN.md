# YOLO 与中央凹视觉研究

[English](README.md) · [完整指标附录](docs/zh-CN/tables.md) · [全部历史文档](docs/zh-CN/archive_index.md) · [证据及 SHA 清单](results/evidence_manifest.csv)

本仓库按完整时间线整理 YOLO26 双头监督、跨领域 YOLOv8n 基线、目标保护背景模糊、VisDrone 特征平滑与凹视推理、CUB-200-2011 图像表示、全局/局部分类和已停止的归一化实验。保留正向信号、负结果、工程审计和未完成工作，不将其包装为已完成的最先进方法。

## 核心结果

| 实验 | 配对参照 | 结果 | 判断 |
|---|---|---|---|
| YOLO26 投影式 O2M→O2O 迁移，50 轮 | B50，seed42 / seed0 | AP95 差值 **+0.670 / −0.215 点** | 单种子收益未复现，停止该路线 |
| 基于分配稳定性的增强控制器 | 插桩固定强度 A0 | AP95 **50.847% vs 51.357%** | 机制探针通过，但训练没有收益 |
| 跨领域 YOLOv8n 从零基线 | 11 个数据版本 × 2 个种子 × 125 轮 | **22/22 完成并验收** | 是参考基线集，不是 22 项算法改进 |
| VOC 学习式安全背景模糊 | 相同初始化，额外 125 轮 | AP95 **49.5201% vs 50.0451%** | 负结果，实测预测延迟约 1.9 倍 |
| VisDrone SET-HBS-only 移植 | 相同预算 identity 额外损失对照 | AP95 **21.4733% vs 22.9070%** | 负结果，不是完整 SET 复现 |
| 冻结检测器＋预测热度四裁剪 | 单次整图 | AP95 **22.6803% vs 23.0872%** | 模型调用增加且 AP 降低，当前原型 no-go |
| CUB 原图 / GT 保护背景淡出 / GT 保护背景模糊 | 从零 ResNet50、seed2005、第125轮 | Top1 **49.9655 / 53.3310 / 53.4518%** | 正向 **Oracle** 表示证据，测试需要真值框 |
| CUB 原图＋局部 / 空白＋局部，共享 ResNet50 | 相同双卡协议，第125轮 | Top1 **52.2782 / 64.6876%** | 本实现全局分支无益，但 BN 等混杂尚未排除 |
| 按视图分离 BN 统计量 G6 | 计划中的 G4 消融 | **用户要求于第50轮停止** | 未完成125轮，不能给出完整消融结论；未启动GPU6重跑 |

AP95 指 IoU=0.50:0.05:0.95 的 mAP，不是历史 VOC2007 的评测定义。各行数据集、checkpoint、检测头和评价器不同，**不是可以直接横比的排行榜**。分类百分比和 AP 点也不能混用。

![CUB 表示结果与历史耗时](figures/cub_A_F_endpoints_and_cost.png)

## 完整研究内容

1. [YOLO26：基线、双头监督、梯度、分配与因果筛选](docs/zh-CN/01_yolo26.md)
2. [数据集下载、质量审计与22组跨领域基线](docs/zh-CN/02_datasets.md)
3. [VOC：可微模糊、Oracle 对照与125轮学习式前端](docs/zh-CN/03_voc_blur.md)
4. [VisDrone：分辨率、SET-HBS、凹视、误报与目标保护](docs/zh-CN/04_visdrone.md)
5. [CUB：数据概况、A–F精确变换与完整结果](docs/zh-CN/05_cub.md)
6. [CUB全局/局部输入、共享BN、G4/G5与G6停止记录](docs/zh-CN/06_global_local_bn.md)
7. [工程、指标定义、复现条件与限制](docs/zh-CN/07_reproducibility.md)
8. [研究时间线、交付物和未实施项目](docs/zh-CN/08_progress.md)
9. [文献、移植边界与研究启示](docs/zh-CN/09_literature.md)

## 数据与代码

- [训练登记表](results/training_run_registry.csv)：全部采集到的服务器训练CSV，包含工程smoke和中止实验，本地快照不重复计为实验。
- [全部逐轮曲线](results/all_canonical_epoch_curves.csv)：保留原始loss、学习率、AP和准确率列，本次收录96份CSV、6,056行epoch记录。
- [22组末轮/best](results/dataset_22_run_endpoints.csv)、[CUB末轮/best/耗时](results/cub_primary_and_secondary_endpoints.csv)、[JSON指标账本](results/detection_metric_ledger.csv)、[逐类AP](results/detection_classwise_ledger.csv)、[TP/FP与覆盖率分母](results/diagnostic_counts_ledger.csv)。
- [代码索引](results/code_index.csv)、[最初38.5K参数可微PGB原型](code/original_pgb_core)、[分类训练/变换/审计](archives/remote/classification/src)、[模糊模型与测试](archives/remote/progressive_blur/pgb)、[早期YOLO26研究代码](code/historical/yolo26_research)。
- [374份历史研究Markdown记录](docs/zh-CN/archive_index.md)保留开发过程，不混入无关上游框架文档；早期计划、实时状态可能已被取代，当前章节明确引用最终审计和停止证据。
- [21组图表及来源，PNG+SVG](docs/zh-CN/figures.md)对应可追溯的CSV/JSON证据。[生成脚本](tools/build_report.py)只重建表格、图表，不启动训练。

不分发原始数据集、权重文件、私人凭证或第三方论文PDF。大型FCA逐对象表和冻结检测器预测记录以压缩数值证据保留，见[公开范围](PUBLICATION_SCOPE.md)与[授权说明](LICENSING.md)。

## 当前状态及成果边界

A–F与G4/G5分类实验已完成；G6在50轮按用户要求停止。本次整理没有新开训练。test/dev反复查看、分类单种子、运行环境变化及硬件故障历史限制结论强度。Oracle使用测试标注，不能当作无标注部署能力。当前最强正向观察是CUB框保护与局部裁剪信号，尚未建立可发表的新算法或可靠的计算节省模型。
