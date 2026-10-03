# 图表及数据来源

图内英文标签便于两版本共享。所有检测AP先换算为百分数；差值用百分点。没有合成实验结果、未保存的置信区间或额外训练。

## CUB A–F 测试准确率与训练/测试损失

![CUB A–F 测试准确率与训练/测试损失](../../figures/cub_A_F_learning_curves.png)

[SVG](../../figures/cub_A_F_learning_curves.svg) · [数据来源](../../results/cub_primary_and_secondary_endpoints.csv)

## CUB 固定125轮结果与历史耗时

![CUB 固定125轮结果与历史耗时](../../figures/cub_A_F_endpoints_and_cost.png)

[SVG](../../figures/cub_A_F_endpoints_and_cost.svg) · [数据来源](../../results/cub_primary_and_secondary_endpoints.csv)

## G4/G5及未完成G6逐轮轨迹

![G4/G5及未完成G6逐轮轨迹](../../figures/cub_global_local_and_incomplete_bn.png)

[SVG](../../figures/cub_global_local_and_incomplete_bn.svg) · [数据来源](../../results/cub_primary_and_secondary_endpoints.csv)

## CUB 全部200类 train/test 数量

![CUB 全部200类 train/test 数量](../../figures/cub_class_distribution.png)

[SVG](../../figures/cub_class_distribution.svg) · [数据来源](../../results/cub_all_200_class_counts.csv)

## 22组从零检测基线

![22组从零检测基线](../../figures/dataset_22_baseline_endpoints.png)

[SVG](../../figures/dataset_22_baseline_endpoints.svg) · [数据来源](../../results/dataset_22_run_endpoints.csv)

## YOLO26短训对照，种子42/0；横轴截断

![YOLO26短训对照，种子42/0；横轴截断](../../figures/yolo26_short_ablation_endpoints.png)

[SVG](../../figures/yolo26_short_ablation_endpoints.svg) · [数据来源](../../results/yolo26_run_endpoints.csv)

## YOLO26基线/零增益轨迹，评测头不同

![YOLO26基线/零增益轨迹，评测头不同](../../figures/yolo26_150_baseline_reproduction.png)

[SVG](../../figures/yolo26_150_baseline_reproduction.svg) · [数据来源](../../results/yolo26_run_endpoints.csv)

## ASAC A2/A0 AP及实际q

![ASAC A2/A0 AP及实际q](../../figures/asac_paired_training.png)

[SVG](../../figures/asac_paired_training.svg) · [数据来源](../../results/asac_controller_epoch_states.csv)

## VOC独立评测20类AP差值

![VOC独立评测20类AP差值](../../figures/voc125_classwise.png)

[SVG](../../figures/voc125_classwise.svg) · [数据来源](../../results/voc125_classwise.csv)

## HBS独立评测10类对照

![HBS独立评测10类对照](../../figures/visdrone_hbs_classwise.png)

[SVG](../../figures/visdrone_hbs_classwise.svg) · [数据来源](../../results/visdrone_hbs_classwise.csv)

## VOC长训AP与模糊辅助损失

![VOC长训AP与模糊辅助损失](../../figures/voc125_curves_and_map_loss.png)

[SVG](../../figures/voc125_curves_and_map_loss.svg) · [数据来源](../../results/all_canonical_epoch_curves.csv)

## 三档推理模糊，GT/预测保护

![三档推理模糊，GT/预测保护](../../figures/testtime_blur_GT_vs_prediction.png)

[SVG](../../figures/testtime_blur_GT_vs_prediction.svg) · [数据来源](../../results/detection_metric_ledger.csv)

## 凹视val/test-dev结果；纵轴截断

![凹视val/test-dev结果；纵轴截断](../../figures/foveation_val_and_testdev_endpoints.png)

[SVG](../../figures/foveation_val_and_testdev_endpoints.svg) · [数据来源](../../results/foveation_same_protocol_endpoints.csv)

## 固定阈值逐类真阳性/误报

![固定阈值逐类真阳性/误报](../../figures/foveation_TP_FP_by_class.png)

[SVG](../../figures/foveation_TP_FP_by_class.svg) · [数据来源](../../results/foveation_fixed_threshold_classwise_TP_FP.csv)

## CUB训练准确率及泛化差距

![CUB训练准确率及泛化差距](../../figures/cub_train_accuracy_and_generalization_gap.png)

[SVG](../../figures/cub_train_accuracy_and_generalization_gap.svg) · [数据来源](../../results/cub_primary_and_secondary_endpoints.csv)

## 矩形保护几何示意，非真实实验图

![矩形保护几何示意，非真实实验图](../../figures/cub_protection_geometry_illustration.png)

[SVG](../../figures/cub_protection_geometry_illustration.svg) · [数据来源](../../tools/build_extended_figures.py)

## 已实现的输入、模型及真值使用路径

![已实现的输入、模型及真值使用路径](../../figures/implemented_method_pipelines.png)

[SVG](../../figures/implemented_method_pipelines.svg) · [数据来源](../../docs/en/03_voc_blur.md)

## 目标保护与背景处理许可

![目标保护与背景处理许可](../../figures/voc_gate_protection_and_permission.png)

[SVG](../../figures/voc_gate_protection_and_permission.svg) · [数据来源](../../results/voc_gate_size_protection.csv)

## HBS AP曲线与按尺寸召回

![HBS AP曲线与按尺寸召回](../../figures/hbs_learning_and_size_recall.png)

[SVG](../../figures/hbs_learning_and_size_recall.svg) · [数据来源](../../results/hbs_size_recall.csv)

## 掩膜保护率/面积跨集变化，仅统计合格目标

![掩膜保护率/面积跨集变化，仅统计合格目标](../../figures/foveation_mask_val_test_transfer.png)

[SVG](../../figures/foveation_mask_val_test_transfer.svg) · [数据来源](../../results/foveation_mask_transfer_counts.csv)

## FCA全量覆盖审计及启动阈值

![FCA全量覆盖审计及启动阈值](../../figures/fca_coverage_gap_screening.png)

[SVG](../../figures/fca_coverage_gap_screening.svg) · [数据来源](../../results/fca_o2o_checkpoint_coverage_gap.csv)

