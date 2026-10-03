# 完整训练与指标附录

检测 AP 为 0–1；分类 Top1/Top5 为百分数。每张表明确区分数据划分、checkpoint 和评测口径，不能横向混用不同协议。best 仅为次要结果。

## CUB A–F、G4/G5及未完成G6

| 组别 | 实际轮数 | 训练Top1(%) | 训练Top5(%) | 训练loss | 测试Top1(%) | 测试Top5(%) | 测试loss | best轮 | best测试Top1(%) | 平均秒/轮 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 125 | 100.000000 | 100.000000 | 0.014418 | 49.965482 | 76.216776 | 2.062928 | 105 | 50.638592 | 49.860479 |
| B | 125 | 100.000000 | 100.000000 | 0.016150 | 29.668623 | 57.386952 | 3.210199 | 92 | 30.704177 | 66.303955 |
| C | 125 | 100.000000 | 100.000000 | 0.013626 | 53.331032 | 79.064550 | 1.900559 | 89 | 54.055920 | 60.333303 |
| D | 125 | 100.000000 | 100.000000 | 0.014064 | 53.451847 | 79.703141 | 1.887603 | 124 | 54.159475 | 142.671809 |
| E | 125 | 100.000000 | 100.000000 | 0.014786 | 27.096997 | 55.799103 | 3.247952 | 83 | 28.667587 | 49.841539 |
| F | 125 | 100.000000 | 100.000000 | 0.015649 | 17.742492 | 41.491198 | 4.088853 | 79 | 19.054194 | 104.072672 |
| G4 | 125 | 100.000000 | 100.000000 | 0.009257 | 52.278219 | 78.736624 | 1.960690 | 72 | 53.883328 | 131.998655 |
| G5 | 125 | 100.000000 | 100.000000 | 0.009684 | 64.687608 | 86.555057 | 1.456469 | 118 | 65.429755 | 69.258093 |
| G6 | 50 | 91.508175 | 99.449449 | 0.321017 | 43.182603 | 73.023818 | 3.051473 | 46 | 43.803935 | 210.417175 |

## 22组YOLOv8n从零跨数据集训练

| 数据版本 | 种子 | 末轮P | 末轮R | 末轮AP50 | 末轮AP95 | best轮 | best AP95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP-10K | 2005 | 0.815190 | 0.733790 | 0.814040 | 0.614160 | 118 | 0.615330 |
| AP-10K | 2027 | 0.803160 | 0.729430 | 0.814390 | 0.607940 | 122 | 0.608880 |
| DUO | 2005 | 0.808320 | 0.738720 | 0.816330 | 0.615080 | 123 | 0.615560 |
| DUO | 2027 | 0.839750 | 0.737530 | 0.818790 | 0.619640 | 125 | 0.619640 |
| ExDark | 2005 | 0.673470 | 0.541670 | 0.594130 | 0.356060 | 115 | 0.356740 |
| ExDark | 2027 | 0.682100 | 0.542390 | 0.589340 | 0.351640 | 115 | 0.354470 |
| HIT-UAV | 2005 | 0.902360 | 0.778610 | 0.851270 | 0.556130 | 110 | 0.566650 |
| HIT-UAV | 2027 | 0.879740 | 0.820310 | 0.863590 | 0.560220 | 122 | 0.564820 |
| HRSID-JPG | 2005 | 0.905800 | 0.824810 | 0.897700 | 0.635660 | 125 | 0.635660 |
| HRSID-JPG | 2027 | 0.907640 | 0.815430 | 0.886140 | 0.624750 | 119 | 0.627750 |
| LLVIP-infrared | 2005 | 0.951210 | 0.883640 | 0.947240 | 0.611150 | 52 | 0.623700 |
| LLVIP-infrared | 2027 | 0.944670 | 0.892530 | 0.944710 | 0.613560 | 47 | 0.620410 |
| LLVIP-visible | 2005 | 0.858950 | 0.778990 | 0.844390 | 0.455790 | 37 | 0.467640 |
| LLVIP-visible | 2027 | 0.855500 | 0.780850 | 0.852960 | 0.472830 | 114 | 0.474270 |
| ORCA-class-level-seen | 2005 | 0.455320 | 0.343570 | 0.351230 | 0.284530 | 121 | 0.285210 |
| ORCA-class-level-seen | 2027 | 0.402070 | 0.346940 | 0.352490 | 0.284900 | 124 | 0.285110 |
| SeaClear | 2005 | 0.855260 | 0.740310 | 0.815140 | 0.574050 | 123 | 0.574590 |
| SeaClear | 2027 | 0.882130 | 0.712180 | 0.795480 | 0.560020 | 124 | 0.562980 |
| VisDrone2019-DET | 2005 | 0.372590 | 0.292420 | 0.254670 | 0.137210 | 114 | 0.139410 |
| VisDrone2019-DET | 2027 | 0.366290 | 0.284570 | 0.250070 | 0.135050 | 115 | 0.136850 |
| ZeroWaste-f | 2005 | 0.402990 | 0.366520 | 0.308530 | 0.205920 | 93 | 0.224050 |
| ZeroWaste-f | 2027 | 0.380870 | 0.402610 | 0.324040 | 0.219750 | 92 | 0.238090 |

## YOLO26全部训练末轮记录

| 原始run/CSV | 完成轮数 | 评测头 | P | R | AP50 | AP95 |
| --- | --- | --- | --- | --- | --- | --- |
| asac_a0_instrumented_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.743560 | 0.649140 | 0.720610 | 0.513570 |
| asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1 | 50 | O2O | 0.744500 | 0.641070 | 0.719080 | 0.508470 |
| b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1 | 50 | O2O | 0.751730 | 0.647250 | 0.723890 | 0.513590 |
| b50_no_kd_e50_g4567_s0_w4_pin_numa1 | 50 | O2O | 0.751730 | 0.647250 | 0.723890 | 0.513590 |
| b50_no_kd_e50_g4567_s42 | 50 | O2O | 0.740530 | 0.651660 | 0.722160 | 0.512760 |
| d0_diag_yolo26n_voc_e150_g4567_s42 | 150 | O2O | 0.790950 | 0.697890 | 0.778510 | 0.576920 |
| d1_ungated_matched_kd_g05_e50_g4567_s42 | 50 | O2O | 0.736890 | 0.647910 | 0.718930 | 0.512110 |
| d2_gm_quality_gated_kd_g033_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.733490 | 0.654980 | 0.717740 | 0.512770 |
| d2_quality_gated_kd_g05_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.740010 | 0.647120 | 0.723210 | 0.515840 |
| sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.741230 | 0.646270 | 0.720650 | 0.511750 |
| tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.740530 | 0.651660 | 0.722160 | 0.512760 |
| tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1 | 50 | O2O | 0.730360 | 0.658090 | 0.719000 | 0.511440 |
| tqgb_t1_cls_m2o_proj_g008_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.746190 | 0.652310 | 0.728620 | 0.519460 |
| tqgb_t1_np_cls_m2o_g008_e50_g4567_s0_w4_pin_numa1 | 50 | O2O | 0.741260 | 0.655670 | 0.723120 | 0.513030 |
| tqgb_t1_np_cls_m2o_g008_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.746580 | 0.641810 | 0.723370 | 0.514740 |
| tqgb_t2_cls_bidir_proj_g008_g0015_e50_g4567_s42_w4_pin_numa1 | 50 | O2O | 0.742510 | 0.648530 | 0.724190 | 0.516850 |
| voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42 | 150 | O2M+NMS | 0.796290 | 0.718990 | 0.793390 | 0.587050 |

## 冻结模型凹视评测

| 评测图片数 | 处理条件 | P | R | AP50 | AP95 | 总预测数 |
| --- | --- | --- | --- | --- | --- | --- |
| 548 | full | 0.495904 | 0.403318 | 0.401083 | 0.230872 | 492024 |
| 548 | full_fixed4 | 0.460974 | 0.392365 | 0.376783 | 0.217609 | 529844 |
| 548 | full_adaptive4 | 0.463859 | 0.409765 | 0.392441 | 0.226803 | 541555 |
| 1610 | full | 0.450111 | 0.343734 | 0.322967 | 0.179863 | 1366078 |
| 1610 | full_adaptive4 | 0.429730 | 0.358506 | 0.317616 | 0.178484 | 1595348 |
| 1610 | full_adaptive4_small_only | 0.434143 | 0.360188 | 0.321098 | 0.180785 | 1593195 |
| 1610 | full | 0.450111 | 0.343734 | 0.322967 | 0.179863 | 1366078 |
| 1610 | full_raw4 | 0.429730 | 0.358506 | 0.317616 | 0.178484 | 1595348 |
| 1610 | full_predicted_background_blur4 | 0.429113 | 0.355263 | 0.314890 | 0.176827 | 1597902 |
| 548 | full | 0.495904 | 0.403318 | 0.401083 | 0.230872 | 492024 |
| 548 | full_raw4 | 0.463859 | 0.409765 | 0.392441 | 0.226803 | 541555 |
| 548 | full_predicted_background_blur4 | 0.459990 | 0.409256 | 0.390437 | 0.225561 | 542253 |
| 548 | full_gt_oracle_background_blur4 | 0.469904 | 0.410450 | 0.396973 | 0.228695 | 540254 |
| 548 | full | 0.495904 | 0.403318 | 0.401083 | 0.230872 | 492024 |
| 548 | full_raw4 | 0.463859 | 0.409765 | 0.392441 | 0.226803 | 541555 |
| 548 | full_blur4 | 0.453950 | 0.388068 | 0.370515 | 0.211691 | 529966 |

## 规范CSV登记表（含短测/未完成）

| 研究阶段 | 原始run/CSV | 完成轮数 | 记录类型 | 轮次连续 |
| --- | --- | --- | --- | --- |
| yolo26 | tqgb_smoke_cls_bidir_proj_e3_f002_w4_pin_numa1 | 3 | engineering_smoke_or_pilot | True |
| yolo26 | asac_a0_instrumented_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1 | 50 | training_run | True |
| yolo26 | b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | b50_no_kd_e50_g4567_s0_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | b50_no_kd_e50_g4567_s42 | 50 | training_run | True |
| yolo26 | d0_diag_yolo26n_voc_e150_g4567_s42 | 150 | training_run | True |
| yolo26 | d1_ungated_matched_kd_g05_e50_g4567_s42 | 50 | training_run | True |
| yolo26 | d2_gm_quality_gated_kd_g033_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | d2_quality_gated_kd_g05_e50_g4567_s42_w0_nopin_numa1 | 2 | engineering_smoke_or_pilot | True |
| yolo26 | d2_quality_gated_kd_g05_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | smoke_qgat_cls_g05_e1_g4567_s42 | 1 | engineering_smoke_or_pilot | True |
| yolo26 | smoke_sqgat_state_e6_f002_g4567_s314159_w4_pin_numa1 | 6 | engineering_smoke_or_pilot | True |
| yolo26 | smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v2 | 7 | engineering_smoke_or_pilot | True |
| yolo26 | smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v3_force_active | 7 | engineering_smoke_or_pilot | True |
| yolo26 | sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t1_cls_m2o_proj_g008_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t1_np_cls_m2o_g008_e50_g4567_s0_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t1_np_cls_m2o_g008_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | tqgb_t2_cls_bidir_proj_g008_g0015_e50_g4567_s42_w4_pin_numa1 | 50 | training_run | True |
| yolo26 | voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42 | 150 | training_run | True |
| yolo26 | voc_yolo26n_smoke_4x3090_b64_g4567_s42 | 1 | engineering_smoke_or_pilot | True |
| yolo26 | asac_a2_ddp_2e_g4567_s42_w4_pin_numa1 | 2 | engineering_smoke_or_pilot | True |
| dataset_baselines | yolov8n_scratch_AP-10K_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_AP-10K_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_DUO_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_DUO_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ExDark_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ExDark_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_HIT-UAV_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_HIT-UAV_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_SeaClear_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_SeaClear_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2027 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2005 | 125 | training_run | True |
| dataset_baselines | yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2027 | 125 | training_run | True |
| progressive_blur | tiny_pgb_2gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | oracle_blackout_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | oracle_gtblur_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | oracle_strong_p50_trainonly_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | oracle_strong_trainonly_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | baseline_voc_train90_e125_s2027_g0123_numa0 | 125 | training_run | True |
| progressive_blur | sbg_joint_voc_train90_e125_s2027_g0123_numa0 | 125 | training_run | True |
| progressive_blur | sbg_aug_4gpu_numa0_e1_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_baseline_matched_f10_e3_g12_s2027 | 3 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_frozen_feedback_f10_e2_g12_s2027 | 2 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_gate_full_e5_g123_s2027 | 5 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_gate_map_f10_e3_g12_s2027 | 3 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_joint_f10_e3_g12_s2027 | 3 | engineering_smoke_or_pilot | True |
| progressive_blur | joint_tiny_2gpu_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | joint_tiny_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | joint_tiny_v2_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | voc_map_only_f001_v4_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | hbs_only_s2027_g0123_numa0 | 125 | training_run | True |
| progressive_blur | identity_aux_s2027_g0123_numa0 | 125 | training_run | True |
| progressive_blur | hbs_only_1pct_e1_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | identity_aux_1pct_e1_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | resume_audit | 2 | engineering_smoke_or_pilot | True |
| progressive_blur | yolov8n_oracle_blur_p50_official_960_s2027_e125 | 125 | training_run | True |
| progressive_blur | yolov8n_oracle_blur_p50_official_960_smoke4_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | yolov8n_p2_scratch_official_640_s2027_e125 | 125 | training_run | True |
| progressive_blur | yolov8n_scratch_official_640_s2027_e125 | 125 | training_run | True |
| progressive_blur | yolov8n_scratch_official_960_s2027_e125 | 125 | training_run | True |
| progressive_blur | yolov8n_p2_scratch_official_640_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | yolov8n_scratch_official_640_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | yolov8n_scratch_official_640_view_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | yolov8n_scratch_official_960_smoke_4gpu | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | a0_yolov8n_scratch_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | d_pgb_scratch_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | oracle_blackout_scratch_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | oracle_gtblur_scratch_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | oracle_strong_p50_trainonly_s16_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | oracle_strong_trainonly_s16_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | oracle_weak_trainonly_s8_e20_b64_s2027 | 20 | training_run | True |
| progressive_blur | d_pgb_scratch_e1_f001_b64_s2027 | 1 | engineering_smoke_or_pilot | True |
| progressive_blur | d_pgb_scratch_e1_f001_b64_s2027_isolated | 1 | engineering_smoke_or_pilot | True |
| classification | resnet50_A_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_B_global_blur_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_C_context_removed_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_D_gradual_background_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_E_grayscale_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_F_scharr_native500_s2005_e200 | 125 | training_run | True |
| classification | resnet50_G4_global_crop_native500_s2005_e125_ddp23 | 125 | training_run | True |
| classification | resnet50_G5_blank_crop_native500_s2005_e125_ddp23 | 125 | training_run | True |
| classification | resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23 | 50 | user_stopped_incomplete | True |

## 全部服务器CSV末轮数值（含短测，非独立复评）

种子以run名称/原始args为依据；每个run名称链接到完整CSV。不能把不同预算、smoke或未完成组当作方法对照。

### yolo26

| 原始run/CSV | 记录种子 | 完成轮数 | 原args输入 | 原args batch | 评测头 | P | R | AP50 | AP95 | 训练box loss | 训练cls loss | 训练dfl loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tqgb_smoke_cls_bidir_proj_e3_f002_w4_pin_numa1](../../archives/remote/yolo26/runs/_smoke/tqgb_smoke_cls_bidir_proj_e3_f002_w4_pin_numa1/results.csv) |  | 3 | 640 | 64 | O2O | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |
| [asac_a0_instrumented_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/asac_a0_instrumented_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.743560 | 0.649140 | 0.720610 | 0.513570 |  |  |  |
| [asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1](../../archives/remote/yolo26/runs/asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.744500 | 0.641070 | 0.719080 | 0.508470 |  |  |  |
| [b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1](../../archives/remote/yolo26/runs/b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1/results.csv) | 0 | 50 | 640 | 64 | O2O | 0.751730 | 0.647250 | 0.723890 | 0.513590 |  |  |  |
| [b50_no_kd_e50_g4567_s0_w4_pin_numa1](../../archives/remote/yolo26/runs/b50_no_kd_e50_g4567_s0_w4_pin_numa1/results.csv) | 0 | 50 | 640 | 64 | O2O | 0.751730 | 0.647250 | 0.723890 | 0.513590 |  |  |  |
| [b50_no_kd_e50_g4567_s42](../../archives/remote/yolo26/runs/b50_no_kd_e50_g4567_s42/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.740530 | 0.651660 | 0.722160 | 0.512760 |  |  |  |
| [d0_diag_yolo26n_voc_e150_g4567_s42](../../archives/remote/yolo26/runs/d0_diag_yolo26n_voc_e150_g4567_s42/results.csv) | 42 | 150 | 640 | 64 | O2O | 0.790950 | 0.697890 | 0.778510 | 0.576920 |  |  |  |
| [d1_ungated_matched_kd_g05_e50_g4567_s42](../../archives/remote/yolo26/runs/d1_ungated_matched_kd_g05_e50_g4567_s42/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.736890 | 0.647910 | 0.718930 | 0.512110 |  |  |  |
| [d2_gm_quality_gated_kd_g033_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/d2_gm_quality_gated_kd_g033_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.733490 | 0.654980 | 0.717740 | 0.512770 |  |  |  |
| [d2_quality_gated_kd_g05_e50_g4567_s42_w0_nopin_numa1](../../archives/remote/yolo26/runs/d2_quality_gated_kd_g05_e50_g4567_s42_w0_nopin_numa1/results.csv) | 42 | 2 | 640 | 64 | O2O | 0.473770 | 0.043890 | 0.008640 | 0.002430 |  |  |  |
| [d2_quality_gated_kd_g05_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/d2_quality_gated_kd_g05_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.740010 | 0.647120 | 0.723210 | 0.515840 |  |  |  |
| [smoke_qgat_cls_g05_e1_g4567_s42](../../archives/remote/yolo26/runs/smoke_qgat_cls_g05_e1_g4567_s42/results.csv) | 42 | 1 | 640 | 64 | O2O | 0.635800 | 0.025220 | 0.012180 | 0.003800 |  |  |  |
| [smoke_sqgat_state_e6_f002_g4567_s314159_w4_pin_numa1](../../archives/remote/yolo26/runs/smoke_sqgat_state_e6_f002_g4567_s314159_w4_pin_numa1/results.csv) | 314159 | 6 | 640 | 64 | O2O | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |
| [smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v2](../../archives/remote/yolo26/runs/smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v2/results.csv) | 314159 | 7 | 640 | 64 | O2O | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |
| [smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v3_force_active](../../archives/remote/yolo26/runs/smoke_sqgat_state_e7_f002_g4567_s314159_w4_pin_numa1_v3_force_active/results.csv) | 314159 | 7 | 640 | 64 | O2O | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |
| [sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.741230 | 0.646270 | 0.720650 | 0.511750 |  |  |  |
| [tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.740530 | 0.651660 | 0.722160 | 0.512760 |  |  |  |
| [tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1/results.csv) | 0 | 50 | 640 | 64 | O2O | 0.730360 | 0.658090 | 0.719000 | 0.511440 |  |  |  |
| [tqgb_t1_cls_m2o_proj_g008_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t1_cls_m2o_proj_g008_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.746190 | 0.652310 | 0.728620 | 0.519460 |  |  |  |
| [tqgb_t1_np_cls_m2o_g008_e50_g4567_s0_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t1_np_cls_m2o_g008_e50_g4567_s0_w4_pin_numa1/results.csv) | 0 | 50 | 640 | 64 | O2O | 0.741260 | 0.655670 | 0.723120 | 0.513030 |  |  |  |
| [tqgb_t1_np_cls_m2o_g008_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t1_np_cls_m2o_g008_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.746580 | 0.641810 | 0.723370 | 0.514740 |  |  |  |
| [tqgb_t2_cls_bidir_proj_g008_g0015_e50_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/runs/tqgb_t2_cls_bidir_proj_g008_g0015_e50_g4567_s42_w4_pin_numa1/results.csv) | 42 | 50 | 640 | 64 | O2O | 0.742510 | 0.648530 | 0.724190 | 0.516850 |  |  |  |
| [voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42](../../archives/remote/yolo26/runs/voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42/results.csv) | 42 | 150 | 640 | 64 | O2M+NMS | 0.796290 | 0.718990 | 0.793390 | 0.587050 | 0.859210 | 0.640630 |  |
| [voc_yolo26n_smoke_4x3090_b64_g4567_s42](../../archives/remote/yolo26/runs/voc_yolo26n_smoke_4x3090_b64_g4567_s42/results.csv) | 42 | 1 | 640 | 64 | O2O | 0.649680 | 0.043810 | 0.022130 | 0.006350 | 2.798180 | 5.279220 |  |
| [asac_a2_ddp_2e_g4567_s42_w4_pin_numa1](../../archives/remote/yolo26/smoke/asac_a2_ddp_2e_g4567_s42_w4_pin_numa1/results.csv) | 42 | 2 | 640 | 64 | O2O | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |

### dataset_baselines

| 原始run/CSV | 记录种子 | 完成轮数 | 原args输入 | 原args batch | 评测头 | P | R | AP50 | AP95 | 训练box loss | 训练cls loss | 训练dfl loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [yolov8n_scratch_AP-10K_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_AP-10K_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.815190 | 0.733790 | 0.814040 | 0.614160 | 0.639590 | 0.543820 | 1.161500 |
| [yolov8n_scratch_AP-10K_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_AP-10K_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.803160 | 0.729430 | 0.814390 | 0.607940 | 0.638020 | 0.547510 | 1.151160 |
| [yolov8n_scratch_DUO_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_DUO_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.808320 | 0.738720 | 0.816330 | 0.615080 | 0.738030 | 0.474770 | 0.915870 |
| [yolov8n_scratch_DUO_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_DUO_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.839750 | 0.737530 | 0.818790 | 0.619640 | 0.750310 | 0.489000 | 0.926340 |
| [yolov8n_scratch_ExDark_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ExDark_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.673470 | 0.541670 | 0.594130 | 0.356060 | 1.154740 | 0.938730 | 1.361810 |
| [yolov8n_scratch_ExDark_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ExDark_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.682100 | 0.542390 | 0.589340 | 0.351640 | 1.172220 | 0.982470 | 1.375080 |
| [yolov8n_scratch_HIT-UAV_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_HIT-UAV_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.902360 | 0.778610 | 0.851270 | 0.556130 | 1.182020 | 0.578200 | 0.976390 |
| [yolov8n_scratch_HIT-UAV_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_HIT-UAV_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.879740 | 0.820310 | 0.863590 | 0.560220 | 1.189990 | 0.581330 | 0.971040 |
| [yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.905800 | 0.824810 | 0.897700 | 0.635660 | 0.910760 | 0.471140 | 0.905110 |
| [yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_HRSID-JPG_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.907640 | 0.815430 | 0.886140 | 0.624750 | 0.903720 | 0.468570 | 0.915660 |
| [yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.951210 | 0.883640 | 0.947240 | 0.611150 | 1.055030 | 0.425250 | 1.334290 |
| [yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_LLVIP-infrared_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.944670 | 0.892530 | 0.944710 | 0.613560 | 1.048190 | 0.425170 | 1.314710 |
| [yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.858950 | 0.778990 | 0.844390 | 0.455790 | 1.299870 | 0.619100 | 1.517970 |
| [yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_LLVIP-visible_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.855500 | 0.780850 | 0.852960 | 0.472830 | 1.299080 | 0.617250 | 1.506240 |
| [yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.455320 | 0.343570 | 0.351230 | 0.284530 | 0.766320 | 0.986930 | 1.182370 |
| [yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ORCA-class-level-seen_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.402070 | 0.346940 | 0.352490 | 0.284900 | 0.756150 | 1.001360 | 1.183710 |
| [yolov8n_scratch_SeaClear_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_SeaClear_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.855260 | 0.740310 | 0.815140 | 0.574050 | 0.911990 | 0.600750 | 1.028960 |
| [yolov8n_scratch_SeaClear_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_SeaClear_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.882130 | 0.712180 | 0.795480 | 0.560020 | 0.890280 | 0.576220 | 1.011930 |
| [yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.372590 | 0.292420 | 0.254670 | 0.137210 | 1.439840 | 0.968520 | 0.907450 |
| [yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_VisDrone2019-DET_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.366290 | 0.284570 | 0.250070 | 0.135050 | 1.448020 | 0.986210 | 0.913940 |
| [yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2005](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2005/results.csv) | 2005 | 125 | 640 | 64 |  | 0.402990 | 0.366520 | 0.308530 | 0.205920 | 0.830270 | 0.853900 | 1.182320 |
| [yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2027](../../archives/remote/dataset_baselines/runs/yolov8n_scratch_baselines/yolov8n_scratch_ZeroWaste-f_e125_b64_i640_s2027/results.csv) | 2027 | 125 | 640 | 64 |  | 0.380870 | 0.402610 | 0.324040 | 0.219750 | 0.843270 | 0.871820 | 1.220930 |

### progressive_blur

| 原始run/CSV | 记录种子 | 完成轮数 | 原args输入 | 原args batch | 评测头 | P | R | AP50 | AP95 | 训练box loss | 训练cls loss | 训练dfl loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tiny_pgb_2gpu](../../archives/remote/progressive_blur/runs/ddp_audit/tiny_pgb_2gpu/results.csv) |  | 1 | 128 | 4 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.854130 | 4.422680 | 4.679640 |
| [oracle_blackout_smoke_4gpu](../../archives/remote/progressive_blur/runs/oracle_smoke/oracle_blackout_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.940900 | 5.132170 | 4.283220 |
| [oracle_gtblur_smoke_4gpu](../../archives/remote/progressive_blur/runs/oracle_smoke/oracle_gtblur_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.959590 | 5.131220 | 4.254240 |
| [oracle_strong_p50_trainonly_smoke_4gpu](../../archives/remote/progressive_blur/runs/oracle_smoke/oracle_strong_p50_trainonly_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.975460 | 5.136370 | 4.227250 |
| [oracle_strong_trainonly_smoke_4gpu](../../archives/remote/progressive_blur/runs/oracle_smoke/oracle_strong_trainonly_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.973650 | 5.115220 | 4.256270 |
| [baseline_voc_train90_e125_s2027_g0123_numa0](../../archives/remote/progressive_blur/runs/sbg_long_e125/baseline_voc_train90_e125_s2027_g0123_numa0/results.csv) | 2027 | 125 | 640 | 64 |  | 0.747020 | 0.634890 | 0.702500 | 0.500620 | 0.937480 | 0.942950 | 1.313750 |
| [sbg_joint_voc_train90_e125_s2027_g0123_numa0](../../archives/remote/progressive_blur/runs/sbg_long_e125/sbg_joint_voc_train90_e125_s2027_g0123_numa0/results.csv) | 2027 | 125 | 640 | 64 |  | 0.762620 | 0.616680 | 0.696880 | 0.495290 | 0.935980 | 0.945100 | 1.311090 |
| [sbg_aug_4gpu_numa0_e1_s2027](../../archives/remote/progressive_blur/runs/sbg_long_smoke/sbg_aug_4gpu_numa0_e1_s2027/results.csv) | 2027 | 1 | 640 | 64 |  | 0.057780 | 0.146740 | 0.025210 | 0.010120 | 2.154860 | 3.251530 | 2.671460 |
| [voc_baseline_matched_f10_e3_g12_s2027](../../archives/remote/progressive_blur/runs/sbg_pilot/voc_baseline_matched_f10_e3_g12_s2027/results.csv) | 2027 | 3 | 640 | 32 |  | 0.382990 | 0.282610 | 0.256340 | 0.137980 | 0.715990 | 0.798410 | 1.234410 |
| [voc_frozen_feedback_f10_e2_g12_s2027](../../archives/remote/progressive_blur/runs/sbg_pilot/voc_frozen_feedback_f10_e2_g12_s2027/results.csv) | 2027 | 2 | 640 | 32 |  | 0.389010 | 0.283940 | 0.264500 | 0.140330 | 0.736130 | 0.837290 | 1.270540 |
| [voc_gate_full_e5_g123_s2027](../../archives/remote/progressive_blur/runs/sbg_pilot/voc_gate_full_e5_g123_s2027/results.csv) | 2027 | 5 | 640 | 48 |  | 0.377320 | 0.290270 | 0.263680 | 0.140440 | 0.000000 | 0.000000 | 0.000000 |
| [voc_gate_map_f10_e3_g12_s2027](../../archives/remote/progressive_blur/runs/sbg_pilot/voc_gate_map_f10_e3_g12_s2027/results.csv) | 2027 | 3 | 640 | 32 |  | 0.377320 | 0.290270 | 0.263680 | 0.140440 | 0.000000 | 0.000000 | 0.000000 |
| [voc_joint_f10_e3_g12_s2027](../../archives/remote/progressive_blur/runs/sbg_pilot/voc_joint_f10_e3_g12_s2027/results.csv) | 2027 | 3 | 640 | 32 |  | 0.344150 | 0.292520 | 0.253290 | 0.136190 | 0.713630 | 0.801020 | 1.235830 |
| [joint_tiny_2gpu_s2027](../../archives/remote/progressive_blur/runs/sbg_smoke/joint_tiny_2gpu_s2027/results.csv) | 2027 | 1 | 128 | 4 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.864830 | 4.425460 | 4.672360 |
| [joint_tiny_s2027](../../archives/remote/progressive_blur/runs/sbg_smoke/joint_tiny_s2027/results.csv) | 2027 | 1 | 128 | 2 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5.671840 | 3.793330 | 3.778490 |
| [joint_tiny_v2_s2027](../../archives/remote/progressive_blur/runs/sbg_smoke/joint_tiny_v2_s2027/results.csv) | 2027 | 1 | 128 | 2 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5.674830 | 3.795120 | 3.780630 |
| [voc_map_only_f001_v4_s2027](../../archives/remote/progressive_blur/runs/sbg_smoke/voc_map_only_f001_v4_s2027/results.csv) | 2027 | 1 | 640 | 8 |  | 0.376090 | 0.293370 | 0.264330 | 0.139880 | 0.000000 | 0.000000 | 0.000000 |
| [hbs_only_s2027_g0123_numa0](../../archives/remote/progressive_blur/runs/set_hbs_visdrone_960_e125/hbs_only_s2027_g0123_numa0/results.csv) | 2027 | 125 | 960 | 64 |  | 0.506340 | 0.379300 | 0.380690 | 0.214740 | 2.562810 | 1.712400 | 1.879410 |
| [identity_aux_s2027_g0123_numa0](../../archives/remote/progressive_blur/runs/set_hbs_visdrone_960_e125/identity_aux_s2027_g0123_numa0/results.csv) | 2027 | 125 | 960 | 64 |  | 0.491580 | 0.403800 | 0.396320 | 0.229070 | 2.687110 | 1.842740 | 1.899120 |
| [hbs_only_1pct_e1_s2027](../../archives/remote/progressive_blur/runs/set_hbs_visdrone_960_smoke/hbs_only_1pct_e1_s2027/results.csv) | 2027 | 1 | 960 | 64 |  | 0.000000 | 0.000140 | 0.000000 | 0.000000 | 11.016300 | 10.847000 | 8.734770 |
| [identity_aux_1pct_e1_s2027](../../archives/remote/progressive_blur/runs/set_hbs_visdrone_960_smoke/identity_aux_1pct_e1_s2027/results.csv) | 2027 | 1 | 960 | 64 |  | 0.000000 | 0.000140 | 0.000000 | 0.000000 | 11.020700 | 10.841800 | 8.732430 |
| [resume_audit](../../archives/remote/progressive_blur/runs/tiny_trainer_audit/resume_audit/results.csv) |  | 2 | 128 | 4 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.776930 | 4.579010 | 4.450180 |
| [yolov8n_oracle_blur_p50_official_960_s2027_e125](../../archives/remote/progressive_blur/runs/visdrone_official/yolov8n_oracle_blur_p50_official_960_s2027_e125/results.csv) | 2027 | 125 | 960 | 64 |  | 0.505470 | 0.392020 | 0.395130 | 0.225420 | 1.341560 | 0.896030 | 0.954870 |
| [yolov8n_oracle_blur_p50_official_960_smoke4_s2027](../../archives/remote/progressive_blur/runs/visdrone_official/yolov8n_oracle_blur_p50_official_960_smoke4_s2027/results.csv) | 2027 | 1 | 960 | 64 |  | 0.000000 | 0.000140 | 0.000000 | 0.000000 | 5.461100 | 5.482020 | 4.629400 |
| [yolov8n_p2_scratch_official_640_s2027_e125](../../archives/remote/progressive_blur/runs/visdrone_official/yolov8n_p2_scratch_official_640_s2027_e125/results.csv) | 2027 | 125 | 640 | 64 |  | 0.453420 | 0.354240 | 0.348120 | 0.196890 | 1.471260 | 1.117250 | 1.036640 |
| [yolov8n_scratch_official_640_s2027_e125](../../archives/remote/progressive_blur/runs/visdrone_official/yolov8n_scratch_official_640_s2027_e125/results.csv) | 2027 | 125 | 640 | 64 |  | 0.416320 | 0.323560 | 0.307910 | 0.169990 | 1.456830 | 0.991310 | 0.913430 |
| [yolov8n_scratch_official_960_s2027_e125](../../archives/remote/progressive_blur/runs/visdrone_official/yolov8n_scratch_official_960_s2027_e125/results.csv) | 2027 | 125 | 960 | 64 |  | 0.499820 | 0.399610 | 0.401990 | 0.231670 | 1.348980 | 0.925320 | 0.961300 |
| [yolov8n_p2_scratch_official_640_smoke_4gpu](../../archives/remote/progressive_blur/runs/visdrone_smoke/yolov8n_p2_scratch_official_640_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5.116290 | 5.645570 | 4.301380 |
| [yolov8n_scratch_official_640_smoke_4gpu](../../archives/remote/progressive_blur/runs/visdrone_smoke/yolov8n_scratch_official_640_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5.938420 | 5.796900 | 4.338020 |
| [yolov8n_scratch_official_640_view_smoke_4gpu](../../archives/remote/progressive_blur/runs/visdrone_smoke/yolov8n_scratch_official_640_view_smoke_4gpu/results.csv) |  | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5.938420 | 5.796900 | 4.338020 |
| [yolov8n_scratch_official_960_smoke_4gpu](../../archives/remote/progressive_blur/runs/visdrone_smoke/yolov8n_scratch_official_960_smoke_4gpu/results.csv) |  | 1 | 960 | 64 |  | 0.000000 | 0.000140 | 0.000000 | 0.000000 | 5.510340 | 5.420900 | 4.366210 |
| [a0_yolov8n_scratch_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/a0_yolov8n_scratch_e20_b64_s2027/results.csv) | 2027 | 20 | 640 | 64 |  | 0.377990 | 0.291100 | 0.263710 | 0.140230 | 0.792920 | 0.950510 | 1.299440 |
| [d_pgb_scratch_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/d_pgb_scratch_e20_b64_s2027/results.csv) | 2027 | 20 | 640 | 64 |  | 0.404530 | 0.259300 | 0.256060 | 0.137110 | 0.786200 | 0.952760 | 1.292490 |
| [oracle_blackout_scratch_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/oracle_blackout_scratch_e20_b64_s2027/results.csv) | 2027 | 20 | 640 | 64 |  | 0.516070 | 0.502020 | 0.492690 | 0.438160 | 0.357050 | 0.602930 | 0.900600 |
| [oracle_gtblur_scratch_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/oracle_gtblur_scratch_e20_b64_s2027/results.csv) | 2027 | 20 | 640 | 64 |  | 0.389020 | 0.318290 | 0.288400 | 0.153460 | 0.789600 | 0.935450 | 1.293660 |
| [oracle_strong_p50_trainonly_s16_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/oracle_strong_p50_trainonly_s16_e20_b64_s2027/results.csv) | 16 | 20 | 640 | 64 |  | 0.370420 | 0.291760 | 0.268590 | 0.142920 | 0.803750 | 0.967270 | 1.306200 |
| [oracle_strong_trainonly_s16_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/oracle_strong_trainonly_s16_e20_b64_s2027/results.csv) | 16 | 20 | 640 | 64 |  | 0.368830 | 0.293720 | 0.263250 | 0.141870 | 0.790030 | 0.934010 | 1.294200 |
| [oracle_weak_trainonly_s8_e20_b64_s2027](../../archives/remote/progressive_blur/runs/voc_p1/oracle_weak_trainonly_s8_e20_b64_s2027/results.csv) | 8 | 20 | 640 | 64 |  | 0.354600 | 0.309160 | 0.265940 | 0.144120 | 0.789600 | 0.935450 | 1.293660 |
| [d_pgb_scratch_e1_f001_b64_s2027](../../archives/remote/progressive_blur/runs/voc_smoke/d_pgb_scratch_e1_f001_b64_s2027/results.csv) | 2027 | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.006220 | 5.124480 | 4.241260 |
| [d_pgb_scratch_e1_f001_b64_s2027_isolated](../../archives/remote/progressive_blur/runs/voc_smoke/d_pgb_scratch_e1_f001_b64_s2027_isolated/results.csv) | 2027 | 1 | 640 | 64 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.006220 | 5.124480 | 4.241260 |

### classification

| 原始run/CSV | 记录种子 | 完成轮数 | 原args输入 | 原args batch | 训练Top1(%) | 训练Top5(%) | 训练loss | 测试Top1(%) | 测试Top5(%) | 测试loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [resnet50_A_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_A_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.014418 | 49.965482 | 76.216776 | 2.062928 |
| [resnet50_B_global_blur_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_B_global_blur_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.016150 | 29.668623 | 57.386952 | 3.210199 |
| [resnet50_C_context_removed_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_C_context_removed_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.013626 | 53.331032 | 79.064550 | 1.900559 |
| [resnet50_D_gradual_background_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_D_gradual_background_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.014064 | 53.451847 | 79.703141 | 1.887603 |
| [resnet50_E_grayscale_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_E_grayscale_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.014786 | 27.096997 | 55.799103 | 3.247952 |
| [resnet50_F_scharr_native500_s2005_e200](../../archives/remote/classification/runs/resnet50_F_scharr_native500_s2005_e200/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.015649 | 17.742492 | 41.491198 | 4.088853 |
| [resnet50_G4_global_crop_native500_s2005_e125_ddp23](../../archives/remote/classification/runs/resnet50_G4_global_crop_native500_s2005_e125_ddp23/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.009257 | 52.278219 | 78.736624 | 1.960690 |
| [resnet50_G5_blank_crop_native500_s2005_e125_ddp23](../../archives/remote/classification/runs/resnet50_G5_blank_crop_native500_s2005_e125_ddp23/results.csv) | 2005 | 125 |  |  | 100.000000 | 100.000000 | 0.009684 | 64.687608 | 86.555057 | 1.456469 |
| [resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23](../../archives/remote/classification/runs/resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23/results.csv) | 2005 | 50 |  |  | 91.508175 | 99.449449 | 0.321017 | 43.182603 | 73.023818 | 3.051473 |

完整逐轮数值与列名保存在 [CSV](../../results/all_canonical_epoch_curves.csv)，JSON 中所有可识别 AP 结果和逐类结果分别保存在 [指标账本](../../results/detection_metric_ledger.csv)、[逐类账本](../../results/detection_classwise_ledger.csv)，TP/FP/覆盖率分母保存在 [诊断账本](../../results/diagnostic_counts_ledger.csv)。源文件与 SHA 见 [证据清单](../../results/evidence_manifest.csv)。
