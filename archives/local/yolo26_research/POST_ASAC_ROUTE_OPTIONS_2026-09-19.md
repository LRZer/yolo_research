# ASAC NO-GO 后的候选研究路线

日期：2026-09-19  
状态：供讨论决定；未启动新训练、未下载新数据

## 总体判断

TQGB、强度匹配后的质量门控 KD、PBGA、clean-VOC FCA 与当前 ASAC 均已提供明确的停止证据。继续在这些方向上扫权重、阈值或 seed，不是可靠研究策略。下一轮应只做能在数小时内证伪问题假设的 Phase-0，不直接投入长训。

## 排名 1：CSPC——可折叠跨尺度预测编码

定位：通用检测；当前成本/新颖性比最好。

假设：FPN 中 nearest-upsampled coarse feature 与 lateral feature 存在大量可预测冗余，真正有用的是 lateral 中无法由 coarse 预测的 boundary innovation。训练期把 lateral 写成 `l-A(u)`；若 `A` 是无 bias 的线性 1×1 predictor，可在部署前精确折回原 post-concat convolution，推理图、参数量和 FLOPs 与基线相同。

最小零训练 P0：冻结 B50，在 32 个 train batches 上拟合 `coarse→lateral` ridge predictor，使用 held-out 图像报告 GT interior/boundary/background 的 whitened R²、residual energy enrichment、canonical correlation 与 effective rank。

GO 必须同时满足：

- 至少一个 fusion 点的 GT-interior `R² >= 0.45`；
- residual boundary/interior energy ratio `>=1.5`。

不达线立即停止。通过后只做 10-epoch 三元组：C0 原 concat、C1 同可折叠参数化但无正确 prediction target、C2 完整 CSPC。C2 必须胜 C1，才能支持 predictive-coding 语义，而不是一般重参数化。

风险：跨尺度融合、频率感知融合和 residual calibration 已有大量工作；可防守边界只能是“可精确折回原卷积的跨尺度预测残差参数化”，正式实现前还需专项查重。

## 排名 2：RS-SPAR——遥感微小目标亚像素相位恢复

定位：领域主线；优先把遥感作为放大微小目标/下采样问题的压力场景，而不是泛化地“改一个遥感 YOLO”。建议 AI-TOD-v2 做 tiny-object 压力测试，DIOR-HBB 做外部验证。

最小零训练 P0：测 stride-2 前后 object-ring phase detail 的 held-out 可解码性，以及一像素平移误差与 miss/assignment quality 的关系，并匹配面积、边界距离和背景纹理控制。

GO 必须同时满足：

- 至少两个层存在 object-specific detail gap；
- object-ring held-out R² 相对匹配背景或低频控制至少低 `0.10`；
- shift-error 最高四分位目标的 miss rate 至少是最低四分位的 `1.5×`；
- 关系不能由目标面积、图像边界或背景纹理解释。

通过后才做 baseline、phase-shuffle placebo、正确 phase target 三元组；正确相位不胜 shuffle 就停止。该方向的新颖性风险高，只能主张 object-gated、phase-specific、训练期恢复、推理删除，不能主张首次频域增强或抗混叠。

## 排名 3：跨切片目标身份一致性

定位：遥感系统论文；问题真实，但工程量和公平评测风险最高。

先对同一原图构造两组错位重叠切片，零训练审计同一 GT 在完整可见与部分截断切片中的 recall、分数、定位方差、类别不一致和重复框，并按边缘距离与截断比例分组。在相同整图推理预算下与固定 overlap、edge filtering、普通 SAHI/NMS 对比。

只有简单 overlap/edge filtering 仍无法消除主要失败，才研究训练期 identity-aware consistency；否则不进入方法实现。可防守边界是“同一目标跨错位切片的训练期身份一致性，在固定推理预算下改善边界目标”，而不是 overlap 或 NMS 本身。

## 已停止或降级

- **停止**：TQGB、质量门控 KD/D2、PBGA、clean-VOC FCA、VOC tiny/STAL/P2、恢复 DFL、普通注意力/大核/IoU-loss/FPN 拼装、当前 ASAC。
- **降级储备**：IGB（与动态监督预算/instance normalization 近邻碰撞明显）；SARC（可能主要反映 positive multiplicity，结构近邻也拥挤）；遥感 FCA（必须在新领域重新预注册，不能事后挽救 clean-VOC NO-GO）。

## 建议

优先讨论并批准 **CSPC-P0**，因为它不需要下载数据或训练，成本最低，失败也能快速结束；若用户更希望明确转入遥感，则选择 **RS-SPAR P0**，先小规模准备 AI-TOD-v2/DIOR-HBB，仍不直接长训。一次只运行一个 P0，完成并裁决后再决定下一步。

