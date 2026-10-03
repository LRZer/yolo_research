# VisDrone/YOLOv8n：SET-HBS 单模块移植与双分支等损失对照（预注册）

记录：2026-09-24 约 05:47 CST，正式训练前。前一轮 VOC 安全背景输入模糊 v0 在严格125轮配对后低于原图基线0.525 AP点，且推理约1.9倍耗时；因此不再追其门控阈值，改测试已发表的**训练期特征背景平滑**路线。

## 来源与研究边界

- SET（CVPR2025）官方 FCOS 实现的 HBS：训练时根据 GT 框做前景/背景特征掩码，每个检测特征尺度用两层卷积的残差 denoiser 修改背景特征，再在目标区域恢复原特征；对修改后的特征增加第二检测损失。其完整方法还包括 API（检测损失梯度扰动）。官方实现和结果基于 FCOS/AI-TOD 等，**本次仅是 HBS 子模块的 YOLOv8n 移植**，不是完整 SET 复现，更不能将模块原理称为本研究创新。[官方代码](https://github.com/HuixinSun/SET/blob/master/mmdet/models/detectors/fcos_set.py)、[仓库](https://github.com/HuixinSun/SET)。
- 要检验的具体问题：在高密度小目标 VisDrone2019-DET 上，把额外训练分支的 YOLO 特征背景平滑用于学习，是否比“相同第二分支但不平滑”带来独立增益？两组正常推理均只走原 YOLO 特征路径，不读取框、不调用 denoiser，因此设计上零额外推理运算，但须用实际测速验证。
- 不能直接把 SET 原论文 AP 值与本试验比较：检测器、数据集、输入分辨率、训练日程不同。已有本项目 VisDrone 960/125ep 原图基线只作外部上下文；**主因果对照是本轮同时训练的 identity extra-loss 组**。

## 固定试验设置

数据沿用已经验收的 VisDrone2019-DET official train/val 视图，不碰 test-challenge；图像与标签不修改。模型均 YOLOv8n、10类、随机初始化，不用 COCO 权重；seed2027。两个配置仅 `name`、`hbs_mode` 不同，其余一致：`imgsz=960`、125 epoch、batch64、物理 GPU0–3、NUMA0、workers4/rank、SGD、Ultralytics 默认日程、Mosaic末10轮关闭、每轮同一val548、AMP、相同NMS/增强。主指标为第125轮 `last.pt` 在同一独立验证设置下的 mAP50–95；还报告 mAP50、P/R、逐类 AP、AP_small/召回或明确不可得、训练时长和推理时延。`best.pt` 仅次要。

| 组 | 训练期损失 | 推理 |
|---|---|---|
| HBS-only | 原检测损失 + 权重1的 HBS 特征第二分支检测损失 | 原始 YOLO 路径 |
| Identity-extra-loss | 原检测损失 + 权重1的未修改特征第二分支检测损失 | 原始 YOLO 路径 |

两组第二分支使用同一个检测头和同样的两次损失计算，避免把“仅因额外监督而提升”误认 HBS 效果。HBS-only 的训练期 denoiser 额外参数不在正式推理路径调用；checkpoint 仍保存它。两组检测器从相同随机种子初始化，单测已验证所有原 YOLO detector 参数逐 tensor 相同。当前端到端单图输入和无标签推理同样逐 tensor 与原 YOLO 输出一致的 GPU smoke 已通过；伪标签训练 batch 的 HBS 梯度非零且有限。另须做实际 1%/1轮四卡 smoke，才能正式开跑。

仅占用任务专用项目 `runs/set_hbs_visdrone_960_e125/`；不会修改共享 Ultralytics、数据、其他进程或全局环境。启动前查物理 GPU0–3 空闲、NUMA0 可用内存、EDAC mc0/mc1及新 MCE/SIGBUS；GPU7他人进程不触碰。两组**顺序**跑，不同时占同四卡。若 1% smoke 报错先修工程，若正式每轮耗时推算无法在09:00前验收则报告预计，不能为截止强杀健康任务或声称完成。重启/硬件错误时按既有安全规则复核，不能机械续训。

## 判据与限制

两组 CSV 必须各连续125轮、权重可加载、配置和输入相同；独立验证统一同一 val548。探索性 go 条件：HBS-only 比 identity-extra-loss 至少高0.005 mAP50–95（0.5 AP点），且不显著损害小目标或训练稳定性；即使达标也必须补种子、正常原图同预算控制和其他数据集才有研究说服力。若未达标，则不在相同验证集上反复挑卷积核/权重，不将本移植称为新方法。

本轮只测试 **HBS-only**，没有 API、没有新的目标损失或自创适配点。论文潜力取决于之后能否定位可重复的特定失效并提出、验证有归因的改进；单纯移植已有方法即使提点，也不能直接写作原创论文贡献。
