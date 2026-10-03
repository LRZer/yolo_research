# 8. 时间线、交付成果与未实施工作

[目录](../../README.zh-CN.md) · [English](../en/08_progress.md) · [全部历史文档](archive_index.md)

## 完整研究进展

| 日期（北京时间） | 实际工作 | 状态 |
|---|---|---|
|9月17–18|恢复成熟scratch YOLO26n/VOC150，核查O2M/O2O、尺度、源码，D0零gain复现|基线及数值复现完成|
|9月18–19|D1/D2、KD量级/方向/clipping，保留硬件失败启动；D2-GM、S-QGAT|D2小增益被量级匹配消除|
|9月19|TQGB T0/T1/T1-NP/T2、第二种子、同开机重复，F1回溯|双种子未确认，路线关闭|
|9月19|PBGA、尺度探针、SARC实现/真实源码CPU集成、IGB代码、FCA全量图审计|朴素接回与clean VOC coverage否决；SARC/IGB未正式训练|
|9月19–20|ASAC P0、golden/DDP门禁、无效中断、重试/live-state、A2/A0 50轮|有效两组完成，A2低0.510AP点|
|9月20–22|跨领域筛选、下载、传输、转换、完整性/类别/几何审计|10源包11视图；JODD未取得|
|9月22|22组YOLOv8n scratch125，seed2005/2027，权重/CSV独立审计|22/22完成，保留全部曲线|
|9月23|PGB接入、smoke、P1 scratch20原图/学习/Oracle，强预览、涂黑、仅训练弱/强/p50|学习式和raw泛化未改善，涂黑raw严重失败|
|9月23–24|VisDrone官方640/P2/960、训练Oracle125、预测/GT推理三档、YOLO26 O2O150 Oracle|GT特权信号存在，预测模糊降低|
|9月24凌晨|安全背景预热/小试、额外125配对、独立dev评测、保护/配对召回/成本|负结果，公开loss广播偏差|
|9月24上午|HBS-only/identity各125、严格验证、尺寸召回、成本|负结果，未完整SET|
|9月24|四重点论文、冻结固定/自适应4裁剪、修正multi-label、来源/几何/FP/面积审计|更多调用更低AP，纠正口径错误|
|9月24–25|small-only、test-dev预测背景模糊、一次固定val GT-mask诊断|微小不一致收益，裁剪路线no-go|
|9月24–25|新分类问题、CUB主包+mask下载/传输/profile、BCD预览、A–E顺序训练、200改实际125、D重启恢复|五组验收，C/D Oracle正向|
|9月30–10月1|Scharr预览确认、恢复环境/查资源、F共用GPU6 125轮|独立边缘表示负结果|
|10月1|局部GT放大尺寸预览、原图+局部/空白+局部、CPU/AMP/DDP/sampler，GPU2/3 G4/G5|125均验收，G5高G4 12.409Top1点|
|10月2|分离BN统计设计、CPU/归一化/DDP smoke、G6启动、G4初始化控制|50轮用户停止，非完整消融|
|10月2–3|确认任务停止、查GPU资源、考虑GPU6重跑|未实际启动GPU6|
|10月4|全历史提取、服务器数值归档、source/hash清单、双语表图整理|仅公开整理，无新训练|

部分事件时间重叠，[374份研究文档索引](archive_index.md)和原始记录保留精确时间。存在计划文档不代表计划实验已做。

## 实际交付成果

- 分头评测的成熟YOLO26及22组跨领域基线轨迹。
- 质量迁移变体/因果控制，分配、梯度、保护、误报诊断。
- 原始可微PGB与较大安全背景模块，训练期Oracle/涂黑/强度对照。
- HBS-only训练分支移植与identity-extra-loss对照，不冒称原创。
- 冻结凹视/融合实现、候选评价政策核查、逐类TP/FP、mask分母与迁移分析。
- CUB native500六表示基准、精确变换、mask审计、train/test全部曲线及严格恢复。
- 双视图共享主干分类、精确DDP采样/累积、共享/分离统计归一化代码与用户停止来源。
- 源码/配置/数据/权重指纹、单测/smoke、管理/恢复/审计工具及依据证据no-go记录。
- 双语完整现状、原历史记录、数据表、可追溯图和可重建报告工具。

这些是实现与研究产物。记录中**没有**已经完成论文、专利、录用workshop、SOTA提升或可部署无标注凹视分类器。

## 未实施或未完成

| 分支 | 未完成内容 | 边界 |
|---|---|---|
|TQGB|更多种子/150轮/COCO、BRC训练|双种子短训未通过|
|SARC|R1/R2/R3真实GPU训练/部署/导出|CPU等价不等于AP|
|IGB|真实CUDA VJP及quota训练|单测不证明机制|
|FCA|coverage-first matcher训练|预设效应门槛未达|
|ASAC|A1匹配日程、A3打乱反馈、其他种子长训|A2/A0未通过|
|初始PGB|identity/单尺度/无map/shuffle完整矩阵|计划不当已跑|
|YOLO26 GT模糊|O2M三档完整系列|只有O2O三档完成|
|分类|Oxford Pet、径向/random/半径、非均匀计算、学习注视、多glimpse、分割模型|现证据仅窄Oracle输入干预|
|全局/局部|真正单局部模型、多种子、BN样本数配平|G5仍含活动零分支|
|G6|剩75轮、125终审、GPU6重启|50用户停止，不算完整负结果|
|部署|无标注盲测、严格端到端成本、多模型族|实际/论文主张仍需验证|

## 查阅路径

当前章节给最终解释；[历史索引](archive_index.md)包含全部计划/报告；[训练登记](../../results/training_run_registry.csv)列CSV实际运行；[逐轮表](../../results/all_canonical_epoch_curves.csv)看曲线；[指标账本](../../results/detection_metric_ledger.csv)看JSON上下文；[代码索引](../../results/code_index.csv)查实现；[manifest](../../results/evidence_manifest.csv)查字节来源。历史材料保留原语言，两份新整理正文覆盖完整时间线和科学边界。
