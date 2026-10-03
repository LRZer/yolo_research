# 7. 工程、评测与复现边界

[目录](../../README.zh-CN.md) · [English](../en/07_reproducibility.md)

## 仓库能够复现什么

包含研究源码/配置、原始数值表、诊断、model/data/config指纹、启动/停止/恢复、测试、严格审计及图表重建脚本；不分发原数据、模型权重或完整第三方框架。复现需要合法取得数据并匹配冻结源码。历史绝对路径用于来源核验，**不是可移植路径或让人直接在别人的服务器启动作业的指令**。

| 阶段 | 原运行证据 | 变化 |
|---|---|---|
|YOLO26双头|`/home/ubuntu/rz/yolo_study/.conda/yolo26`、冻结研究checkout|有研究改动，单看版本号不够，需源码SHA|
|VOC/VisDrone前端/HBS|隔离项目实现、配置和manager清单|自定义模块/训练器影响权重加载，loss reduction关键|
|CUB A–E|原记录`/home/ubuntu/miniconda3/envs/LZCyolo/bin/python`|后来路径消失，不能称A–E用的是当前环境|
|CUB F/G4/G5/G6|`/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python`|PyTorch2.6.0+cu118、torchvision0.21.0+cu118，F环境变化公开|
|本地原型/测试|本机conda PyTorch、CPU报告|CPU通过不代表真实GPU训练/AP已验证|

本次公开整理未修改共享环境。单卡累积变DDP/SyncBN，即使有效batch和seed相同，BN统计、增强顺序和浮点归约也可能改变。只在特定YOLO26控制上证明了精确复现，**不对跨卡数/跨环境普遍保证数值一致**。

## 历史源码完整性

[362份Python索引](../../results/code_index.csv)不仅包括最新分类器，也保留了[数据集工程](../../code/dataset_engineering)中的下载、转换、完整性审计、断点传输和22组队列，以及[原YOLO26基线启动器/配置](../../code/historical/baseline_launcher)。各阶段训练、分析、GPU短测和管理器按原项目目录保留。

[精选框架覆盖文件](../../code/framework_overlays)保留质量迁移、ASAC的loss/config/loader/trainer修改，以及SARC候选的卷积、模型解析和结构配置。这些是**不同历史源码状态，不是混合后即可安装的统一框架**，复现必须对应run协议及源码哈希。SARC源码存在不代表其正式GPU训练已经完成。无关框架文档/完整vendor副本不收录；[原AGPL全文](../../LICENSES/Ultralytics-AGPL-3.0.txt)随派生源码保留。

## 指标字典

检测AP50是IoU.5类平均AP，AP95平均IoU.50:.05:.95。Ultralytics/内部插值、匹配及候选NMS政策不能自动等同VOC2007十一点或VisDrone官方工具。P/R通常为评价器选择工作点，不是共同conf的计数。$H=2PR/(P+R)$仅为派生量，不是原生最大F1。

固定阈值TP/FP声明conf和IoU，按类别兼容一对一匹配；只有同一工作点才有FN=GT−TP、pred=TP+FP。AP输入conf.001总框不同于conf.25框。size recall不是AP_small；保护率/覆盖率必须明确eligible GT分母。

分类Top1/Top5为正确类进入前1/5预测的图片比例；交叉熵还衡量概率，不仅正确与否。train loss下降、test loss恶化可能表示错误更自信。当前CUB未保存逐类准确率、校准、混淆矩阵，不能从数据集类别数“重建”。

主结果始终标注数据划分、模型/头、seed、预算、初始化、checkpoint和评价器。**不能混比**O2M/O2O；VisDrone合并训练test-dev/官方val；训练CSV/独立评测；best/last；G6第50轮/G4G5第125轮完整消融。

## 真实采用的验收标准

1. 启动前配置、训练/数据/stats/源码SHA及初始参数检查。
2. 核查真实进程/GPU，状态文件或停滞显卡画面不能证明在跑/空闲；早期识别了显示停滞，改用direct NVML。
3. 要求目标轮数恰好连续且有限，样本/update/LR正确、匹配checkpoint可读。
4. 关键长训检测配对独立同口径复评，保留训练验证器差异，不覆盖原数值。
5. 辅助模块进入优化器、EMA、保存恢复和predict；学习式前端与检测器之间不转numpy/uint8、不detach。
6. 检查epoch/hash/硬件后才续训，保留无效启动，不从损坏/空权重强行恢复。
7. 停止/清理只针对严格确认的本任务进程，不触碰其他人的作业。

例：T0/B50和seed0重跑精确相同；SARC融合保留SiLU外gamma；ASAC A0等价失败后如实改称插桩对照；模糊aux发生3倍广播偏差；CUB D从完整15轮恢复；G4/G5精确DDP尾组及AMP门禁；G6完整50后按用户停止。

## 硬件与时间限制

八RTX3090服务器存在不可纠正内存/MCE/SIGBUS历史、NUMA关联故障和整机重启。重启后EDAC计数归零**不证明修复**。保留早期NUMA1、后续NUMA0及GPU变化；部分窗口无新明确错误，另有未解码BERT或MC1新增事件。应保留探索性硬件限制，不编造损坏结果，也不只删不利实验。

预处理、batch、共卡、NUMA和竞争负载不同。完整test秒数、forward-only毫秒、预解码完整predict和粗略FLOPs是不同口径。尺寸不变的高斯/多视图不会天然减少CNN计算，5次960裁剪不能因为源区域小就称效率模型。

## 公开证据与检查

[清单](../../results/evidence_manifest.csv)记录source SHA、published SHA、大小及归一化/脱敏/压缩；重复副本保留来源，[canonical登记](../../results/training_run_registry.csv)不重复算实验。gzip原内容SHA列在source_sha256。公开验收脚本检查文件hash、派生表、完整/未完整CUB轮数、链接与凭证模式。

仅重建本地报告：

```bash
python -m pip install -r tools/requirements-report.txt
python tools/build_report.py
python tools/validate_publication.py
```

不训练、不续训。训练复现应读取对应协议，调整本地路径并匹配运行栈，而非盲抄服务器绝对路径launcher。

## 科学主张限制

- test/dev反复看过；CUB长度观察test后修订，非干净盲测。
- 多数方法和全部CUB单种子；T1第二种子反向。双基线种子不等于广泛稳健性。
- GT Oracle提供普通推理没有的位置，还可能通过框边界给图像加入线索。
- E/F/G5等为后追加诊断；没跑的预设控制在进展章节明确列出。
- 架构、归一化、计算和运行环境混杂跨系列比较。
- 选best、好看的注意图、小aux loss均不能替代合适对照。
- 记录中没有已证实SOTA新算法、完成部署流水线或录用论文；交付的是实验基准、代码、审计、机制分析及受限结论。
