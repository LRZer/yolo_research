# 非通用目标检测数据集筛选

日期：2026-09-20  
状态：只读调研；尚未下载数据、尚未启动训练  
目标：为 YOLO26n 选择规模适中、有公信力、具高频领域失效且能支撑论文验证的数据集

## 1. 选择原则

领域数据并不自动意味着容易发表。主要筛选标准为：

1. 公开且来源可信，尽量有固定划分、评测服务器或同行评议的数据论文；
2. 数百 MB 至约 10 GB，优先不超过约 2 万张图；
3. 不是仅靠随机切分就能接近饱和的小数据集；
4. 存在高频、可分层、可证伪的领域失效；
5. 至少能找到一个同领域外部数据集，避免只在单一 benchmark 刷分；
6. 主方法仍以机制问题为中心，而不是“加注意力/换卷积”。

## 2. 第一梯队

### 2.1 SeaDronesSee Object Detection v2：海上无人机搜救（首推）

- 规模：14,227 图，8,930 train + 1,547 val + 3,750 test；train+val 约 67,390 个 HBB；
- 类别：swimmer、boat、jetski、life-saving appliance、buoy，另有 ignore；
- 任务难点：水中人员极小，浪花/反光造成高假阳性，尺度随高度和俯仰角显著变化，类别长尾；
- 公信力：WACV 2022 benchmark，官方评测服务器，官方页面提供高度和俯仰角等元数据；
- 许可：官方数据页标注 CC0；
- 资源：压缩版 train+val 工作副本约 7.4 GB；原始全分辨率版更大，第一阶段不使用；
- 主指标：COCO AP@[.5:.95]、AP50/AP75、AR，并按高度、视角和小目标分层；
- 外部验证：TinyPerson 或 VisDrone 的 tiny-person 子问题。

可研究问题：高度/视角条件下，固定像素尺度的标签分配与回归目标是否系统性失配。任何使用元数据的方法都必须加入元数据随机打乱、等正样本数、等分辨率与等推理预算控制，避免先验泄漏。

来源：

- 官方数据页：<https://seadronessee.cs.uni-tuebingen.de/dataset>
- 论文：<https://openaccess.thecvf.com/content/WACV2022/papers/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.pdf>
- 官方仓库：<https://github.com/Ben93kie/SeaDronesSee>

### 2.2 DUO：水下机器人目标检测（最快进入纯 HBB 机制研究）

- 规模：7,782 图，官方固定 6,671 train / 1,111 test，74,515 个 COCO bbox；
- 类别：海参、海胆、扇贝、海星，共 4 类；
- 难点：目标很小、平均约 9.57 个目标/图，浑浊、偏色、模糊、遮挡明显；类别极不均衡，海胆约占 67.3%，扇贝约 2.6%；
- 优点：作者整合、去重并重新标注多个 URPC/UDD 来源，解决旧水下数据缺少公开 test 标签与近重复帧的问题；
- 风险：官方仓库未发现独立的数据许可证文件；公开学术使用广泛，但发表和再分发前应向作者确认条款；
- 外部验证：RUOD-R（14,000 图、10 类、高质量重标注）或 UODD。

可研究问题：低质成像下的小目标分配、长尾类别梯度预算、定位不确定性，或水下增强是否真的改善检测而非只改善视觉质量。不能再做普通注意力/特征融合堆叠。

来源：

- 论文：<https://arxiv.org/abs/2106.05681>
- 官方仓库与下载：<https://github.com/chongweiliu/DUO>
- RUOD-R：<https://github.com/RSSL-MTU/RUOD-R>

### 2.3 GWHD 2021：全球小麦穗检测（实验协议最规范）

- 规模：官方 2021 release 为 6,515 张 1024×1024 RGB 图、275,187 个 wheat-head bbox；官方网页概述为 6,000+ 图、300k+ heads；
- 类别：1 类；
- 来源：12 个国家、16 个机构、44 个采集 session；官方完整 PNG 包约 10.2 GB；
- 难点：密集重叠、风致模糊、成熟度/品种/朝向变化，以及跨国家、设备和平台的域偏移；
- 优点：官网明确允许自由使用，具有官方 Challenge split 与 GlobalWheat-WILDS OOD split；
- 风险：如果使用随机图像切分，问题会显得过于容易；必须按国家、机构或 acquisition session 做 group/OOD protocol；
- 外部验证：不同国家/session 的 held-out domain，或 GWHD 2020→2021 跨版本验证。

可研究问题：跨采集域的密集目标泛化、计数一致性、域不变的尺度/密度建模。论文主读数应是 unseen-domain AP/AR，而不是随机切分 AP。

来源：

- 官方网站：<https://www.global-wheat.com/gwhd.html>
- 2021 release：<https://zenodo.org/records/5092309>
- 论文：<https://doi.org/10.34133/2021/9846158>

### 2.4 HRSID：高分辨率 SAR 舰船检测（最小算力的遥感路线）

- 规模：5,604 张 800×800 SAR 图，16,951 艘船；官方 train/test 为 3,642/1,962；
- 类别：1 类 ship；同时带实例轮廓标注；
- 难点：SAR speckle、旁瓣、海杂波、近岸建筑/船坞误检，分辨率、极化、海况和近岸/离岸域变化；
- 优点：体量小、test 标签公开，并提供 400 张纯背景图和 inshore/offshore 属性，可专门评估 false positives；
- 风险：单类且 SAR ship detection 已有大量专用网络；普通多尺度/注意力方法新颖性不足；
- 外部验证：SSDD，1,160 图、约 2.5k ship instances、官方 inshore/offshore test 子集。

可研究问题：近岸 clutter-aware false-positive suppression、跨传感器/分辨率鲁棒性，或不增加推理开销的 SAR-specific training regularization。

来源：

- HRSID 官方仓库：<https://github.com/chaozhong2010/HRSID>
- SSDD 官方仓库：<https://github.com/TianwenZhang0825/Official-SSDD>

## 3. 第二梯队

| 数据集 | 规模与任务 | 价值 | 暂不首选原因 |
|---|---|---|---|
| VisDrone2019-DET | 8,629 个常用带 GT 图、10 类、约 2 GB、HBB | YOLO 自动下载转换；小目标/密集/遮挡明显 | 工程最省事，但 YOLO 改进论文极度拥挤 |
| TinyPerson | 1,610 标注图、72,651 tiny-person boxes | 极小人员、海边搜救、专用尺度评测 | 数据偏小，许可与视频相关性需核验，适合作外部压力集 |
| DOTA-v1.0/1.5 | 2,806 超高分辨率图、15/16 类、OBB、约 2.7 GB raw | 遥感 OBB 公认 benchmark | 切片后规模膨胀、需切换 YOLO26-OBB，标准赛道拥挤 |
| RUOD-R | 14,000 图、10 类、原 RUOD 的高质量重标注 | 水下退化、标注噪声、跨质量研究 | 图像获取与许可链更复杂，适合作 DUO 外部验证 |
| ExDark | 7,363 低照度图、12 类 | 低光退化明确，规模小 | 划分和评价协议不如第一梯队统一，增强+检测赛道拥挤 |
| D-Fire | 21,527 图、2 类、原生 YOLO labels | 大量 hard negatives，烟雾边界模糊 | 类别窄且随机切分容易高 AP，原始图片版权链仍需谨慎 |

VisDrone 的官方 Ultralytics 版本约 2 GB，6,471 train / 548 val / 1,610 test-dev，可作为最快工程基线，但不建议把“在 VisDrone 加模块提 AP”本身作为论文命题。

来源：

- VisDrone：<https://docs.ultralytics.com/datasets/detect/visdrone>
- TinyPerson：<https://openaccess.thecvf.com/content_WACV_2020/html/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.html>
- DOTA：<https://docs.ultralytics.com/datasets/obb/dota-v2>
- ExDark：<https://cs-chan.com/doc/cviu.pdf>
- D-Fire：<https://github.com/gaia-solutions-on-demand/DFireDataset>

## 4. 不建议作为唯一主数据集

- **NEU-DET**：1,800 张 200×200 灰度图、6 类×300；太小、无统一官方 split、被大量 YOLO 模块论文反复使用；
- **DeepPCB**：1,500 对 template/test，部分缺陷人工植入；如果不用 template pairing，就浪费数据集最重要的结构；
- **GC10-DET**：常见下载版数量、实例数和原论文描述不一致，缺少统一 split 与清晰数据许可；
- **原始 URPC**：跨年重复、test 标签缺失、各论文自切分；应直接使用 DUO；
- **SeaShips**：连续视频帧相关性很强，随机按帧划分容易泄漏，完整原始下载和许可不够稳定；
- **VinDr-CXR**：18,000 DICOM、22 个局部框类别，标注质量高，但需要 PhysioNet credential、CITI training 和 DUA，存储较大且需要医学合作解释；当前不适合作第一条路线；
- **AI-TOD-v2/DIOR**：任务有价值，但分别约 28k/23k 图；AI-TOD 还需下载 xView 原图并重建，第一阶段成本偏高；
- **RDD2022**：47,420 图，跨国道路损伤很有价值，但超过当前“小而可快速迭代”的预算。

## 5. 当前推荐顺序

1. **SeaDronesSee ODv2**：最佳综合选择。领域问题清楚、规模可控、元数据允许构造强因果控制、文献拥挤度低于 VisDrone/DOTA；
2. **DUO**：最快建立多类 HBB 水下基线，test 标签公开，适合小目标+长尾+图像退化机制，但先解决许可证确认；
3. **GWHD 2021**：最适合严谨的跨域泛化论文，数据协议最干净，但单类且约 10 GB；
4. **HRSID + SSDD**：算力最低、遥感特色强；适合明确选择 SAR 舰船课题时使用。

建议首先在 SeaDronesSee、DUO、GWHD 三者中讨论选一个。选定后先执行 Dataset-P0：只下载 annotation/metadata 或小样本，核验许可、完整性、split 泄漏、目标尺度/密度/长尾、基线预估成本；审计通过后才下载完整图像并训练 YOLO26n baseline。

