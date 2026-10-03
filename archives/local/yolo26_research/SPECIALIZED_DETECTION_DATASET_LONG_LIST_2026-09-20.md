# 小领域目标检测数据集深筛：第二轮长名单

> 后续筛选与更正：见同目录 `DATASET_QUALITY_SHORTLIST_UNDER_20GB_2026-09-20.md`。本文件保留调研历史；JODD背景图并未被论文明确纳入视频分组保证，SortWaste框数官网表格存在加总冲突，IP102小包为其他研究者转换版，相关结论以新报告为准。

日期：2026-09-20  
状态：只读文献与数据页审计；尚未下载数据、尚未启动训练  
目标：寻找比 VOC/COCO 类通用检测更不拥挤，同时具备可信协议、真实失效和外部验证条件的 YOLO26n 研究场景

## 1. 先给结论

本轮共审计 40 余个候选，保留 30 余个可用数据集。结论不是“小领域一定更容易”，而是：

1. 小领域通常更少论文直接竞争，但也更常出现视频相邻帧泄漏、同一设备跨集合、测试集过小、缺少负样本、许可不清和 AP 已饱和等问题；
2. 真正值得选的是“领域失效可测量、划分可信、数据不大、至少有一个外部数据集”的小领域；
3. 研究题不能只是“在小数据集上给 YOLO26 加模块提 AP”，而应围绕一个可复现的失效机制，并报告分层指标和跨域验证；
4. 扩大检索后，当前最值得进入 Dataset-P0 的不是单一数据集，而是四种不同取向：

| 取向 | 首选数据集 | 为什么值得先看 |
|---|---|---|
| 最适合快速入门 | JODD | 343 MB、8,151 图、官方按源视频分组切分、2026 新数据、背景图和相机/潜器/潜次信息齐全 |
| 评测协议最干净 | HRP4K | 6,003 张 4K 图，官方按 170 个视频组切分，含 2,000 张负样本，CC BY 4.0 |
| 机制论文潜力最高 | RUOD-R | 同一批 14,000 图拥有旧/新两套标注，可直接研究缺标造成的错误负监督 |
| 最贴近当前 YOLO26 | XWOD | 2026 极端天气集，官方仓库已有 stock YOLO26n 表格基线和 leave-one-weather-out 协议 |

如果现在只选一个最稳妥的起点，本轮建议 **JODD**；如果愿意承担更高数据与许可审计成本来换取更强论文命题，则优先 **RUOD-R**。

## 2. 第一梯队：建议进入 Dataset-P0

### 2.1 JODD：深海生物检测，当前综合首推

- 规模：8,151 图、15,621 框、20 类；7,151 张目标图和 1,000 张纯背景图；
- 官方划分：4,864 / 847 / 2,440 train/val/test；
- 数据成本：343 MB，图像分辨率主要为 480×270、480×320 或 480×360；
- 协议质量：每个源视频最多取 2 帧且至少相隔 60 帧；先按 video ID 分组，再做分层 6:1:3 划分，同一视频不跨集合；
- 许可：CC BY-NC-SA 4.0，仅科研或教育用途；
- 领域失效：marine snow、字幕/水印、低照度、压缩、长尾、小目标以及不同潜器/相机/潜次域偏移；
- 研究入口：小目标×长尾质量校准、背景误报预算、跨 vehicle/camera/dive 训练期一致性；
- 推荐指标：COCO AP、AP_S/M/L、每类 AP、macro AP、每千张背景图误报、held-out vehicle/camera/dive AP；
- 外部验证：SeaClear、Brackish、UODD；TrashCan 与 JODD 都来自 J-EDI 上游，若使用必须做源视频重叠审计。

来源：

- 官方数据页：<https://www.godac.jamstec.go.jp/jedi/e/dataset/jedi_organism_detection_dataset.html>
- 数据论文：<https://doi.org/10.1016/j.dib.2026.112462>

### 2.2 HRP4K：坑洞检测，协议最干净

- 规模：6,003 张 4K 透视道路图、7,217 个坑洞框；4,003 张正样本、2,000 张负样本；
- 官方划分：4,203 / 900 / 900 train/val/test；
- 协议质量：正样本来自 170 段视频，正负样本都按视频级分组，避免相邻帧跨集合；
- 场景：约 1,100 km 城乡道路，覆盖三座城市和沥青/混凝土路面；
- 许可：论文和 Zenodo 公开，CC BY 4.0；
- 领域失效：超小坑洞、强透视尺度变化、路面纹理硬负样本、稀疏检测与固定误报预算；
- 研究入口：透视位置/尺度感知的训练期正样本预算，或训练期动态切片；推理网络保持不变；
- 风险：单类，且 4K 图需要预注册整图缩放或切片协议；只提总体 AP 容易缺少方法深度；
- 外部验证：RDD2022 的 pothole 子集或另一地区坑洞数据。

来源：

- Scientific Data：<https://doi.org/10.1038/s41597-026-07317-w>
- Zenodo：<https://doi.org/10.5281/zenodo.17522874>
- 官方代码：<https://github.com/hanshenChen/HRP4K>

### 2.3 RUOD-R：水下缺标因果研究，方法潜力最高

- 规模：与原 RUOD 使用相同的 14,000 张图、10 类；
- 天然对照：原 RUOD 为 74,903 个框，RUOD-R 为 263,263 个框，约增至 3.5 倍；小目标标注由 522 增至 73,601；
- 协议：train/test 文件名、图像 ID、分辨率均与原 RUOD 保持一致；
- 关键价值：同一批图只改变标注完整性，能够直接验证“缺标→把真目标当负样本→错误负梯度→方法是否缓解”的因果链；
- 研究入口：对增强一致且由 O2M/O2O 持续高置信预测、但未匹配的区域减弱负梯度或转为软忽略；推理不增加模块；
- 推荐实验：完整 RUOD-R、旧 RUOD、人工丢标 10%/30%/50% 三条曲线；报告 AP_S、背景误报和旧/新评测翻转率；
- 风险：图像和 annotation repository 的数据许可链尚未在仓库中明确；密集小框让训练和验证比 JODD 更重；不能混用 full 与 filtered 版。

来源：

- RUOD-R 官方仓库：<https://github.com/RSSL-MTU/RUOD-R>
- IEEE Access 论文：<https://doi.org/10.1109/ACCESS.2026.3685121>
- 原 RUOD 仓库：<https://github.com/xiaoDetection/RUOD>

### 2.4 XWOD：极端天气，最贴近 YOLO26

- 规模：10,010 图、42,924 框、6 个交通类别、7 种极端天气；
- 官方划分：6,206 / 1,744 / 2,060 train/val/test，并提供 7 个 leave-one-weather-out 划分；
- 许可：CC BY-NC 4.0；
- 官方仓库表格基线：stock YOLO26n 的 P/R/mAP50/mAP50-95 约为 57.92/45.27/46.29/26.73；
- 研究入口：训练期 weather-balanced dual-assignment curriculum，目标同时提高 overall、weather-macro、worst-weather 和 LOWO 泛化；
- 优点：2026 年新发布，研究尚不拥挤，且与当前 YOLO26n 代码链衔接最快；
- 风险：目前是 arXiv v1、仓库历史很短；图像来自不同网络/事件源，必须先做 pHash 和来源簇跨 split 审计；仓库说明表格数可能有约 ±1–2 AP50 的单跑波动。

来源：

- 官方仓库：<https://github.com/kuanting/xwod>
- 论文：<https://arxiv.org/abs/2605.11521>

### 2.5 SortWaste：工业垃圾分拣，密集与遮挡机制很强

- 规模：5,261 图、87,252 个 HBB、8 类；
- 官方划分：3,705 / 780 / 776 train/val/test；
- 场景：真实材料回收厂传送带，存在重叠、形变、污染、纹理相似和类别不均衡；
- 额外价值：作者提供 ClutterScore，可按目标数、类别/尺度熵和空间重叠对场景难度分层；
- 已知基线：plastic-only 最佳基线 mAP 约 59.7%，高 clutter 下显著下降；
- 研究入口：clutter-conditioned assignment、重复预测抑制或 O2M→O2O 训练期质量桥接；报告各 ClutterScore 分位点结果；
- 外部验证：ZeroWaste-f 与 PackWISE；
- 风险：数据来自工业视频，需核查相邻帧是否跨 split；数据归档许可与下载体积还需在 Dataset-P0 复核。

来源：

- 官方项目页：<https://sortwaste.di.ubi.pt/>
- 官方仓库：<https://github.com/sarainacio/SortWaste>
- WACV 2026 论文：<https://openaccess.thecvf.com/content/WACV2026W/WasteVision/html/Inacio_SortWaste_A_Densely_Annotated_Dataset_for_Object_Detection_in_Industrial_WACVW_2026_paper.html>

### 2.6 SeaClear：水下垃圾，跨相机/跨地点泛化

- 规模：8,610 张 1920×1080 ROV 图、40 个原始类别，提供 bbox 与 polygon；
- 域结构：5 个地点、11 个 site-camera 组合；
- 许可：CC BY 4.0；
- 已有证据：混合随机基线明显高于跨 camera/site 结果，说明主要问题不是随机 split 上再提一点 AP，而是部署域移；
- 研究入口：training-only camera/site consistency、域平衡标签分配和长尾质量校准；
- 评测：必须同时报告 in-domain 与 cross-camera/site；不允许只用随机帧划分；
- 风险：视频近帧泄漏仍需复核；下载后必须从 COCO 标注重新统计实例数。

来源：

- 数据：<https://doi.org/10.4121/4f1dff25-e157-4399-a5d4-478055461689.v1>
- Scientific Data：<https://doi.org/10.1038/s41597-024-03759-2>
- 代码：<https://github.com/adjuras/seaclear-dataset>

### 2.7 RAIL-BENCH Object：铁路场景，隐藏测试集最可信

- 规模：2,500 张实景 RGB 图，1,500 train / 500 val / 500 hidden test；
- 类别：train、signal、signal pole、catenary pole、road vehicle、bicycle、person，共 7 类；
- 标注：每框带遮挡等级，另有 ignore/crowd；
- 评测：官方 COCO 风格 easy/moderate/hard AP，CodaBench 隐藏测试；
- 优点：2026 年新基准，结果不容易被反复看 test 调参污染；
- 风险：图像为高分辨率，官方影像压缩包约 22 GB；训练图仅 1,500；许可为研究/非商用口径；
- 研究入口：遮挡等级质量目标、O2M→O2O 课程或 small/occluded object supervision；
- 外部验证：RailSem19 或 RAILOD。

来源：

- 官方页：<https://www.mrt.kit.edu/railbench/object_detection>
- 工具包：<https://github.com/KIT-MRT/RAIL-BENCH_toolkit>
- 论文：<https://arxiv.org/abs/2604.22507>

### 2.8 MBDD2025：建筑表面缺陷，最容易从普通 HBB 过渡

- 规模：14,471 张最高 1280×720 图，5 类：裂缝、渗漏、腐蚀、脱落、鼓包；
- 场景：6 类建筑材料/结构，多个采集区域；
- 官方划分：70/20/10，提供 VOC XML 与 YOLO；
- 许可与体积：CC BY 4.0，约 2.53 GB；
- 基线：YOLOv8n P/R/mAP50/mAP50-95 约为 .858/.808/.867/.486；严格定位仍有空间；
- 研究入口：细长/不规则缺陷的定位质量、材料条件化或材料不变训练、建筑级 OOD；
- 风险：官方随机图像切分可能让同建筑、同航线或同区域跨集合；需要重建 building/site group split；
- 外部验证：CUBIT-Det、COCO-Bridge。

来源：

- Scientific Data：<https://www.nature.com/articles/s41597-025-06318-5>
- Zenodo：<https://zenodo.org/records/15622584>

## 3. 强候选，但应先解决一个明显风险

| 数据集 | 规模/任务 | 最有价值的问题 | 主要风险 |
|---|---|---|---|
| RFDD | 1,350 张 2048×2021、8,100+ 扣件、6 类 | 同场景正常扣件作训练期缺陷原型；已有 YOLO26m 基准 | 约 1,000 个重构缺陷，test 仅 100；需审计重构图、重复和许可 |
| HIT-UAV | 2,898 热红外 UAV 图、24,899 对象、4 类+DontCare | 高度/视角/昼夜条件化的小目标正样本预算 | 来自视频帧；必须确认 flight/video 不跨 split，DontCare 应为 ignore |
| AEWD | 7,483 高分辨率 UAV 帧、27,194 框、4 种濒危动物 | 按植被覆盖和遮挡难度研究 worst-group AP | 无官方 split；必须按视频/航线/园区分组 |
| WTBD | 1,065 张、1,568 框、6 种风机叶片缺陷，约 75 MB | 细微缺陷的层级/顺序监督 | 全为有缺陷图，缺负样本；同风机和随机划分风险 |
| ORCA | 14,647 图、42,217 框、478 species/670 taxonomy categories | 细粒度、长尾、层级和开放词汇 | 对入门固定类 YOLO26n 过于复杂，更适合第二阶段 |
| D-Fire | 21,527 图、2 类，其中 9,838 张纯负图，CC0 | 固定误报预算下的困难负样本课程 | fire-YOLO 论文很多；网络/视频近重复和烟雾边界主观 |
| MD-AS-2025 | 4,000 显微图、5,821 框、5 类 | 透明、细长、弯曲目标的定位质量 | 随机帧拆分可能泄漏污水厂/视频域；数据许可字段需确认 |
| Big Bird | 4,824 标注原图、49,990 鸟、101 species | GSD/地点/分类层级条件下的跨域检测 | 细粒度标签复杂，许可不是普通 CC，成本中等 |
| DuckNet | 正式检测集 1,444 图、6,812 框、9 类 | 极相似水禽类别与高 IoU 定位 | test 仅 73 图；同一 survey 可跨 split，需重新分组 |
| iSOOD | 10,481 排污口图、10 类、约 9.5 GB | 两大流域域移与极小排污口 | 使用政策有额外联系/署名要求；随机 split 与中心偏置 |
| InsPLAD | 约 10,607 电力巡检图、约 28,933 框、17 类 | 长尾、小目标、视角与尺度条件化 | 已发现重复文件名/异常类别，必须先清洗 |
| DsPCBSD+ | 10,259 PCB 图、20,276 框、9 类，约 123 MB | 跨生产批次泛化 | 无独立 test；随机混合 326 批次会制造虚高 AP |

关键来源：

- RFDD：<https://www.nature.com/articles/s41597-026-07851-7>
- HIT-UAV：<https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset>
- AEWD：<https://github.com/maml1116/AEWD>
- WTBD：<https://www.nature.com/articles/s41597-026-06762-x>
- ORCA：<https://openaccess.thecvf.com/content/WACV2026/html/Wong_ORCA_Object_Recognition_and_Comprehension_for_Archiving_Marine_Species_WACV_2026_paper.html>
- D-Fire：<https://github.com/gaia-solutions-on-demand/DFireDataset>
- MD-AS-2025：<https://doi.org/10.1038/s41597-025-06228-6>
- Big Bird：<https://doi.org/10.1002/rse2.70059>
- DuckNet：<https://github.com/mkmitchell/DuckNet>
- iSOOD：<https://doi.org/10.1038/s41597-024-03574-9>
- InsPLAD：<https://andreluizbvs.github.io/InsPLAD/>
- DsPCBSD+：<https://www.nature.com/articles/s41597-024-03656-8>

## 4. 其他可用长名单

### 4.1 工业、基础设施与特殊交通

| 数据集 | 大致规模 | 适合角色 | 备注 |
|---|---:|---|---|
| ZeroWaste-f | 4,503 帧、4 类 | SortWaste 外部验证 | 约 7 GB、CC BY-NC 4.0；必须按视频序列防泄漏 |
| PackWISE | 586 张 4096²、23 类 | SortWaste 小型外测 | 355 MB、COCO masks+boxes；太小，不宜单独主训 |
| CUBIT-Det | 5,527 超高分辨率图、3 类缺陷 | MBDD 外部验证 | 原图很大，需统一 tiling 协议 |
| COCO-Bridge | 1,470 图、4 类桥梁细部 | MBDD 外部验证 | 约 648 MB，单独训练太小 |
| TTPLA | 1,100 张 4K 航拍图、8,987 实例 | 输电线路 pilot | 同时有检测/分割；适合与 TIC 外验组合 |
| Apron Dataset | 10,098 图、169k+ 框、43 细类 | 机场长尾/环境鲁棒 | 按序列划分；需申请、非商用 |
| ABOships/PLUS | 9,880 图、约 33k–42k 船框 | 港口船舶检测 | 连续视频采样，需按航线/时间组划分 |
| FishEye8K | 8,000 图、157k 框、5 类 | 鱼眼畸变/边缘目标 | 比普通交通小众，但不像缺陷/生态那样领域专一 |
| OPIXray | 8,885 X-ray 图、5 类 | 安检遮挡检测 | 学术使用；已有较多工作，拥挤度中高 |
| CLCXray | 9,565 图、12 类 | 安检长尾/遮挡 | 真实+模拟混合，应分开报告 |

### 4.2 生态、水下与动物

| 数据集 | 大致规模 | 适合角色 | 备注 |
|---|---:|---|---|
| SeaDronesSee ODv2 | 14,227 图、5 类 | 海上极小目标完整验证 | 公信力强但约 7.4 GB 压缩训练版；仍是一线备选 |
| DUO | 7,782 图、74,515 框、4 类 | 水下小目标+长尾 | 固定 train/test；许可链需再确认 |
| TrashCan-Instance | 7,212 图、22 instance 类 | 水下垃圾层级外验 | 与 JODD 共享 J-EDI 上游，必须查重 |
| Amphitrite v1.0 | 1,583 水下图、16 类 | 小型外部验证 | 无官方 split，CC BY-NC-SA |
| SealDetectionDataset | 495 tiles、7,085 seals | 极低成本 pilot | 只有正图；按日期+原始大图分组，否则严重泄漏 |
| Sherbrooke Red Deer | 4,071 超高分辨率图、9,337 鹿 | 季节域移外验 | 按 flight 划分较可靠；单类主 AP 空间有限 |
| WildLive | 19,139 4K 帧、215,800 框/掩码 | 后期 UAV 野生动物验证 | 22 段视频，需严格 sequence split |
| AP-10K | 10,015 图、13,028 动物、54 species | 跨物种外测 | 框+关键点，检测本身可能偏容易 |
| SA-FARI | 11,609 camera-trap videos | 大型后期验证 | 近百万 frame boxes/masks，当前快速研究太重 |

### 4.3 热红外、低光与恶劣环境

| 数据集 | 大致规模 | 适合角色 | 备注 |
|---|---:|---|---|
| ExDark | 7,363 图、12 类 | 低光快速集 | 固定 3,000/1,800/2,563；第三方转换版本常漂移 |
| LLVIP | 15,488 RGB/IR 对 | IR-only 部署、RGB 训练期教师 | pair/scene 必须同组；只从 train 划 val |
| DAWN | 1,000 图、4 种天气 | XWOD 外部测试 | 无官方 split，最好 external-only |
| RTTS | 4,322 真实雾图 | XWOD 外部测试 | 本来就是测试集，不应用于训练 |
| WEDGE | 3,360 合成天气图 | 合成域偏移外验 | 不能替代真实世界外验 |
| AU-AIR | 32k+ UAV 视频帧、8 类 | 高度/IMU 条件化后期验证 | 仅 8 段视频，逐帧随机切分风险极高 |
| FLIR ADAS Expanded | 26,442 帧、约 520k 框 | 热红外后期验证 | 版本、条款和固定 test 不够清楚 |
| IFSOD | 4,815 红外图、23,333 框、18 类 | 红外 few-shot | 三组 base/novel split；协议比普通检测复杂 |

### 4.4 农业、显微与科学影像

| 数据集 | 大致规模 | 适合角色 | 备注 |
|---|---:|---|---|
| GWHD 2021 | 6,515 图、275k wheat heads | 跨国家/session OOD | 协议强但单类，约 10.2 GB |
| CropAndWeed | 8,034 高分辨率图、约 112k 实例、74 种 | 物种长尾/跨 session | 必须按录制 session 分组 |
| MAD Apple | 105 标注 4K 帧+4,440 未标注帧 | 半监督双头机制 | 单类、数据少，但视频/树级划分规范 |
| IP102 detection | 约 19k bbox 图、102 pest 类 | 害虫长尾 | 细粒度复杂且已有不少工作 |
| FAIRHiveFrames-1K | 1,265 图、124,669 框、7 类 | 蜂巢密集小目标 | 部分标注不完整会污染 FP 指标 |
| EMDS-7 | 2,365 显微图、13,216 框、42 类 | 极长尾/unknown 类 | 协议与许可表述不一致，需先澄清 |
| NuCLS | 1,744 病理 FOV、59,485 nuclei boxes | 标注不确定性/医院域 | 病理赛道较拥挤，最好有医学合作者 |
| FracAtlas | 4,083 X-ray、922 fracture boxes | 医学 pilot | 患者/医院 ID 被移除，无法排除患者泄漏 |

## 5. 暂不建议作为论文唯一主集

- SSDD/HRSID 单独使用：训练很便宜，但 SAR ship YOLO 改进已高度拥挤；更适合 pilot 或外部验证；
- NEU-DET、DeepPCB、GC10-DET：数据太小、合成/配对特性未被正确使用、统一 split 与许可不足或被反复使用；
- DINS：公开基线 mAP50 已接近饱和，不适合把“提 AP”作为主问题；
- SKU-110K：单类极密集但约 1.7M 框、十几 GB，成熟且算力不低；
- DroneVehicle：约 95 万框，双模态/OBB，仓库许可不明确，第一阶段过重；
- NightOwls：约 285 GB，非商用且强时序相关，超过当前预算；
- LAMOST-DET：2026 天文预印本很新，但数据实际可获取性和许可尚未核实；
- ORCA 直接作为第一站：新颖性很高，但 478 species/670 taxonomy categories 使固定类 YOLO26n 研究难度陡增。

## 6. 三条可发表路线草案

### 路线 A：JODD，小目标长尾与背景误报

研究问题：

> 深海低质成像中，YOLO26 的双分配头是否把有限的正样本质量预算过度集中给头部类和较大目标，从而同时损伤稀有小类召回与背景误报控制？

最小证据链：

1. 官方 video-grouped split 的 YOLO26n baseline，至少 3 seeds；
2. 每类/尺度/vehicle/camera/dive 的 O2M/O2O assignment 和错误类型审计；
3. 一个 training-only、zero-inference-cost 的质量/预算校准机制；
4. 背景图误报、macro AP、AP_S 和 held-out domain 同时改善；
5. SeaClear 或另一个无源视频重叠的水下数据外验。

### 路线 B：RUOD-R，缺标鲁棒的双头检测

研究问题：

> 当密集小目标被漏标时，O2M/O2O 两个头在哪些位置产生系统性错误负梯度，能否只依靠训练期的跨增强/跨头稳定证据安全地将这些位置转为软忽略？

这条线的优势是同图双标注能形成强因果对照；短板是许可确认和 263k 小框带来的训练成本。

### 路线 C：XWOD，最差天气组泛化

研究问题：

> 统一的标签分配与增强强度是否在不同极端天气下产生不公平的正样本覆盖，能否用训练期 weather-balanced dual assignment 提高 worst-weather 与 leave-one-weather-out 泛化，而不是只提高平均 AP？

这条线最贴近现有 YOLO26n 工程，但必须先验证图像来源去重与官方基线可复现性。

## 7. 下一步：只做 Dataset-P0，不急着下载完整数据

建议同时对 JODD、HRP4K、RUOD-R、XWOD 做一个只读/小下载 Dataset-P0：

1. 核对数据许可、版本、下载体积和引用要求；
2. 只取 annotations、metadata 与少量样例；
3. 统计类别、box area/aspect ratio、每图密度、空图、重复与损坏文件；
4. 检查 video/site/flight/device/source 是否跨 split；
5. 核对 YOLO26 label 转换和 ignore/crowd/DontCare 语义；
6. 估算 50/150 轮、3 seeds 的时间和磁盘预算；
7. 冻结主指标、分层指标、外部验证集与停止条件；
8. 审计通过后再选一个主集下载并训练官方 YOLO26n baseline。

本轮没有下载任何数据，也没有在服务器上启动训练。
