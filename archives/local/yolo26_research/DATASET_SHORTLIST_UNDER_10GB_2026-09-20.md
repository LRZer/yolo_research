# 10 GB以内的跨领域目标检测数据集候选

核对日期：2026-09-20。按用户要求仅检索与记录，没有下载完整数据集、连接服务器、改变数据或训练。阈值采用十进制10,000,000,000字节；限定表中版本，不代表同名所有版本都小于10GB。解压、保留原包和缓存另需空间。

质量指作者/机构可追溯、目标级标注、公开技术说明及可复用协议；并非保证零错标、零近重复。新候选尚未全量验收。主实验适配性与原始标注质量分开评价。

## 候选清单

| 数据集 | 领域 | 指定材料体积 | 图片数/类别 | 质量依据与用途限制 |
|---|---|---|---|---|
| [HRSID](https://github.com/chaozhong2010/HRSID) | SAR雷达船舶 | 0.614 GB，作者JPG+近岸/远岸标注版本 | 5604图，1类，16951船实例 | 作者开放检测/实例标注和论文；海杂波、近岸背景、小船问题清晰。需固定JPG版本和原始划分，不能混用PNG版大小或把另400张背景扩展算进主包 |
| [ZeroWaste-f](https://ai.bu.edu/zerowaste/) | 工业垃圾分拣 | 7.518 GB，zerowaste-f-final.zip | 4503图，4类 | CVPR2022，真实工业传送带、精细多边形标注；适合重叠/变形/透明物体。检测框提取规则须统一，不把分割分数当检测AP。不需要整站30GB所有扩展 |
| [HRP4K](https://github.com/hanshenChen/HRP4K) | 道路坑洞 | 8.659 GB，HRP4K.zip | 6003图，1类，7217实例 | Scientific Data2026，人工复核，4003正例+2000负例，4K原图；适合远距离小目标与误检问题。较新，不声称已有成熟广泛使用生态；高分辨率训练计算量仍高 |
| [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face) | 人脸检测 | 全部ZIP约3.68GB；train+val及标注约1.84GB | 32203图，1类；train12880/val3226/test16097 | CVPR2016经典基准，393703人脸，交叉复核标注。非身份识别；test框不公开，需遵循Easy/Medium/Hard及忽略协议；竞争很成熟 |
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | 航拍交通 | train+val+test-dev约1.79GB；加challenge约2.07GB | 前三部分8629图，10类；含challenge10209图 | 天津大学公开挑战与TPAMI论文；小目标、密集及遮挡。这里是静态DET而不是视频全包；test-dev标签可用，challenge不公开 |
| [NEU-DET](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) | 钢材表面缺陷 | 27.607MB，NEU-DET.zip | 1800图，6类，200×200灰度图 | 东北大学作者发布，IEEE TIM检测研究；细长纹理与低对比度。不是NEU-CLS分类版。很适合快速实验，但规模小，不能仅凭此集提点声称广泛泛化 |
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | 电路板缺陷 | 当前仓库全部文件载荷127.793MB（不是ZIP实测） | 1500对模板/待检图，6类 | 作者提供配准图、框和评测程序；对齐参考图有研究价值。作者评测IoU阈值0.33，不能直接混比普通AP50；二值化场景较窄，原报告已很高 |
| [WGISD](https://github.com/thsant/wgisd) | 农业葡萄串 | 核心data文件398.024MB；当前整个仓库1.484GB文件载荷 | 300图，通常1类葡萄串，5种葡萄品种 | Embrapa作者发布，YOLO框、部分实例掩码与固定划分；242训练/验证，58测试。品种不等于5个标准检测类；体量小，优先辅助外验。README框总数4432/4431差1，下载后重算 |
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 医疗内镜息肉 | 官网ZIP44.1MB（页面另列数据46.2MB） | 1000图，1类 | 人工标注并由胃肠专家核验，掩码+框JSON，MMM2020。不是仅分类的Kvasir；不等于临床验证，需要关注患者/序列独立性和外部测试 |
| [GTSDB](https://github.com/houbensebastian/GermanTrafficSignBenchmarks) | 道路交通标志 | FullIJCNN2013.zip 1.662GB | 900图，train600/test300；经典检测3大组 | IJCNN2013正式基准；原标签有细类ID，标准检测通常分禁止/指令/危险3组，不混淆GTSRB43类裁剪分类。很小且较饱和，更适合补充测试 |
| [PackWISE v2](https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en) | 包装废物分拣 | 官网355.24MB，指定v2 | 586图，23类，4096×4096 | Fraunhofer实物采集、COCO框/实例掩码、70/15/15划分及框转换脚本。标注丰富但小样本，适合作为工业分拣补充验证；与ZeroWaste类别不一，不能直接无映射外测 |

前五项更适合考虑主实验；后六项来源可信且轻量，但小样本、评测特殊或任务范围窄，更适合作为快速验证/补充证据。这是选题建议而非实验结论。

## 本轮大小核验记录

- NEU-DET：作者官网链接Google Drive ID `1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k`。仅请求bytes=0-0并流式关闭，HTTP206，Content-Range `bytes 0-0/27607059`，未下载归档。
- HRSID：作者仓库明确列出的JPG近岸/远岸标签版Drive ID `1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms`。相同只读探测HTTP206，Content-Range `bytes 0-0/614069765`。不是同页PNG高保真版，未验包内清单。
- GTSDB：[作者存档](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)的[published-files.json](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-files.json)：FullIJCNN2013.zip=1661866983 bytes。历史train=1104177081，test=555674823；不重复下载完整包与分包。
- HRP4K：[Zenodo记录17522874](https://zenodo.org/records/17522874) API返回HRP4K.zip=8659158643 bytes，MD5=07bd59989052db4fe3d7ad4c8ab7ad67。网页抓取工具失败但官方API成功；不能把README遗留“soon”理解为API包不存在。[数据论文](https://www.nature.com/articles/s41597-026-07317-w)支持标注与样本说明。
- ZeroWaste：[官方Zenodo记录6412647](https://zenodo.org/records/6412647) API返回zerowaste-f-final.zip=7518242799 bytes。选f完整标注子集，不把单个分卷冒充整个s版本。
- DeepPCB GitHub master完整树truncated=false，全部文件size和127793202 bytes。只统计当前快照，不含Git历史；这是载荷不是压缩包大小。完整影像标注均在作者仓库。
- WGISD GitHub master完整树truncated=false，全部文件size和1483718639 bytes，data/下737个文件398024216 bytes；extras含视频帧扩展，核心300图不需把额外帧当标注样本。没有.gitattributes条目；不按LFS指针大小冒充原图体积。
- WIDER FACE：[CUHK-CSE数据文件页](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data)列train1.47GB、val363MB、test1.84GB、标注3.59MB，全目录3.68GB。
- Kvasir-SEG与PackWISE大小采用作者页面标示，未伪装成全包下载实测。

## 既有数据资源复用

本机archives目录只读统计：DUO=2940459407 bytes（2.940GB），HIT-UAV=814751578（0.815GB），SeaClear=1711829309（1.712GB），ExDark=1518029453（1.518GB，14个材料文件），AP-10K=3403294327（3.403GB）。此前完整性与标注工程验收见各数据集VALIDATION.json。本轮未重复执行全量解码。

优先复用HIT-UAV（2898图/4有效类）、SeaClear（8610图/40原始类）、DUO（7782图/4类）、ExDark（7363图/12类）。AP-10K是姿态来源，检测穷尽性需要额外核对。ORCA仍有语义使用警告；LLVIP由用户自行下载，当前未验收其新包；JODD源仍阻塞，均不列为无条件准备就绪。

## 暂缓与剔除

- GWHD2021原始包约10.215GB、CropAndWeed约11.470GB，超过本轮10GB阈值；不是把GB/GiB单位换一下当作符合。
- KITTI左目约12GB、SKU110K_fixed约13.6GB，COCO训练验证约20GB量级，排除。
- D-Fire：作者[官方仓库](https://github.com/gaia-solutions-on-demand/DFireDataset)链接的Kaggle发行版[sayedgamal99/smoke-fire-detection-yolo](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)，Kaggle元数据totalBytes=3049605157（3.050GB文件载荷）。作者21527图/2类和大量负例有价值，但本轮没有解决先前提出的来源簇/近重复跨划分风险，因此保留“条件候选”，不能因为更新了大小就升格为已严格审计高质量。
- IP102原作者确认约19000检测框图/102类，但这轮未核实原始v1.1完整包大小；不沿用798MB第三方COCO转包作为原包大小。
- RDD2020作者确认26336图/4类，但本轮官方文件API403，完整包体积暂未复核，不用第三方不一致的1.13/3.74GB口径填入主表。
- CrowdHuman作者Google文件页本轮404；未核实完整包元数据，暂不作为本轮新增确定项。不意味着原数据质量低。
- MinneApple、TTPLA等保留后续考察，不为凑数量纳入未确认版本/体积的名单。

## 路线判断

1. 低成本迭代：现成HIT-UAV，或新增HRSID；均是明确成像域，但SAR与热红外物理机制不同，不应混成同一个问题。
2. 工业分拣：ZeroWaste-f主实验，PackWISE补充任务验证；类目差异需要独立训练/映射，不称作直接零样本外验。
3. 高分辨率道路小目标：HRP4K，关注小坑洞召回与负例误报。低于10GB不表示训练便宜。
4. 农业：WGISD可作葡萄遮挡的小型验证，但仅300图，暂不建议当唯一主实验。
5. 经典工业缺陷：NEU-DET/DeepPCB方便，但不能把数据小或已有高AP视为创新容易。应另外找独立生产来源数据证明泛化。

本轮不确定研究路线，不承诺论文成果，不启动下载或训练。下一步应由用户从领域中选少量候选后做样本和协议审计。
