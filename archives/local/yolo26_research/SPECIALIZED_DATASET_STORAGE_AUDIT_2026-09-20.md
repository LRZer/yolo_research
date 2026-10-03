# 小领域目标检测数据集体量审计

日期：2026-09-20  
状态：只读核验；未下载数据，未启动训练  
范围：上一轮长名单中的 55 个实际候选数据集（不含“暂不建议”段落里仅顺带提到的淘汰项）

## 1. 结论

- 这批候选并非都大。55 个候选中，51 个已经得到官方精确值、官方标称或明确注明来源的镜像参考值；只剩 4 个目前无法得到可信体积。
- 51 个可估算实用包合计约 505 GB；其中部分是格式转换镜像，而不是官方原始包，因此这个合计只用于磁盘量级判断，也不表示应该全部下载。
- 最大的并不是图片最多的数据集：Sherbrooke Red Deer 只有约 4,071 张图，却因超高分辨率达到 131.49 GB；SA-FARI 的 6 fps 帧+标注约 66.30 GB；随后是 Big Bird 标注训练包 44.87 GB、WildLive 31.13 GB、RAIL-BENCH 24.16 GB。
- 当前四个首选候选并不重：JODD 0.343 GB、HRP4K 8.659 GB、RUOD-R 约 3.4–3.7 GB、XWOD 2.690 GB，压缩下载合计约 15–16 GB。
- 真正训练时还要给解压、标签转换、图像缓存、切片图和 run 结果留空间。普通 JPEG 数据建议按下载包的 2–3 倍预留；4K、PNG、切片或 `cache=disk` 建议按 3–5 倍预留。当前服务器不建议使用 `cache=ram`。

## 2. 口径

- `GB` 按十进制计算：1 GB = 1,000,000,000 bytes；官方页面若只给近似值，则保留其原始标称。
- 体量分级：极轻 `≤1 GB`；小型 `>1–5 GB`；中型 `>5–15 GB`；大型 `>15–50 GB`；超大 `>50 GB`。
- “官方精确”表示官方 API、文件清单或服务器 `Content-Length`；“官方标称”表示官方页面显示的近似值；“第三方参考”不用于最终磁盘承诺。
- 多个替代下载包不相加；只列完成检测实验所需的实用包。原始视频、额外模态、模型权重等可选材料单独说明。

## 3. 55 个候选逐项体量

### 3.1 第一梯队与强候选（20 个）

| 数据集 | 图像规模 | 实用下载体积 | 体量 | 口径与说明 |
|---|---:|---:|---|---|
| [JODD](https://www.godac.jamstec.go.jp/jedi/e/dataset/jedi_organism_detection_dataset.html) | 8,151 | 343 MB；解压 405 MB | 极轻 | 官方标称，完整数据集 |
| [HRP4K](https://zenodo.org/records/17522874) | 6,003 | 8.659 GB | 中型 | 官方精确，`HRP4K.zip` |
| [RUOD-R](https://github.com/RSSL-MTU/RUOD-R) | 14,000 | 原 RUOD 图像包约 3.4 GB；新标注约 5.8–58 MB | 小型 | RUOD-R 复用原图；标注大小随 YOLO/COCO 格式而异 |
| [XWOD](https://github.com/kuanting/xwod) | 10,010 | 2.690 GB | 小型 | 官方仓库所链接 Kaggle 数据包元数据 |
| [SortWaste](https://sortwaste.di.ubi.pt/) | 5,261 | 预划分包 12.424 GB；full 包 11.106 GB | 中型 | 两者是替代包，不相加；原始视频另为 2.180 GB |
| [SeaClear](https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689/1) | 8,610 | 1.712 GB | 小型 | 官方精确，完整 RAR |
| [RAIL-BENCH Object](https://www.mrt.kit.edu/railbench_data/object_and_rail/) | 2,500 | 24.156 GB | 大型 | 图像 24.152 GB + 标注 3.56 MB |
| [MBDD2025](https://zenodo.org/records/15622584) | 14,471 | 2.534 GB | 小型 | 官方精确，完整 ZIP |
| [RFDD](https://doi.org/10.57760/sciencedb.msdc.00071) | 1,350 | 8.849 GB | 中型 | 官方元数据精确值，图像分辨率高 |
| [HIT-UAV](https://doi.org/10.5281/zenodo.7633134) | 2,898 | 814.8 MB | 极轻 | 官方精确，完整 ZIP |
| [AEWD](https://github.com/maml1116/AEWD) | 7,483 | 约 4.03 GB | 小型 | 官方 GitHub 图像文件载荷求和；不含另存的原始视频 |
| [WTBD](https://figshare.com/articles/dataset/30210175) | 1,065 | 79.0 MB | 极轻 | 官方精确，完整 ZIP |
| [ORCA](https://huggingface.co/datasets/WongYukKwan/ORCA) | 14,647 | 约 1.1 GB | 小型 | 官方 Hugging Face 页面标称 |
| [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) | 21,527 | 约 3.118 GB | 小型 | 官方 README 直接链接的 Kaggle 副本；非 OneDrive 原包精确值 |
| [MD-AS-2025](https://zenodo.org/records/15210103) | 4,000 | 2.011 GB | 小型 | 官方精确，完整 ZIP |
| [Big Bird](https://doi.org/10.48610/27809f1) | 4,824 标注图 | 44.871 GB | 大型 | 常用 `annotated_dataset.zip`；官方目录全部文件约 313 GB，但含重复/替代归档，不能相加当作必需量 |
| [DuckNet](https://github.com/mkmitchell/DuckNet) | 1,444 | 未知 | 未定 | 官方 Google Drive 链接当前失效 |
| [iSOOD](https://zenodo.org/records/10903574) | 10,481 | 9.470 GB | 中型 | 官方 ZIP+XLSX 精确求和 |
| [InsPLAD](https://data.mendeley.com/datasets/5n3fjgvfyz/1) | 10,607 | 6.400 GB | 中型 | 官方整套外层包；检测-only 子包大小未单列 |
| [DsPCBSD+](https://figshare.com/articles/dataset/DsPCBSD_/24970329) | 10,259 | 128.5 MB | 极轻 | 官方精确，完整 ZIP |

### 3.2 工业、基础设施与特殊交通补充（10 个）

| 数据集 | 图像规模 | 实用下载体积 | 体量 | 口径与说明 |
|---|---:|---:|---|---|
| [ZeroWaste-f](https://zenodo.org/records/6412647) | 4,503 | 7.518 GB | 中型 | 官方精确，final benchmark 包 |
| [PackWISE](https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en) | 586 | 355.24 MB | 极轻 | 官方页面标称 v2 |
| [CUBIT-Det](https://github.com/BenyunZhao/CUBIT) | 5,527 | 约 16.283 GB | 大型参考 | 第三方完整 Kaggle 镜像；官方网盘未列汇总体积，使用前需复核内容 |
| [COCO-Bridge 2021+](https://figshare.com/articles/dataset/COCO-Bridge_2021_Dataset/16624495/1) | 1,470 | 679.9 MB | 极轻 | 官方文件求和，含 ZIP、说明 PDF 和 README |
| [TTPLA](https://github.com/R3ab/ttpla_dataset) | 约 1,100–1,234 | 4.511 GB | 小型 | 官方当前原始图 ZIP；图像数存在版本漂移 |
| [Apron Dataset](https://github.com/apronai/apron-dataset) | 10,098 | 约 5.001 GB | 中型参考 | 第三方 Kaggle 镜像；官方申请制且未公开体积 |
| [ABOships-PLUS](https://zenodo.org/records/10469672) | 9,880 | 8.149 GB | 中型 | 官方精确，完整 ZIP |
| [FishEye8K](https://github.com/MoyoG/FishEye8K) | 8,000 | 13.363 GB | 中型 | 官方 2024-01 整包 HEAD 精确值；YOLO 标签另约 3.4 MB |
| [OPIXray](https://github.com/OPIXray-author/OPIXray) | 8,885 | 约 2.634 GB | 小型参考 | 第三方 YOLO 镜像；官方需联系作者且未公开体积 |
| [CLCXray](https://github.com/GreysonPhoenix/CLCXray) | 9,565 | 约 6.406 GB | 中型参考 | 第三方 YOLO 镜像；官方网盘未显示汇总体积 |

### 3.3 生态、水下与动物补充（9 个）

| 数据集 | 图像规模 | 实用下载体积 | 体量 | 口径与说明 |
|---|---:|---:|---|---|
| [SeaDronesSee ODv2](https://seadronessee.cs.uni-tuebingen.de/dataset) | 14,227 | 全压缩包 9.959 GB；只取有标注 train+val 约 7.422 GB | 中型 | 官方云盘文件元数据精确求和；全分辨率解压版约 126 GB，不建议首轮使用 |
| [DUO](https://github.com/chongweiliu/DUO) | 7,782 | 约 2.7 GB | 小型 | 官方 Google Drive 页面标称完整包 |
| [TrashCan-Instance](https://doi.org/10.13020/g1gx-y834) | 7,212 | 约 0.54–0.61 GB | 极轻参考 | Dataset Ninja/Kaggle 镜像；官方原文件体积当前不可取 |
| [Amphitrite v1.0](https://figshare.com/articles/figure/Amphitrite_v1_0_An_underwater_database_for_marine_debris_and_biodiversity/28908176) | 1,583 | 1.318 GB | 小型 | 官方 Figshare API 精确值 |
| [SealDetectionDataset](https://doi.org/10.21334/NPOLAR.2025.C4E51ECE) | 495 tiles | 952.9 MB | 极轻 | 官方 API 992 个图像/标注/附属文件的原始字节求和 |
| [Sherbrooke Red Deer](https://doi.org/10.5683/SP3/TELIYD) | 4,071 | 131.485 GB | 超大 | 官方 Dataverse 精确求和；图片少但单张超高分辨率 |
| [WildLive](https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/) | 19,139 | 31.133 GB | 大型 | 官方 Bristol 目录整包精确值 |
| [AP-10K](https://github.com/AlexTheBad/AP-10K) | 10,015 | 3.403 GB | 小型 | 官方 Google Drive 文件 Range header 精确值（约 3.17 GiB） |
| [SA-FARI](https://huggingface.co/buckets/jivagorolo/SA-FARI-bucket) | 11,609 个视频 | 检测训练约 66.30 GB；全部约 194.23 GB | 超大 | 6 fps 帧 64.443 GB + 标注 1.86 GB；原视频另 127.928 GB |

### 3.4 热红外、低光与恶劣环境（8 个）

| 数据集 | 图像规模 | 实用下载体积 | 体量 | 口径与说明 |
|---|---:|---:|---|---|
| [ExDark](https://researchdata.um.edu.my/dataset.xhtml?persistentId=doi%3A10.22452%2FRD%2FJUSQEK) | 7,363 | 1.518 GB | 小型 | 官方 Dataverse 14 个归档文件求和 |
| [LLVIP](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md) | 15,488 RGB/IR 对 | 主数据 4.004 GB | 小型 | 官方 Drive 精确值；可选原始未配准/视频另约 10.331 GB |
| [DAWN](https://data.mendeley.com/datasets/766ygrbt8y/3) | 1,000 | 未知 | 未定 | 官方页面当前不展示文件体积 |
| [RTTS](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/dataset_prepare.md) | 4,322 | 1.091 GB | 小型 | 官方/主流准备链接对应包；本质是测试集 |
| [WEDGE](https://github.com/Infernolia/WEDGE) | 3,360 | 129.0 MB | 极轻 | 官方仓库所链接 detection ZIP 的 Range header 精确值 |
| [AU-AIR](https://github.com/freeridering/auair-dataset) | 32k+ 帧 | 图像约 2.2 GB + 标注 3.9 MB | 小型 | 官方 README 标称 |
| [FLIR ADAS Expanded](https://www.flir.eu/oem/adas/adas-dataset-form/) | 26,442 帧 | 官方未知；当前第三方镜像约 12.738 GB | 未定/中型参考 | 官方需填表且不显示体积；镜像值不能当官方精确值 |
| [IFSOD](https://github.com/RuihengZhang/IFSOD-dataset) | 4,815 | 67.1 MB | 极轻 | 官方 Drive 文件精确值 |

### 3.5 农业、显微与科学影像（8 个）

| 数据集 | 图像规模 | 实用下载体积 | 体量 | 口径与说明 |
|---|---:|---:|---|---|
| [GWHD 2021](https://zenodo.org/records/5092309) | 6,515 | 10.215 GB | 中型 | 官方精确，完整 ZIP |
| [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset) | 8,034 | 11.470 GB | 中型 | 官方四个图像包+标注包精确求和 |
| [MAD Apple](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms/mad.html) | 105 标注 + 4,440 未标注帧 | 5.878 GB | 中型 | 官方完整 ZIP 精确值 |
| [IP102 detection](https://zenodo.org/records/15546419) | 原版检测约19k；转换包需复核 | 798.3 MB | 极轻参考 | 其他研究者发布的COCO转换包，非IP102原作者官方包；需确认内容与划分完整性 |
| [FAIRHiveFrames-1K](https://zenodo.org/records/19241078) | 1,265 | 未知 | 未定 | Zenodo 只有 2 KB 说明文件；所指 Box ZIP 当前失效，不能把 2 KB 当数据体积 |
| [EMDS-7](https://figshare.com/articles/dataset/EMDS-7_Dataset/16867114) | 2,365 | 5.308 GB | 中型 | 官方 Figshare API 精确值 |
| [NuCLS corrected single-rater](https://sites.google.com/view/nucls/single-rater) | 1,744 FOV | 未知 | 未定 | 官方 Google Drive 文件夹未显示汇总体积 |
| [FracAtlas](https://figshare.com/articles/dataset/FracAtlas_A_Dataset_for_Fracture_Classification_Localization_and_Segmentation_of_Musculoskeletal_Radiographs/22363012) | 4,083 | 338.4 MB | 极轻 | 官方 Figshare API 精确值 |

## 4. 对当前选题最有用的磁盘结论

| 候选 | 压缩下载 | 建议预留工作空间 | 判断 |
|---|---:|---:|---|
| JODD | 0.343 GB | 5–10 GB | 最轻，几乎没有磁盘压力，最适合先做 Dataset-P0 和完整 baseline |
| XWOD | 2.690 GB | 15–25 GB | 较轻，直接接 YOLO26n 的工程成本低 |
| RUOD-R | 约 3.4–3.7 GB | 20–35 GB | 图像不大，但 263k 密集小框会增加标签缓存与评估时间 |
| HRP4K | 8.659 GB | 35–60 GB | 下载量中等；4K 图和切片实验才是实际空间大头 |
| SortWaste | 约 11–12.4 GB | 40–70 GB | 可做，但视频帧查重、解压和 cache 要额外留量 |
| RAIL-BENCH | 24.156 GB | 70–120 GB | 已进入“大型”，不适合作为最快第一站 |
| Big Bird | 44.871 GB | 120 GB 以上 | 标注训练包本身就大，暂不适合当前快速研究 |

若先并行保存 JODD、HRP4K、RUOD-R、XWOD 四套原始压缩包，准备 **至少 60 GB 空闲磁盘** 比较稳妥；若要做 HRP4K 切片、多个转换副本和多次训练，建议准备 **100 GB**。不建议一次性下载全部 55 套。

## 5. 待补核条目

目前完全没有可信体积的只剩 4 个：DuckNet、DAWN、FAIRHiveFrames-1K、NuCLS。

另外 6 个虽然已经能给出量级，但不是官方原始包的精确值：D-Fire、CUBIT-Det、Apron、OPIXray、CLCXray、TrashCan-Instance。它们的数字来自官方 README 直接链接的副本或明确标注的第三方转换镜像，实际立项下载前仍需重新核对内容清单。

这些“未知”不是说数据一定很大，而是目前没有可审计的数值。真正决定使用其中某个数据集时，应先做一次只读 HEAD/API/网盘元数据核验，再决定下载。
