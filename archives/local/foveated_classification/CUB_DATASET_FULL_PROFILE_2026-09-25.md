# CUB-200-2011 数据集全量信息（本机原始文件复算）

日期：2026-09-25。统计来源：已校验的 Caltech 官方原始包及解压文件；只读扫描，未修改图片或标注，未运行模型。

来源：[Caltech 官方页面](https://www.vision.caltech.edu/datasets/cub_200_2011/)、[主包记录](https://data.caltech.edu/records/65de6-vp158)、[分割包记录](https://data.caltech.edu/records/w9d68-gec53)。图片限非商业研究和教学使用。官方提醒它与 ImageNet 图片有重叠；若用 ImageNet 预训练权重，测试可能存在预训练污染。

## 1. 它是什么、掩膜是什么

这是 200 种鸟的细粒度**单标签图像分类**数据集。每张图有一个物种类别、一个主要鸟的矩形框、15 类部位点和属性标注。官方另行发布的 `segmentations.tgz` 为逐像素鸟体掩膜：图片与掩膜同名、同尺寸；黑色(0)表示背景、白色(255)表示鸟体，轮廓处可见中间灰度。实际文件不是严格二值，出现 0/51/102/153/204/255 六档，因此阈值规则须在实验前说明；2 张为 `LA` 模式但 alpha 恒为 255。它是**发布的参考前景标注**，不同于我们未来由模型预测的模糊强度图。第一阶段按 GT 框中心做实验不依赖分割包；掩膜可用于诊断被模糊的鸟体像素比例，正式未知图像推理不能读取它。

## 2. 总量与划分

| 指标 | 实测值 |
|---|---:|
| 鸟类物种 | 200 |
| 图像 | 11,788 |
| 官方训练集 | 5,994 |
| 官方测试集 | 5,794 |
| 矩形框 | 11,788（每图1个） |
| 独立分割掩膜 | 11,788（与图像逐一匹配） |
| 每类图像数 | 41–60 |
| 每类训练图 | 29–30 |
| 每类测试图 | 11–30 |
| 图像文件合计 | 1,150,650,529 B (1.151 GB; 1.072 GiB) |
| 分割PNG合计 | 40,953,594 B (0.041 GB; 0.038 GiB) |
| 主压缩包 | 1,150,585,339 B (1.151 GB; 1.072 GiB) |
| 分割压缩包 | 39,272,883 B (0.039 GB; 0.037 GiB) |

官方 `train_test_split.txt` 中 1 表示训练、0 表示测试。若要有开发验证集，只从 5,994 张官方训练图内再划；不要把官方测试集拿来反复调半径或模糊强度。

## 3. 图像、框与掩膜的技术细节

- 图像共有 1,152 种原始尺寸；宽度最小/中位/最大为 121/500/500 像素，高度为 120/375/500 像素。最常见尺寸：500×333（1,820张）、500×375（1,192张）、500×400（366张）、500×334（364张）、333×500（305张）、500×357（239张）、500×332（235张）、500×500（205张）、400×500（166张）、500×335（157张）。
- JPG 模式：RGB 11,780张、L 8张；PNG 模式：L 11,786张、LA 2张。全部 11,788 对都能完整像素解码且尺寸相同。
- 类别规模相对均衡：144类恰好60张；194类有30张官方训练图，另6类有29张；每类官方测试图为 11–30 张。
- 原始框采用像素坐标 `(x, y, width, height)`，框面积占图像面积的第10/50/90百分位分别约为 13.9%/31.7%/59.2%。有 30 张原始框越过图片边缘，最大越界 32 像素；不能直接假定框始终在图内，原始文件未改动。
- 分割图的六档像素数量：0→1,774,236,633、51→46,938,495、102→19,284,786、153→17,518,731、204→35,578,230、255→219,878,240；共 2,113,435,115 个掩膜像素。将哪一级当作前景，必须在具体实验中预先定义。

## 4. 额外标注

- 部位：`parts/parts.txt` 定义 15 种，`parts/part_locs.txt` 有 176,820 条聚合记录，正好每图每部位一条；其中可见标记为1的有 141,407 条（80.0%）。不可见点通常坐标为0，不应当作真实位置。`parts/part_click_locs.txt` 有 794,258 条众包原始点击记录。
- 属性：实际 `attributes.txt` 有 312 个属性；`attributes/image_attribute_labels.txt` 有 3,677,856 条记录，每图最少/最多 312/312 条，标为存在的记录 371,047 条。每条含属性是否存在、确定度和标注耗时。`class_attribute_labels_continuous.txt` 是 200×312 的类级属性比例矩阵。包内 README 概览写作“322”，但实际名单与矩阵及官方网页均为 312，使用时以文件内容为准。
- 属性源文件另有 606 条六列记录（图像ID 2275、9364各303条，均在第5列多一个`0`）；其它记录为 README 描述的五列。本报告只把前四列用于标签统计、最后一列视为耗时，不改动原文件。未来若使用逐图属性，必须显式处理并审计这一格式异常。
- 15个部位的可见次数：back 9,064、beak 11,745、belly 10,347、breast 10,967、crown 11,580、forehead 11,603、left eye 5,857、left leg 8,491、left wing 6,578、nape 11,311、right eye 6,079、right leg 8,517、right wing 6,677、tail 10,961、throat 11,630。
- 属性确定度编码：1=not visible（394,906条）、2=guessing（228,021条）、3=probably（1,021,155条）、4=definitely（2,033,774条）。

## 5. 文件结构与字段

- `images/<species>/<file>.jpg`：原始照片；`images.txt`：图像 ID 到相对路径；`classes.txt`：200 类 ID 与名称；`image_class_labels.txt`：图像 ID 到类别。
- `train_test_split.txt`：官方训练/测试标记；`bounding_boxes.txt`：每图一个 `(x,y,w,h)` 框；`parts/part_locs.txt`：每图每部位 `(x,y,visible)`。
- `attributes/`：逐图属性判定、确定度及类级属性比例；压缩包顶层另有 `attributes.txt` 属性名称表。
- 独立分割包解压到 `segmentations/<species>/<file>.png`，与 JPG 相对路径对应。完整官方包与解压件在服务器 `/home/ubuntu/rz/foveated_classification_research/data/CUB-200-2011/`，本机在 `E:\datasets\CUB-200-2011\`。

## 6. 200 类逐类图片、官方划分、框与掩膜数量

每图恰好一个类别、一个框、一张掩膜，所以每类框数和掩膜数均等于该类图片总数。以下数字直接由 `image_class_labels.txt` 与 `train_test_split.txt` 复算。

| ID | 官方类名 | 全部图片 | 训练 | 测试 | 框 | 掩膜 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 001.Black_footed_Albatross | 60 | 30 | 30 | 60 | 60 |
| 2 | 002.Laysan_Albatross | 60 | 30 | 30 | 60 | 60 |
| 3 | 003.Sooty_Albatross | 58 | 30 | 28 | 58 | 58 |
| 4 | 004.Groove_billed_Ani | 60 | 30 | 30 | 60 | 60 |
| 5 | 005.Crested_Auklet | 44 | 30 | 14 | 44 | 44 |
| 6 | 006.Least_Auklet | 41 | 30 | 11 | 41 | 41 |
| 7 | 007.Parakeet_Auklet | 53 | 30 | 23 | 53 | 53 |
| 8 | 008.Rhinoceros_Auklet | 48 | 30 | 18 | 48 | 48 |
| 9 | 009.Brewer_Blackbird | 59 | 30 | 29 | 59 | 59 |
| 10 | 010.Red_winged_Blackbird | 60 | 30 | 30 | 60 | 60 |
| 11 | 011.Rusty_Blackbird | 60 | 30 | 30 | 60 | 60 |
| 12 | 012.Yellow_headed_Blackbird | 56 | 30 | 26 | 56 | 56 |
| 13 | 013.Bobolink | 60 | 30 | 30 | 60 | 60 |
| 14 | 014.Indigo_Bunting | 60 | 30 | 30 | 60 | 60 |
| 15 | 015.Lazuli_Bunting | 58 | 30 | 28 | 58 | 58 |
| 16 | 016.Painted_Bunting | 58 | 30 | 28 | 58 | 58 |
| 17 | 017.Cardinal | 57 | 30 | 27 | 57 | 57 |
| 18 | 018.Spotted_Catbird | 45 | 30 | 15 | 45 | 45 |
| 19 | 019.Gray_Catbird | 59 | 30 | 29 | 59 | 59 |
| 20 | 020.Yellow_breasted_Chat | 59 | 30 | 29 | 59 | 59 |
| 21 | 021.Eastern_Towhee | 60 | 30 | 30 | 60 | 60 |
| 22 | 022.Chuck_will_Widow | 56 | 30 | 26 | 56 | 56 |
| 23 | 023.Brandt_Cormorant | 59 | 30 | 29 | 59 | 59 |
| 24 | 024.Red_faced_Cormorant | 52 | 30 | 22 | 52 | 52 |
| 25 | 025.Pelagic_Cormorant | 60 | 30 | 30 | 60 | 60 |
| 26 | 026.Bronzed_Cowbird | 60 | 30 | 30 | 60 | 60 |
| 27 | 027.Shiny_Cowbird | 60 | 30 | 30 | 60 | 60 |
| 28 | 028.Brown_Creeper | 59 | 30 | 29 | 59 | 59 |
| 29 | 029.American_Crow | 60 | 30 | 30 | 60 | 60 |
| 30 | 030.Fish_Crow | 60 | 30 | 30 | 60 | 60 |
| 31 | 031.Black_billed_Cuckoo | 60 | 30 | 30 | 60 | 60 |
| 32 | 032.Mangrove_Cuckoo | 53 | 30 | 23 | 53 | 53 |
| 33 | 033.Yellow_billed_Cuckoo | 59 | 30 | 29 | 59 | 59 |
| 34 | 034.Gray_crowned_Rosy_Finch | 59 | 30 | 29 | 59 | 59 |
| 35 | 035.Purple_Finch | 60 | 30 | 30 | 60 | 60 |
| 36 | 036.Northern_Flicker | 60 | 30 | 30 | 60 | 60 |
| 37 | 037.Acadian_Flycatcher | 59 | 30 | 29 | 59 | 59 |
| 38 | 038.Great_Crested_Flycatcher | 60 | 30 | 30 | 60 | 60 |
| 39 | 039.Least_Flycatcher | 59 | 30 | 29 | 59 | 59 |
| 40 | 040.Olive_sided_Flycatcher | 60 | 30 | 30 | 60 | 60 |
| 41 | 041.Scissor_tailed_Flycatcher | 60 | 30 | 30 | 60 | 60 |
| 42 | 042.Vermilion_Flycatcher | 60 | 30 | 30 | 60 | 60 |
| 43 | 043.Yellow_bellied_Flycatcher | 59 | 30 | 29 | 59 | 59 |
| 44 | 044.Frigatebird | 60 | 30 | 30 | 60 | 60 |
| 45 | 045.Northern_Fulmar | 60 | 30 | 30 | 60 | 60 |
| 46 | 046.Gadwall | 60 | 30 | 30 | 60 | 60 |
| 47 | 047.American_Goldfinch | 60 | 30 | 30 | 60 | 60 |
| 48 | 048.European_Goldfinch | 60 | 30 | 30 | 60 | 60 |
| 49 | 049.Boat_tailed_Grackle | 60 | 30 | 30 | 60 | 60 |
| 50 | 050.Eared_Grebe | 60 | 30 | 30 | 60 | 60 |
| 51 | 051.Horned_Grebe | 60 | 30 | 30 | 60 | 60 |
| 52 | 052.Pied_billed_Grebe | 60 | 30 | 30 | 60 | 60 |
| 53 | 053.Western_Grebe | 60 | 30 | 30 | 60 | 60 |
| 54 | 054.Blue_Grosbeak | 60 | 30 | 30 | 60 | 60 |
| 55 | 055.Evening_Grosbeak | 60 | 30 | 30 | 60 | 60 |
| 56 | 056.Pine_Grosbeak | 60 | 30 | 30 | 60 | 60 |
| 57 | 057.Rose_breasted_Grosbeak | 60 | 30 | 30 | 60 | 60 |
| 58 | 058.Pigeon_Guillemot | 58 | 30 | 28 | 58 | 58 |
| 59 | 059.California_Gull | 60 | 30 | 30 | 60 | 60 |
| 60 | 060.Glaucous_winged_Gull | 59 | 30 | 29 | 59 | 59 |
| 61 | 061.Heermann_Gull | 60 | 30 | 30 | 60 | 60 |
| 62 | 062.Herring_Gull | 60 | 30 | 30 | 60 | 60 |
| 63 | 063.Ivory_Gull | 60 | 30 | 30 | 60 | 60 |
| 64 | 064.Ring_billed_Gull | 60 | 30 | 30 | 60 | 60 |
| 65 | 065.Slaty_backed_Gull | 50 | 30 | 20 | 50 | 50 |
| 66 | 066.Western_Gull | 60 | 30 | 30 | 60 | 60 |
| 67 | 067.Anna_Hummingbird | 60 | 30 | 30 | 60 | 60 |
| 68 | 068.Ruby_throated_Hummingbird | 60 | 30 | 30 | 60 | 60 |
| 69 | 069.Rufous_Hummingbird | 60 | 30 | 30 | 60 | 60 |
| 70 | 070.Green_Violetear | 60 | 30 | 30 | 60 | 60 |
| 71 | 071.Long_tailed_Jaeger | 60 | 30 | 30 | 60 | 60 |
| 72 | 072.Pomarine_Jaeger | 60 | 30 | 30 | 60 | 60 |
| 73 | 073.Blue_Jay | 60 | 30 | 30 | 60 | 60 |
| 74 | 074.Florida_Jay | 60 | 30 | 30 | 60 | 60 |
| 75 | 075.Green_Jay | 57 | 30 | 27 | 57 | 57 |
| 76 | 076.Dark_eyed_Junco | 60 | 30 | 30 | 60 | 60 |
| 77 | 077.Tropical_Kingbird | 60 | 30 | 30 | 60 | 60 |
| 78 | 078.Gray_Kingbird | 59 | 30 | 29 | 59 | 59 |
| 79 | 079.Belted_Kingfisher | 60 | 30 | 30 | 60 | 60 |
| 80 | 080.Green_Kingfisher | 60 | 30 | 30 | 60 | 60 |
| 81 | 081.Pied_Kingfisher | 60 | 30 | 30 | 60 | 60 |
| 82 | 082.Ringed_Kingfisher | 60 | 30 | 30 | 60 | 60 |
| 83 | 083.White_breasted_Kingfisher | 60 | 30 | 30 | 60 | 60 |
| 84 | 084.Red_legged_Kittiwake | 53 | 30 | 23 | 53 | 53 |
| 85 | 085.Horned_Lark | 60 | 30 | 30 | 60 | 60 |
| 86 | 086.Pacific_Loon | 60 | 30 | 30 | 60 | 60 |
| 87 | 087.Mallard | 60 | 30 | 30 | 60 | 60 |
| 88 | 088.Western_Meadowlark | 60 | 30 | 30 | 60 | 60 |
| 89 | 089.Hooded_Merganser | 60 | 30 | 30 | 60 | 60 |
| 90 | 090.Red_breasted_Merganser | 60 | 30 | 30 | 60 | 60 |
| 91 | 091.Mockingbird | 60 | 30 | 30 | 60 | 60 |
| 92 | 092.Nighthawk | 60 | 30 | 30 | 60 | 60 |
| 93 | 093.Clark_Nutcracker | 60 | 30 | 30 | 60 | 60 |
| 94 | 094.White_breasted_Nuthatch | 60 | 30 | 30 | 60 | 60 |
| 95 | 095.Baltimore_Oriole | 60 | 30 | 30 | 60 | 60 |
| 96 | 096.Hooded_Oriole | 60 | 30 | 30 | 60 | 60 |
| 97 | 097.Orchard_Oriole | 59 | 30 | 29 | 59 | 59 |
| 98 | 098.Scott_Oriole | 60 | 30 | 30 | 60 | 60 |
| 99 | 099.Ovenbird | 60 | 30 | 30 | 60 | 60 |
| 100 | 100.Brown_Pelican | 60 | 30 | 30 | 60 | 60 |
| 101 | 101.White_Pelican | 50 | 30 | 20 | 50 | 50 |
| 102 | 102.Western_Wood_Pewee | 60 | 30 | 30 | 60 | 60 |
| 103 | 103.Sayornis | 60 | 30 | 30 | 60 | 60 |
| 104 | 104.American_Pipit | 60 | 30 | 30 | 60 | 60 |
| 105 | 105.Whip_poor_Will | 49 | 30 | 19 | 49 | 49 |
| 106 | 106.Horned_Puffin | 60 | 30 | 30 | 60 | 60 |
| 107 | 107.Common_Raven | 59 | 29 | 30 | 59 | 59 |
| 108 | 108.White_necked_Raven | 60 | 30 | 30 | 60 | 60 |
| 109 | 109.American_Redstart | 60 | 30 | 30 | 60 | 60 |
| 110 | 110.Geococcyx | 60 | 30 | 30 | 60 | 60 |
| 111 | 111.Loggerhead_Shrike | 60 | 30 | 30 | 60 | 60 |
| 112 | 112.Great_Grey_Shrike | 60 | 30 | 30 | 60 | 60 |
| 113 | 113.Baird_Sparrow | 50 | 30 | 20 | 50 | 50 |
| 114 | 114.Black_throated_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 115 | 115.Brewer_Sparrow | 59 | 30 | 29 | 59 | 59 |
| 116 | 116.Chipping_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 117 | 117.Clay_colored_Sparrow | 59 | 30 | 29 | 59 | 59 |
| 118 | 118.House_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 119 | 119.Field_Sparrow | 59 | 30 | 29 | 59 | 59 |
| 120 | 120.Fox_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 121 | 121.Grasshopper_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 122 | 122.Harris_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 123 | 123.Henslow_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 124 | 124.Le_Conte_Sparrow | 59 | 30 | 29 | 59 | 59 |
| 125 | 125.Lincoln_Sparrow | 59 | 30 | 29 | 59 | 59 |
| 126 | 126.Nelson_Sharp_tailed_Sparrow | 59 | 29 | 30 | 59 | 59 |
| 127 | 127.Savannah_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 128 | 128.Seaside_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 129 | 129.Song_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 130 | 130.Tree_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 131 | 131.Vesper_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 132 | 132.White_crowned_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 133 | 133.White_throated_Sparrow | 60 | 30 | 30 | 60 | 60 |
| 134 | 134.Cape_Glossy_Starling | 60 | 30 | 30 | 60 | 60 |
| 135 | 135.Bank_Swallow | 59 | 29 | 30 | 59 | 59 |
| 136 | 136.Barn_Swallow | 60 | 30 | 30 | 60 | 60 |
| 137 | 137.Cliff_Swallow | 60 | 30 | 30 | 60 | 60 |
| 138 | 138.Tree_Swallow | 60 | 30 | 30 | 60 | 60 |
| 139 | 139.Scarlet_Tanager | 60 | 30 | 30 | 60 | 60 |
| 140 | 140.Summer_Tanager | 60 | 30 | 30 | 60 | 60 |
| 141 | 141.Artic_Tern | 58 | 29 | 29 | 58 | 58 |
| 142 | 142.Black_Tern | 60 | 30 | 30 | 60 | 60 |
| 143 | 143.Caspian_Tern | 60 | 30 | 30 | 60 | 60 |
| 144 | 144.Common_Tern | 60 | 30 | 30 | 60 | 60 |
| 145 | 145.Elegant_Tern | 60 | 30 | 30 | 60 | 60 |
| 146 | 146.Forsters_Tern | 60 | 30 | 30 | 60 | 60 |
| 147 | 147.Least_Tern | 60 | 30 | 30 | 60 | 60 |
| 148 | 148.Green_tailed_Towhee | 60 | 30 | 30 | 60 | 60 |
| 149 | 149.Brown_Thrasher | 59 | 30 | 29 | 59 | 59 |
| 150 | 150.Sage_Thrasher | 60 | 30 | 30 | 60 | 60 |
| 151 | 151.Black_capped_Vireo | 51 | 30 | 21 | 51 | 51 |
| 152 | 152.Blue_headed_Vireo | 60 | 30 | 30 | 60 | 60 |
| 153 | 153.Philadelphia_Vireo | 59 | 30 | 29 | 59 | 59 |
| 154 | 154.Red_eyed_Vireo | 60 | 30 | 30 | 60 | 60 |
| 155 | 155.Warbling_Vireo | 60 | 30 | 30 | 60 | 60 |
| 156 | 156.White_eyed_Vireo | 60 | 30 | 30 | 60 | 60 |
| 157 | 157.Yellow_throated_Vireo | 59 | 30 | 29 | 59 | 59 |
| 158 | 158.Bay_breasted_Warbler | 60 | 30 | 30 | 60 | 60 |
| 159 | 159.Black_and_white_Warbler | 60 | 30 | 30 | 60 | 60 |
| 160 | 160.Black_throated_Blue_Warbler | 59 | 30 | 29 | 59 | 59 |
| 161 | 161.Blue_winged_Warbler | 60 | 30 | 30 | 60 | 60 |
| 162 | 162.Canada_Warbler | 60 | 30 | 30 | 60 | 60 |
| 163 | 163.Cape_May_Warbler | 60 | 30 | 30 | 60 | 60 |
| 164 | 164.Cerulean_Warbler | 60 | 30 | 30 | 60 | 60 |
| 165 | 165.Chestnut_sided_Warbler | 60 | 30 | 30 | 60 | 60 |
| 166 | 166.Golden_winged_Warbler | 59 | 30 | 29 | 59 | 59 |
| 167 | 167.Hooded_Warbler | 60 | 30 | 30 | 60 | 60 |
| 168 | 168.Kentucky_Warbler | 59 | 30 | 29 | 59 | 59 |
| 169 | 169.Magnolia_Warbler | 59 | 30 | 29 | 59 | 59 |
| 170 | 170.Mourning_Warbler | 60 | 30 | 30 | 60 | 60 |
| 171 | 171.Myrtle_Warbler | 60 | 30 | 30 | 60 | 60 |
| 172 | 172.Nashville_Warbler | 60 | 30 | 30 | 60 | 60 |
| 173 | 173.Orange_crowned_Warbler | 60 | 30 | 30 | 60 | 60 |
| 174 | 174.Palm_Warbler | 60 | 30 | 30 | 60 | 60 |
| 175 | 175.Pine_Warbler | 60 | 30 | 30 | 60 | 60 |
| 176 | 176.Prairie_Warbler | 60 | 30 | 30 | 60 | 60 |
| 177 | 177.Prothonotary_Warbler | 60 | 30 | 30 | 60 | 60 |
| 178 | 178.Swainson_Warbler | 56 | 30 | 26 | 56 | 56 |
| 179 | 179.Tennessee_Warbler | 59 | 30 | 29 | 59 | 59 |
| 180 | 180.Wilson_Warbler | 60 | 30 | 30 | 60 | 60 |
| 181 | 181.Worm_eating_Warbler | 59 | 30 | 29 | 59 | 59 |
| 182 | 182.Yellow_Warbler | 60 | 30 | 30 | 60 | 60 |
| 183 | 183.Northern_Waterthrush | 60 | 30 | 30 | 60 | 60 |
| 184 | 184.Louisiana_Waterthrush | 60 | 30 | 30 | 60 | 60 |
| 185 | 185.Bohemian_Waxwing | 60 | 30 | 30 | 60 | 60 |
| 186 | 186.Cedar_Waxwing | 60 | 30 | 30 | 60 | 60 |
| 187 | 187.American_Three_toed_Woodpecker | 50 | 30 | 20 | 50 | 50 |
| 188 | 188.Pileated_Woodpecker | 60 | 30 | 30 | 60 | 60 |
| 189 | 189.Red_bellied_Woodpecker | 60 | 30 | 30 | 60 | 60 |
| 190 | 190.Red_cockaded_Woodpecker | 58 | 29 | 29 | 58 | 58 |
| 191 | 191.Red_headed_Woodpecker | 60 | 30 | 30 | 60 | 60 |
| 192 | 192.Downy_Woodpecker | 60 | 30 | 30 | 60 | 60 |
| 193 | 193.Bewick_Wren | 60 | 30 | 30 | 60 | 60 |
| 194 | 194.Cactus_Wren | 60 | 30 | 30 | 60 | 60 |
| 195 | 195.Carolina_Wren | 60 | 30 | 30 | 60 | 60 |
| 196 | 196.House_Wren | 59 | 29 | 30 | 59 | 59 |
| 197 | 197.Marsh_Wren | 60 | 30 | 30 | 60 | 60 |
| 198 | 198.Rock_Wren | 60 | 30 | 30 | 60 | 60 |
| 199 | 199.Winter_Wren | 60 | 30 | 30 | 60 | 60 |
| 200 | 200.Common_Yellowthroat | 60 | 30 | 30 | 60 | 60 |

注：这些是数据集统计，不是模型准确率或检测 AP；本研究尚未跑 CUB 分类模型。
