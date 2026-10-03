# 遥感方向条件转向评估：让领域问题驱动方法，而不是给 YOLO26 换皮

日期：2026-09-19  
状态：路线讨论后的冻结评估；不代表已下载数据或已把主线强行改为遥感  
当前并行实验：TQGB-T1-NP seed0 继续按原冻结协议运行，不因路线讨论中断

## 1. 执行结论

遥感方向**更容易找到明确、强烈、可测量的问题**，因此比“继续在通用 VOC/COCO 上给官方 YOLO26n 刷少量 AP”更有研究抓手；但它**不天然更容易发论文**。如果只是把注意力、大核、IoU loss 或额外 P2 检测头拼到 YOLO26 上，再在一个遥感数据集上提高一点，仍然属于高度拥挤的改进路线。

本项目采用“条件转向”，而不是立即推翻已有工作：

1. VOC 继续作为低成本机制发现和因果证伪平台；
2. 遥感成为优先应用候选，首先选择水平框数据集，避免一开始同时承担 OBB、切片和新机制三种复杂度；
3. 方法问题优先定义为可迁移机制，遥感只负责把问题放大并提供分组指标；
4. 第一推荐主线是**密集/小目标场景中的实例监督预算失衡**，第二推荐是有严格阴性控制的**亚像素相位信息丢失**；
5. 只有 train-only 探针先证明问题存在，才下载并训练相应遥感基准；不通过探针就停止，不用测试集反复试模块。

因此目前最合适的论文问题不是：

> 如何改进 YOLO26 在遥感数据集上的 mAP？

而是：

> 在密集微小目标检测中，原生动态分配是否让少数实例获得过量监督质量与共享梯度，而微小/困难实例被系统性饿死？能否在不改变推理图的前提下守恒地重分配训练预算？

YOLO26 是实验平台，遥感是高强度场景，机制本身应有机会迁移到其他 dense/dual-assignment detector。

## 2. 为什么遥感更有抓手

遥感目标检测具有通用图像中没有那么集中的结构性困难：

- 一幅超大图中同时存在大量极小目标；
- 目标密集度变化极端，同一图内可能从大片空背景跳到密集车辆/船舶；
- 下采样一次就可能抹掉目标的大部分像素证据；
- 高分辨率图像通常需要切片，边缘截断、重复框和跨片上下文缺失成为可单独度量的错误；
- DOTA 一类数据还带来任意朝向、极端长宽比与 OBB 表示问题。

DOTA 官方基准本身包含 2,806 幅约 4k×4k 图像、188,282 个实例、15 类及任意四边形标注，正说明尺度、方向、形状与密度不是人为编造的问题：<https://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html>。

这些困难的价值在于：我们可以报告 AP_tiny、不同尺寸四分位、不同密度、切片边界距离、旋转角度等细粒度证据，而不是只看总 mAP。若方法真正解决一个机制问题，因果链更容易讲清楚。

## 3. 为什么遥感也不“简单”

近两年的高水平工作已经覆盖了许多直觉路线：

- CVPR 2025 SET 从频域分析微小目标特征，并在 AI-TOD 上报告对既有方法的显著提升：<https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html>；
- TGRS 2024 ORFENet 已使用仅训练期存在、推理时删除的目标重建分支来缓解微小目标信息丢失：<https://ieeexplore.ieee.org/document/10478927/>；
- TGRS 2025 CADDN 已直接研究下采样导致的小目标空间信息损失，并做内容感知下采样：<https://ieeexplore.ieee.org/document/11030746/>；
- CVPR 2025 ABBSPO 研究航空图像弱监督旋转框尺度与方向学习：<https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html>；
- CVPR 2025 AeroGen 已把遥感检测的数据稀缺与长尾问题推进到布局可控的扩散数据生成，并在 DIOR/DIOR-R/HRSC 上验证：<https://openaccess.thecvf.com/content/CVPR2025/html/Tang_AeroGen_Enhancing_Remote_Sensing_Object_Detection_with_Diffusion-Driven_Data_Generation_CVPR_2025_paper.html>。

这意味着以下题目应直接降级：普通注意力、普通大核/多尺度融合、单纯增加 P2、小幅修改 IoU、没有机制证据的频域模块、以及只有一个数据集的一次性 YOLO 改进。

遥感论文通常还需要更复杂的数据预处理、两个或以上基准、强遥感专用 baseline、尺度/密度分组结果和速度/显存开销。若进入 DOTA OBB，还必须处理切片合并、旋转框评测和角度表示。领域更聚焦，但实验义务更多。

## 4. 数据集选择

| 数据集 | 规模与特点 | 与当前 YOLO26 兼容性 | 研究用途 | 当前决定 |
|---|---|---:|---|---|
| DIOR-HBB | 23,463 幅 800×800 图像、约 192k 实例、20 类；尺度和成像条件多样 | 高：水平框，可先沿用 detection 管线 | 第一阶段广义遥感基准；检验机制是否不只对极小目标有效 | **首选进入数据准备** |
| AI-TOD-v2 | 原 AI-TOD 为 28,036 幅、700,621 实例、8 类，平均目标约 12.8 px；v2 对明显漏标/噪声做了重标 | 中：水平框，但实例极密且数据需组装 | 微小目标压力测试；最适合检验 gradient budget 与相位丢失 | **第二数据集/机制压力测试** |
| DOTA-v1.0 | 2,806 幅约 4k×4k 大图、188,282 实例、15 类、任意四边形 | 低到中：需切片与 OBB 管线 | 方向、密集目标和超大图最终强验证 | **方法成熟后再进入** |

DIOR 统计来源：<https://arxiv.org/abs/1909.00133>。  
AI-TOD 官方统计与组装说明：<https://github.com/jwwangchn/AI-TOD>。  
AI-TOD-v2 重标动机与基准：<https://chasel-tsui.github.io/AI-TOD-v2/>。

选择 DIOR-HBB 起步不是因为它最容易刷分，而是它能把“领域变化”与“OBB 工程变化”分开。若直接从 DOTA 开始，一次负结果无法判断是方法无效、切片不当、角度表示问题还是评测合并错误。

服务器只读检查显示 `/home/ubuntu/rz/yolo_study/datasets` 目前没有 DIOR/DOTA/AI-TOD，现有数据约 2.2 GB；所在文件系统剩余约 437 GB。容量足够，但正式下载前仍应记录来源、许可、文件哈希和划分，不能把共享服务器的空闲空间当作无限授权。

## 5. 候选路线排序

### 5.1 RS-IGB：密集微小目标的实例梯度预算（优先级 A）

核心假设：native O2M/TAL 根据匹配质量产生的 soft target mass 在 GT 之间高度不均衡，少量大/易/高质量实例支配 P3/P4/P5 的共享梯度，微小或困难实例得到的训练预算不足。

方法边界：

- 不改变原生 assignment 和正样本集合；
- 根据每个 GT 已获得的 soft positive mass 做解析、detached、batch-conserving 的 quota correction；
- 分类正样本与定位共同受同一实例预算约束；
- 推理网络完全不变；
- 必须胜过保留 quota 数值多重集与总质量、只打乱 GT 对应关系的 permutation placebo。

为什么适合遥感：AI-TOD 的微小、高密度实例会放大 GT 间监督预算与梯度竞争；若 VOC 上信号弱而 AI-TOD 上强，反而支持“场景条件触发”的机制解释。

新颖性风险：ATSS、ObjectBox、LapNet/PONO、GFL、ICCV 2025 UGS 等已经覆盖 assignment balancing、per-object normalization、quality target 与小目标梯度稳定。可防守边界只能是“冻结 native TAL + batch-conserving soft-mass correction + exact per-GT VJP 证据 + permutation 因果对照”，不能声称首个目标公平训练。

首个 GO 门：先完成已有 IGB-P0 exact per-GT VJP ledger。若 gradient Gini、top-20% share 与 mass-gradient 相关性任一不过预注册门槛，RS-IGB 不进入训练。

### 5.2 RS-SPAR：亚像素相位保持（优先级 B，条件性较强）

核心假设：stride-2 下采样对目标边界/相位细节的丢失不是一般高频现象，而是对微小目标特别严重，并与一像素平移不稳定和低 assignment quality 同时相关。

必须保留的窄边界：object-gated、subpixel phase-specific、只在训练期恢复、推理删除；并以 phase-shuffle 控制证明正确相位语义，而不是普通 auxiliary reconstruction/deep supervision。

风险：SET、ORFENet 与 CADDN 已分别覆盖频域增强、训练期目标重建和内容感知下采样。若 SPAR-P0 只发现“高频比低频难重建”，没有 object-vs-background 特异性和 shift-quality 关联，则立即 NO-GO。

### 5.3 跨切片边界一致性（优先级 C，先诊断后命名）

真实问题：高分辨率图像切片后，目标可能被截断，重叠切片会产生重复或不一致预测。常用 SAHI 通过重叠切片保证边界目标至少在一个 patch 中完整出现：<https://obss.github.io/sahi/guides/sliced-inference/>，说明该问题存在，但也说明“使用重叠切片”本身不新。

可能的研究边界：在训练期对同一原图的相邻/错位切片建立 identity-aware consistency，只在完整可见区域传递定位/类别信息；推理仍采用标准切片，不增加检测器结构。先定义 Boundary Object Recall、截断比例分组 AP 与跨切片一致性误差，再判断是否存在足够大的失败面。

风险：切片、动态切片和边界处理已有大量工作；若只做 overlap、NMS 或多尺度 tile，贡献不足。该路线只有在“训练期跨片语义一致性 + 边界专用诊断 + 零模型推理开销”成立时才值得进入。

### 5.4 OBB/旋转等变（暂不作为入门首线）

方向高度匹配遥感，但近期工作密集且工程、评测成本高。ICCV 2025 已专门测量 rotation equivariance 对航空检测的影响：<https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Measuring_the_Impact_of_Rotation_Equivariance_on_Aerial_Object_Detection_ICCV_2025_paper.html>。本项目先用 HBB 分离机制，只有方法在 DIOR/AI-TOD-v2 成立后才考虑迁移 DOTA OBB。

## 6. 冻结的双轨执行路线

```text
现有 VOC 因果闭环
  ├─ 完成 T1-NP seed0
  └─ 完成同开机 B50 seed0，量化重启/故障主机漂移
              ↓
train-only 机制探针（不读 test AP）
  ├─ IGB-P0：exact per-GT gradient budget
  ├─ SPAR-P0：object-specific phase loss
  └─ 轻量遥感数据/标注审计
              ↓ GO
DIOR-HBB baseline + 10-epoch baseline/placebo/semantic 三元组
              ↓ GO
DIOR 完整固定末轮多种子 + AI-TOD-v2 tiny 压力测试
              ↓ GO
健康硬件复现 + DOTA/另一检测器泛化
```

VOC 不再承担论文最终主张，只承担便宜的因果筛选。遥感也不允许成为“换个数据集让负方法变正”的补救：机制先在 train-only 指标上通过，遥感实验才启动。

## 7. 论文潜力的最低标准

一个候选只有同时满足以下条件，才进入“可发文成果”讨论：

1. 问题测量：明确证明哪类遥感实例失败，以及失败与 proposed quantity 的关联；
2. 因果控制：semantic method 胜过等强度、等计算、等数值分布的 placebo；
3. 多种子：至少三组预注册 seed，固定末轮报告；
4. 多数据集：DIOR + AI-TOD-v2，或 DIOR + DOTA；
5. 泛化：至少另一模型尺度或另一 native assignment 检测器；
6. 效率：参数/FLOPs/延迟或“训练期额外、推理零开销”明确；
7. 健康硬件：服务器当前结果只作 faulty-host exploratory，最终数值从随机初始化复现。

## 8. 当前裁决

**遥感是比继续纯通用刷分更有希望的主应用方向，但最佳策略是“机制主线 + 遥感压力场景”，不是“YOLO26 遥感魔改”。**

当前优先选择 RS-IGB 作为第一候选，理由是已有 exact VJP 探针、推理零开销、与密集微小目标高度吻合且仍保留窄新颖边界。RS-SPAR 保留为第二候选，但由于与近期频域/重建/下采样工作碰撞更强，必须通过更严格的 phase-shuffle 语义控制。DOTA OBB 和切片边界研究暂不抢占第一阶段资源。

这个裁决不是服从某个单一想法，而是一个可证伪的工作假设：后续探针和小预算三元组可以把它推翻；如果 IGB/SPAR 都无机制信号，我们再讨论数据、开放词汇、弱监督或跨传感器路线，而不是继续堆模块。
