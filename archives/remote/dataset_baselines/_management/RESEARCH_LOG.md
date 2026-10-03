# YOLO26 通用目标检测持续研究日志

### 2026-09-21 03:34 ExDark误判闭环与LLVIP交付

LLVIP远端原包SHA256一致，46,465成员逐项校验、30,976张图片全解码通过；5个等横坐标XML框仍保留为读取策略警告。ExDark前三个失败包通过只读恢复复验：不覆盖原解压文件，对每个tar成员做内容哈希，真实图片全部解码，AppleDouble元数据保留但不误判为图片。由此证明早先错误来自验收器类型识别，不是原图片损坏；ExDark 19/19、LLVIP 5/5、HRSID 5/5均完成。现63/68，仅剩ZeroWaste-f五项及最后独立全量审计。

### 2026-09-21 02:22 ORCA远端图片验收

ORCA原始`images.zip`的远端SHA256与本机冻结值一致，14,646个成员全部CRC、解压大小及图片解码通过。主标注引用14,645张，额外1张未引用图片原样保留；caption `-1`含义仍是使用前需确认的语义边界，未擅自改标注。其余ORCA划分与说明文件仍在队列，最终全清单独立复验尚未进行。

### 2026-09-21 01:58 ExDark验收器修正与接续

远端ExDark前三个分类包因把AppleDouble辅助文件当图片解码而失败。包成员`._`名称及文件头`0005160700020000`共同确认其类型；原包、辅助文件和标注均保留。验收器仍检查每个成员的CRC或SHA256，仅对确认的元数据不做图片解码；实际图像继续全量解码。修正部署后bus527图、car638图、cat包已通过。前三包待只读恢复复验，不能提前计成功。已启动有限接续进程36576，在现上传批次释放工作锁后只重试已审阅失败并传清单剩余数据；其他新错误停下审阅。当前四套完整交付、15/68文件完成，最终独立审计尚未开始。

### 2026-09-21 数据集上传目标与验收

用户明确要求持续上传全部已下载数据至`gpu8-team:/home/ubuntu/rz/object_detection_dataset_test`。已创建目标和任务专属目录，保留原包、安全解压、上传前后SHA256及全图片解码；不训练、不用GPU、不改共享环境。10套68文件共25.565GB已冻结入队（JODD无包未计）。HIT-UAV已远端验收完成，其余上传中；短暂SSH拒绝连接后自行恢复，DUO经前缀哈希核对从1.725GB续传。

新验收发现：VisDrone三条零高度原框（含一普通汽车框），LLVIP五个xmin=xmax框，HRSID分split标注16969条而合并版16951条、目标框集合实际不同，ZeroWaste138个原框越界。全部保留原始标签并记录使用约束，没有为“通过验收”改标签。新VisDrone test-dev1610图/1610标注通过图像完整性检查；ZeroWaste4503图+4503掩码及26766标注全部配对核验，3处跨split同名图的原文件和RGB像素哈希均不相同。

最新进程、清单、错误恢复、待做最终远端独立验收见`DATASET_UPLOAD_HANDOFF_2026-09-21.md`；本阶段不宣称所有上传已完成。

### 2026-09-20 23:52 三数据集网盘查询与ZeroWaste本机下载

- 用户选定HRSID/ZeroWaste-f/VisDrone2019-DET，要求优先提供网盘，找不到再代下到电脑。核实HRSID作者JPG百度分享及bs64提取码，VisDrone作者静态DET train/val/test-dev三分享；记录在E:/datasets对应DOWNLOAD_LINKS.md。百度分享实际内容未登录核验；HRSID百度版不能直接承诺等同此前614MB Google新增标签版。
- ZeroWaste-f作者官网/仓库只指向Zenodo，本轮针对百度/夸克搜索未找到可核实分享。按授权启动本机后台下载PID35768，E:/datasets/ZeroWaste-f/archives/zerowaste-f-final.zip.part。包7518242799字节，官方MD5 e26e31a58080bca6782dca0e56074c5d；23:52已观察收到数据，未完成。
- 复用任务下载器，增加可选公开ZIP密码支持；下载后自动MD5/SHA256及ZIP CRC检查，不自动解压、不称作逐图验收。仅进程级已有代理，未碰服务器、未训练、未恢复旧自动复查。

### 2026-09-20 10GB跨领域数据集补充筛选

- 仅检索与只读元数据核验，未连接服务器、下载完整包、训练或恢复自动复查。完整名单见 `DATASET_SHORTLIST_UNDER_10GB_2026-09-20.md`。
- 明确11个指定版本候选：HRSID、ZeroWaste-f、HRP4K、WIDER FACE、VisDrone-DET、NEU-DET、DeepPCB、WGISD、Kvasir-SEG、GTSDB、PackWISE v2；小样本与主实验候选分开，不承诺均适合唯一论文主集。
- 新核验：NEU-DET官方ZIP27607059字节，HRSID作者JPG近岸/远岸版614069765字节，GTSDB全包1661866983字节；官方API复核HRP4K8659158643、ZeroWaste-f7518242799字节。DeepPCB/WGISD明确记录仓库文件载荷而非伪造压缩包大小。
- 10GB阈值剔除GWHD2021/CropAndWeed/KITTI/SKU/COCO。D-Fire虽作者链接版约3.05GB，尚未解决来源近重复问题，维持条件备选；IP102/RDD2020原包大小未确认，不用第三方转包替代。
- 优先考虑已有HIT-UAV/SeaClear/DUO/ExDark，或新增HRSID/ZeroWaste-f/HRP4K；这是路线建议，不是用户已批准的方向或已完成的新实验。

> 本文档是本项目唯一的持续研究总账。新增文献结论、代码改动、实验结果、失败原因和方向调整都追加到这里；未经核验的论文或数字不会当作事实使用。

## 1. 项目目标

- 基础模型：Ultralytics YOLO26n，端到端 O2O 推理。
- 当前数据集：PASCAL VOC 2007+2012。
- 研究目标：围绕 YOLO26 原生训练机制提出可解释、可消融、推理零额外开销的改进，并通过真实训练验证是否提高检测精度。
- 首要指标：VOC07 test 的 mAP@0.5:0.95；同时报告 Precision、Recall、mAP@0.5 和逐类别结果。
- 约束：只使用物理 GPU 4、5、6、7；不污染官方基线源码和已完成结果。

## 2. 固定实验协议

### 2.1 数据划分

- 训练：VOC2007 trainval + VOC2012 trainval，共 16,551 张图像。
- 统一评测：VOC2007 test，共 4,952 张图像。
- 用户已决定使用全部 trainval 训练。该协议接近经典 VOC07+12 设置，但 VOC07 test 必须只承担统一最终评测：不得根据 test 表现反复挑选超参数、挑 epoch 或删除失败实验。
- 当前所有正式比较固定训练 150 epochs，并优先报告 epoch 150 的 `last.pt`；`best.pt` 只作为补充结果，避免不同方法因 test 上的最佳轮次不同而产生选择偏差。

### 2.2 公平性与复现

- 模型从随机初始化开始，和现有基线保持一致。
- 输入尺寸、增强、优化器、batch、workers、随机种子、确定性设置均继承基线；除被研究变量外不改动其他条件。
- 正式主对比：150 epochs，batch 64，4 卡分布式，seed 42。
- 初筛采用短训时，只用于淘汰明显无效/不稳定方案；进入主结果的模型必须完整训练 150 epochs。
- 若提升小于约 0.5 mAP，后续至少补 3 个随机种子，报告均值与标准差。
- 每项训练型改进同时记录训练时间与峰值显存；推理型结论同时记录参数量、GFLOPs 和真实延迟。

## 3. 已锁定基线 B0

| 项目 | 数值/状态 |
|---|---|
| 实验 ID | B0-O2M-Y26N-VOC-E150-S42 |
| 模型 | 官方 YOLO26n 配置，随机初始化 |
| epochs / batch | 150 / 64 |
| GPU | 4,5,6,7 |
| Precision | 0.79629 |
| Recall | 0.71899 |
| mAP@0.5 | 0.79339 |
| mAP@0.5:0.95 | 0.58705（O2M + NMS） |
| 最佳轮次 | 150 |
| 总训练时间 | 1 h 31 min 24 s |

基线目录：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42`

### 3.1 补测端到端 O2O 基线

现有训练参数 `nms: null` 会在验证阶段选择 O2M 头并执行 NMS，因此原先的 0.58705 不是端到端 O2O 指标。使用同一个 epoch-150 `last.pt`，显式设定 `nms: false` 后得到：

| 实验 ID | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| B0-O2M（NMS） | 0.79629 | 0.71899 | 0.79339 | 0.58705 |
| B0-O2O（NMS-free） | 0.7910313 | 0.6971400 | 0.7786028 | 0.5768901 |

O2O 比 O2M 低 0.01016 mAP@0.5:0.95（约 1.02 个百分点），主要同时损失 Recall 与 mAP@0.5。这一差距成为双头协同研究的直接、可量化目标。后续正式实验必须同时报告 O2M+NMS 和 O2O NMS-free，不能再把二者混为一个“YOLO26 指标”。

## 4. 官方代码机制核查

以下结论来自当前锁定源码，而不是二手论文描述：

1. YOLO26 训练时有 O2M 与 O2O 两套独立检测头；推理使用 O2O 头。
2. O2O 头接收 `detach()` 后的 Neck 特征，因此 O2O 损失只更新自己的检测头，不反向更新 Backbone/Neck。
3. O2M 使用 Task-Aligned Assigner 的 top-k=10；O2O 先在扩大后的候选区域内选 top-7，再收缩为 top-1。tiny box 的候选区域会被扩到中间层 stride 尺度，这两部分共同构成当前 STAL 行为。
4. ProgLoss 当前是纯时间调度：O2M 权重从 0.8 线性衰减到 0.1，O2O 权重从 0.2 线性增长到 0.9；它不观察两个头的质量、分歧或梯度冲突。
5. 当前端到端 Head 的 `reg_max=1`，即 direct box regression，无 DFL 分布回归。
6. 当前训练日志只返回 O2O 的 box/cls/L1 三项，而不是加权总损失或 O2M 损失；旧曲线不能用于判断 O2M/O2O 的相对学习状态。

这些事实指出了三个真实接口：双头间知识没有显式传递；ProgLoss 不感知学习状态；direct regression 缺少显式定位分布信息。

### 4.1 VOC 尺度审计与 STAL 方向裁决

对 VOC 标签做了独立只读审计，确认图像、标签和类别索引完整有效：没有空标签、孤立文件、越界框或非法类别。统计口径是保持宽高比缩放到 `imgsz=640`，未包含 mosaic 与随机缩放造成的动态尺寸变化。

| 划分 | 图像数 | GT 数 | 任一边 `<16 px`（STAL 静态触发） | 含触发 GT 的图像 | COCO-small `<32²` | medium | large |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOC07+12 trainval | 16,551 | 40,058 | 294（0.7339%） | 189（1.1419%） | 2.724% | 24.355% | 72.922% |
| VOC07 test | 4,952 | 12,032 | 25（0.2078%） | 23（0.4645%） | 1.853% | 25.025% | 73.122% |

训练集缩放后目标最短边的 p1/p5/p10/中位数分别为 16.64/26.88/37.12/136.96 px，最短边小于 32 px 的目标占 7.317%。这说明当前数据集以中大目标为主，原生 STAL 的 `<16 px` 静态覆盖极低。训练增强会改变单批次的实际触发率，因此这里不能替代运行时统计；但它足以说明“围绕 tiny-box STAL 做主创新”与当前 VOC 的证据密度不匹配。

**当前裁决：STAL 专项改进在本项目首轮为 no-go/降级方向。** 先研究覆盖全部目标、且直接针对已观测 O2O 差距的双头协同；若将来转向小目标数据集，再重新评估 STAL。

## 5. 候选方向审查（2026-09-18）

| 候选 | 核心假设 | 新颖性初判 | 实现风险 | 推理开销 | 当前状态 |
|---|---|---:|---:|---:|---|
| A. 质量感知 O2M→O2O 协同学习 | 用 O2M 的密集高质量监督缓解 O2O 稀疏监督 | 中高；必须区别于普通双头蒸馏、混合匹配与质量感知分配 | 中 | 0 | 已完成 v0、单测和 smoke；D0 诊断中 |
| B. 状态/冲突感知 ProgLoss | 用头部质量与分歧代替纯 epoch 线性权重 | 较高且 YOLO26 原生 | 中 | 0 | 优先诊断与初试 |
| C. 训练期分布定位教师 | 训练时学习位置分布，蒸馏给 direct head | 低于原判断：与 YOLOv6 DLD 高度重合 | 中高 | 0 | 降级；除非重构为 YOLO26 双路径质量门控定位迁移 |
| D. 通用质量/不确定性 STAL | 按尺度、质量、不确定性动态分配正样本 | 中高，标签分配赛道拥挤 | 高 | 0 | VOC 静态触发仅 0.73%，首轮 no-go |
| E. Foundation-guided assignment | 大模型只在训练时提供目标先验 | 高 | 很高 | 0 | 后续资源充足时研究 |
| F. Conditional P2 | 仅对需要小目标细节的样本激活 P2 | 中高 | 很高 | 可变 | 不适合当前 VOC 首轮 |

### 暂不作为主创新

CBAM/EMA/ECA、Ghost/DWConv、静态 P2、简单 Neck 融合、替换 IoU loss、普通特征 KD、直接换 DINO Backbone。这些可以成为辅助对照，但缺少足够清晰的新研究问题。

## 6. 当前首轮研究路线

围绕一个可验证问题展开：**能否只在 O2M 的预测质量明确高于 O2O 时，将密集头知识安全迁移给部署使用的 O2O 头，从而缩小当前约 1.02 mAP 的双头差距？**

执行顺序：

1. 在不改变优化行为的前提下增加诊断记录：分别记录 O2M/O2O 的 box、cls、L1 损失、正样本数量、同 anchor/同 GT 覆盖率、预测分歧和 ProgLoss 权重。
2. 先实现最安全的 A1：只在 O2O 正样本中满足“两个头匹配同一 GT、O2M 质量高于 O2O 且超过置信阈值”的位置，做 stop-gradient 的 O2M→O2O 分类蒸馏。新增梯度只进入 O2O head，推理结构不变。
3. 用无门控 KD 作为必要负/弱基线，验证质量门控是否真的避免 O2M 密集重复预测污染 O2O 的唯一预测特性。
4. 做等设置短训 A/B；只有达到预设门槛且训练稳定，才进入 150-epoch 正式对比。
5. A1 确认有效后，再研究 A2 定位迁移以及 B 状态感知 ProgLoss；第一轮不同时修改 STAL，确保能归因。

预注册初筛门槛：相同短训预算下，候选方法不得出现 NaN/发散，且 mAP@0.5:0.95 至少不低于短训基线；若提升小于 0.3，则优先修改假设而不是直接消耗 150 epochs。

### 6.1 已实现原型：质量门控的非对称 O2M→O2O 迁移

暂定工作名为 **QGAT / QG-AKT**（尚未锁定论文命名）。为避免把普通“双头自蒸馏”包装成创新，当前更准确的研究表述是：

> Quality-Gated Asymmetric O2M-to-O2O Transfer for Consistent Dual-Assignment Detectors

代码只放在研究副本 `/home/ubuntu/rz/yolo_study/ultralytics_research` 的 `research/dual-head-diagnostics` 分支，基于官方提交 `9d45ee4613e79d7bceef7a39e42c15c278b0d4e8`；官方目录和原始基线均未改动。当前修改仅涉及 `ultralytics/utils/loss.py`、`ultralytics/cfg/default.yaml` 与 `ultralytics/cfg/__init__.py`。

当前 v0 分类迁移规则：

1. 只考察 O2M 与 O2O **同一 anchor 且匹配同一 GT** 的共同正样本，避免跨目标或重复框的错误传递。
2. 两个头的目标质量统一计算为 `q = sqrt(p_gt × IoU(pred, GT))`。
3. 仅当 O2M 的 `q` 达到最低质量、比 O2O 高出指定 margin，且 O2M 的 GT 类分数也高于 O2O 时开放门控。
4. 对被选位置仅做 assigned-class 的 Bernoulli KL；O2M 教师与权重全部 stop-gradient，新增梯度只进入 O2O 检测头。
5. 蒸馏损失按教师相对优势加权，并支持 warmup、ramp 与训练末期 fade；推理图、参数量和 NMS-free 路径完全不变。

新增配置默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `o2o_kd_gain` | 0 | 默认关闭，保证诊断版不改变优化行为 |
| `o2o_kd_temp` | 1.0 | KL 温度 |
| `o2o_kd_min_quality` | 0.25 | O2M 教师最低质量 |
| `o2o_kd_margin` | 0.05 | 教师相对 O2O 的最小质量优势 |
| `o2o_kd_warmup` | 5 | 前期禁用轮数 |
| `o2o_kd_ramp` | 10 | 蒸馏权重爬升轮数 |
| `o2o_kd_fade` | 10 | 末期淡出轮数 |
| `o2o_kd_gate` | true | 是否启用质量门控 |

诊断版同时记录 O2M/O2O 各自损失、正样本/GT 覆盖、`same_gt_rate`、`teacher_better`、`teacher_cls_better`、`kd_active`、`quality_gap`、`head_box_iou`、`kd_scale`、`o2o_kd_loss` 与 ProgLoss 权重。由于官方 O2O 特征已 detach，O2O 本身不更新共享骨干，因此当前不把“双头共享骨干梯度余弦”当作有效诊断量。

当前实现仍是可证伪原型，不是最终论文方法。一个待检验的强化版本是：门控不仅比较教师与学生预测质量，还要求教师质量超过 O2O 的监督目标/匹配质量；是否必要由 D0 诊断分布与后续消融决定。

### 6.2 安全性单元测试

测试脚本：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts/verify_dual_head_research.py`。

- `gain=0` 时，原有 box/cls/L1 三项损失与未改版逐项完全一致。
- 空目标 batch 的所有损失均为有限值，无 NaN/Inf。
- 构造可触发门控的合成样本时，KD loss 为 `4.45059e-06`，`kd_active=1`。
- KD 反传的 Backbone 梯度为 0、O2M 头梯度为 0，O2O 头梯度为 `9.52023e-06`，符合“只教部署头”的设计。
- Python 编译与 `git diff --check` 均通过；环境未安装 Ruff，因此没有把“未运行 Ruff”误记为 lint 通过。

### 6.3 四卡 1-epoch 强制激活 smoke test

实验目录：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/smoke_qgat_cls_g05_e1_g4567_s42`。

该实验只用于验证 DDP/BF16、日志和梯度路径，不用于精度比较。为确保第一轮就能观察到 KD，临时使用 `gain=0.5`、质量阈值与 margin 为 0、无 warmup/ramp/fade；1 epoch 短任务由自动优化器选择 AdamW，因此不可与 150-epoch MuSGD 基线比较。

| 诊断量 | epoch 1 结果 |
|---|---:|
| `same_gt_rate` | 0.9415 |
| `teacher_better` | 0.9003 |
| `teacher_cls_better` | 0.9086 |
| `kd_active` | 0.8729 |
| `quality_gap` | 0.0142 |
| `head_box_iou` | 0.7903 |
| `o2o_kd_loss` | 0.005352 |

训练在物理 GPU 4–7 上稳定完成，每卡约 2.64 GB，未出现 NaN。共同同-GT位置占比很高，说明研究接口真实存在；但宽松门控激活率过高，不能据此判断默认阈值或方法有效性。

### 6.4 独立代码审计与方法假设升级（正式 KD 实验前必须处理）

独立审计确认当前实现没有致命的 anchor/GT 对齐、坐标单位、温度 KL 或 stop-gradient 错误，正在运行的 D0 指标有效，不需要中止。但审计发现以下正式消融前必须修正或量化的问题：

1. **类别索引必须来自 assigner 的真实 `target_labels`。** 当前原型通过 `target_scores.argmax()` 恢复类别；若极端正样本的 TAL soft score 全为 0，非 0 类会被误当成类别 0。后续将保存 `target_labels` 并补全零 soft-score 单测。
2. **质量更高不等于蒸馏方向与原监督一致。** 当前门控要求教师类别概率高于学生，所以分类 KL 总是把 O2O logit 往上推；但 O2O BCE 使用 soft TAL target `y`，当学生概率 `p_s > y` 时，原监督正把 logit 往下推，二者直接冲突。对任意正温度，已有 `p_t > p_s` 条件下，两项梯度同向当且仅当 `p_s < y`。
3. **新增两种待验证门控。** 方向一致门控要求 `p_s < y`；更严格的教师接近度门控要求 `|p_t-y| < |p_s-y|`，即教师确实比学生更接近 O2O 的 soft target。先利用 D0 每 10 轮 checkpoint 做固定批次的离线冲突轨迹分析，再决定主方法采用“质量门控”“质量+方向一致”还是“质量+教师接近度”。
4. **诊断量需要全局计数语义。** 当前曲线是逐批比例的平均，适合看趋势，但不是四卡全局、按正样本计数的精确总体比例。正式实验前将记录并汇总 `common/selected/agree` 原始计数及质量差求和，同时把 `kd_active` 拆成 candidate/effective 两个概念。
5. **验证侧调度字段不可解释。** EMA 验证副本会新建停留在初始状态的 criterion，因此 `val/prog_o2m_w≈0.8`、`val/prog_o2o_w≈0.2`、`val/kd_scale=0`。这不影响 O2O 的 P/R/mAP 或原始验证预测，但这些调度/KD-loss 列必须忽略；训练侧字段才表示真实调度。
6. **补强基线等价性测试。** 当前 D0 与旧基线的前期损失在日志精度下逐项一致，但新代码的浮点乘法次序与官方路径不同。正式实验前将恢复官方运算顺序，并加入参数梯度级对照、温度有限差分、同类不同实例、warmup/ramp/fade 边界和完整梯度隔离测试。

这个发现把研究问题从“高质量教师就可以迁移”推进为更严格的可证伪假设：**O2M 只有在预测质量更高、且所提供的更新不违背 O2O 自身 soft supervision 时，才是安全教师。** 这也为无门控 KD 可能负迁移提供了可直接测量的机制解释。

### 6.5 v1 加固与固定 checkpoint 监督冲突轨迹

审计项已在研究副本中落实为 v1：类别索引改为直接使用 assigner 的 `target_labels`；恢复官方基础损失的运算次序；KD 改为先按图像聚合再对 batch 求均值；候选数量与最终生效数量分开记录；增加 `none/direction/closer` 三种监督门控，并补齐非法配置检查。完整 CPU 测试确认 `gain=0` 时基础损失和参数梯度逐项完全一致；GPU 测试的基础损失完全一致，重复 CUDA reduction 的梯度相对 L2 误差为 0.00105355、最大绝对差 0.0146492，在显式容差内。空目标和 KD 梯度隔离测试均通过：新增梯度只进入 O2O 分类头，不进入 Backbone、O2M 或 O2O box 分支。

随后对 D0 的 7 个固定 checkpoint 做无增强、无权重更新的 VOC07 test 探针。统计按跨 batch 原始计数汇总，避免逐批比例平均造成偏差：

| epoch | 同-GT共同位置 | 质量门控候选率 | KD/原监督方向一致 | 教师更接近 soft target | 教师越过 target |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,873 | 0.46% | 100.00% | 100.00% | 18.52% |
| 11 | 11,131 | 25.03% | 99.68% | 99.25% | 3.41% |
| 31 | 11,561 | 16.68% | 99.79% | 99.17% | 2.65% |
| 51 | 11,663 | 12.24% | 99.86% | 99.51% | 2.31% |
| 101 | 11,709 | 8.61% | 100.00% | 99.50% | 1.69% |
| 141 | 11,724 | 4.98% | 99.83% | 99.49% | 1.71% |
| 150 | 11,728 | 4.48% | 99.81% | 99.62% | 1.14% |

这组结果否定了一个原先候选贡献：在当前 `min_quality=0.25, margin=0.05` 的质量门控之后，额外的 `direction` 或 `closer` 条件仅会删掉约 0%–0.83% 的候选，几乎没有独立作用，不能作为主创新。它们保留为安全诊断/附录消融。更有证据价值的现象是候选覆盖率在第 11 轮达到 25.03%，随后持续衰减到 4.48%，支持“教师可靠性具有明显阶段性”的假设。

**当前实验决策：** 第一轮只比较三项可归因设置：D0 无 KD、D1 同 anchor/同 GT 的无质量门控 KD、D2 质量门控的非对称 KD。三者使用相同 seed、训练预算和 KD 时间日程；先做损失量级校准与短训筛选。监督方向门控不并入 D2，以免增加一个已被数据证明基本冗余的组件。

轨迹探针使用了 VOC07 test 标签，因此这里只把它作为探索性机制证据，不能据此选择截止轮次或调门槛。后续动态调度若进入正式方法，必须由训练期在线统计或独立验证集决定；VOC07 test 只用于按预注册轮次报告结果。

### 6.6 KD 量级校准与首轮短训预注册

在不反向传播、不更新权重的条件下，以 D0 固定 checkpoint 在 VOC07+12 训练划分上测量 `gain=1` 的 KD 原始量级。该校准只用于避免新增损失数量级失衡，不根据 test 精度选参：

| 设置 | epoch | 候选率 | KD loss | O2O cls loss | KD/cls | 与原监督方向一致 |
|---|---:|---:|---:|---:|---:|---:|
| 质量门控 | 11 | 24.84% | 0.068109 | 2.009213 | 3.39% | 99.80% |
| 质量门控 | 51 | 12.93% | 0.031524 | 0.946490 | 3.33% | 99.90% |
| 质量门控 | 150 | 4.69% | 0.012125 | 0.617862 | 1.96% | 100.00% |
| 无门控 | 11 | 100% | 0.055697 | 2.009213 | 2.77% | 63.02% |
| 无门控 | 51 | 100% | 0.039806 | 0.946490 | 4.21% | 46.04% |
| 无门控 | 150 | 100% | 0.058227 | 0.617862 | 9.42% | 32.19% |

因此锁定一个保守且统一的 `gain=0.5`，D1 与 D2 均不再扫 gain。两组短训均从随机初始化开始，固定 50 epochs、batch 64、4×GPU 4–7、MuSGD（`optimizer=auto` 在 12,950 iterations 下选择 MuSGD）、seed 42、BF16；KD 均使用 warmup 5、ramp 10、末期 fade 10。短训只用于筛掉明显无效方向，主结论仍需 150 epochs 和多种子验证。

公平性复核发现，ProgLoss 按配置的总 epochs 完成 O2M→O2O 权重迁移：50-epoch 任务会在第 50 轮到达 O2M/O2O=`0.1/0.9`，而原 150-epoch D0 的第 50 轮仅到 `0.5698/0.4302`。因此原 D0 epoch 50 不是有效短训对照。首轮矩阵补充 B50（50 epochs、gain=0），并只在 B50/D1/D2 三个完全同预算任务之间比较最终轮结果；这项修正是在读取任何 D1/D2 最终精度前完成的。

另预注册一个实现级因果校验：官方 O2O 分支接收 detach 后的特征，当前单向 KD 的教师与权重也全部 detach，因此新增损失的直接梯度只能进入 O2O 分类头，并可通过改变 O2O assignment 间接改变 O2O box 头；正常情况下 Backbone、Neck 和 O2M 头应保持相同轨迹。B50/D1/D2 完成后将逐张量比较 `last.pt`，并配对评测 O2M+NMS。需要注意训练器对全模型执行 `max_norm=10` 的全局梯度裁剪：若某些 step 触发裁剪，O2O 新增梯度会改变统一缩放因子，从而间接造成共享参数的微小分叉。因此“完全逐位相等”是强检查而非无条件保证；若出现差异，必须先量化差异和 O2M 指标，再区分正常的裁剪耦合与实现异常。

### 6.7 新候选：质量路由的双向分配头协同（待查重、待 D1/D2 后验证）

D0 不只说明 O2M 教师会变弱，还显示角色可能反转：训练早期多数共同位置是 O2M 更优，而末轮只有 24.4% 的共同位置仍是 O2M 更优。由此形成一个比“固定 O2M 教 O2O”更强的可证伪假设：

> 在同 anchor、同 GT 的共同正样本上，由 GT 锚定质量更高的一侧临时担任 stop-gradient 教师；O2M 更优时迁移到部署 O2O 头，O2O 更优时反向迁移到 O2M 头及其共享特征路径，让教学方向随分配头的真实学习状态自行切换。

该设计与普通双向 KL 的关键区别应是：分类与定位分别比较 GT 锚定质量，方向由相对质量逐位置、逐任务路由，两向 mask 互斥；O2M→O2O 只更新 O2O Head，而 O2O→O2M 会通过 O2M 学生路径更新 Head 与 Backbone/Neck。由于锁定代码中的 O2O 监督自身不能穿过 `detach()` 更新共享特征，反向路由构成一条 **sparse-to-dense gradient bridge**：把稀疏部署头知识间接送入共享 trunk。它仍保持推理结构不变。

目前只把它记为候选 D3，不宣称创新成立。集中查重尚未发现“同一双分配模型 + exact same-anchor/same-GT + per-anchor/per-task 选优 + stop-gradient 单向路由 + 方向自切换”的完全同类方法，但每个宽泛组成部分都有强先例。风险包括双向反馈导致共适应、O2M/O2O soft target 不一致、反向迁移破坏 O2M 的密集多样性，以及观察到的角色反转只是 ProgLoss 预编程权重迁移的结果。进入实现的条件是 D1/D2 证明教师方向与选择确实是性能敏感变量；正式验证必须增加 ProgLoss 固定/关闭对照，并证明 O2O→O2M 允许回传共享 trunk 优于在 O2M 输入处再次 detach 的 head-only 版本。

### 6.8 D3 细化：Task-Decoupled Quality-Routed Gradient Bridge（TQGB）

进一步方法审计把 D3 收敛为“逐任务质量路由 + 监督兼容投影 + detached 部署头反向梯度桥”，而不是泛泛的双向蒸馏。YOLO26 的拓扑写为：

\[
F=N(B(x)),\qquad u_m=H_m(F),\qquad u_o=H_o(\operatorname{sg}(F)).
\]

在 `same anchor + same GT` 的共同正样本集合上，分类与定位独立判断教师方向：O2M 更好时执行 O2M→O2O，只更新部署头；O2O 更好时执行 O2O→O2M，使梯度经 O2M student 进入 O2M Head、Neck 与 Backbone。一个位置可以出现“O2M 教分类、O2O 教定位”，这是 task-decoupled 的核心。

为防止 KD 与 student 自己的 GT/TAL 监督冲突，分类教师概率不直接使用，而是投影到 student 当前概率与 soft target 的闭区间：

\[
\tilde p_{t\rightarrow s}=\operatorname{clip}\left(\operatorname{sg}(p_t),\min(\operatorname{sg}(p_s),y_s),\max(\operatorname{sg}(p_s),y_s)\right).
\]

使用 Bernoulli KL/BCE 时，原监督梯度为 `p_s-y_s`，KD 梯度为 `p_s-\tilde p`；该投影保证两者方向不冲突。有冲突的教师不是整点丢弃，而是被截断成安全目标。定位侧把框写成相对 GT 的中心偏移与 log 尺度残差，并把教师残差逐坐标投影到 student 残差与 0（GT）之间，再做 SmoothL1。它只保证在该残差空间中不把坐标推离 GT，不能声称与官方 CIoU 梯度严格同向。

最有研究价值的部分是反向路由：O2O 自身监督因 `detach()` 不能更新共享 trunk，而 O2O→O2M 会形成一条受控的 deployment-to-trunk gradient bridge。稳妥的潜在表述是：**面向 detached deployment head 的双分配检测器，进行 exact-correspondence、task-decoupled 的质量路由，并通过监督兼容投影实现安全方向自适应迁移。** 不能声称首个双头自蒸馏、首个动态教师、首个质量门控、首个 O2M/O2O 对齐或首个零推理开销 KD。

实现前的 go/no-go 探针必须只用训练划分和固定 checkpoint，分别统计分类/定位的 O2M→O2O 与 O2O→O2M 有效覆盖、两任务教师方向不一致率、投影后仍非零的迁移率，以及 reverse route 随训练是否增加。若中后期反向有效覆盖长期低于约 1%，停止双向方案，回到训练状态感知的单向调度。最小消融至少包含：仅分类动态双向、分类+定位完整版本、阻断 reverse 到 trunk 的 HeadOnly、He et al. 式对称 L1、取消投影、以及 ProgLoss 固定权重控制。还需记录分组梯度范数/余弦和全局梯度裁剪触发率。

## 7. 文献核验与创新边界

### 7.1 证据分级

- A 级：正式会议/期刊官网、作者官方代码、官方模型论文。
- B 级：arXiv/SSRN 等预印本，有可检查方法和实验但尚未正式同行评审。
- C 级：二手网页、仅标题/摘要可见、无法确认正文或来源。
- 用户提供的汇总材料只作为检索入口；只有逐篇确认作者、标识符、全文方法、实验与发表状态后，才作为论文依据。

### 7.2 直接 YOLO26 文献的核验结果

| 工作 | 核验状态 | 对本项目的实际价值 |
|---|---|---|
| [YOLO26 官方论文](https://arxiv.org/abs/2606.03748) | 官方 arXiv 与源码；目前不能称顶会/顶刊论文 | 定义基线、STAL、ProgLoss、端到端路径与官方 KD 边界 |
| *Toward a Deeper Understanding of YOLO26* | *Applied Sciences* 2026，真实论文 | 提供 COCO 逐模块消融线索；增益较小且主要是单种子，不能当稳健性证明 |
| *YOLO26 Under the Microscope* | SSRN working paper，DOI `10.2139/ssrn.6890055`，未同行评审 | 提示 STAL×ProgLoss 可能存在负交互；只能作为待复现实验假设 |
| Fire-YOLO26 / AquaYOLO26 | 分别发表于 *Frontiers* / *Symmetry*，均为领域任务改造 | 可学习完整实验组织，但其 STAL/ProgLoss 结论不能直接外推到通用 VOC |
| *DINOv3 Meets YOLO26* | *Computers and Electronics in Agriculture*，真实论文 | Foundation feature 融合的领域案例；实现与算力成本高，不适合作为首轮 |
| MD-YOLO | *Algorithms*，真实论文 | 文中 DFL 相关表述与官方 `reg_max=1` 事实存在疑点，引用前必须复核实现 |
| DSG-YOLO26 | *Signal Processing: Image Communication*，真实论文 | 仓库当前信息有限，暂不能依赖其复现性 |
| SDD-YOLO | 仅 arXiv，正文信息存在内部不一致 | 高风险证据，不作为方法成立或数值对比依据 |

结论：目前没有证据支持“已有大量 YOLO26 顶刊顶会改进论文”这一说法。直接 YOLO26 文献适合了解问题与实验写法，但创新查重必须回到更成熟的端到端检测、标签分配和蒸馏谱系。

### 7.3 与当前构想最接近的工作及碰撞点

| 既有工作 | 已占据的核心思想 | 对当前方法的限制/启发 |
|---|---|---|
| [YOLOv10，NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html) | 一致双标签分配，以 O2M 辅助 O2O | “双分配协同”本身不新；必须证明质量门控迁移解决了什么额外冲突 |
| [O2F，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_One-to-Few_Label_Assignment_for_End-to-End_Dense_Detection_CVPR_2023_paper.html) | 动态 soft one-to-few 正样本分配 | 软化/动态正样本数量不能作为我们的独占贡献 |
| [H-DETR，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Jia_DETRs_With_Hybrid_Matching_CVPR_2023_paper.html) 与 [DATE](https://arxiv.org/abs/2211.13859) | O2O/O2M 混合匹配或双分配训练 | 再次说明“加一个密集辅助分支”不具新颖性 |
| [DEIM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html) | Dense O2O、Matchability-Aware Loss 和质量过滤 | 质量感知匹配/过滤赛道已有强工作；我们的贡献要落在双头间的非对称、同-GT迁移及其冲突证据 |
| [CrossKD，CVPR 2024](https://arxiv.org/abs/2306.11369) | 外部 teacher/student 的 cross-head prediction mimicking，减少监督冲突 | “避免蒸馏与 GT 监督冲突”已有先例；需强调本方法是单模型内部 O2M→O2O、同 anchor/同 GT 且部署零开销 |
| [BTKD++，IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02924-x) | 以教师分类正确性和框 IoU 划分检测正样本的可靠/偏置知识，并过滤、纠正不可靠教师信号 | “根据 GT/IoU 判断教师可靠性”也已被占据，不能主张首个可靠性门控检测 KD；当前剩余区别是同一模型内共同训练的 O2M→O2O、同 anchor/同 GT、相对教师-学生质量优势及 O2O 一致分配场景 |
| [Dual-head self-distillation，Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S0925231226011471) | YOLO11 单代双头自蒸馏；主 Head 通过分类/定位目标指导带轻量 Neck 的辅助 Head，部署删除辅助路径 | “单模型双头自蒸馏、零推理开销”已被占据；当前区别只能落在原生 O2M/O2O 分配头、相反的 O2M→部署 O2O 方向、同 GT 对齐与相对质量选择 |
| [DATE，Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320325005370) | 训练期并行 O2M/O2O 分配、推理只保留 O2O，用稠密监督改善收敛 | “训练期 O2M 辅助 O2O”本身已非常成熟；必须用显式跨头迁移、负迁移诊断和门控消融证明新增价值 |
| [RT-SFOD，arXiv 2026](https://arxiv.org/abs/2606.31834) | 在源数据不可用的域适应中选择性融合 O2O/O2M 预测生成伪标签 | 虽任务不同且不是同锚点监督 KD，但已说明“双头预测要选择性使用”不是独占概念；不能做过宽的新颖性表述 |
| [Learning Efficient Object Detection Models with KD，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html) | 当学生框回归已优于教师一定 margin 时停止额外模仿 | “学生优于教师时不再蒸馏/按相对质量选择”早已有先例；我们的相对优势门控本身不能单独构成创新，只能作为原生双分配协同机制的一部分 |
| [A Good Student Is Cooperative and Reliable，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_A_Good_Student_is_Cooperative_and_Reliable_CNN-Transformer_Collaborative_Learning_ICCV_2023_paper.html) | 在 CNN/Transformer 语义分割协同中，用 GT 交叉熵判断可靠侧并动态决定双向选择性蒸馏方向 | “由 GT 质量动态选择谁教谁”这一抽象也已有先例；D3 若推进，贡献必须明确限定为 O2M/O2O 分配角色、同 anchor/同 GT 对齐和 YOLO26 detach 梯度拓扑，而不能声称首个质量路由双向蒸馏 |
| [He et al.，JKSUCIS 2025](https://link.springer.com/article/10.1007/s44443-025-00236-7) | YOLOv10+ 的 O2M top-k 与对应 O2O targets 做 box/logit 对称 L1 matching，联合 `L_OM+L_OO+0.5L_align` | 这是“同模型 O2M/O2O 跨头预测一致性”的最近直接先例，阻断首次跨头对齐的主张；其无 stop-gradient、GT 相对质量、winner/loser 或方向切换，且论文描述的梯度拓扑不同 |
| [CoLAD，WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf) | 两个在线检测器按 assignment-cost separability 动态互换 teacher/student，迁移标签分配 | 动态检测教师角色已有先例；D3 的剩余区别是单模型双分配头、逐 anchor/逐任务 logit/box 迁移和 exact correspondence |
| [Self-Competitive Distillation，Algorithms 2026](https://www.mdpi.com/1999-4893/19/4/262) | 两个共同训练实例按即时监督损失选较优者为 detached teacher，只更新较差者，角色动态切换 | 几乎占据“winner teaches loser”的通用原理；D3 不能以动态角色本身主张新颖性 |
| [Task Adaptive Regularization](https://arxiv.org/abs/2006.13108) | 外部检测教师的质量选择回归蒸馏，并将 KD 权重随训练线性衰减 | 质量选择和时间衰减的组合也已有先例；固定 early-to-late KD 日程不足以成为贡献 |
| [What Makes for End-to-End Object Detection?，ICML 2021](https://proceedings.mlr.press/v139/sun21b.html) | 分析 O2M 重复预测与分类 score gap 对 NMS-free 的影响 | 支持“不能无条件把 O2M 分布灌给 O2O”的研究动机 |
| [YOLOv6 v3.0 / DLD](https://arxiv.org/abs/2301.05586) | 训练期分布定位教师，推理移除重分支 | “训练期分布定位、推理零开销”已被占据，不能作为独立主创新 |

另有两条明确边界：YOLO26 官方方案已经包含 KD，因此“给 YOLO26 加知识蒸馏”不新；YOLO26-Refined 已恢复 DFL/`reg_max=16`，因此“首个给 YOLO26 加分布定位”也不可主张。

### 7.4 当前可辩护的创新假设

当前不主张“首个双头自蒸馏”。更窄、可验证且与既有工作有区别的假设是：

> 无门控 O2M→O2O 迁移会因重复预测和标签分配差异而污染唯一预测；只在同 anchor、同 GT 且 O2M 的 GT 锚定质量明确更高时做非对称迁移，能改善 O2O NMS-free 性能而不增加推理成本。

这个表述仍需 D0→无门控 KD→质量门控 KD 的完整消融、至少多种子稳定性，以及后续 COCO/另一通用数据集外部验证。VOC 只承担快速形成和证伪机制假设，单个 VOC 结果不足以支持正式“通用改进”结论。

## 8. 实验登记表

| ID | 日期 | 改动 | 预算 | 状态 | 主要结果 | 决策 |
|---|---|---|---|---|---|---|
| B0-O2M-Y26N-VOC-E150-S42 | 已完成 | 官方基线，O2M+NMS | 150 ep | 完成 | mAP50-95 0.58705 | 锁定 O2M 基线 |
| B0-O2O-Y26N-VOC-E150-S42 | 2026-09-18 | 同一权重，O2O NMS-free 补测 | eval | 完成 | mAP50-95 0.57689 | 锁定 O2O 基线 |
| UT-QGAT-V0 | 2026-09-18 | gain=0 等价性、空目标、梯度隔离 | unit | 完成 | 全部通过 | 允许进入 smoke |
| SMOKE-QGAT-G05-E1-S42 | 2026-09-18 | 宽松门控强制激活分类 KD | 1 ep | 完成 | DDP/BF16 稳定；`kd_active=0.8729` | 仅验证工程路径，不比较精度 |
| D0-DIAG-Y26N-VOC-E150-S42 | 2026-09-18 | gain=0，仅增加双头诊断 | 150 ep | **完成** | O2O 0.57689；O2M 0.58707；权重与 B0 逐张量完全一致 | 基线复现通过；进入监督冲突探针 |
| AUDIT-QGAT-V0 | 2026-09-18 | 独立审计索引、梯度、DDP 统计与 EMA 验证语义 | code audit | 完成 | 无致命错误；发现 soft-target 梯度冲突与统计语义问题 | D0 继续；正式 KD 前先修复并做离线冲突探针 |
| PROBE-SUPERVISION-TRAJECTORY | 2026-09-18 | 7 个固定 checkpoint、VOC07 test、无增强/无更新 | diagnostic | 完成 | 质量候选的方向一致率 99.68%–100%，teacher-closer 99.17%–100% | 额外监督门控降级为附录；主线保留质量门控与时间状态 |
| CAL-KD-SCALE | 2026-09-18 | 固定 checkpoint 测量 gain=1 的原始 KD 量级 | diagnostic | 完成 | 质量门控 KD/cls 为 1.96%–3.39%；无门控末轮 9.42% | 锁定统一 `gain=0.5`，不做精度驱动扫参 |
| B50-NO-KD-E50-S42 | 2026-09-18 | gain=0 的 50 轮短训控制 | 50 ep | **完成，待健康硬件复现** | 固定末轮 O2O mAP50-95 0.51276 | 暂作 50 轮公平对照，不作论文最终数值 |
| D1-UNGATED-MATCHED-KD-E50-S42 | 2026-09-18 | 同 anchor/同 GT，无质量门控分类 KD | 50 ep | **完成，待健康硬件复现** | 固定末轮 O2O mAP50-95 0.51211，较 B50 -0.00065 | 暂未见正收益；结论仅作方向筛选 |
| D2-QUALITY-GATED-KD-E50-S42 | 2026-09-18 | 相对质量门控的非对称分类 KD | 50 ep | **硬件阻塞，未产生 epoch** | 三次原设置及一次 recovery 启动均因主机不可纠正 RAM 错误触发 SIGBUS | 管理员处理内存硬件前不再强行训练 |
| D2-QUALITY-GATED-KD-E50-S42-R1 | 2026-09-18 | 与 D2 原配置完全相同；服务器所有者知情授权后的重启 | 50 ep | **硬件失败，未产生 epoch** | 修正研究源码路径后，workers=4 启动于 18:58 再次触发多条 MCE/SIGBUS | 保留失败日志；不作方法结论 |
| D2-QUALITY-GATED-KD-E50-S42-W0-NOPIN | 2026-09-18 | workers=0 且关闭 pinned host memory 的故障隔离 | 50 ep | **硬件失败，未产生 epoch** | 19:02 的 SIGBUS 直接击中 DDP Python rank，且同时影响无关进程 | 证伪“仅 DataLoader workers/shm 故障”解释 |
| D2-QUALITY-GATED-KD-E50-S42-NUMA1-W0 | 2026-09-18 | workers=0、关闭 pin memory，CPU/内存绑定 GPU4–7 本地 NUMA 1 | 50 ep | **按计划中止于 epoch 1 后** | epoch 1 完整落盘：68.44 s，无本次启动新增 MCE；证明 NUMA 规避可行 | 用户要求恢复原始数据管线；保留为故障隔离证据，不作精度对照 |
| D2-QUALITY-GATED-KD-E50-S42-NUMA1-W4 | 2026-09-18 | 原始 workers=4/pinned memory，仅增加 GPU4–7 本地 NUMA 1 绑定 | 50 ep | **完成；faulty-host exploratory** | 固定末轮 O2O 0.51584，较 B50 +0.00308、较 D1 +0.00373；O2M 0.53164，较 B50 +0.00340 | 窄幅通过预注册筛选线；进入多种子与强度/裁剪控制，不作论文最终数值 |

D0 目录：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/d0_diag_yolo26n_voc_e150_g4567_s42`。固定使用 4×GPU 4–7、batch 64、MuSGD、seed 42、`nms=false`、`o2o_kd_gain=0`；因此诊断门控即使记录潜在激活位置，也不会产生 KD 梯度或改变训练目标。

运行中复现核对：截至已同步的 epoch 50，D0 的 `train/o2o_box_loss`、`train/o2o_cls_loss`、`train/o2o_l1_loss` 与原始 B0 CSV 对应三列逐轮完全相同，三列最大绝对差均为 0（比较的是日志保存精度）。这为“新增诊断没有改变当前实际训练轨迹”提供了比单轮 smoke 更强的证据；最终仍以 150 轮权重和指标复核为准。

epoch 50 阶段快照（仅观察，不作最终结论）：O2O NMS-free 的 P/R/mAP50/mAP50-95 为 0.7322/0.6728/0.7379/0.5258；共同同-GT率 94.0%，O2M 质量更优率 48.8%，默认质量门控候选率 15.9%，共同位置平均质量差 0.0085，双头框 IoU 91.6%。验证侧 O2M 质量更优率仅 31.3%，门控候选率 12.3%，进一步表明 O2M 不是全局可靠教师。完整中途诊断另见 `D0_DIAGNOSTIC_REPORT.md`；所有数值仍可能随剩余 100 轮变化。

### 8.1 D0 完整结果与可复现性结论

D0 已完整训练 150 epochs，用时 1.541 h。对 `last.pt` 使用相同单卡、batch 64 评测设置显式切换两个推理路径，结果如下：

| 路径 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| O2O NMS-free | 0.7910313 | 0.6971400 | 0.7786028 | 0.5768901 |
| O2M + NMS | 0.7951196 | 0.7199211 | 0.7935888 | 0.5870695 |

O2M 相对 O2O 高 0.0101793 mAP50-95、0.0149860 mAP50 和 0.0227811 Recall，确认双头差距稳定存在。更重要的是：

- D0 与原 B0 的 150 轮 O2O box/cls/L1 日志逐轮完全相同，三列最大绝对差均为 0。
- 两个 `last.pt` 共 708 个状态张量，差异张量数为 0，最大绝对差为 0，键集合也完全一致。
- 因此 D0 的诊断改动没有改变模型权重，D0 可作为后续消融的可信机制基线。

双头关系呈现清楚的阶段变化：epoch 5 时训练侧 O2M 更优率 77.9%、门控候选率 18.6%；epoch 50 分别为 48.8%/15.9%；epoch 100 为 41.2%/14.2%；epoch 150 仅为 24.4%/9.7%。与此同时同-GT率从中期约 84%–94% 升到 97.9%，双头框 IoU 升到 96.3%，但共同位置平均质量差最终变为 -0.0116。验证侧末轮 O2M 更优率更低，仅 16.2%，质量门控候选率 4.4%。

对完整训练曲线进一步按固定规则查询发现：训练侧质量门控候选率在 epoch 7 达峰值 20.379%；O2M 更优率在 epoch 40 首次低于 50%；共同位置平均质量差到 epoch 118 才首次转负。该结论只使用训练统计，可用于形成“早期教师窗口”的后续假设；但它还不足以证明某个硬截止轮次最优。

这给出一个新的、可检验的机制假设：**O2M 作为 O2O 教师的可靠性会随训练推进明显衰减；静态、全程、无门控 KD 会在后期大量传递较差知识。** 因而后续方法不只需要“空间上的质量门控”，还可能需要由教师优势/候选覆盖驱动的状态感知时间调度，而不是仅按 epoch 固定 fade。该想法先作为候选 B1，不与第一轮监督一致性门控同时改动，避免无法归因。

### 8.2 D1 无质量门控 KD：固定 50 轮结果

D1 已按预注册配置完成 50 epochs，用时 1908.36 s（31.81 min）。固定第 50 轮 O2O NMS-free 结果为：P 0.73689、R 0.64791、mAP50 0.71893、mAP50-95 0.51211。完整曲线和原始配置已归档，当前不以训练中间的最佳轮次替代固定末轮。

末轮的 KD scale 已按预注册的 fade 日程降到 0，因此最终轮不含额外 KD 损失；训练期共同位置全部接受无门控 KD，但末轮仅 50.844% 的 KD 更新方向与 O2O soft target 监督一致、44.757% 的 O2M 教师更接近该 soft target。这进一步确认“无门控信号包含大量潜在冲突”，但是否造成精度损害必须等 B50 同预算控制和 D2 质量门控均完成后才能判断，不能拿原 150 轮 D0 的 epoch 50 作对照。

### 8.3 B50 公平对照与第一项短训差值

B50 按完全相同的 50-epoch ProgLoss 日程完成，用时 1865.69 s（31.09 min）。固定第 50 轮 O2O NMS-free 结果为：P 0.74053、R 0.65166、mAP50 0.72216、mAP50-95 0.51276。D1 相对 B50 的固定末轮差值为：P -0.00364、R -0.00375、mAP50 -0.00323、mAP50-95 -0.00065。

因此，无质量门控 KD 在本次单种子短训中没有正收益，但 AP50-95 的负差只有 0.00065，远不足以单独证明稳定负迁移。它的主要价值是提供机制对照：KD 激活阶段的平均监督方向一致率只有约 60.3%、教师更接近 soft target 的比例约 58.7%。是否由质量选择恢复收益，仍必须由 D2 同设置实验回答。

### 8.4 D2 启动故障：主机物理内存错误，不是方法失败

D2 在任何有效 epoch 产生之前，三次原配置启动与一次 `workers=2` recovery 均收到 `SIGBUS`。只读系统检查显示内存、磁盘与 `/dev/shm` 均有大量余量，而内核日志明确记录：`Uncorrected hardware memory error`、`Memory failure`，并向 `python`、`pt_data_worker` 和 `pt_data_pin` 发送 SIGBUS；错误涉及多个物理页和 CPU 核。由此确认故障来自服务器 RAM/内存子系统，不是 D2 loss、GPU 显存或共享内存容量。

D1 与 B50 均在本次开机首次用户态内存 MCE 记录出现前完整结束，但随后取得的轮转 `kern.log` 证明 9 月 16–17 日已存在 fatal/UE/SIGBUS，且本次开机启动阶段也有 fatal 硬件记录。因此二者虽完整落盘、校验和一致且曲线无明显异常，也只能作为暂定筛选结果，正式论文数值必须在健康硬件上复现。所有失败启动目录只包含参数或尚未形成有效 epoch，已/将移动到 `runs/_failed_stubs/` 保留审计证据，没有伪造或续接为有效实验。为避免静默权重/数据损坏，在管理员完成硬件检查、故障内存下线或重启并确认 MCE 不再出现前暂停远程训练；本地文献、方法和分析工作继续。

用户要求谨慎再试后，额外执行了一个仅数秒、限定物理 GPU 4–7 的四进程 CUDA/NCCL 卷积与 all-reduce smoke test。它在 rank 3 再次以 `SIGBUS` 退出；内核于同一时刻新增 `Uncorrected hardware memory error`，并明确记录 `Sending SIGBUS to python ... due to hardware memory corruption`。因此故障仍在活动，不是历史残留或 Dataloader 特例；不再启动 D2，等待管理员处理。

用户随后在自己的终端只读复核，18:22:35–18:24:17 仍出现连续新 MCE，且 `sed`、`fgrep`、`modinfo`、`tokio-rt-worker` 等普通进程也被 `due to hardware memory corruption` 的 SIGBUS 终止。这把根因进一步锁定为整机 RAM/内存子系统活动故障，与 YOLO、Conda、Dataloader 和特定 GPU 无关。共享服务器上停止一切研究命令与写操作，立即通知管理员。

完整 24 小时日志进一步给出一致硬件定位：47 次不可纠正用户态内存错误对应 47 次 SIGBUS，至少 14 次 EDAC UE；所有可定位源均为 `P0MC0 / EDAC MC0 / mc#0 csrow#1 channel#0 / Unified Memory Controller`。这使故障范围从泛化的“主机 RAM”收窄到 Processor 0 的 Memory Controller 0 对应内存路径，具体 DIMM/通道仍需管理员结合 BMC/EDAC/主板映射确认。启动后约 10 秒还存在 `fatal`、`Processor Context Corrupt` 的处理器错误记录，说明平台健康问题早于本研究训练。

轮转 `kern.log` 将时间范围扩展到前两天后，结论再次升级：9 月 16 日启动即有 fatal DRAM/processor record；9 月 17 日有 42 次 EDAC UE、20 次用户态不可纠正错误和 20 次 SIGBUS，分布于 `P0MC0/P0MA0/P1MI0/P1MG0` 四条路径、跨 EDAC MC0 与 MC1。因而不能再把故障收窄为今天单个 `P0MC0` DIMM，也不能用简单 NUMA 绑定规避；这是整个平台级硬件问题。

### 8.5 服务器所有者授权后的临时实验政策

用户已将上述硬件证据告知服务器所有者（其导师），所有者明确答复可继续使用、暂不处理硬件。基于这项授权恢复探索性实验，但它不改变数据可信度判断：本机结果统一标注为 **faulty-host exploratory**，不可直接作为论文最终数值；所有入选方法都必须在健康硬件从随机初始化完整复现。每次作业启动前后记录 GPU 占用、进程状态与新增 MCE/SIGBUS；出现 SIGBUS 时保留失败现场，不从可能损坏的 checkpoint 续训，不通过反复重启制造选择性存活结果。为维持公平性，D2-R1 使用原始 D2 配置（workers=4），而不是排障时改为 workers=2 的 recovery 配置。

同时完成 TQGB 的只读 go/no-go 探针脚本设计：在固定 checkpoint、无增强、无反向传播条件下，分别统计分类/定位的 O2M→O2O 与 O2O→O2M 路由覆盖、两任务方向不一致率、监督线段投影后的有效信号保留率。该探针先判断双向梯度桥是否有足够证据密度，再决定是否投入 50/150 轮训练；脚本需在健康或经所有者授权的主机上运行后才能形成数值结论。

### 8.6 D2 故障隔离与 NUMA 1 恢复运行

所有者授权后的原配置重启先暴露了一个独立环境问题：父进程没有指向隔离研究源码，启动器按设计在训练前主动中止，未创建结果或触发 MCE；补上锁定的 `PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research` 后才进入真实 D2 启动。真实启动在 18:58 再次于首个 epoch 前触发多个不可纠正内存错误，rank 0 收到 SIGBUS。随后使用 `workers=0` 并增加可审计的 `ULTRALYTICS_PIN_MEMORY=0` 开关，仍在 19:02 由 DDP rank 1 的 Python 主进程触发 SIGBUS；同秒附近 `MainThread` 和无关 `conda` 也被硬件内存损坏终止。因此 workers、pinned-memory 和 `/dev/shm` 都不是根因。

NUMA 拓扑核对给出一个暂时可用但不保证安全的规避依据：物理 GPU 0–3 的 CPU affinity 属于 NUMA 0，GPU 4–7 属于 NUMA 1；当前连续故障 PFN 约位于 64–128 GiB 地址段，与 NUMA 0 对应，而 DDP 在 NUMA 1 空闲内存低时会回退分配到 NUMA 0。用 `numactl --cpunodebind=1 --membind=1` 将父进程及其 DDP 子进程限定在 GPU 4–7 本地 NUMA 1，同时保留 workers=0/no-pin 后，D2 首次真正进入训练。epoch 1 已完整训练和验证并落盘：耗时 68.44 s，显存约 2.6 GB/卡，O2O P/R/mAP50/mAP50-95 为 0.00021/0.00317/0.00001/0；前 5 轮按预注册 warmup 禁用 KD，因此 `kd_scale=0`、`o2o_kd_loss=0` 正常。本次 NUMA 1 启动到 epoch 1 落盘期间未观察到新增 MCE/SIGBUS。

该恢复不能被解释为硬件修复，也不能删除历史上 NUMA 1/第二插槽曾出现 UE 的证据。由于 workers 与内存放置相对 B50/D1 已变化，这个 run 只用于判断 D2 是否有明显机制/精度信号；即使完成，也不得把小幅差值写成严格公平提升。若它产生值得继续的结果，健康硬件上必须把 B50、D1、D2 用同一原始 workers=4 协议全部从随机初始化重跑。

用户随后要求尽可能恢复原始 `workers=4` 与 pinned memory。workers=0/no-pin 作业在完整 epoch 1 后按计划发送 SIGINT 停止，结果和日志保留，没有续接其 checkpoint。新作业从随机初始化开始，保留 NUMA 1 CPU/内存绑定，但恢复 `workers=4` 和 pinned memory 默认开启；Ultralytics 在四个 DDP rank 上合计创建 16 个 worker。该作业 epoch 1 用时 38.4917 s，相比 no-pin/workers=0 的 68.44 s 快约 43.8%，并且未出现新 MCE/SIGBUS。

更关键的是，D2 的前 5 轮属于预注册 KD warmup，理论上应与同 seed 的 B50/D1 完全同轨迹。对 epochs 1–5 的六项双头训练损失和 P/R/mAP50/mAP50-95 共 10 个字段逐轮比较，D2 相对 B50、D1 的最大绝对差均为 0；其中 epoch 1 的 O2O box/cls/L1 为 3.62606/7.44770/0.06128，O2M 为 4.29480/4.98195/0.08162，P/R/mAP50/mAP50-95 为 0.00038/0.02461/0.00005/0.00001。这提供了当前最强的运行级证据：NUMA 绑定改变的是主机内存放置，不是模型数学条件或随机训练轨迹。warmup 等价性检查已通过；从 epoch 6 开始的轨迹分叉才可归因于质量门控 KD（仍受故障硬件复现限制）。

### 8.7 D2 epoch-10 预定中期读数

在不挑选最佳轮次的前提下，固定读取 epoch 10。D2 的 O2O P/R/mAP50/mAP50-95 为 0.48149/0.40397/0.40674/0.24695；相对 B50 同轮分别为 +0.02785/+0.00048/+0.00777/+0.01108，相对无门控 D1 的 mAP50-95 为 +0.00701。收益目前主要来自 Precision，Recall 基本持平。该轮质量门控候选率为 18.908%，`kd_scale=0.5`，记录到的 `o2o_kd_loss=0.01671`。

训练量的变化提供了一个需要后续解释、不能过度简化的信号：D2 相对 B50 的 O2O box/L1 loss 分别低 0.02585/0.00047，但 O2O cls loss 高 0.07159。检测精度仍更高，说明额外 KD 使“基础分类损失”与最终排序质量并非单调对应，不能通过 loss 大小直接断言方法好坏。当前 +1.108 个 mAP50-95 百分点是值得继续的早期信号，但不是最终结论；固定后续观察点为 epoch 25 和 epoch 50，不使用中途最优轮次替代末轮结果。

### 8.8 D2 完成：固定末轮正信号与边界

D2 已完整训练 50/50 epochs，用时 1862.85 s，并写出 `last.pt`、`best.pt`、5 个阶段 checkpoint、配置和完整 CSV；本地归档与服务器 SHA-256 一致。固定第 50 轮 O2O NMS-free 为 P 0.74001、R 0.64712、mAP50 0.72321、mAP50-95 0.51584。相对 B50 的差值为 -0.00052/-0.00454/+0.00105/**+0.00308**，相对 D1 的 mAP50/mAP50-95 为 +0.00428/+0.00373。D2 因而成为当前第一项固定末轮为正的候选，但只比预注册 +0.003 筛选线多 0.00008，属于边界性通过。

曲线并非全程占优：warmup 后 45 轮只有 10 轮高于 B50，中期多数时间落后；epochs 41–50 的平均 mAP50-95 为 0.51080，较 B50 仅 +0.00116，epochs 46–50 平均为 0.51470，较 B50 +0.00186。随着 KD fade 接近结束，最后 7 轮有 6 轮高于 B50。这提示质量选择可能有用，但当前 gain 或持续时间偏强；不能把末轮正差简化为全程加速收敛。

机制统计与该解释一致：KD 激活期 D2 只选择 18.242% 的共同正样本，其中与原监督方向一致率 99.663%、教师更接近 soft target 率 99.323%；D1 无门控分别只有 60.329% 和 58.749%。但 D2 平均记录 KD loss 为 0.02302，高于 D1 的 0.01187，说明两者并未在总 KD/梯度量级上匹配。因此这轮支持门控假设，却还不能隔离门控与强度的因果贡献。

对三个 `last.pt` 的 O2M+NMS 补测得到：B50/D1/D2 的 mAP50-95 分别为 0.52824/0.52821/0.53164；D2 相对 B50 +0.00340。checkpoint 范围比较也显示共享/O2M 参数发生轨迹分叉。虽然直接 KD 梯度只进入 O2O 分类头，全模型 gradient clipping 仍可能通过梯度范数产生间接耦合，因此不能声称“只有 O2O 头变化”；后续需记录裁剪触发并加入 head-local 或量级控制。

从 NUMA 1 作业启动到三组补测结束，19:14–19:55 未出现新的匹配 MCE/SIGBUS；这只说明该次运行未观察到显式硬件事件，不推翻历史硬件故障。最终判定为 **faulty-host exploratory 的窄幅正信号**：进入健康硬件至少 3 seeds 复现与 150 轮前置控制，不作为可发表性能结论。详细数据见 `D2_FINAL_COMPARISON.md`、`short_ablation_final/` 与 `d2_o2m_scope_eval.json`。

### 8.9 后续研究路线冻结

对 D2 曲线按阶段对齐后发现：epochs 6–10 的平均 KD scale 为 0.30，D2-B50 平均 mAP50-95 为 +0.00350；epochs 11–20/21–30/31–40 的 scale 为 0.90/1.00/1.00，对应差值 -0.01096/-0.00768/-0.00674；epochs 41–50 fade 到平均 0.45 后重新变为 +0.00116。同期训练侧 O2M 整体更优率从 67.81% 下降到 43.44%，共同位置平均质量差从 0.02744 降到 0.00542。这是探索性相关，不是以 test 曲线调参的授权，但足以形成待独立验证的“教师可靠性状态控制”假设。

源码复核确认当前 D2 对 advantage 权重做 selected-count 归一化，并最终再除以 selected count。这会保持不同候选数 batch 的损失尺度，却也使候选覆盖和绝对质量差的下降无法自然减弱 KD；D2 的平均 KD loss 因而约为 D1 的两倍。下一阶段首先做无更新的 reduction、分组梯度和 global-clipping 探针，再做质量选择与总强度匹配的负对照，不能直接把 +0.00308 归因于门控。

从下一方法开始冻结 VOC07 test。研发阶段暂定使用 VOC2012 trainval + VOC2007 train 训练、VOC2007 val 开发；方法锁定后再恢复完整 VOC07+12 trainval，从头训练并只在固定末轮评测 VOC07 test。主候选命名为 S-QGAT：保留 local same-anchor/same-GT quality gate，并按训练期 O2M 相对 O2O 的全局可靠性 EMA 自动衰减 KD。TQGB 只有在训练划分探针显示中后期反向有效覆盖具有足够密度后才进入实现。

执行顺序冻结为：A）无更新 reduction/clipping/TQGB 探针；B）内部 dev 上的 C0 baseline、C1 当前 D2、C2 强度/分母匹配 D2、C3 Random-Matched、C4 S-QGAT 最小因果矩阵；C）唯一胜出版本与 baseline 做 `{0,1,42}` 三组配对 seeds；D）通过后才做健康硬件 150-epoch VOC；E）再进入 COCO2017 YOLO26n，并在成立后扩展 YOLO26s。每阶段均设 go/no-go，详细协议见 `NEXT_RESEARCH_ROADMAP.md`。

## 9. 变更记录

### 2026-09-18

- 用户明确选择“快速研发模式”：继续使用完整 VOC07+12 trainval，并允许每轮 VOC07 test 作为探索性曲线；不以 test 最优轮次报告主结果，所有对照统一固定末轮。该选择覆盖先前立即拆分内部 dev 的执行计划，但不改变最终论文必须在健康硬件采用干净协议复现的要求。
- 完成完整 train split 的 TQGB 无更新探针：分类反向 O2O→O2M 路由从 18.93% 升至 39.17%，定位反向路由末轮仍有 7.11%，约 48%–54% 共同位置的分类/定位原始赢家不同；机制覆盖显著超过 1% go/no-go 门槛。
- 完成固定批次 KD 梯度/裁剪探针：所有 batch 均触发全局 `max_norm=10` 裁剪；D2 在仅保留 14.9%–23.3% 候选时，KD 梯度仍比 D1 大 32%–53%，并使共享更新缩小 0.64%–1.56%。确认 active-count normalization 是首要强度混杂。
- 用训练数据梯度校准预注册并启动 D2-GM：仅将 D2 gain 从 0.5 改为 1/3，其余协议不变；目标是隔离质量选择与总梯度强度。配置 ID 为 `D2-GM-QUALITY-GATED-KD-Y26N-VOC-E50-S42-W4-PIN-NUMA1`。
- 在隔离 staging 副本实现 S-QGAT v1：新增 active/common reduction 和基于上一完整 epoch 全局 `teacher_better` 的因果状态控制；CPU 合成回归测试全部通过，尚未写入正在训练作业使用的源码，避免干扰 D2-GM。
- 完成第二轮最近工作查重并写入 `NOVELTY_POSITIONING.md`：YOLOv10、One-to-Few、Group/Mr. DETR、Co-learning LAD、CrossKD、BTKD++ 与 2026 dual-head YOLO self-distillation 已覆盖宽泛的双头、动态角色、质量过滤和训练期辅助叙事。将可防守边界收窄为“原生双分配角色反转 + 同 anchor/同 GT 因果退休”；若 S-QGAT 精度不足，则升级为分类/定位逐任务双向路由和 deployment-to-trunk bridge。
- 冻结 `S_QGAT_METHOD_SPEC.md`：gain=1/3、warmup=5、ramp=10、无手工 terminal fade，状态只使用上一训练 epoch 的全局胜率，并采用单调退休；首轮固定第 50 轮判定，不依据 test 曲线改规则。
- D2-GM 完成 50/50：固定末轮 O2O mAP50-95=0.51277，相对 B50=0.51276 仅 +0.00001；last-5 均值反而 -0.00144。O2M+NMS 为 0.52878，相对 B50 仅 +0.00054，而原 D2 为 +0.00340。强度匹配后表面收益基本消失，因此 D2-GM 独立方向 no-go，原 D2 不再视为已证实的质量门控提升。完整结论见 `D2_GM_FINAL_COMPARISON.md`。
- D2-GM 前 5 轮相对 B50 的十项损失/精度字段最大绝对差为 0；50 轮训练、旁路评测与归档期间未观察到新增匹配 MCE/SIGBUS。所有 checkpoint、结果、配置和日志已校验 SHA-256。

### 2026-09-19

- 将 S-QGAT v1 从隔离 staging 同步到研究源码；本地/服务器 SHA-256 一致，Git 基线仍为 `9d45ee4`，修改范围仍仅限已允许的 `cfg/__init__.py`、`cfg/default.yaml`、`data/build.py`、`utils/loss.py`。
- 真模型 CUDA 回归测试通过：gain=0 官方 loss 等价，基础梯度相对 L2 误差 0.00114；KD 直接梯度仍只进入 O2O cls head，backbone/O2M/O2O-box 为 0。两进程 CPU DDP 的 reduction、状态同步、单调退休测试全部通过。
- 完成三次工程 smoke：v1 验证当 warmup 参考胜率低于 0.5 时控制器会安全退休；v2 验证四 rank 状态汇总和非零 state scale；v3 仅在 smoke 中将质量阈值/间隔置 0，产生 24.7–40.8 个/批有效候选并记录非零 KD loss，完成真实 DDP 状态控制 + collaborative backward 联合覆盖。正式配置仍固定 0.25/0.05。
- 在 GPU4–7、NUMA1、batch64、workers=4、pinned memory、BF16、seed42、完整 VOC07+12 trainval 上启动 S-QGAT 50 轮筛选；run=`sqgat_state_quality_kd_g033_e50_g4567_s42_w4_pin_numa1`。开跑前 GPU4–7 无计算进程，30 分钟窗口无新增匹配 MCE/SIGBUS。
- 完成 S-QGAT v1 独立代码审计：从头且无 recovery 的当前 run 未发现致命错误；确认 epoch 5 冻结 reference、epoch 6–15 ramp、上一轮训练统计控制下一轮和 DDP epoch-end 汇总均正确。发现 controller 状态未保存到 checkpoint，故任何 resume/NaN recovery 均会使实验失效；同时修正规范分母为 `max(rho_ref-0.5, eps)`，并明确验证侧 `kd_*` 不代表 live controller。完整审计见 `S_QGAT_CODE_AUDIT.md`。
- S-QGAT 50/50 完成：固定末轮 O2O mAP50-95=0.51175，较 B50 0.51276 为 -0.00101，较 D2-GM 0.51277 为 -0.00102；last-5 相对 B50 -0.00088，按预注册明确 no-go。controller 在 epoch 7 首次收缩、epoch 40 完全退休，证明状态机制工作，但固定方向的早期分类迁移没有产生净收益。独立 `last.pt` O2O/O2M 复评为 0.51167/0.53008；O2M 较 B50 +0.00184 但部署 O2O 为负，不能作为成功。50 轮及双路径复评均无新增 MCE/SIGBUS，归档哈希逐项与服务器一致；详细结论见 `S_QGAT_FINAL_REPORT.md`。
- 实现 TQGB 分类最小版本：exact same-GT/same-class 双向路由、教师 stop-gradient、接收头监督线段投影、common-anchor normalization，默认关闭。CPU 合成与真模型 CUDA 测试通过；O2M→O2O 只进入 O2O cls，O2O→O2M 只进入 O2M cls 与 shared trunk，两个 box head 均为零梯度。
- 完成不读取新 test 精度的 TQGB 训练梯度校准。gain=1 时正/反分类路线相对接收 cls 基础梯度最高 1.1855/5.8808，故拆分方向增益并冻结为 0.08/0.015。复测后接收头新增梯度最高 9.48%/8.82%，shared 最高 3.17%，global clipping 额外缩放不超过约 0.61%。允许进入 T1/T2 smoke 与固定 50 轮因果对照。
- 完成 TQGB 分类实现独立审计：当前 T1 无致命错误，exact correspondence、双向 detach、监督线段投影、common normalization 和方向独立增益均正确。收紧表述：当前仅是分类版 TQGB；校准百分比是 checkpoint 级 batch 均值而非硬上界；CSV 路由率为 rank-0 局部均值；global clipping 仍会造成小幅跨分支耦合。完整结论见 `TQGB_CODE_AUDIT.md`。
- 审计后补充 same-GT/different-class 显式合成回归：相同 GT 索引但不同离散类别时，common count 与路由损失均严格为 0，测试通过。梯度校准探针已扩展为逐 batch mean/P95/max/min 输出，等待正式训练释放 GPU 后按原固定 checkpoints 重跑；旧 9.48%/8.82% 数值在此之前继续按“最大 checkpoint 批均值”口径报告。
- 在 T1 固定末轮未知时冻结 `TQGB_DECISION_PROTOCOL.md`：T1/T2 达到筛选线后分别进入 projection、方向、clipping 与三配对 seed 控制；若分类两组均失败，不扫 test 阈值，改测结果前已有 7.11%–18.26% 覆盖且投影存活率 100% 的 localization-only O2O→O2M bridge；再失败则整条 TQGB 停止并换机制。
- TQGB-T1 完成 50/50：固定末轮 O2O P/R/mAP50/mAP50-95=`0.74619/0.65231/0.72862/0.51946`，相对 B50 的 mAP50-95 `+0.00670`；last-5/last-10 分别 `+0.00525/+0.00505`，不是单轮尖峰。独立 `last.pt` O2O/O2M 复评为 `0.5193847/0.5334306`，相对 B50 `+0.0064838/+0.0051909`。逐类别 O2O 为 12 类升、8 类降；完整结果见 `TQGB_T1_FINAL_REPORT.md` 与 `tqgb_t1_final/`。
- T1 训练与四次独立复评无新增 MCE/SIGBUS；`last.pt` SHA-256=`fdba78326273242130e3ef243b4164ff174a708b4d19d2ca406595d768afe68a`，关键归档远端/本地哈希一致。裁决为单 seed 机制筛选 go，不升级为论文结论。
- 在未修改 T1 核心源码且哈希核对一致后，顺序启动 TQGB-T2 双向分类路由 50 轮：同一 0.08 M→O 加上预冻结 0.015 O→M，继续 GPU4–7、NUMA1、workers=4、pinned memory、BF16、seed42。T2 回答 reverse deployment-to-trunk bridge 是否在 T1 之上贡献。
- 完成 head 之外的广域创新筛选并写入 `BROAD_DIRECTION_SCREEN.md`。优先候选为 PBGA（金字塔边界非冲突梯度仲裁），备选 SARC（分配校准可重参数化感受野）和 ASAC（分配稳定性增强课程）；均先做训练侧零更新探针和匹配成本控制，不做普通注意力/大核堆叠。PBGA-P0 已在隔离 staging 开始实现，不占用正式训练 GPU、不修改运行源码。
- T1 独立结果审计未发现作废级错误，但指出旧 B50 并非在最终 TQGB 源码快照下完整重跑，且 O2M 正差可能来自 global clipping/轨迹分叉，不能视为 trunk bridge 证据。已预注册 `TQGB-T0-CURRENT-NO-ROUTE` seed42 完整 50 轮控制，顺序排在 T2 之后、T1-NP 之前；详细审计见 `TQGB_T1_INDEPENDENT_AUDIT.md`。
- SARC 探针实现审计发现 `get_dataloader(mode="val")` 会把 workers=4 自动翻倍成 8；正式 T1/T2 train loader 不受影响。PBGA 和新版 TQGB 梯度探针均已改为直接构建固定无增强 loader，强制断言实际 workers=4 与 pinned memory。旧 TQGB 校准的数学批次结果保留为预注册来源，但其 loader workers 口径已在方法/审计文件中纠正，待 GPU 空闲后按新版重跑。
- PBGA-P0 与 SARC assignment-histogram 两个隔离探针已实现并同步为项目脚本，但未修改 core。PBGA CPU-only projection/norm-cap/surrogate topology 自测通过；SARC 的 JS/gate/panel 五项合成测试通过。两者正式运行仍等待 T2 释放 GPU，且将强制 physical GPU4、actual workers=4、pinned memory、NUMA1、固定 8 batches、无 optimizer/无更新。
- TQGB-T2 双向分类路由完成 50/50：固定末轮 O2O mAP50-95=`0.51685`，相对 B50 `+0.00409`，但相对 T1 `-0.00261`；last-5 相对 B50 `+0.00269`。独立 `last.pt` O2O/O2M 为 `0.5165675/0.5334126`，其中 O2M 与 T1 的 `0.5334306` 几乎相同，O2O 却低 `0.0028172`。反向路由覆盖非零但没有额外收益，按结果前协议不升级 T2-R、多种子或 T2 no-projection；详细见 `TQGB_T2_FINAL_REPORT.md`。
- PBGA-P0 在 B50 五个 checkpoint、固定 8 batches 上完成：P3/P4/P5 total-gradient negative-dot rate 全部为 0，自动裁决 `NO_GO`。反事实 O2O shared gradient 后期远大于 O2M，naive reattach 会强烈改变 global clipping，但不存在 PBGA 所需的负点积，故停止该路线，不投入正式训练。
- SARC assignment-histogram 探针完成并裁决 `GO`：pooled positive-anchor JS(P3,P5)=`0.12339`、JS(P4,P5)=`0.06871`，去重 GT 后三对 JS 仍均大于 `0.02`；相邻 checkpoint 最大同层 JS 仅 `0.007918`，层级结构清楚且时间稳定。正式进入 R1 固定 DW7、R2 无校准 reparameterization、R3 完整 SARC 的因果实现与矩阵准备，探针详见 `BROAD_PROBE_RESULTS_2026-09-19.md`。
- 完成 T1-BRC global-clipping 因果控制的本地 staging：默认仍为 combined clipping；BRC 只用 route-free base gradient norm 计算裁剪系数，再施加到保留完整 route 的 combined gradient。默认关闭、route-off 等价、route-active base 重建和两进程 Gloo DDP SUM 合成测试通过；尚需真模型单卡、NCCL/BF16 四卡 smoke 后才能正式训练，审计见 `T1_BRC_STAGING_AUDIT.md`。
- 在 GPU4–7、NUMA1、batch64、workers=4、pinned memory、BF16、seed42 下启动最终 TQGB 源码快照的 T0-current 50 轮零路由控制；所有 KD/TQGB gains 为 0，run=`tqgb_t0_current_no_route_e50_g4567_s42_w4_pin_numa1`。其首要任务是检验最终源码是否精确复现旧 B50，再决定 T1 因果归因。
- PBGA 的 NO-GO 同时产生一个透明标记为 post-hoc 的备选假设 ANOR：既然 O2O/O2M shared gradient 无负冲突但严重幅值失衡，可研究 agreement-gated、per-level norm-capped、base-referenced clipping 的 O2O feature reattachment。该想法不改变 PBGA 的 NO-GO，也不立即占 GPU；先做查重与训练侧预算校准，详见 `POST_PBGA_ALIGNED_REATTACH_HYPOTHESIS.md`。
- SARC 隔离 staging 完成并由主任务复跑：14/14 CPU 测试通过，dilation 1/2/3 kernel 嵌入误差为 0，R1 与 R2/R3 eval→deploy 的 float64 最大误差为 `4.441e-16/8.882e-16`。`gamma*SiLU(z)` 的 gamma 不可穿过非线性，部署保留 frozen scalar multiply；错误折叠反例差 `0.23363`。融合推理增量为 22,400 参数和 0.07025 GFLOPs，约占融合基线 `0.942%/1.325%`；详见 `SARC_STAGING_REPORT.md`。
- 识别 SARC 的下一层因果缺口：R3>R2 只能证明 calibration KL 有效，不能单独证明 assignment 语义有效。若 R3 首筛为正，最低必要语义阴性控制是保持同一 KL 强度、边际熵和 EMA 时间平滑，仅固定置换 P3/P4/P5 histogram；另以 GT-only/heuristic-level histogram 区分模型 assignment 信息与静态数据集尺度先验。uniform target 只能排除一般均衡/熵正则，单独不足。
- TQGB-T0-current 零路由控制完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.74053/0.65166/0.72216/0.51276`。排除 wall-clock `time` 后，T0 与旧 B50 的 50×75 个共同 CSV 字段逐值完全相同（3,750 个数值、非零差异 0、max abs 0）；`last.pt` 的 shared+O2M、O2O-box、O2O-cls 共 708 个状态张量、2,530,706 个元素也逐元素完全相同。覆盖运行窗口的内核日志无新增 MCE/SIGBUS。由此关闭最终 TQGB 源码在 route-off 时的基线漂移混杂；T1 的单 seed 正差仍需 T1-NP 与 BRC 回答 projection 和 global-clipping 因果。完整审计见 `TQGB_T0_CURRENT_FINAL_REPORT.md`。
- TQGB-T1-NP 同 gain 无投影对照完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.74658/0.64181/0.72337/0.51474`，较精确 T0/B50 为 `+0.00198`，但比 projected T1 的 `0.51946` 低 `0.00472`。projected T1 在最后 10 轮 10/10 高于 T1-NP，last-10/last-5 平均优势为 `+0.00393/+0.00430`；两组路由覆盖接近，T1-NP 并非机制未激活。按结果前固定末轮逻辑，supervision-segment projection 在 seed 42 上获得方向性支持，但仍需非 42 seed、T1-BRC、健康硬件和干净 test 复现。运行窗口无新增 MCE/SIGBUS，完整 run、日志和配置已逐哈希归档；详见 `TQGB_T1_NP_FINAL_REPORT.md`。
- ASAC-P0 v3 通过本地独立审计与远端 staging 的 33/33 单元测试，但真实 CPU provenance preflight 在读取历史 B50 `last.pt` 时按设计硬失败：该 checkpoint 的 `train_args` 缺失冻结字段 `save_dir`。因此未启动任何 ASAC GPU smoke 或正式 80-forward 探针，也未放宽 manifest。该分支退回 v4 设计复核；必须解释最终 `last.pt` 的 strip/metadata 语义并重新独立审计后才能继续，不影响正在进行的 TQGB 配对训练。
- 只读源码与五 checkpoint metadata 核对确认 v3 失败来自 Ultralytics 终态 `strip_optimizer(last.pt)` 的确定性行为：四个 numbered checkpoint 保留 `save_dir`，只有 `epoch=-1` 的 `last.pt` 丢失该字段。完成 provenance v4 窄化修订：仅允许固定哈希、固定路径、终态 stripped `last.pt` 从自身 `.../run/weights/last.pt` 路径恢复唯一 `save_dir`，随后仍与 `project/name`、独立 `args.yaml` 和 run path 交叉验证；任意第二缺字段或非终态缺字段仍拒绝。37/37 本地测试通过，远端真实 preflight 等待正式训练释放资源。
- IGB-P0 external probe 完成本地实现；主任务复跑 7/7 tests 与 `py_compile` 通过。二次核对发现首版错误绑定了 T1-BRC staging 的 `loss.py` 哈希而非完成 B50 的精确 TQGB 源码，已改为精确 `6dae8e...` 并绑定 provenance v4；未把错误版本同步远端或运行 GPU。IGB 仍需远端 exact-source CPU preflight 和真实 BF16 smoke 后才允许正式 640-VJP 诊断。
- 完成 IGB 针对性排重更新：ATSS、ObjectBox、LapNet/PONO、GFL、Bucketed Ranking Loss 与 ICCV 2025 UGS 已覆盖 assignment balancing、per-object normalization、soft quality target、正负不平衡和小目标梯度稳定等宽泛叙事。因此 IGB 新颖性降为 conditional，只保留“固定 native TAL assignment、解析式且 batch-conserving 的 per-GT soft target-mass correction、并胜过 quota permutation placebo”这一窄边界；详见 `IGB_NOVELTY_UPDATE_2026-09-19.md`。
- TQGB 配对 seed-0 的 B50 完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.75173/0.64725/0.72389/0.51359`；epochs 46–50 的 mAP50-95=`0.51158/0.51175/0.51290/0.51269/0.51359`，均值=`0.512502`，对应 last-5 P/R/mAP50 均值=`0.739854/0.654162/0.723438`。CSV 恰好 50 行且 epoch 1–50 连续，日志记录 50 epochs 正常结束；trainer/DDP/loader 退出后，重启前的实时 NVML 显示 GPU4–7 无计算进程。完整 run、训练日志和配置已归档到 `outputs/yolo26_research/runs/b50_no_kd_e50_g4567_s0_w4_pin_numa1/`，11 个远端/本地 SHA-256 全部一致。
- 上述 B50 seed-0 **不是干净硬件窗口**：训练 PID 存活期间 04:36:24–04:36:25 出现 5 条 CPU0/MC21、FRU `P0MC0`、EDAC MC0 `UECC|Deferred|Scrub` 事件，物理地址位于 `0x615bc...–0x615bf...`。未观察到 `Sending SIGBUS`，训练日志也无 traceback、OOM、NCCL、NaN 或 bus error，但不能证明未受静默影响；即使地址/FRU 疑似 NUMA0，也不能据此证明 NUMA1 绑定的进程未受影响。因此该组只标记为 **hardware-contaminated / faulty-host exploratory**，所有论文数值仍须在健康硬件从随机初始化复现。主机于 04:47:05（产物落盘约 4 分钟后）重启；此前的 T1 seed-0 提交没有形成 run，只有 0-byte 日志，不能视为实验失败或重复续跑。
- 主机恢复后于 04:50 从头启动 `projected T1 seed0`，不是 resume；配置固定为 GPU4–7、NUMA1、batch64、BF16、workers=4/rank、pinned memory、随机初始化、M2O gain=0.08、projection=true。到 epoch 20 的 paired 曲线在正负之间波动：epoch 18 为 `+0.00345`，epoch 20 为 `-0.00667`；epoch 20 时 last-5/last-10 paired mean 分别为 `-0.00145/+0.00046`。该快照只用于证明不能挑中间轮，正式结论仍固定第 50 轮；从 04:50 启动到快照时无新增 `Memory failure`、`Sending SIGBUS` 或 EDAC UE。
- 配对完整性复核发现：旧 B50 seed0 与当前 T1 seed0 跨越了一次服务器重启，且旧 B50 末段有 deferred EDAC UE；T1 第 1 轮的路由计数/损失均为 0，但训练损失相对旧 B50 已有约 `1e-5` 的微小漂移，说明 seed 相同也不能把跨开机轨迹视为 bitwise 配对。为避免把重启或 faulty-host 漂移误归因于方法，新增完全零路由的同开机 B50 seed0 重复，冻结配置 SHA256=`1f3588848bca74bf716455bca4a8207fefe939557dc2357a0d5ed4efac9a8632`；排在 T1-NP seed0 后，且不会替代最终健康硬件复现要求。
- `projected T1 seed0` 完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.73036/0.65809/0.71900/0.51144`，相对旧 B50 seed0 的 mAP50-95 为 `-0.00215`；last-5/last-10 paired mean 为 `-0.00220/-0.00306`，最后 10 轮 0/10 获胜。独立 `last.pt` O2O/O2M 为 `0.51145839/0.52846890`，相对 B50 seed0 的 `0.51346802/0.53142374` 分别为 `-0.00200963/-0.00295483`。路由全程真实激活，平均 route rate=`12.447%`、projection survival=`94.367%`，故负结果不是开关失效。seed42/seed0 固定末轮 paired delta 为 `+0.00670/-0.00215`，仅 1/2 为正，均值 `+0.002275`、样本标准差约 `0.00626`；当前明确判定为负复现/跨种子不稳定，不能称稳定提点。训练及四次独立复评窗口无新增匹配 MCE/SIGBUS/EDAC，19 项远端/本地哈希一致；完整报告见 `TQGB_T1_SEED0_FINAL_REPORT.md`。
- 在 T1 seed0 完整归档后，`T1-NP seed0` 通过冻结配置、`train_research.py`、`loss.py`、`default.yaml` 哈希门禁，确认 GPU4–7 空闲且当前开机无合格内存故障后从头启动。PID 66785，NUMA maps 显示 `bind:1`，args 确认 50 epochs、batch64、workers4/rank、BF16、seed0、M2O gain0.08、projection=false；用于回答同开机 projection 因果差异，不因 T1 负结果改参数。
- 接纳用户提出的战略判断：YOLO26 已围绕 COCO 风格通用检测做过联合优化，继续把课题定义为“给 YOLO26n/VOC 刷一点 AP”难度高、泛化与新颖性均弱。结合 T1 seed42 `+0.00670`、seed0 `-0.00215` 的直接反例，正式将论文对象改为可迁移的训练机制；YOLO26 只作为快速发现/证伪平台。执行上完成 TQGB 冻结因果闭环但不救结果式调参；独立主线优先 IGB exact per-GT gradient-budget probe 与 SPAR object-gated subpixel phase-loss probe，SARC 保留为必须胜过同成本 R2 的小预算结构候选。完整战略见 `STRATEGY_PIVOT_GENERAL_MECHANISMS_2026-09-19.md`。
- 冻结 `SPAR_P0_SPEC.md`：只用 B50 train-only panel，hook backbone layer 3/5/7 的 stride-2 pre/post features，以 image-level split 的 frozen ridge probe 测 Haar detail/LL 可解码性，并用一像素平移误差与 native TAL per-GT quality 的关联作第二道门。GO 必须同时出现至少两层的 object-specific detail gap 与 shift-error/low-quality 关联；一般高频难预测、背景同样难预测或完整性不足均 no-go/invalid。P0 通过前不写 SPAR 训练模块。
- 响应用户“是否转向遥感领域”的新思路，完成条件转向评估并写入 `REMOTE_SENSING_PIVOT_ASSESSMENT_2026-09-19.md`。结论不是立即把 YOLO26 改成遥感专用网络，而是将遥感作为放大密集微小目标、切片与方向问题的优先压力场景；VOC 继续承担低成本机制筛选。数据顺序冻结为 DIOR-HBB 起步、AI-TOD-v2 微小目标压力测试、DOTA OBB 方法成熟后再进入。候选优先级为 RS-IGB > 条件性 RS-SPAR > 跨切片边界一致性 > OBB/旋转等变。
- 遥感近期工作排重进一步收紧主张：CVPR 2025 SET、TGRS 2024 ORFENet、TGRS 2025 CADDN 已覆盖频域增强、训练期目标重建与内容感知下采样，普通“遥感 + 频域/重建/下采样模块”不足以成为贡献。IGB 只保留冻结 native TAL、解析 batch-conserving per-GT soft-mass correction、exact VJP 证据和 permutation placebo 的窄边界；SPAR 必须以 object-specific phase loss 与 phase-shuffle 阴性控制证明语义作用。
- 服务器只读容量审计显示遥感数据尚不存在，`datasets` 约 2.2 GB，个人目录所在文件系统剩余约 437 GB；存储不是主要阻碍，但路线 GO 前不盲目下载数据。当前 T1-NP seed0 在该评估期间继续正常运行，未因战略讨论中断。
- `T1-NP seed0` 完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.74126/0.65567/0.72312/0.51303`。相对旧 B50 seed0 为 `-0.00056`，last-5/last-10 为 `-0.00039/-0.00075`；相对 projected T1 seed0 固定末轮为 `+0.00159`，last-5/last-10 为 `+0.00182/+0.00232`。独立 `last.pt` O2O/O2M 为 `0.51332068/0.52876220`；O2O 相对旧 B50 仅 `-0.00014734`，但 O2M 为 `-0.00266154`。路由全程激活且量级与 projected T1 接近，排除开关失效。
- projection 消融在 seed42/seed0 的 fixed-e50 effect（projected - NP）为 `+0.00472/-0.00159`，仅 1/2 为正，均值约 `+0.001565`、样本标准差约 `0.00446`。因此早期“投影获得方向性支持”的单 seed 表述被直接推翻；projection 和 TQGB 当前均是 trajectory-sensitive，不能作为稳定改进。训练与独立复评窗口无新增匹配 MCE/SIGBUS/EDAC；详见 `TQGB_T1_NP_SEED0_INTERIM_REPORT.md`。
- 按用户要求，先完成当前 TQGB 因果闭环，之后只提交下一阶段路线供讨论决定，不在用户决定前下载遥感数据或启动 IGB/SPAR/其他新路线。为量化重启与故障主机自身漂移，已在哈希、GPU 空闲和当前开机硬件门禁通过后启动同开机 `B50 seed0 reboot-r1`；PID `122077`，确认 GPU4–7、batch64、BF16、workers=4/rank（16 total）、pinned memory、随机初始化、全部 TQGB/KD gain=0。

- 建立持续研究总账。
- 锁定数据、硬件、公平性规则和基线 B0。
- 完成官方源码第一轮核查，确认 O2O 特征 detach、STAL 的 top-7→top-1 行为和 ProgLoss 的 0.8→0.1 线性调度。
- 将首轮工作从“直接拼模块”改为“先测量双头状态，再设计状态感知监督调度”。
- 开始核验用户提供的 YOLO26 直接论文与方法谱系。
- 发现旧验证实际使用 O2M+NMS；以相同 `last.pt` 补测 O2O NMS-free，mAP50-95 为 0.57689，比 O2M 低 0.01016。
- 文献查重发现“训练期分布定位头、推理删除”的宽泛表述与 YOLOv6 DLD 高度重合，因此将其从前三优先方向降级。
- 服务器 `/usr/bin/nvidia-smi` 被替换为 2026-08-03 的静态备份输出，不能用于判断实时占用；以后所有开跑前检查改用 NVML/PyTorch 实时查询。
- 完成 VOC 标签与目标尺度审计：train/test 标签均有效；640 静态缩放后 STAL `<16 px` 触发仅占训练 GT 的 0.7339%、测试 GT 的 0.2078%，因此将 STAL 专项创新从首轮降为 no-go。
- 完成直接 YOLO26 论文核验和相邻方法谱系查重。确认 YOLOv10、O2F、H-DETR/DATE、DEIM、CrossKD 与 YOLOv6 DLD 对宽泛创新表述构成先例；将当前假设收窄为同 anchor、同 GT、质量门控的非对称 O2M→O2O 迁移。
- 在隔离研究分支完成 QGAT/QG-AKT v0：增加双头诊断与门控分类 KL，默认 `gain=0`，新增梯度只进入 O2O 头，推理结构不变。
- 单元测试确认 gain=0 等价、空目标稳定以及 Backbone/O2M 零 KD 梯度；四卡 1-epoch 强制激活 smoke test 稳定完成，无 NaN。
- 启动 D0 150-epoch gain=0 完整诊断复现；训练结束前不宣称精度改善，也不依据 smoke test 选择论文结论。
- 完成 QGAT v0 独立代码审计：确认当前 D0 无需中止；发现真实类别索引、EMA 验证 criterion 状态、逐批比例统计及 KD 与 soft TAL 监督方向冲突等问题。将正式方法候选扩展为质量门控、质量+梯度方向一致、质量+教师接近度三种可消融版本。
- D0 完成 150 轮：O2O/O2M mAP50-95 分别为 0.57689/0.58707；150 轮 O2O 损失轨迹及 708 个权重张量均与原 B0 完全一致。观察到 O2M 教师更优率从早期 77.9% 衰减到末轮 24.4%，形成“教师可靠性随训练衰减”的新候选假设。
- 完成 QGAT v1 加固与完整 CPU/GPU 等价、空目标、梯度隔离测试；固定 checkpoint 冲突探针显示额外监督方向/接近度门控在质量门控后几乎完全冗余，因此将其降级，不作为主贡献。
- 新近 IJCV 2026 的 BTKD++ 已在检测中使用基于 GT 正确性和 IoU 的教师可靠性划分；据此进一步收窄创新边界，不再主张首个“GT/质量门控检测蒸馏”。
- 完成 gain=1 的无更新量级校准，仅用训练划分锁定统一 `gain=0.5`；预注册 D1 无门控与 D2 质量门控两组 50 轮筛选实验，并启动 D1。
- 在读取短训最终精度前发现 ProgLoss 随总 epochs 改变，原 150-epoch D0 的 epoch 50 不能作为 50-epoch 方法对照；补充预注册 B50 无 KD 控制，避免调度混杂。
- D1 与 B50 完成固定 50 轮：mAP50-95 分别为 0.51211/0.51276，无门控 KD 未见正收益（-0.00065），但单种子差异过小，不能下稳定结论。
- 将 D3 细化为 TQGB：分类/定位逐任务质量路由、监督线段投影和 O2O→O2M 的 deployment-to-trunk gradient bridge；同时锁定 go/no-go 探针与必要消融。
- D2 三次原设置及一次 recovery 均在首个 epoch 前因 SIGBUS 失败；内核证实为不可纠正的硬件内存错误。失败目录保留审计，管理员处理主机内存前暂停训练，避免产生静默损坏结果。
- 谨慎的四卡 CUDA/NCCL smoke test 也触发新的不可纠正内存 MCE 并失败，进一步确认主机硬件仍不安全；停止所有远程训练重试。
- 用户端只读复核显示随后连 `sed`、`fgrep`、`modinfo` 等普通进程也持续被硬件内存损坏触发 SIGBUS；故障为整机级活动事件，需立即由管理员接管。
- 用户将硬件故障告知服务器所有者后获得“可继续使用、暂不处理”的明确授权；恢复 D2 探索性训练，但新增 faulty-host 标记、MCE 审计和健康硬件从头复现要求。
- 完成 TQGB 路由/投影 go-no-go 探针脚本，先测有效反向覆盖与逐任务方向分歧，避免在未证实机制覆盖前盲目长训。
- 进一步证伪 DataLoader/shm 根因：workers=0 与关闭 pinned memory 后 DDP Python rank 仍被硬件 SIGBUS 杀死；NUMA 拓扑显示 GPU 4–7 本地于 NUMA 1，而当前故障页集中在 NUMA 0 地址段。
- 使用 NUMA 1 CPU/内存硬绑定后，D2 首次完成 epoch 1 且无本次运行新增 MCE；以 faulty-host exploratory 身份继续，严格结果留待健康硬件统一复现。
- 按用户要求恢复原始 workers=4 与 pinned memory，只保留 NUMA 1 绑定；新 D2 的 epoch 1 六项训练损失和四项精度与 B50/D1 完全一致，且速度恢复到 38.49 s/epoch，现继续运行。
- D2 完整结束 50 轮：固定末轮 O2O mAP50-95 0.51584，较 B50 +0.00308、较无门控 D1 +0.00373；结果窄幅跨过预注册筛选线，但仅为单种子故障主机探索。
- 完成 B50/D1/D2 的 `last.pt` O2M+NMS 旁路评测：0.52824/0.52821/0.53164，D2 较 B50 +0.00340；发现共享/O2M 轨迹也变化，后续增加 KD 强度匹配与 gradient-clipping 控制。
- D2 权重、阶段 checkpoint、CSV 与配置已归档并逐项校验；本次 NUMA 1 训练及补测时段无新增匹配 MCE/SIGBUS。完整结论写入 `D2_FINAL_COMPARISON.md`。
- 完成后续路线设计：识别 selected-count normalization 与 global clipping 两个首要混杂，冻结 VOC07 test，建立无更新探针→内部 dev 因果矩阵→三组配对 seeds→150-epoch VOC→COCO2017 的逐级 go/no-go 方案，并写入 `NEXT_RESEARCH_ROADMAP.md`。

### 2026-09-19 本轮闭环追加

- 同开机 `B50 seed0 reboot-r1` 完成 50/50：固定末轮 P/R/mAP50/mAP50-95=`0.75173/0.64725/0.72389/0.51359`，独立 O2O/O2M=`0.5134680219/0.5314237361`；训练和独立评估窗口无新增 qualifying MCE/SIGBUS/EDAC。
- 排除 wall-clock `time` 后，reboot-r1 与旧 B50 的 50 轮 5,350 个可比数值全部严格相等，非零差异数 0、最大绝对差 0；最终模型 shared/O2M、O2O-box、O2O-cls 共 708 个张量、2,530,706 个元素逐位相等。raw CSV/checkpoint SHA 差异来自耗时和序列化元数据，不是模型数值差异。
- same-boot B50 的 18 个远端源产物共 97,375,400 bytes 已归档，本地/远端 SHA256 全部一致；归档位于 `runs/b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1/`。
- baseline 精确复现关闭了 seed0 的重启/漂移混杂。projected TQGB 两 seed fixed-e50 delta=`+0.00670/-0.00215`，no-projection=`+0.00198/-0.00056`，projection effect=`+0.00472/-0.00159`；三者均只有 1/2 seed 为正。TQGB 正式裁决 STOP/NO-GO，不进入 seed1、150 轮、COCO 或结果驱动阈值搜索。详见 `TQGB_CAUSAL_CLOSURE_2026-09-19.md`。
- 复核用户新增第三轮研究材料：实际锁定源码确认 O2O 训练特征 detach、默认 O2M+NMS / `nms=False` O2O、E2ELoss O2O `topk=7, topk2=1`。在当前真实 `TaskAlignedAssigner` 上运行合成反例，确认两个 GT 初始均有 7 个候选时，冲突消解与 topk2 后可能变成 `[1,0]` 个最终正样本；该结果只证明逻辑可能，真实数据发生率未知。
- 只读清点 VOC07+12 trainval：16,551 张图、40,058 个 GT，平均 2.4203 个目标/图；至少 5/10/20/30 个目标的图像分别为 2,000/301/21/2 张。7,416 张图（44.81%）至少存在一对相交 GT，5,224 张图（31.56%）至少存在一对 IoU>=0.1 的 GT。该统计只说明存在重叠样本，不能替代真实 candidate-competition 测量。
- 起草 `work/yolo26_research/FINAL_COVERAGE_AUDIT_P0_SPEC.md`。协议把 coverage 精确拆为 STAL 几何、task-aligned top-k、共享 anchor 冲突和 topk2 四个阶段，并用原候选图最大匹配基数差定义“理论可恢复缺口”，避免把“丢失 GT 有候选”误当作必然可恢复。预注册完整性、质量约束、总体/定向 GO、NO-GO 与 Phase-1 简单对照；当前状态仅为待用户讨论草案，未实现、未执行、未启动训练。
- 对 FCA-P0 草案完成反向方法学复审：明确六个 checkpoint 仅是同一轨迹时间切片；加入 image-cluster bootstrap、`last.pt` FP32 敏感性、仪器化/原生 criterion 逐位等价和 O2M sidecar control。tiny/dense 子组不再允许直接把方法送入 Phase 1，只能触发另行预注册的独立压力数据只读审计，避免用子组事后救活主假设。
- 补充查重发现 IEICE 2025 已直接研究 UAV 点竞争与标签重分配；OTA、RFLA、DCFL、DDQ 等也覆盖全局/微小目标分配。故候选主线收窄为“STAL/O2O 候选图的最终覆盖审计；若成立，再研究不创建新候选的局部 coverage-first augmenting-path matching”，不能声称首次发现点竞争。
- 新材料同时使旧优先级调整：CQDA 2026 与 IGB 的动态监督预算叙事直接碰撞；detection shift-equivariance、tiny anti-aliasing 和 TOLF 分别收紧 SPAR/shift 与定位噪声路线。下一步只提交 A 覆盖审计、B shift/noise 诊断、C 跨切片系统路线供用户讨论；用户决定前不启动新训练或下载数据。完整复核见 `NEW_RESEARCH_MATERIAL_AUDIT_AND_ROUTE_OPTIONS_2026-09-19.md`。
- 对路线 A 完成最近邻深查并写入 `COVERAGE_ROUTE_NOVELTY_DEEPCHECK_2026-09-19.md`。IEICE 2025 已描述与当前反例高度同型的 intra-group/invasion point competition，并用 extended points + label reassignment 修复；OTA/POTO/One-to-Few/DDQ/YOLOv10/DEIM 也使“全局匹配、O2O 稀疏监督、零推理开销”等宽泛表述无法成立。路线只保留为“STAL 候选覆盖是否能穿过原生 conflict→topk2 成为最终监督覆盖；若不能，在不扩展候选的局部图中做 coverage-first/quality-second matching”的窄命题。当前裁决仍是只做 FCA-P0 诊断，不承诺方法或训练。
- 在未获 FCA-P0 执行确认的边界内完成只读 provenance 预检，写入 `FCA_P0_READONLY_PREFLIGHT_2026-09-19.md`：远端六个固定 checkpoint 与本地归档逐 SHA256 一致；源码仍为 Ultralytics 8.4.146 / HEAD `9d45ee...`，真实 criterion 为 `E2ELoss`、O2M/O2O `10/10` 与 `7/1`、stride `8/16/32`、20 类；train split 仍为 16,551 图/40,058 GT，并冻结图像清单与 label-content 指纹。GPU4-7 无 compute process，当日只有 EDAC 驱动初始化、无新 MCE/SIGBUS/Memory-failure/EDAC-error。未修改远端、未运行 forward/探针/训练。
- 完成用户批准的 FCA-P0 全量零训练审计。首次正式尝试在全量结果产生前因错误的 `S2⊆S1` 假设被保护断言终止；源码复核和合成反例确认原生 conflict 对所有 GT 做 overlap argmax，能够新增 S1 外边。于揭盲前冻结协议 A1：不修改 TAL、不把 raw S2/S3 投影回 S1，主 matching 图仍为 S1，并以 native-covered identity surrogate rematch 阻断潜在 false-GO。最终 21 项测试、32-batch 回归 smoke 与原生/插桩逐位控制全部通过。
- FCA-P0 A1 正式扫描完成：6×BF16 + last FP32，每次 16,551 图/40,058 GT。O2O pooled BF16 avoidable gap=`31/240348=0.0129%`，image-cluster bootstrap 95% CI=`[0.0087%,0.0176%]`，远低于预注册 0.25%；逐 checkpoint gap=`28/1/0/0/0/2`，31 个事件全部是不同 GT，28 个来自 epoch0，epoch20/30/40 为 0。last BF16/FP32 均为 2，裁决不受精度口径影响。自动 verdict=`NO-GO_ON_CLEAN_VOC`，不进入 Phase 1 matcher 或训练。
- 阶段分解显示 O2O 的所有 coverage loss 均发生在 S1→S2 conflict，S2→S3 coverage loss 恒为 0；六个 BF16 checkpoint 观察到 152 条 S2 新增 S1 外边，但 O2O final off-graph edge 为 0，且 coverage gain 为 0。O2M sidecar gap=`169/240348=0.0703%`，高于 O2O，故撤回“O2O 特有”表述，改称一般 TAL conflict-order 现象；其效应仍低于门槛，不自动转为新主线。
- 31 个 O2O gap 中 28 个为 >=64 px，29 个与其他 GT 最大 IoU>=0.3；没有 dense-tiny cell 达到触发条件。这只否定 clean-VOC coverage-first 路线，不证明遥感 dense-tiny 场景不存在问题；若转向领域数据，必须另写预注册而不能事后救活本轮。
- 正式运行 458.54 s，只使用物理 GPU4、workers=4、pinned memory、NUMA1；7 次扫描几何 SHA 相同，无 optimizer/backward/grad/参数或 buffer 改变，运行窗口新增 MCE/SIGBUS/EDAC=0。正式目录 372 MiB 已完整归档到 `fca_p0_formal_A1_20260919_1912/`，本地/远端全部受检文件 SHA256 一致。中文结论见 `FCA_P0_FINAL_AUDIT_2026-09-19.md`。

### 2026-09-19 ASAC-P0 正式闭环

- 更正 GPU 占用判断：服务器 `/usr/bin/nvidia-smi` 展示的是停滞旧画面，不能用于当前调度；以服务器时间和 direct NVML 为准。正式作业前后 GPU4–7 均为 0%、无 compute PID，约 452.8 MiB 是驱动上下文，不代表训练占用。
- ASAC provenance revision 5 完成最终加固：冻结 launch/runtime 两份 127 字段 manifest、五个 checkpoint 内容、model/data YAML、反序列化前后 checkpoint hash，并拒绝 YAML 重复键和浮点 epsilon 伪等价。进一步在第一次正式 forward 前封存固定 256 张 train 图及对应标签的逐文件字节内容，ordered manifest SHA256=`8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`。本地和服务器 CPU 回归最终为 47/47 PASS。
- 最终 GPU smoke 使用物理 GPU4、NUMA1、batch64、workers4、pinned memory、BF16；原生 E2ELoss 的 O2M `10/10`、O2O `7/1` 接口、双 head 模块和 64 图 assignment 均通过，参数/BN hash 前后不变，无 optimizer，运行窗口新增硬件事件 0。
- ASAC-P0 正式零训练审计完成：五个 B50 checkpoint、固定 256 图、三种 scale/translate 强度。657 个 clean GT 中 624 个在所有视图存活；每个统计 cell 624 对象，最低 assignment coverage 99.84%。O2M/O2O 在 5/5 checkpoint 均随强度单调下降，q0.25→q0.75 平均稳定度下降分别为 0.04537/0.03655；最强视图 `rho(S,Q-retain)>=0.2` 也均为 5/5，自动 verdict=`GO`。
- ASAC 的 GO 只批准 scale/translate 训练期反馈原型，不是 AP 改进结果。33 个 dropped GT 显著更小、更靠边；五个 checkpoint 属于一条轨迹；固定 256 图并非全量随机样本；Mosaic 未测试。下一步限定为 A0 固定增强、A1 匹配强度调度、A2 真实 ASAC、A3 打乱反馈时序四组因果矩阵，不允许同时堆其他模块。详细结果见 `ASAC_P0_FINAL_AUDIT_2026-09-19.md` 与 `ASAC_P0_B50_V2/`。
- 正式运行约 30 s，仅使用物理 GPU4；GPU5–7 未使用。正式窗口内核日志无任何新增条目，结束后四卡均无 compute PID。本结果仍统一标记为 faulty-host exploratory，健康硬件复现要求不变。

### 2026-09-19 ASAC Phase-1 工程门禁闭环

- 在不覆盖失败证据的前提下定位 golden v2 门禁错误：YOLO26 训练格式 checkpoint 确实含 O2M/O2O 双分支，但 P0 的 `head.end2end` 运行标志为 false。修正测试契约后用全新 v3 路径重跑；v3 对固定 B50 epoch50/q=.50 得到 O2M=`0.9545479413122414`、O2O=`0.9192160195274237`，相对 P0 两项差值均严格为 0。
- v3 同时确认 E2ELoss、O2M `10/10`、O2O `7/1`、stride `8/16/32`、BF16/inference、624 GT、固定 panel/population、参数/全部 buffer/Detect 额外 tensor/RNG 零副作用以及无 assigner OOM fallback。artifact SHA256=`aabeb03b02da7e8e5c7d8f4b26efe8314b933816beb4364fc08c8e20ef1d0d41`，A0/A2 formal launcher 均实测成功消费该 artifact。
- 4 卡 DDP 两轮联调在物理 GPU4–7、NUMA1、batch64、每 rank workers4、pinned memory、BF16 下成功退出。两轮 `q=.5` 的联合稳定性为 `0.9364471628/0.9364590350`，O2M/O2O coverage 均为 1.0，624 对象，panel/population/RNG hash 恒定，OOM fallback 均未触发。
- 2% 数据、2 轮随机初始化产生的 AP=0 明确仅为工程 smoke，不作性能解释。运行窗口无新增 MCE/SIGBUS/Memory failure，结束后 GPU4–7 无 compute PID。完整报告与逐文件证据位于 `ASAC_PHASE1_ENGINEERING_GATE_2026-09-19/`。
- 工程门禁裁决为 PASS，只批准按预注册进入 seed42 Phase-1a A2/A0 50 轮筛选；目前仍无 ASAC AP 改进结论。所有当前主机结果继续标记 faulty-host exploratory，正结果必须健康硬件从头复现。
- 重新打包冻结代码为 `asac_integration_bundle_v3.tgz`，本地/远端 SHA256=`31144ad7a8932925b143640155154dd9bb933e7e50ea0f7ecaf0d979f01210d6`；包内 smoke fraction=.02，且无 Python/pytest cache。旧 v2 包不删除但明确停用。
- 首次 formal A2 在 22:12 完成 provenance/golden 门禁并进入四卡 DDP 初始化后遭遇整机重启，未产生 epoch 1。重启后旧 run 仅有 0 字节 `args.yaml`，外部 launch sidecar SHA256=`7e36ab5167c8e73294b329ebff5b53d2ab62088f8f06a2f1f32ea508c2c50604`；裁决 INVALID、禁止 resume，所有残留保留。
- 22:14 新启动内核记录 severity=recoverable 但 `Uncorrected: true` 的 CPU303/socket1 L1 cache data-read Machine Check。该事件不能归因于 ASAC，也不能忽略。按服务器所有者此前授权，可在稳定检查后以全新 run 名从头继续 faulty-host exploratory；任何运行窗口新增 MCE/SIGBUS 仍立即判废。详细记录见 `ASAC_PHASE1_INTERRUPTED_2026-09-19_2212/INCIDENT.md`。
- 使用全新 run `asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1` 从随机初始化重启 A2；采用 detached tmux wrapper，普通 SSH 断开不会杀死训练，整机重启仍按无 exit marker 判废。pre-import source tree、golden、空输出目录与 GPU4–7 空闲再次验证通过。
- 对 A2 epoch1–17 的 live state 做独立只读复算：Sref、EMA、指数 q target、±.05 log-rate clip、[.25,1] clamp 与跨轮 q 传递均逐位相等，最大误差 0；最低 coverage=623/624，OOM fallback 全 false，协议/源码/数据 hash 一致。详见 `ASAC_A2_LIVE_STATE_AUDIT_E17_2026-09-19.md`。
- A2 reboot-r1 正式完成 50/50、exit=0；q 从 .5 起步，最低 .4895，epoch31 首次达到 1.0，epochs31–50 共 20 个训练轮位于上限（其中 epoch50 不再生成 next action）。全程最低 coverage=623/624，无 OOM fallback、无新增 MCE/SIGBUS；当前只确认执行成功，必须等配对 A0 完成后才作精度裁决。
- A0 运行到 epoch24 时完成前缀等价审计：与旧官方 B50 的 epoch1 75/75 非 time 字段完全一致，但 epoch2 起明显分叉；24×75 共 1,800 格有 1,276 格非零差异，故 bitwise/numeric equivalence 门禁 FAIL。A0 自身 q=1、panel/hash/coverage/OOM 均正常。A0 改称“ASAC 基础设施下固定 q=1 的配对仪器化控制”，旧 B50 单列为官方外部参考；A2-vs-A0 仍是同基础设施的必要 q 策略对照。详见 `ASAC_A0_PREFIX_EQUIVALENCE_AUDIT_E24_2026-09-19.md`。
- A0 正式完成 50/50、exit=0；固定末轮 P/R/mAP50/mAP50-95=`0.74356/0.64914/0.72061/0.51357`。A2 对应为 `0.74450/0.64107/0.71908/0.50847`，配对 mAP50-95 差值 `-0.00510`；last-5/last-10 差值为 `-0.004564/-0.004721`，50 轮仅 7 轮胜 A0。相对旧 B50，A2 末轮为 `-0.00429`。按预注册 `>=+0.003` 与 last-5 同号门禁明确 **NO-GO**，不进入 seeds0/1、A1/A3/A4、150 轮或 COCO，也不调 beta/q 边界救结果。
- A2 的控制器真实动态但后期饱和：`q_min=0.489525`、均值 `0.809132`、epoch31 首次到 1，epochs31–50 共 20 个训练 epoch 在上限。A0/A2 最低 panel coverage 均为 623/624，无 OOM fallback；墙钟为 1973.60/1973.95 s，A2 相对配对 A0 仅 `+0.0177%`。完整报告及机器可读裁决见 `ASAC_PHASE1A_SEED42_FINAL_2026-09-19/`。
- A2-A0 的阶段均值在 epochs1–5/6–15/16–30/31–50 分别为 `-0.004852/-0.004661/-0.001528/-0.004336`；最后 20 轮两组都已 `q=1`，A2 却 0/20 获胜，说明早期课程产生的路径差异未被后期同强度训练恢复。固定 panel stability 更像训练成熟度状态而非充分的增强控制误差信号；该解释只作为机制教训，不授权事后修改控制映射继续试验。
- 代码级定位 A0 与旧 B50 分叉：ASAC 从 epoch2 起为清除旧 q 预取而逐轮 `train_loader.reset()`，会丢弃每 rank 最多约 16 个预取 batch、重建 worker RNG 并立即应用新 sampler epoch；旧 B50 平时不逐轮 reset。该时点与 epoch1 全等、epoch2 分叉精确吻合。A0/A2 使用完全相同 reset，故二者仍是公平配对；详见 `ASAC_A0_DATALOADER_DIVERGENCE_DIAGNOSIS_2026-09-19.md`。
- A0/A2 正式窗口均无新增 MCE/SIGBUS/qualifying EDAC，结束后 GPU4–7 direct NVML 均为 0% 且无 compute process。两个远端 run 各 12 个原始文件已完整归档，本地/远端 SHA256 逐项一致、mismatch=0；当前主机结果继续保留 faulty-host exploratory 标签。
- ASAC NO-GO 后只提交候选路线、不立即启动新实验：优先级为 CSPC 可折叠跨尺度预测编码 P0、RS-SPAR 遥感微小目标相位 P0、跨切片身份一致性审计。详细 go/no-go 阈值与停止路线见 `POST_ASAC_ROUTE_OPTIONS_2026-09-19.md`，等待用户讨论选择。
- 最终独立冷复核逐项重算 A0/A2/B50 CSV、两组 state、分阶段曲线、墙钟与门禁；全部核心数值和哈希与正式报告一致。A2 对 A0 为 7 胜、1 平、42 负；唯一规范备注是预注册的“q 动态范围非退化”未给硬数值阈值，但本次范围 0.5105 明显非退化，不影响两项性能门禁失败与总 NO-GO。

### 2026-09-20 非通用数据集路线筛选

- 接纳“避开纯通用 benchmark，优先选择领域高频失效”的战略，但明确领域数据不等于可以只追求容易的高 AP；选择标准改为中等规模、公信力、固定协议、真实高频失效和同领域外部验证。
- 只读筛选海上无人机、水下、农业、SAR、工业缺陷、火灾、低照度与医学检测数据；未下载任何数据、未启动训练。第一梯队为 SeaDronesSee ODv2、DUO、GWHD 2021、HRSID，分别对应 maritime tiny/metadata、水下小目标+长尾、跨国家农业 OOD、SAR clutter/false positives。
- 当前综合推荐顺序为 SeaDronesSee > DUO > GWHD 2021 > HRSID+SSDD。VisDrone 工程最容易但论文拥挤；DOTA 需 OBB/切片；AI-TOD/DIOR/RDD 超当前快速预算；NEU-DET/DeepPCB/GC10-DET 不足以独立支撑机制论文；VinDr-CXR 因访问、存储与医学解释门槛暂不首选。
- 选定数据集前先做 Dataset-P0，只取 annotations/metadata 或小样本核验许可、哈希、划分泄漏、目标尺度/密度/长尾及训练成本；通过后才下载完整数据并建立 YOLO26n baseline。完整比较见 `SPECIALIZED_DETECTION_DATASET_SHORTLIST_2026-09-20.md`。

### 2026-09-20 非通用数据集第二轮扩展检索

- 将候选范围扩展到深海生态、野生动物、工业分拣、道路/铁路/建筑缺陷、显微科学、极端天气、热红外、低光与特殊交通，共审计 40 余个数据集并保留 30 余个可用候选；仍为只读调研，未下载数据、未启动训练。
- 扩大检索后首选从单一 SeaDronesSee 调整为分取向决策：JODD 是最适合快速入门的综合首选；HRP4K 的视频级切分与负样本使协议最干净；RUOD-R 的同图双版本标注最适合研究缺标错误负监督；XWOD 有 stock YOLO26n 表格基线和 LOWO 协议，与当前工程衔接最快。
- JODD 仅 343 MB，包含 8,151 图、15,621 框、20 类和 1,000 张背景图，官方按 source video 分组为 4,864/847/2,440，且保留潜器/相机/潜次信息；因此当前建议把它作为第一候选进行 Dataset-P0。
- 明确小领域不自动等于容易发表。视频相邻帧、同建筑/设备/航线、随机生产批次跨 split、仅正样本、重构缺陷、测试集过小、许可不清和 AP 饱和均可能制造虚假提点；所有候选必须先通过 group leakage、license、duplicate 与 protocol 审计。
- 冻结三个高潜路线草案：JODD 的小目标长尾与背景误报；RUOD-R 的缺标鲁棒双头训练；XWOD 的 worst-weather/LOWO 双分配公平性。三者均要求 training-only、zero-inference-cost 的机制主张、分层指标、至少一个外部验证集和多种子复现。
- 完整长名单、数据规模、许可、风险、外验与来源写入 `SPECIALIZED_DETECTION_DATASET_LONG_LIST_2026-09-20.md`。下一步建议先对 JODD、HRP4K、RUOD-R、XWOD 做 annotations/metadata 级 Dataset-P0，用户确认前不下载完整影像。

### 2026-09-20 候选数据集存储体量审计

- 对上一轮长名单中的 55 个实际候选逐项核查下载体积；51 个得到官方 API、文件服务器、官方页面/仓库或明确标注的镜像参考值，只剩 DuckNet、DAWN、FAIRHiveFrames-1K、NuCLS 四个没有可信体积。对第三方/格式转换镜像均显式标注，不伪称官方原始包精确值。
- 51 个可估算实用包合计约 505 GB，但不建议批量下载。关键反例是 Sherbrooke Red Deer：仅约 4,071 图却因超高分辨率达到 131.485 GB；SA-FARI 检测训练材料约 66.30 GB、Big Bird 标注训练包 44.871 GB、WildLive 31.133 GB、RAIL-BENCH 24.156 GB。
- 四个当前首选的压缩下载合计仅约 15–16 GB：JODD 343 MB、HRP4K 8.659 GB、RUOD-R 约 3.4–3.7 GB、XWOD 2.690 GB。考虑解压、4K 切片、缓存、转换副本和训练输出，四套并存建议至少 60 GB 空闲，做 HRP4K 切片则建议约 100 GB。
- 体量审计未下载任何完整数据、未改动服务器、未启动训练。逐项结果和来源写入 `SPECIALIZED_DATASET_STORAGE_AUDIT_2026-09-20.md`。

### 2026-09-20 用户截图五个检测基准的信息核对

- 汇总VOC2007、VisDrone2019-DET、KITTI 2D、SKU110K_fixed、COCO2017的领域、类别、划分、下载规模与协议。详见FIVE_DETECTION_DATASETS_OVERVIEW_2026-09-20.md，含官方/实现方来源。
- 关键口径：SKU为1检测类、fixed版11743图；VisDrone常用8629有标注图，不与视频全库混用；KITTI主要评测3类并按难度/指定IoU/AP_R40；COCO trainval约20GB边界，不当小包；VOC2007不等于此前VOC07+12训练设置。
- 建议VisDrone/SKU作为明确领域失效的候选实验场景，并不据此断言容易提点或不拥挤。仅检索及落盘，无下载、训练、转换、服务器或监控变更。

### 2026-09-20 21:37 删除LLVIP断点并检索JODD网盘

- 按用户明确要求删除本机LLVIP.zip.part（2383264080 bytes），已验证删除目标为本任务普通文件且无活动下载进程，删除后复核不存在。未进回收站；不能从原断点续传，但可重新下载。保留说明/日志和其它数据集，未触碰服务器。
- JODD按完整名称、文件名、DOI与百度/夸克/HF关键词检索，未找到可核实的网盘备用源；[官方DOI](https://www.jamstec.go.jp/datadoi/doi/10.17596/0004035.html)及论文仍指向JAMSTEC官方包。不把“未找到”表述为“绝对没有”。
- 对YOLO格式转换仅提供解释，未执行转换或训练；原数据及划分保持，未来转换需单独建立派生标注与类别映射，而不是改图片尺寸。自动复查继续暂停。

### 2026-09-20 用户接管百度/夸克网盘下载

- 用户决定有百度/夸克网盘的自行下载；当前提供已核实的LLVIP作者百度主包链接与提取码14lc。不自动操作网盘账号/客户端，不重启LLVIP替代下载，现有断点保留，自动复查保持暂停。
- 用户完成后再核验本地原包、版本及更新标签，百度文件不未经核实就假定与Google包字节相同。其它六套已下载数据不重复下载；JODD尚无已核实官方百度/夸克备用源。

### 2026-09-20 20:48 LLVIP官方配额阻塞与复查暂停

- LLVIP保留2383264080/4003858172 bytes。实际8MiB断点请求返回Google Drive Quota exceeded HTML，下载器正确拒绝拼接；有限重试仍失败。32字节小探测206不等于实际文件下载可用，已纠正初步判断。
- 作者百度备用链接已从冻结文档核对（https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg ，14lc），但本轮web失败、浏览器入口超时，未实际进入或登录，不能断言其账号需求。提供给用户自行选择，不规避官方限额。
- 全部本任务下载/验收Python进程已结束，无无关进程受影响。按既有指令将自动复查automation暂停，产品工具确认PAUSED，等待用户操作或选择恢复时机。
- 五套通过；ORCA文件与结构完成但保留源数据读取警告；LLVIP配额阻塞；JODD源阻塞。原文件、断点与验收/审阅记录全部保留于E:/datasets，未连接服务器，未宣称全部成功。

### 2026-09-20 20:16 ExDark说明插图误计修正

- **20:18最终结果：v3通过**，errors/warnings为空。7363数据图+1说明图全量解码、7363标注/23710框、12类及官方3000/1800/2563划分核对通过；大小写映射明确保留，原文件不改。当前无未处理验收问题的原格式数据更新为5/8，ORCA待审、LLVIP继续、JODD阻塞。
- 20:17补充：v2排除说明图后发现一个真实的大小写命名差异（2015_00391.JPG对应2015_00391.jpg.txt）；不改原文件，记录唯一映射，完整casefold匹配无碰撞。已启动v3全量重验PID28712。以后Linux读取必须使用实际文件名/显式映射，细节写入ExDark/READING_NOTES.md，不能把该差异掩盖为说明图问题。
- ExDark第一遍全部7364图片文件均能解码；唯一不在7363样本清单的图是官方标注README所引用的annotations.png说明插图。原标签7363份、官方划分3000/1800/2563，缺图0。此前集合错误来自验证器误计说明图，不是下载缺失。
- 修正为仅将该明确路径作为辅助图片记录，仍检查其解码，不把它当样本，也不删除原始文件。修正版全量重验PID22264已启动，尚待最终结果。
- 标注文本只读审计共12类23710框，合法类名检查无异常。LLVIP约1897/4004MB继续下载。
- JODD官方源有限复查仍TLS失败，保留阻塞。自动检查继续，无服务器操作。

### 2026-09-20 20:03 SeaClear通过，ORCA文件完整与caption语义分开审阅

- 本轮末ExDark16/16清单文件全部完成，已启动全量图片解码、标注与作者划分清单核验；稍早的“剩最后motorbike包”状态已被覆盖。尚未宣布ExDark验收完成。
- SeaClear全部8610图片解码、31555框/40类匹配通过，零错误/警告；当前原格式无警告通过4/8。
- ORCA全部14646图片和主JSON/10份split结构检查通过，缺图/尺寸/无效框错误0；仍保留额外1图未引用和caption label=-1未明确定义的警告。
- [论文§3.1](https://arxiv.org/html/2512.21150v1#S3.SS1)仅对面积>1024框生成描述；实测7465条-1全是面积≤1024的小框，7464无caption、1有caption。支持“小框未描述”推断，但不冒充已得到作者枚举定义。caption正负与bbox有效性必须分开，不能因label=1或-1删除检测框。
- [项目页](https://orca.hkustvgd.com/)图像统计自身出现14647与14645；本地冻结包14646实际图/14645引用图。完整审阅写入E:/datasets/ORCA/REVIEW.md，原数据未修改，VALIDATION仍needs_review。
- ExDark剩最后motorbike包、LLVIP继续下载，JODD阻塞不变；自动复查继续，仅本机操作。

### 2026-09-20 19:48 SeaClear完整包确认与两套全量验收启动

- SeaClear下载完整且发布者MD5一致，但Windows bsdtar错误尝试解压RAR的零字节目录条目。修正验证器为显式选择全部8612普通成员做流式解压测试，全部通过；不是跳过损坏文件、不是改原始包，也未安装全局解压工具。
- SeaClear全量验收PID3432已启动：安全一次性解压、逐输出文件长度核验、全部图片解码和COCO标签匹配。包内8610图、31555标注、40类，尚待全量结果。原格式RAR完整性与数据内容可用性分开记录。
- ORCA全部14个清单文件完整下载；全量验收PID27800启动。此前发现未引用图片及未定义caption标签将保留审阅警告，原图/标签不变。
- 已完成整套验收仍3/8。ExDark和LLVIP继续下载，JODD访问阻塞未变。仅E:/datasets本机数据流程，无服务器操作。

### 2026-09-20 19:34 HIT-UAV通过与ORCA实包计数更正

- HIT-UAV全量验收通过，当前已通过3/8（另DUO/AP-10K）。2898源图，train2029/val290/test579；包内11600图含四格式副本及8示意图，全部解码成功。各套JSON/XML对象核验通过；24751普通目标+148 DontCare，4有效类，不将ignore作为普通第5类。原图/标签未修改。
- ORCA图片包发布者SHA256和ZIPCRC一致。实际14647条目=1目录+14646图片，主标注14645图，缺图0、额外1张阿德利企鹅图片未引用，需保留且在转换时明确处理。此前14647图片的说法得到纠正。
- ORCA未定义label=-1有7465行，7464空caption而1行非空；不能把它直接解释为某类正/负样本。增加验证器覆盖审计和语义警告，不为使验收通过而删除原图/标注。尚缺三份划分文件，整套解码及验收未启动。
- 本机下载持续增长，其他文件无新失败，JODD阻塞未变；自动复查继续，仅本地操作。

### 2026-09-20 19:20 首批两套数据全量验收通过

- DUO与AP-10K的VALIDATION均为verified_usable_original_format，错误/警告0。DUO7782图/4类/74515框，AP-10K10015图/54类/9份划分文件；全部图片实际verify/load，引用、尺寸及框检查通过。没有转换YOLO或启动训练。
- AP-10K额外schema检查9文件图片/框ID无重复、无缺bbox，三套划分train/test文件名不相交；姿态来源不自动等于检测标签穷尽，研究适配仍需独立审计。
- HIT-UAV主包发布者MD5及ZIPCRC通过，启动全量验收PID26744。按实际normal/rotate与JSON/XML四副本结构修正验证器，支持水平annotation单数键、filename字段、五维旋转框和XML；严格区分2898源图、四套图副本与README示意图，不修改原数据。
- ORCA元数据初审：14645图、42217标注行、670类；caption label还包含README未解释的-1（7465条），以及负描述label1（12431条）。不能把标注行数直接当独立正目标数；待全包到齐核实图像计数、负描述及未定义标签。此发现已进入接续清单。
- 仍仅本机E:/datasets，下载正常增长，JODD源阻塞未解除。当前完成2/8，自动复查继续，不宣称全部完成。

### 2026-09-20 19:05 DUO/AP-10K整包完成与LLVIP续传修复

- DUO和AP-10K的官方主包已完整下载，精确字节数及ZIP全CRC通过，SHA256记录在各自checks。无发布者哈希的包只记录本地SHA256，不冒称已通过发布者哈希验证。
- DUO第一遍7782张图片全部解码成功，实际train6671/test1111，4类、74515框。各split图片/框ID无重复、bbox无缺失。train/test复用1.jpg等basename，旧验证器误报只有6671张；修正为split目录精确匹配并全量重验中，未改原图、标注或划分。basename重合不等于实际图片重复或泄漏，内容去重属于后续研究审计。
- AP-10K启动安全解压与全量图片/标注核验（PID20808）。两包下载完成不代表两数据集已验收通过，不提前报告全部成功。
- LLVIP开放式续传收到curl33，但闭区间Range实测206且范围准确；保留613MB前缀，改为8MiB严格分块续传（PID28896），已增长至764MB。每块核对响应范围、总大小及块长度，有限重试，不从头重复下载、不改全局代理。其它四worker继续正常工作。
- JODD仍官方源不可达；HIT-UAV/SeaClear/ExDark继续下载，ORCA主包排队。全部工作仅限本机E:/datasets，无服务器连接或训练。详见下载接续文档及各集VALIDATION/checks。

### 2026-09-20 17:38 本机下载复查与首包解码预检

- 本机下载仍正常，无新增失败状态。ExDark bicycle.tar.gz发布者MD5一致，安全解压后652张图片全量verify/load错误0；只算单包通过，不算整个ExDark完成。
- 本机补齐ExDark固定版本作者划分清单及LLVIP新版来源文档；ExDark官方划分实际3000/1800/2563，验证器新增清单与图像一致性检查，未改原始标签或split。
- 下载载荷约961MB（不含单包预检解压），所有整套数据仍未验收；不连接服务器、不自动上传，JODD继续标为阻塞。

### 2026-09-20 17:22 改为本机下载并删除服务器下载残留

- 用户看到服务器进度缓慢后要求先下载到电脑，并明确要求把本次服务器下载全部删除，之后自行转发。已停止远端任务、确认安全路径与无符号链接后删除7个archives目录约918MiB，不在回收站，可重新下载；保留目录及操作记录，未动旧VOC/训练环境。
- 原专用中转已终止且远端17997无监听；自动复查更新为仅本机，禁止重新启动远端下载或自动上传。因C盘仅15.35GB空闲，选新建E:/datasets（136.34GB空闲）。
- 本机隐藏后台PID24976已实际开始下载，四并发，原始包及part落盘，采用原有本机代理的进程级配置。已适配Windows锁、RAR完整性检查和本地验证入口，不修改系统网络或全局Python环境。
- 所有数据集尚未通过完整验收，JODD仍阻塞。最新路径、脚本与日志见DATASET_DOWNLOAD_HANDOFF_2026-09-20.md顶端。

### 2026-09-20 17:15 下载复查与有限重试修复

- v2因HTTP/2 PROTOCOL_ERROR和提前断流导致4个文件失败，部分文件全部保留。只停止本任务进程并确认退出后，从原进度启动v3，使用HTTP/1.1、300秒连接窗口、最多48次有限续传与文件锁，未更改全局代理或系统软件。现约835MiB，无完整集通过验收。
- Drive主包真实字节数写入清单：DUO2940459407、AP-10K3403294327、LLVIP4003858172；补齐ExDark官方划分清单和LLVIP新版来源说明，均已download_verified。修复下载器未强制ZIP格式/JSON可解析的问题，补正tar根目录解析。
- 服务器17:08–17:11出现14次明确向进程发送SIGBUS的内存硬件错误，未匹配本任务下载PID。按原所有者授权继续限定NUMA1下载，记录硬件风险，不把网络中断推断为硬件所致，不动共享服务。接续详情见DATASET_DOWNLOAD_HANDOFF_2026-09-20.md。

### 2026-09-20 8项数据集下载与验收启动

- 用户进一步授权下载DUO、ORCA、AP-10K、LLVIP、HIT-UAV、ExDark、SeaClear，并要求完整可用后结束。统一在 `/home/ubuntu/rz/datasets` 建8个独立目录，原始包、源元数据、校验记录分开保存。
- 官方发布者MD5/SHA256与文件大小已进入清单；服务器直连Google/HF失败，使用本机已存在网络代理经任务专用临时SSH中转，仅进程级代理，不更改共享系统设置。已开始7项下载，JODD官方源仍阻塞；当前约307MiB，未有整套完成验收。
- 准备了安全解压、全量图片解码与标注匹配校验脚本，尚待完整数据实测，不能把已编译脚本等同验收完成。下载速度波动明显，末次平均约0.1–0.2MiB/s/Google包；不承诺早期1–3小时估算。
- 已设置10分钟任务内自动复查ID `automation`，完成或不可自行解决的阻塞后停止。完整接续、安全边界和每集注意事项见 `DATASET_DOWNLOAD_HANDOFF_2026-09-20.md`。

### 2026-09-20 统一数据目录与JODD下载尝试

- 按用户明确请求在服务器新建 `/home/ubuntu/rz/datasets/JODD`，今后新数据以 `/home/ubuntu/rz/datasets` 为统一存放根目录；没有移动已有VOC等数据或改动训练配置。
- 服务器和本机均无法连接JODD官方数据站，HTTPS重试与HTTP检查失败；DOI页面可达但没有备用数据入口。因此本轮只完成建目录，JODD尚未下载成功，未启动训练。详细状态见 `JODD_DOWNLOAD_STATUS_2026-09-20.md`，并复制到远端JODD目录供后续接续。

### 2026-09-20 F1回溯核验

- 读取全部已归档50轮CSV和T1逐类别独立复评。TQGB-T1 seed42的O2O原生宏平均F1由69.030275%升至69.252815%（+0.222540个百分点），O2M辅助路径为+0.260558个百分点。
- CSV平均P/R调和均值不是原生宏平均F1：T1在seed42为+0.284046个百分点，在seed0为-0.324442；T1-NP seed0仅+0.025439。没有跨种子稳定提升证据，不改变NO-GO裁决。
- 最初150轮基线的平均P/R调和均值为75.566832%，与50轮方法筛选不同预算；不得宣称已超越150轮基线。完整口径、所有末轮数值和来源见 `F1_RETROSPECTIVE_AUDIT_2026-09-20.md`。未启动训练或修改服务器。

### 2026-09-20 20GB上限与质量筛选

- 按用户要求对55项重新筛选：10项优先候选、8项专项备选、28项因证据/协议/访问或任务适配问题暂缓、5项因下载材料超过20GB剔除、4项体积未知暂缓。剔除仅为清单变更，没有删除文件或下载数据。
- 优先10项：HRP4K、JODD、GWHD2021、CropAndWeed、LLVIP、HIT-UAV、FishEye8K、ExDark、SeaClear、ZeroWaste-f；专项8项：DUO、SortWaste、SeaDronesSee ODv2 compressed、ORCA、AP-10K、MAD Apple、PackWISE、RTTS。质量判断为资料级，尚未完成实际样本抽检或查重。
- 重要更正：JODD生物图按视频ID分组，但背景图随机拆分；HIT-UAV为4有效类、DontCare另计；LLVIP为15488对/30976张；CropAndWeed原始74类不等于所有类都足以单独训练；IP102的798.3MB小包是其他研究者的COCO转换版；SortWaste官方框数分项合计87525而总计写87252；GWHD2021/ORCA也需锁定图像计数和标签版本。
- 新的严格筛选把XWOD、RUOD-R等此前高潜路线暂列观察项：新颖性或YOLO26已有基准不能代替标注、划分、版本和使用条件证据。并未断言未入选数据低质量。
- 55项去留、18项候选的大小/领域/类型/类别数/图片数/质量依据完整写入 `DATASET_QUALITY_SHORTLIST_UNDER_20GB_2026-09-20.md`。本轮不启动新训练。

### 2026-09-21 10套数据集上传与独立最终验收完成

- 已将 HIT-UAV、DUO、AP-10K、SeaClear、ExDark、ORCA、HRSID、VisDrone2019-DET、LLVIP、ZeroWaste-f 整理至服务器 `/home/ubuntu/rz/object_detection_dataset_test`；共68个清单文件、25,564,671,681 bytes。
- 本地原文件大小/SHA256复核通过；远端逐文件哈希、压缩成员CRC或SHA256、解压成员集合及实际图片重新解码通过。独立审计状态 `all_manifest_files_and_extracted_content_verified`，errors为空。
- 最终README和完成报告已原子发布并回读哈希确认；README SHA256 `728f51fa25fc5ae5b1d827b30774225f351c10d884d4d252016fbb5d7e0c26cc`。各集的标注边界和协议风险保留在README及VALIDATION/REVIEW记录中，未修改原标签。
- JODD仍缺实际数据包，明确排除在完成计数外；未用不明镜像冒充。全过程未训练、未使用GPU、未修改共享环境或其他进程。

### 2026-09-21 YOLO训练格式准备完成（test作为每轮评测集）

- 按用户指定口径，将有val的数据统一合并为train，所有`dataset.yaml`的`val`和`test`均指向test；不额外建立val，不处理相邻帧泄漏。
- 在服务器`/home/ubuntu/rz/object_detection_dataset_test/processed_yolo`生成11个派生版本：10套数据集，LLVIP分别生成红外/可见光版本。原图以绝对软链接复用，原始数据未修改。
- 派生规则包括：DUO裁剪953个越界框并跳过1个无正面积框；ExDark裁剪50框；HIT-UAV排除148个DontCare框；LLVIP按VOC闭区间保留5个一像素宽框；VisDrone排除ignore/others并跳过1个零高有效框；ZeroWaste-f裁剪138框。
- 显式移除389个派生完全重复标签行（ORCA357、SeaClear6、VisDrone4、ZeroWaste-f22）。当前Ultralytics 8.4.146对11个版本、22个分区全量扫描通过，0损坏、0最终重复警告；22个分区真实管线取样均得到`3×640×640`张量。
- 最终状态`ready_for_yolo_training_test_used_as_val`。完整规模、哈希及证据见`YOLO_DATASET_PREPARATION_2026-09-21.md`。本轮未启动训练、未使用GPU。

### 2026-09-22 YOLOv8n随机初始化基线准备

- 用户明确要求真实训练不使用预训练权重。已停止并清理尚未完成的任务专用`yolov8n.pt`下载，服务器模型目录无`.pt`文件。
- 固化YOLOv8n源码结构到`/home/ubuntu/rz/object_detection_dataset_test/models/yolov8n_scratch.yaml`并CPU实例化验证：`ckpt_path=None`、scale=n、3,157,200参数（3,157,184可训练），确认随机初始化且没有加载checkpoint。
- 主scratch基线预算设为300轮；3轮仅冒烟，150轮为中期观察，若300轮仍明显上升则必须对基线和所有对比方法统一预注册扩展到400轮。统一imgsz640、总batch128、SGD、seed0、关闭早停。
- 启动器`_management/train_yolov8n_scratch.sh`通过shell语法检查，强制物理GPU4–7、workers=4/进程、cache=False、pretrained=False、每25轮保存。尚未启动训练。完整方案见`YOLOV8N_SCRATCH_BASELINE_PLAN_2026-09-22.md`。

### 2026-09-22 YOLOv8n 22次统一基线队列正式启动

- 用户将实验协议更新并冻结为：11个派生数据集全部训练125轮、总batch 64、imgsz 640、两个种子2027与2005，共22次独立训练；该协议取代上一条尚未执行的300轮/batch128计划。
- 模型仍为`yolov8n_scratch.yaml`源码结构随机初始化，模型SHA256为`44c9efc76c9dad388ad1ef6c4efb37e47ec885fe2ad12f110c76f6cdb750889d`；训练参数显示`pretrained=False`、`resume=False`，没有以任何预训练checkpoint初始化模型。
- 只使用物理GPU 4–7。队列管理器和训练/DDP子进程均以`numactl --cpunodebind=1 --membind=1`启动；运行时策略核验为`policy: bind`、`membind: 1`，训练worker的heap与匿名私有内存实际主要分配在NUMA1。近7天及启动后未见新的MCE/SIGBUS或GPU Xid。
- 已于00:47启动首项`AP-10K__seed2027`。启动核验时四卡均正常工作；`results.csv`已写入17个epoch，任务错误日志为空。Ultralytics为AMP数值兼容性检查自动下载并短暂加载了`yolo26n.pt`，此权重只用于AMP自检，不进入YOLOv8n训练模型的参数初始化；真实训练仍从随机权重开始。
- 建立原子状态、单实例锁、每25轮checkpoint、checkpoint可读性验证及服务器重启后的自动续训。30分钟低频heartbeat已启用；只在新任务完成、重启恢复、故障或全队列完成时通知。
- 完整协议和恢复边界见`YOLOV8N_SCRATCH_QUEUE_2026-09-22.md`，远端结果根目录为`/home/ubuntu/rz/object_detection_dataset_test/runs/yolov8n_scratch_baselines`。

### 2026-09-22 严格最终验收工具部署

- 部署`_management/audit_yolov8n_scratch_queue.py`，最终不以“队列显示完成”作为充分证据，而是直接审计22个运行目录。
- 每项要求`results.csv`严格包含epoch 1..125、`best.pt`与`last.pt`均可由PyTorch读取，并从checkpoint内的`train_args`复核epochs125、batch64、imgsz640、workers4、seed、SGD、四卡映射、`pretrained=False`、模型及数据路径。
- 工具会生成逐运行指标、双种子mean±std、权重SHA256和Markdown总表。首次在训练进行中试运行，正确返回`incomplete`（0/22、错误0；首项当时85/125），证明不会把部分结果误判为完成。

### 2026-09-22 AP-10K seed2027完成并通过单项验收

- `AP-10K__seed2027`于01:07完成，return code 0，`results.csv`严格包含epoch 1..125；`best.pt`与`last.pt`均可读取，checkpoint训练参数与冻结协议完全一致，审计错误0。
- 第125轮：Precision 0.80316、Recall 0.72943、F1 0.76452、mAP50 0.81439、mAP50-95 0.60794。
- 按每轮mAP50-95选出的最佳轮为epoch 122：mAP50-95 0.60888、mAP50 0.81365、Precision 0.78881、Recall 0.74178、F1 0.76457。
- 权重SHA256：best `61934b6350ad30c6a56f2706db466ee7a813b808c6b974a5ed16d36e2020b8d2`；last `1c48d46f2bf3f26030f8a6f43b677de0b8b124b346ecdbbed5f9851e332b6113`。
- 队列已自动切换到`AP-10K__seed2005`，首次复查时已完成39/125轮；四卡和NUMA绑定继续正常，未出现新的MCE/SIGBUS或GPU Xid。
- 在第二个独立运行进行到72/125轮时再次抽查，而非沿用首项结论：训练父进程`numa_maps`仍为`bind:1`；DDP worker heap为888.43MB且全部在NUMA1，Private内存3057.85MB位于NUMA1、191.73MB为已存在的文件映射等NUMA0页。新增硬件错误计数仍为0，证明队列切换没有丢失内存绑定策略。

### 2026-09-22 AP-10K双种子完成与汇总

- `AP-10K__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.81519/0.73379/0.77235/0.81404/0.61416。最高mAP50-95为epoch118的0.61533。
- seed2005权重SHA256：best `710603d5bda1bfde97d1429f243baeee576f332fefd635b3084a1f24eb528683`；last `95e09d38ca99ce4f6a5de8e9b2c3d33eb5409aa6a926e3ccad9657ce05b7ea2e`。
- AP-10K两种子末轮mean±sample std：P 0.809175±0.008506、R 0.731610±0.003083、F1 0.768436±0.005536、mAP50 0.814215±0.000247、mAP50-95 0.611050±0.004398。
- 两种子各自最佳epoch的mAP50-95均值为0.612105±0.004561；严格审计累计2/22运行、4/44权重，错误0。
- 队列已自动进入`DUO__seed2027`，首次结果复查为51/125轮；新增硬件错误0。

### 2026-09-22 DUO seed2027完成并通过单项验收

- `DUO__seed2027`于01:46完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读取，checkpoint训练参数与冻结协议一致。
- 第125轮同时为本次最高mAP50-95轮：Precision 0.83975、Recall 0.73753、F1 0.78533、mAP50 0.81879、mAP50-95 0.61964。
- 权重SHA256：best `e90509f6f6a37120f19eb9d96d211c68b4ec5647de5073a991dda94628fb3760`；last `ac3d3c26b7d964bb525654b80a36d3584c32bae490fdc1c54352796019954521`。
- 严格审计累计3/22运行、6/44权重，错误0；队列已自动进入`DUO__seed2005`，本轮检查时102/125。四卡运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 DUO双种子完成与汇总

- `DUO__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.80832/0.73872/0.77195/0.81633/0.61508。最高mAP50-95为epoch123的0.61556。
- seed2005权重SHA256：best `18fa493920e8d74d93902421a69994904b5ea0a8cae1038797709a4b50d9f899`；last `0d8f03b82e5aee904e359acb7fc1629416c0244baa5955f7f17e52b29caed96c`。
- DUO两种子末轮mean±sample std：P 0.824035±0.022224、R 0.738125±0.000841、F1 0.778641±0.009456、mAP50 0.817560±0.001739、mAP50-95 0.617360±0.003224。
- 两种子各自最佳epoch的mAP50-95均值为0.617600±0.002885；严格审计累计4/22运行、8/44权重，错误0。
- 队列已自动进入`ExDark__seed2027`，本轮审计时已写入105/125轮；管理器和训练进程存活，四张指定GPU工作正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 ExDark seed2027完成并通过单项验收

- `ExDark__seed2027`于02:19完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.68210、Recall 0.54239、F1 0.60427、mAP50 0.58934、mAP50-95 0.35164。
- 按mAP50-95选出的最佳轮为epoch115：mAP50-95 0.35447、mAP50 0.59409、Precision 0.69234、Recall 0.53465、F1 0.60336。
- 权重SHA256：best `ddecf765c3f40cd47241b538904261807e14f7adb1c777580e3a632744708e60`；last `6392b4778faa78694afe6c2390ba43cca13a3641f3533d76edec3d2740c0ee73`。
- 严格审计累计5/22运行、10/44权重，错误0；队列已自动进入`ExDark__seed2005`，审计时已写入107/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 ExDark双种子完成与汇总

- `ExDark__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.67347/0.54167/0.60042/0.59413/0.35606。最高mAP50-95为epoch115的0.35674。
- seed2005权重SHA256：best `6d8a0f34995004eeeaa622d31be6d5f693c2ab5a207daade654882728c912fde`；last `ce3355e0a24ee324f532de76531162afd0fa9db46423794dee9ae7e701749cc4`。
- ExDark两种子末轮mean±sample std：P 0.677785±0.006102、R 0.542030±0.000509、F1 0.602348±0.002724、mAP50 0.591735±0.003387、mAP50-95 0.353850±0.003125。各自最佳epoch的mAP50-95均值为0.355605±0.001605。

### 2026-09-22 HIT-UAV seed2027完成并通过单项验收

- `HIT-UAV__seed2027`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.87974/0.82031/0.84899/0.86359/0.56022。最高mAP50-95为epoch122的0.56482。
- 权重SHA256：best `f043f6b9db71dca577dcc82df1aa6e4d31776179603f7e72e1bf8b6b616f7f86`；last `d20136b985d4e15fb2a26129d9c93f0e5e31bb893c32db2f544b8f6e9a67034d`。
- 严格审计累计7/22运行、14/44权重，错误0；队列已自动进入`HIT-UAV__seed2005`，审计前已写入82/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 HIT-UAV双种子完成与汇总

- `HIT-UAV__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.90236/0.77861/0.83593/0.85127/0.55613。最高mAP50-95为epoch110的0.56665。
- seed2005权重SHA256：best `6687f1391acb53e6e2335014c7a777083e433f527dba038a4338fc9c99aa81d3`；last `5a0780f9fc888d25fd6fc195b65eeafa09781bb15f190f5e29875c456eb96585`。
- HIT-UAV两种子末轮mean±sample std：P 0.891050±0.015995、R 0.799460±0.029486、F1 0.842458±0.009232、mAP50 0.857430±0.008712、mAP50-95 0.558175±0.002892。各自最佳epoch的mAP50-95均值为0.565735±0.001294。

### 2026-09-22 HRSID-JPG seed2027完成并通过单项验收

- `HRSID-JPG__seed2027`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.90764/0.81543/0.85907/0.88614/0.62475。最高mAP50-95为epoch119的0.62775。
- 权重SHA256：best `03f318b8d2a6684132b7fe7108c60d99e8f7e72b6d38b9960f9c3f8a64f3ebdd`；last `d50ca7ac6af9c9c50423170537593b8138978f4fa78a1d31ae76bc2a8533da29`。
- 严格审计累计9/22运行、18/44权重，错误0；队列已自动进入`HRSID-JPG__seed2005`，检查时已写入5/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 HRSID-JPG双种子完成与汇总

- `HRSID-JPG__seed2005`完成125轮并通过严格审计；第125轮同时为最佳轮，P/R/F1/mAP50/mAP50-95为0.90580/0.82481/0.86341/0.89770/0.63566。
- seed2005权重SHA256：best `2e8080dac94a3905a74c8ced1fdd0f006ce1a25c7094711a60c5c8d5fece8f21`；last `155a0a5d9048c38fb00378e68a4620cb1f6e13c91270839abae991d9aa43fc96`。
- HRSID-JPG两种子末轮mean±sample std：P 0.906720±0.001301、R 0.820120±0.006633、F1 0.861239±0.003070、mAP50 0.891920±0.008174、mAP50-95 0.630205±0.007715。各自最佳epoch的mAP50-95均值为0.631705±0.005593。
- 严格审计累计10/22运行、20/44权重，错误0；队列已自动进入`LLVIP-infrared__seed2027`。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 LLVIP-infrared seed2027完成并通过单项验收

- `LLVIP-infrared__seed2027`于03:32完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.94467、Recall 0.89253、F1 0.91786、mAP50 0.94471、mAP50-95 0.61356。
- 按mAP50-95选出的最佳轮为epoch47：mAP50-95 0.62041、mAP50 0.95595、Precision 0.95983、Recall 0.89882、F1 0.92832。最佳轮较早，后续在双种子汇总中再判断过拟合是否稳定。
- 权重SHA256：best `afa3e5e8c257829cd03c9daa872a6ddee7f3958701a38c0451b54d4a4c5d36d3`；last `83fa3458197c3e550fe40556c035d3d5198d48bfaaacb7538f60d1a220512204`。
- 严格审计累计11/22运行、22/44权重，错误0；队列已自动进入`LLVIP-infrared__seed2005`，检查时已写入43/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 LLVIP-infrared双种子完成与汇总

- `LLVIP-infrared__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.95121/0.88364/0.91618/0.94724/0.61115。最高mAP50-95为epoch52的0.62370，与seed2027在epoch47达到最佳的趋势一致。
- seed2005权重SHA256：best `9d84cc1314563c9f8563799bccc415d1ae56602915e9666ce95f6d5f29b08991`；last `467b8396559947f6030f39a11487db8bd23b4d5e5e843d767b901e76d7b733ac`。
- LLVIP-infrared两种子末轮mean±sample std：P 0.947940±0.004624、R 0.888085±0.006286、F1 0.917020±0.001187、mAP50 0.945975±0.001789、mAP50-95 0.612355±0.001704。各自最佳epoch的mAP50-95均值为0.622055±0.002326，比末轮均值高0.00970，显示两种子都在中期达峰后回落。
- 严格审计累计12/22运行、24/44权重，错误0；队列已自动进入`LLVIP-visible__seed2027`，检查时已写入22/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 LLVIP-visible seed2027完成并通过单项验收

- `LLVIP-visible__seed2027`于04:28完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.85550、Recall 0.78085、F1 0.81647、mAP50 0.85296、mAP50-95 0.47283。
- 按mAP50-95选出的最佳轮为epoch114：mAP50-95 0.47427、mAP50 0.85610、Precision 0.85604、Recall 0.78535、F1 0.81917。
- 权重SHA256：best `5dae5da81f41b070abaaf607f33388ed73903e39828aad3a39983e436e60dced`；last `47ebfbc02fbaf75ebf43f01ba347fca33103cd962282b0bf933ea770909df5b3`。
- 严格审计累计13/22运行、26/44权重，错误0；队列已自动进入`LLVIP-visible__seed2005`，检查时已写入2/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 LLVIP-visible双种子完成与汇总

- `LLVIP-visible__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.85895/0.77899/0.81702/0.84439/0.45579。最高mAP50-95为epoch37的0.46764，而seed2027在epoch114达到最佳，两种子的峰值轮次差异较大。
- seed2005权重SHA256：best `3b75f57a2dbb98c1fab5101bc6a203419ead22534011ff1e3164c82ce51b0441`；last `fbb67a6471d5178b56dabe31f7b82608ee6eac75272313e9bd25704aab12fe98`。
- LLVIP-visible两种子末轮mean±sample std：P 0.857225±0.002440、R 0.779920±0.001315、F1 0.816745±0.000386、mAP50 0.848675±0.006060、mAP50-95 0.464310±0.012049。各自最佳epoch的mAP50-95均值为0.470955±0.004688。
- 严格审计累计14/22运行、28/44权重，错误0；队列已自动进入`ORCA-class-level-seen__seed2027`，检查时已写入40/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 ORCA-class-level-seen seed2027完成并通过单项验收

- `ORCA-class-level-seen__seed2027`于05:19完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.40207、Recall 0.34694、F1 0.37248、mAP50 0.35249、mAP50-95 0.28490。
- 按mAP50-95选出的最佳轮为epoch124：mAP50-95 0.28511、mAP50 0.35275、Precision 0.40540、Recall 0.35779、F1 0.38011。峰值接近训练末尾，需结合seed2005判断125轮是否充分。
- 权重SHA256：best `0df0346209fca43109a5dfdbabcea7ed9a720d3c30375b5ec06ecf5ad61ae258`；last `5a75178cbaa3cf04c2da39ced4ab8d4a1fd86244a8c7bf2eb2bb6e405080e59e`。
- 严格审计累计15/22运行、30/44权重，错误0；队列已自动进入`ORCA-class-level-seen__seed2005`，检查时已写入47/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 ORCA-class-level-seen双种子完成与汇总

- `ORCA-class-level-seen__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.45532/0.34357/0.39163/0.35123/0.28453。最高mAP50-95为epoch121的0.28521，与seed2027在epoch124达峰的趋势一致。
- seed2005权重SHA256：best `938cd503bf56674c0dc0e7960a073906937e1ad444d1668f7de0e400a2d28fd7`；last `d535c372f5d5293f35ef592ea08c93a5b7d1794098b21db625388dd34249c556`。
- ORCA两种子末轮mean±sample std：P 0.428695±0.037653、R 0.345255±0.002383、F1 0.382053±0.013543、mAP50 0.351860±0.000891、mAP50-95 0.284715±0.000262。各自最佳epoch的mAP50-95均值为0.285160±0.000071，双种子AP结果高度一致，但两者均在末段达峰，后续需将训练长度充分性列为复评项。
- 严格审计累计16/22运行、32/44权重，错误0；队列已自动进入`SeaClear__seed2027`，检查时已写入71/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 SeaClear seed2027完成并通过单项验收

- `SeaClear__seed2027`于05:58完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.88213、Recall 0.71218、F1 0.78810、mAP50 0.79548、mAP50-95 0.56002。
- 按mAP50-95选出的最佳轮为epoch124：mAP50-95 0.56298、mAP50 0.79614、Precision 0.88081、Recall 0.71306、F1 0.78811。
- 权重SHA256：best `0b254ec1f4317674537bc928e587fd548b38a82f4f4cca510677ed27cfa5b0cd`；last `4b963387d8795cfcf954f2f757058589ca1c6fc3af3a6e0a5ea0d715a436ad74`。
- 严格审计累计17/22运行、34/44权重，错误0；队列已自动进入`SeaClear__seed2005`，检查时已写入39/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 SeaClear双种子完成与汇总

- `SeaClear__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.85526/0.74031/0.79364/0.81514/0.57405。最高mAP50-95为epoch123的0.57459，与seed2027在epoch124达峰的趋势一致。
- seed2005权重SHA256：best `94c677db9171345dd3c5fc5707908fce6b61caef5d0ea65a06d7d5103318c600`；last `0453ce4f376ad4303bb3d2844073feabb2433a3b28a51a749a28c530f2f9b796`。
- SeaClear两种子末轮mean±sample std：P 0.868695±0.019000、R 0.726245±0.019891、F1 0.790871±0.003923、mAP50 0.805310±0.013902、mAP50-95 0.567035±0.009921。各自最佳epoch的mAP50-95均值为0.568785±0.008210。
- 严格审计累计18/22运行、36/44权重，错误0；队列已自动进入`VisDrone2019-DET__seed2027`，检查时已写入6/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 VisDrone2019-DET seed2027完成并通过单项验收

- `VisDrone2019-DET__seed2027`于06:35完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.36629、Recall 0.28457、F1 0.32030、mAP50 0.25007、mAP50-95 0.13505。
- 按mAP50-95选出的最佳轮为epoch115：mAP50-95 0.13685、mAP50 0.25318、Precision 0.36661、Recall 0.28874、F1 0.32305。
- 权重SHA256：best `34391207a4c56d0751f3929c957407f29716bc39462c8049e620bd89b1f82a5e`；last `89d20e0d30fc5c01e2195583b797608e9ef82ca65d5cc223d3bebc35760aaa1c`。
- 严格审计累计19/22运行、38/44权重，错误0；队列已自动进入`VisDrone2019-DET__seed2005`，检查时已写入23/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 VisDrone2019-DET双种子完成与汇总

- `VisDrone2019-DET__seed2005`完成125轮并通过严格审计；第125轮P/R/F1/mAP50/mAP50-95为0.37259/0.29242/0.32767/0.25467/0.13721。最高mAP50-95为epoch114的0.13941，与seed2027在epoch115达峰的趋势一致。
- seed2005权重SHA256：best `b8fb89b347628930775bb7c0f3bb60aaef4323a810958740fc3b08d8e56d4b26`；last `863775075a3baae0a38aa330373399201f99d1f0105e0d1b07e32afda0c85b52`。
- VisDrone2019-DET两种子末轮mean±sample std：P 0.369440±0.004455、R 0.288495±0.005551、F1 0.323986±0.005213、mAP50 0.252370±0.003253、mAP50-95 0.136130±0.001527。各自最佳epoch的mAP50-95均值为0.138130±0.001810。
- 严格审计累计20/22运行、40/44权重，错误0；队列已自动进入`ZeroWaste-f__seed2027`，检查时已写入39/125轮。指定GPU 4–7运行正常，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 ZeroWaste-f seed2027完成并通过单项验收

- `ZeroWaste-f__seed2027`于07:17完成，return code 0；`results.csv`严格包含epoch 1..125，best/last权重均可读，checkpoint训练参数与冻结协议一致。
- 第125轮：Precision 0.38087、Recall 0.40261、F1 0.39144、mAP50 0.32404、mAP50-95 0.21975。
- 按mAP50-95选出的最佳轮为epoch92：mAP50-95 0.23809、mAP50 0.34379、Precision 0.38520、Recall 0.38891、F1 0.38705。最佳轮与末轮相差0.01834 AP，需在双种子汇总时复核后期回落是否稳定。
- 权重SHA256：best `e4bc1591e27c8cb5ff4e0b8663314cb3094ce35b5f24e7134b9d5c15b839f57c`；last `92d5099b1dfe4a1b94977f514297fc40c297446fe6ea1d2d0c8ebd42fc6b0926`。
- 严格审计累计21/22运行、42/44权重，错误0；最后一项`ZeroWaste-f__seed2005`正在运行，检查时已写入51/125轮。指定GPU 4–7具有本任务显存占用，近20分钟无新增MCE/SIGBUS或GPU Xid。

### 2026-09-22 22次YOLOv8n随机初始化基线全部完成并通过最终审计

- `ZeroWaste-f__seed2005`完成125轮；第125轮P/R/F1/mAP50/mAP50-95为0.40299/0.36652/0.38389/0.30853/0.20592，最高mAP50-95为epoch93的0.22405。权重SHA256：best `51b0468883def4527015cf4258276acc65cfad0032d169c09928bef846a2c4ea`；last `4ba161a323987e727cc9a3d7778acb5807f8200b6d6a48b815639df8759c44e6`。
- ZeroWaste-f两种子末轮mean±sample std：P 0.391930±0.015641、R 0.384565±0.025519、F1 0.387665±0.005337、mAP50 0.316285±0.010967、mAP50-95 0.212835±0.009779；各自最佳epoch的mAP50-95均值为0.231070±0.009928。
- 最终审计状态为`verified_complete`：22/22个独立运行均严格包含125轮，44/44个`best.pt`/`last.pt`均可读，checkpoint内的epochs、batch、imgsz、workers、seed、SGD、GPU映射、`pretrained=False`、模型及数据路径与冻结协议一致；errors 0、incomplete 0。
- 11个数据集的双种子末轮mAP50-95 mean±sample std：AP-10K 0.611050±0.004398；DUO 0.617360±0.003224；ExDark 0.353850±0.003125；HIT-UAV 0.558175±0.002892；HRSID-JPG 0.630205±0.007715；LLVIP-infrared 0.612355±0.001704；LLVIP-visible 0.464310±0.012049；ORCA 0.284715±0.000262；SeaClear 0.567035±0.009921；VisDrone2019-DET 0.136130±0.001527；ZeroWaste-f 0.212835±0.009779。
- 从00:47至07:39队列连续完成，22项均为首次尝试成功，无中断恢复或损坏checkpoint；全程限定物理GPU 4–7与NUMA node 1，未观察到新的MCE/SIGBUS或GPU Xid。管理器在记录`queue_complete`后以`SystemExit: 0`退出，该文本被泛化异常记录为`manager_fatal`，但退出码为0且不影响队列完成或最终审计。
- 完整审计产物保存于`outputs/yolo26_research/runtime_audit/`，其中`audit_latest.json`、`audit_per_run_metrics.csv`、`audit_two_seed_summary.csv`和`AUDIT_LATEST.md`均已更新为最终版。
