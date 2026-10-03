# ASAC-P0：分配稳定性增强课程的零更新探针规范

日期：2026-09-19  
状态：v2 协议与 C6 来源证明完成第二轮加固；**尚未上传远端、尚未占用 GPU、尚未启动探针或训练**。  
用途：在实现 ASAC 控制器之前，用固定 YOLO26n/VOC 基线 checkpoint 判断“assignment stability”是否真是一个可感知增强强度、并与监督质量有关的反馈信号。

## 1. 假设与裁决范围

待证伪假设：

> 对同一训练图像和同一 GT，增强越强，YOLO26 原生 O2M/O2O assigner 在规范坐标中的正样本候选分布越不稳定；这种不稳定性与该 GT 的分配监督质量保持度正相关。因此它有资格作为 scale/translate 增强课程的因果反馈，而不只是 epoch 或置信度的替身。

ASAC-P0 是固定权重、无更新诊断，不回答“ASAC 能否提 AP”，只回答是否值得实现最小控制器。`GO` 只授权后续 **scale/translate 版 ASAC**；本探针不包含 Mosaic，因此不能据此声称 Mosaic 控制已验证。

## 2. 冻结数据与执行约束

- checkpoint：同一 B50 轨迹的 epoch `11/21/31/41/50`，文件严格为 `epoch10.pt/epoch20.pt/epoch30.pt/epoch40.pt/last.pt`；拒绝 `best.pt`、strip 后的 numbered checkpoint、内部 epoch 不一致或训练行数不足的文件；
- 数据：数据 YAML 的 `train` split，即 VOC07+12 trainval；不读取 `val/test` 指标；
- panel：train split 关闭数据增强、`rect=False`、`shuffle=False` 后最前 `4×64=256` 图；
- 每个 checkpoint 都重新构造 panel，并对文件路径、类别、框、协议参数计算 SHA256；五次指纹必须完全相同，否则直接报错；
- `workers` 参数固定为 `4`，运行时断言 dataloader 的实际 `num_workers==4`；
- 显式 `pin_memory=True`，运行时断言 pinned memory 未被关闭；
- 只暴露一张允许的物理 GPU（默认物理 4）；后续正式执行时仍须等待现有正式训练结束；
- 模型保持 training 输出格式以获得原始双头张量，但所有 BatchNorm 切到 eval；全程 `inference_mode`，无 optimizer、无 backward、无权重或 BN 统计更新；
- 推理使用与现有实验相符的 BF16 autocast；固定 seed 42；
- 输出是诊断数据，不报告 VOC test AP，也不据此调 AP 阈值。

### 2.1 v2 正式运行证明链

正式运行在第一次 forward 前必须同时通过：

- 五个 checkpoint 位于同一个 `weights/` 目录；嵌入的 `train_args`、git/source 元数据、模型结构签名完全一致；“彼此一致”只是必要条件，不能替代下述冻结 B50 精确匹配；
- run 目录存在 `args.yaml` 和完整 50 行、末行 epoch 50 的 `results.csv`，两者路径与 SHA256 落盘；
- 轨迹必须是冻结 B50：真实 `args.yaml` 的 **124 个字段全部逐项冻结**，并额外把历史产物中尚不存在的 `tqgb_cls_m2o_gain/tqgb_cls_o2m_gain/tqgb_direction` 唯一规范化为 `0/0/none`，形成 127 字段 manifest；manifest SHA256 为 `58054c007178b94fccd87d8c6d1ae265a5487cc3b3609cd58301c7fa68c1fe7a`；
- checkpoint `train_args` 与 run `args.yaml` 必须分别独立完成“字段完整、无冻结集合外字段、类型正确、绝对路径规范化、逐项等于冻结值”的校验，再要求二者相等；除上述三个历史 TQGB 字段外，任何字段缺失都硬失败；
- model/data/project 固定为真实 B50 绝对路径；实际 `run_dir` 和 `save_dir` 都必须等于规范化后的 `Path(project)/name`，model/data 文件必须存在且路径、内容 SHA256 落盘；
- 冻结内容不只包含 seed/batch/增强，还覆盖 `amp=bf16`、`cls_remap`、optimizer、完整 LR/warmup/weight-decay schedule、box/cls/dfl 等 detection loss gains、全部 O2O KD 参数、全部 HSV/几何/混合增强与 TQGB gain/direction；
- checkpoint 模型结构不能只在五次之间相同：必须精确为 `DetectionModel + Detect`、VOC 20 个固定类别、`nc=20`、stride `8/16/32`、训练图参数量 `2,511,600`，model YAML 内 `nc=20` 且 scale 为 `n` 或历史上未显式记录；
- 五个 checkpoint 内容 SHA256 采用既有 PBGA 与 SARC 两份只读诊断中相互一致的记录进行 pin：`epoch10 ac6833…`、`epoch20 ea4b17…`、`epoch30 cdbb6e…`、`epoch40 a2f334…`、`last 49076e…`；文件名、内部 epoch、完整 metadata 即使全部伪造，只要内容 hash 不符仍硬失败；
- `last.pt` 即使已 strip，也必须由 checkpoint 内 50 行 `train_results` 及 run 的 50 行 `results.csv` 双重证明完成；不能只信计划的 `epochs=50`；
- criterion 类必须为当前 `ultralytics.utils.loss.E2ELoss`；O2M assigner 必须是 `topk/topk2=10/10`，O2O 必须是 `7/1`；两支 stride 都必须为 `8/16/32`，head 必须存在原生 end-to-end one2one 路径；
- trainer 强制 `rect=False, multi_scale=0, fraction=1, cache=False`；dataset 实际 `rect=False, augment=False`；loader 必须为顺序 sampler、实际 workers 4、pinned memory 开启；
- 最终 panel 必须恰好 `64×4=256` 张且图像路径不重复；不能因最后一个不满 batch 静默缩小；
- 输出 Python/PyTorch/Ultralytics/CUDA/GPU 信息，以及 probe、loss、TAL、data-build、trainer、Detect head 六类关键源码文件的绝对路径和 SHA256；
- 每个 checkpoint 在探针前后分别计算全部参数及所有 BN buffer 的 SHA256；任何变化硬失败。模型保持 training-format 输出，但全部 BN 必须为 eval、optimizer 必须不存在；每次 forward 断言 inference mode 与 CUDA BF16 autocast 真正开启。

以上证明均进入 JSON；关键 trajectory、source bundle、panel、geometry 指纹也进入 decision CSV/Markdown，结果不再只依赖命令说明。

计算量为 `5 checkpoints × 4 views（clean+3强度）× 4 batches = 80` 个 YOLO26n batch forward；一次 forward 同时产生 O2M/O2O，assigner 分别读取，不重复跑模型。

## 3. 受控增强：clean 对三个嵌套 affine view

### 3.1 为什么首轮不放 Mosaic

Mosaic 需要四图 GT 来源、裁剪及逆映射的严格对应。把它直接放进 P0，会同时检验“稳定性机制”和“四图实现是否正确”，一旦失败无法归因。P0 先只检验当前基线中已经存在、且可精确逆映射的 `scale=0.5` 与 `translate=0.1`。若 P0 `GO` 且 affine-ASAC 训练优于强度匹配控制，再单独做 Mosaic-P1。

### 3.2 固定随机方向与嵌套强度

每张图由 `SHA256("asac-v1|seed|absolute_path")` 生成三个固定的 `r_s,r_x,r_y∈[-1,1]`。不同 checkpoint 对同一图使用完全相同的方向；三个强度只放大幅度：

```text
q ∈ {0.25, 0.50, 0.75}
s(q)  = 1 + q · 0.50 · r_s
tx(q) =     q · 0.10 · r_x
ty(q) =     q · 0.10 · r_y
```

因此三个 view 是嵌套的，最大强度范围为 `scale∈[0.625,1.375]`、平移不超过图宽/高的 `7.5%`。这比随机为每个 q 重新抽方向更适合检验单调性。

图像以中心缩放和平移，越界填充值为 `114/255`；标签用同一正向变换并裁剪到画布。标签只有在：

- 裁剪后可见面积比例 `≥0.70`；
- 裁剪后宽、高均 `≥2 px`

时保留。每个 view 的 assigner 仍看到该 view 中全部有效 GT，以保留真实 GT 竞争；但跨强度统计只使用在 clean 和三个 q 中都有效的同一批原始 GT，从而避免不同 q 的样本组成变化伪造单调趋势。

## 4. O2M/O2O 分配稳定性

### 4.1 原生 assignment

每个 view 只调用当前源码的原生 criterion：

- O2M：`one2many` / TAL top-k 10；
- O2O：`one2one` / 当前 YOLO26 top-7→top-1 路径。

每个正样本保留 `target_gt_idx`，通过每图的原始 GT 序号重新关联。P3/P4/P5 映射由 feature-map `H×W` 拼接区间构造，并用 `make_anchors` 返回的 stride 逐锚点交叉断言为 `8/16/32`；不允许靠猜测切分。

### 4.2 规范坐标与平滑候选分布

对 view 中分给 GT `j` 的正样本锚点，先用已知 affine 参数反变换回 clean 坐标，再按原始 GT 尺寸归一：

```text
dx = (x_anchor_clean - x_gt_clean) / w_gt_clean
dy = (y_anchor_clean - y_gt_clean) / h_gt_clean
```

每个 GT、每个 head 构建 `(level,dx,dy)` 占据分布：

- level 为 P3/P4/P5 三个独立平面；
- `dx,dy` 固定裁剪到 `[-1.5,1.5]`；
- 每个平面为 `13×13` 网格；
- 每个正样本以固定 `σ=0.25 GT unit` 的 Gaussian splat；
- 所有正样本权重相同，归一化后得到 `π_j`。

平滑分布使只有一个正样本的 O2O 也能对亚网格位移连续响应。level 平面互不重叠，因此 P3↔P4/P5 改派会被明确计为不稳定。

关键防循环设计：`target_score`、分类概率和 IoU **不参与** `π_j` 的权重，否则后续“稳定性—质量相关”会成为自我证明。

clean 与强度 q 的 object-level 指标为：

```text
JS_j(q) = JS_base2(π_j_clean || π_j_q),  JS∈[0,1]
S_j(q)  = exp(-JS_j(q))
```

若 GT 在任一 view 没有前景分配，有限 JS 的 `exp(-1)` 仍有 0.368，会错误奖励“完全丢失”；因此该情况显式记为 `S=0`，并单独报告 paired assignment coverage。另记录主层翻转率与 GT-unit 质心漂移，便于解释。

## 5. 与稳定性独立的监督质量

每个正锚点计算：

```text
quality_anchor = sqrt(p_assigned_class · IoU(pred_box, assigned_GT))
```

每个 GT 取正锚点中的最大值 `Q_j`，clean/view 的质量保持度为：

```text
Qretain_j = min(Q_clean,Q_view) / max(Q_clean,Q_view)
```

它是对称的 `[0,1]` 指标，避免只因某个增强偶然提高分数就被当作“损坏”。另独立记录 TAL target-score 保持度，但不用于 GO gate。

相关性只在 clean 和 view 都有前景的 GT 上计算 `Spearman ρ(S,Qretain)`，避免“两个都缺失时同时为零”人为制造高相关。主相关门槛只看最强的 `q=0.75`；较弱 q 用于画曲线而不增加多重选择自由度。

## 6. 预注册自动裁决

v2 将完整正式协议作为代码常量冻结：batch/imgsz/workers/panel 大小、seed、三档 q、scale/translate、可见度与最小框、occupancy 网格/范围/σ、stride/epoch、样本门槛以及全部 GO/NO-GO 阈值。CLI 仍允许为方法学敏感性分析传入不同值，但只要任一项偏离，最终 verdict 必须为 `NON_PREREGISTERED`，列出逐项差异，且不得输出或引用自动 GO/NO-GO。输出目录、名称、数据/checkpoint 路径和允许的物理 GPU 不属于统计阈值，但数据与 checkpoint 路径仍受 B50 来源硬校验。

### 6.1 前置完整性；不满足则 INCONCLUSIVE 或硬失败

硬失败：

- checkpoint epoch 不是严格的 `11/21/31/41/50`；
- checkpoint 不是规定文件、来自不同 run、缺少完整 train_results/run 证据或不符合冻结 B50 配置；
- criterion/assigner/head 路径或 top-k/topk2 不等于冻结接口；
- 五个 checkpoint 的 panel SHA256 不一致；
- 五个 checkpoint 的 GT geometry SHA256 或关键源码 bundle SHA256 不一致；
- stride/level 映射不一致；
- dataset 不是非增强、非 rect、全量 fraction=1 的顺序 train loader；
- 实际 workers 不是 4、pinned memory 被关闭、panel 不是恰好 256 个唯一图像；
- BF16/inference mode/BN eval/无 optimizer 断言失败，或参数/BN buffer 前后哈希发生变化。

每个 `epoch × head × q` cell 必须同时满足：

- 可比较 GT 数 `≥100`；
- clean/view 均有前景的 GT 数 `≥100`；
- paired assignment coverage `≥0.50`。

任一 cell 不足，整轮为 `INCONCLUSIVE`，不把低样本噪声判作机制失败。

### 6.2 GO；以下条件 O2M 与 O2O 都必须通过

1. 五个 checkpoint 中至少 `3` 个，其平均 `S(q)` 随 q 非递增；相邻点允许 `0.01` 数值容差。
2. 跨 checkpoint 平均的 `S(0.25)-S(0.75) ≥ 0.02`；否则信号动态范围不足以稳定控制增强。
3. 在 `q=0.75`，至少 `2` 个 checkpoint 的 object-level `Spearman ρ(S,Qretain) ≥ +0.20`。

相关门槛收紧为**正**相关，而不是只看 `|ρ|`：负相关即“越稳定质量越差”，不支持 ASAC 预期机制。

通过时结论只能写为：

> `GO for affine-ASAC prototype`: 两个原生 head 都存在可感知 scale/translate 强度、且与监督质量保持正相关的 assignment-stability 信号。

### 6.3 NO-GO

样本充足但任一 head 未达到单调次数、动态范围或质量相关门槛，即 `NO-GO`。不扫描 q、网格、σ 或相关阈值来翻转结论。此时增强主线应降级，优先继续 PBGA/SARC 或另做不依赖该反馈的固定课程对照。

## 7. 若 GO，后续最小训练矩阵

P0 通过后才允许实现训练侧状态；仍先不纳入 Mosaic：

| ID | scale/translate 策略 | 目的 |
|---|---|---|
| A0 | 官方固定 `scale=.5, translate=.1, close_mosaic=10` | 当前基线 |
| A1 | 固定 epoch schedule，逐 epoch 匹配 A2 实际 q 的均值/边际 | 控制“仅改变强度曲线” |
| A2 | affine-ASAC：train-only panel 的稳定度反馈 q | 检验反馈本身 |
| A3 | 打乱 A2 的 epoch stability 序列、保持 q 边际分布 | 检验真实时序反馈 |

A2 仍使用广域筛选中的门槛：固定 epoch 50 对 A0 `≥+0.003`、对 A1 `≥+0.002`、last-5 同号、训练开销 `<12%`。只有 A2 相对 A1 成立，才进入 Mosaic-P1；不能因为只赢 A0 就声称 assignment-stability 机制成立。

## 8. 实现与输出

实现：

- `work/remote_research_source/scripts/probe_asac_assignment_stability.py`
- `work/remote_research_source/scripts/test_probe_asac_assignment_stability.py`

预计输出：

- `*_objects.csv`：每 GT、checkpoint、head、q 的稳定性、质量保持、主层翻转与质心漂移；
- `*_geometry.csv`：clean panel 的每个原始 GT、面积/宽高、框到边界和中心到边界距离，以及每个 q 与四视图交集的存活标记；
- `*_summary.csv`：每 cell 的覆盖率、均值、Spearman；
- `*_decision.csv`：机器可读裁决；
- `*.json`：完整协议、checkpoint 哈希、汇总和裁决；
- `*.md`：人读报告。

JSON/Markdown 同时报告 clean GT 总数、每个 q 的几何存活数/比例、三视图交集比例，以及 retained/dropped 两组的面积、sqrt-area、框边界距离、中心边界距离的 mean 与 p10/p25/p50/p75/p90。几何行及摘要在五个 checkpoint 间必须逐值同指纹。

未来远端命令模板（本轮未执行）：

```bash
PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research \
CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
  /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python \
  /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts/probe_asac_assignment_stability.py \
  --checkpoints \
    /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/epoch10.pt \
    /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/epoch20.pt \
    /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/epoch30.pt \
    /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/epoch40.pt \
    /home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/last.pt \
  --data /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml \
  --physical-device 4 \
  --output-dir /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2
```

注意：脚本自身在 import torch 前设置 `CUDA_VISIBLE_DEVICES`；外层环境变量只是双重限制。正式运行前须重新核对 checkpoint 路径、GPU 空闲和当时的 MCE/SIGBUS 日志。

## 9. 已完成的本地验证与剩余风险

本地 CPU 合成测试覆盖（v2 共 33 项，另含逐字段 subtest）：

- 路径哈希增强参数的确定性与 q 嵌套比例；
- 同分布 JS=0、跨 level JS=1；
- occupancy 不使用质量加权；
- 前景丢失时 S=0；
- Spearman 正/负/常量边界；
- GO、非单调 NO-GO、质量无关 NO-GO、样本不足 INCONCLUSIVE；
- panel 指纹不一致硬失败；
- criterion 原生 10/10 与 7/1 接口通过、旧 direct-top1 O2O 拒绝；
- numbered strip、`best.pt`、不完整 `last.pt` 拒绝；
- 同 run 精确 B50 来源通过、混 seed/混轨迹拒绝；checkpoint 和 args.yaml 任一非语义默认字段缺失均拒绝；
- 固定 model/data/project/name 及 `run_dir=project/name`；错误 YOLO26n/VOC20 结构拒绝；manifest 自报 hash 不能伪造；
- 五个既有 B50 checkpoint 的内容 SHA256 被逐文件 pin，任一替换或重写拒绝；
- 每个 HSV/几何/混合增强字段、AMP/cls_remap/optimizer/LR/loss gain、两个 TQGB gain 与 direction 逐项偏离均拒绝；
- 复审报告中的整组恶意轨迹（wrong model、`degrees=25`、TQGB 非零、双向 route、wrong project）即使五个 checkpoint 拥有相同 signature 也硬失败；
- 完整冻结参数可裁决、任一参数偏离强制 `NON_PREREGISTERED`；
- retained/dropped GT 几何统计。

主要剩余风险：

1. 这是受控 affine，不是 Ultralytics `RandomPerspective` 的逐像素完全复刻；它用于机制诊断，不能把数值直接解释成正式训练增强强度。
2. 只统计在最强 view 仍保留的 GT，会偏向更居中、可见度更高的对象；v2 已强制输出每个 GT 的几何存活标记及 retained/dropped 尺寸/边界分布，但这种完整报告只能揭示偏差，不能消除偏差。
3. `13×13/extent=1.5/σ=.25` 是预注册测量尺度；虽有平滑避免 O2O 量化跳变，仍应在最终论文中做一次不改结论的敏感性审计，而不能据 test AP 反调。
4. `Qretain` 来自同一 assigner 使用的分类与 IoU，因此是机制相关的外部读数，但不是独立泛化指标；正相关只能支持反馈合理性，不能证明会提升 AP。
5. 五个 checkpoint 属于同一训练轨迹，不是五次独立重复；P0 是机制筛选，真正结论仍需 A0/A1/A2 多 seed 和健康硬件复现。
6. 固定前 256 图可能不是整个 trainval 的无偏抽样。若 P0 临界，不扩展/换 panel 追求 GO；只能标为边界证据，或预注册第二个不重叠 panel 做复核。
7. 当前有已知主机内存硬件故障。任一 SIGBUS 使该次探针无效；即使无 SIGBUS，成果也只能作为故障主机探索证据，最终需健康硬件复现。
8. `2,511,600` 是现有 VOC20 训练格式 checkpoint 的参数口径；正式 GPU 前必须先做一次 `1 checkpoint × 1 batch` 只读 smoke，确认真实历史 checkpoint 的完整 127 字段规范化、结构记录与该口径吻合。若不吻合应停止并审计证据，不能放宽冻结值来追求通过。

## 10. Provenance v4 终态 strip 补充（2026-09-19）

真实远端 CPU preflight 发现：`epoch10/20/30/40.pt` 均含 124 个历史 `train_args` 字段，而训练结束后的 `last.pt` 只有 123 个，唯一缺失 `save_dir` 且 `epoch=-1`。源码核对证明 `final_eval()` 会对 `last.pt` 调用 `strip_optimizer()`；后者把参数过滤为 `DEFAULT_CFG_KEYS`，确定性移除运行期 `save_dir`。

因此 provenance revision 4 只增加一个封闭例外：文件名必须是 `last.pt`、内部 epoch 必须是 `-1`、除三个历史 TQGB no-op defaults 外唯一缺字段必须恰为 `save_dir`；恢复值必须从内容固定 checkpoint 的 `checkpoint_path.parent.parent` 推导。恢复后仍需与完整冻结 manifest、`project/name`、真实 run 目录、独立 `args.yaml`、50 行 `results.csv`、五个 checkpoint 内容哈希和模型结构同时一致。

该例外不适用于 numbered checkpoint、未 strip 的 `last.pt`、任意第二个缺失字段或任意额外字段。新增端到端 synthetic 五 checkpoint panel 与恶意缺字段回归后，本地测试为 37/37 PASS。详细原因、哈希和远端门禁见 `outputs/yolo26_research/ASAC_P0_V4_LOCAL_REVISION.md`。

## 11. Provenance v5 运行期参数转换补充（2026-09-19）

真实远端 v4 CPU preflight 进一步发现：独立 `args.yaml` 中 `warmup_bias_lr=0.1`，而五个内容哈希已固定的 checkpoint 均为 `warmup_bias_lr=0.0`；除该字段以及已知的终态 `last.pt/save_dir` strip 外，其余字段完全一致。源码核对确认这是确定性运行期转换：`args.yaml` 在 optimizer 构建前保存；当前 B50 使用 `optimizer=auto`，`build_optimizer()` 根据迭代数选择具体优化器后明确执行 `self.args.warmup_bias_lr = 0.0`，随后 checkpoint 保存的是已变更的 trainer args。按 `ceil(16551/64)×50=12,950` 次迭代口径，本轨迹进入 MuSGD 分支；这一分支选择不改变上述赋值事实。

provenance revision 5 因此冻结两份相互关联、不可互换的完整 127 字段 manifest：

- launch manifest：独立 `args.yaml` 必须为 `optimizer=auto, warmup_bias_lr=0.1`；SHA256=`58054c007178b94fccd87d8c6d1ae265a5487cc3b3609cd58301c7fa68c1fe7a`；
- checkpoint runtime manifest：只允许在 `optimizer=auto` 前提下，把唯一字段 `warmup_bias_lr` 从 `0.1` 确定性转换为 `0.0`；SHA256=`a6bf4472b2eb23597babdea8190948dfeae56fa45fdcf764dd0196454b466d41`。

checkpoint 写回 launch 值 `0.1`、args.yaml 写成 runtime 值 `0.0`、任何第三值、错误 optimizer 或任何第二个不一致字段均硬失败。v4 的 `last.pt` 唯一 `save_dir` 恢复规则保持不变。该修订只修正来源证明，不改 panel、增强、指标、阈值或 GO/NO-GO 规则；本地与远端 CPU 回归均为 39/39 PASS，真实五 checkpoint CPU provenance/structure preflight 已通过。

## 12. Provenance v5 最终内容封存（2026-09-19）

在第一次正式 ASAC-P0 前向之前，已对顺序固定的前 256 张 train 图像及其逐一对应的 YOLO 标签文件做字节级封存。完整逐文件清单保存在远端 `diagnostics/asac_p0_panel_prereg_v1_20260919_2017/panel_content_manifest.json`，有序 image+label 内容 SHA256 为：

`8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`

正式脚本必须在每个 checkpoint 完成四批读取后重新计算该值；图像、标签、顺序或数量任一变化均硬失败。模型 YAML 与数据 YAML 也分别固定为 `afc4600876cc3a996aa65f5894d2a3005f25baa29b574bf46eff373779451b26`、`5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40`。每个 checkpoint 在反序列化前、反序列化后以及 trainer 构造前后都需保持内容哈希不变，以避免审计窗口内文件变化。
