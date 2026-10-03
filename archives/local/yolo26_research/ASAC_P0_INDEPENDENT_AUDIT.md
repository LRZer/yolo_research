# ASAC-P0 独立实现审计

日期：2026-09-19  
结论：**CONDITIONAL（核心计算可用，但在补齐运行时协议/来源硬校验前，不应把正式远端输出作为有效 ASAC-P0 裁决）**

## 1. 审计范围与边界

完整阅读并逐项对照：

- `work/yolo26_research/ASAC_PROBE_SPEC.md`
- `work/remote_research_source/scripts/probe_asac_assignment_stability.py`
- `work/remote_research_source/scripts/test_probe_asac_assignment_stability.py`
- 当前 `work/remote_research_source/ultralytics/utils/loss.py` 的 detection/E2E loss 接口
- 当前 `work/yolo26_research/build.py` 的 dataset/dataloader 接口
- 当前 `work/remote_research_source/ultralytics/engine/trainer.py` 的 checkpoint 保存及 trainer 公共接口

未同步远端、未启动 GPU、未改动共享实现。由于本机没有 `torch`/`ultralytics`，不能做真实模型 CPU forward；执行了不依赖二者的全部合成单测和 Python 语法编译。结果为 **10/10 单测通过**、两份脚本均可编译。

审计文件 SHA256：

- spec：`D73DB2D2B19EEBF1C6DEEAD72447109B4C04A5DECCF5A46D7E9DFF824B569FB1`
- probe：`51F8BD750D9C59EEF756198ACD370D661CA277CF8BBE3F9DEF2947EA91BA921D`
- tests：`5A0BBA0901D4101DE9CDED84B7951CF9F962BB871B261CE781F6073EB0CB6491`
- loss API：`E4DD9E41FB4AF4AE085A8D6F686AB158C6B64CE4ADF344258E8679CBD59077DB`
- loader API snapshot：`1C3131ADC60DC9763B9EDCDE853D3BC461EC908C2ED9D1965B2610BFD8C27847`

## 2. 逐项结论

| 项目 | 结论 | 审计说明 |
|---|---|---|
| 双视图几何映射 | PASS | `affine_grid` 使用的逆采样矩阵与标签正变换互为逆：单位坐标中 `x_view=.5+s(x-.5)+t`，采样为 `x=(x_view-.5-t)/s+.5`；`align_corners=False` 下像素中心也一致。图像外填充 `114/255`，框裁剪、可见面积和 2 px 门槛实现正确。 |
| GT 配对/存活筛选 | PASS（报告性缺口） | `local_maps` 保留 view 内 GT 的原始局部序号；`common_ids` 取 clean 与全部三个 q 的存活交集，因此各 q 使用同一对象集合，同时每个 view 的 assigner 仍看见该 view 的全部有效 GT。缺口是输出未记录原始 GT 总数、几何存活率及被过滤对象分布，见 C3。 |
| TAL `target_gt_idx` 语义 | PASS | 当前 TAL 在 `get_targets()` 内临时加 batch offset只用于 gather；`_forward()` 最终返回的仍是 `select_highest_overlaps()` 产生的每图局部索引。脚本按 `local_maps[image][local_gt]` 映射正确，并有越界硬失败。 |
| O2M/O2O assignment 提取 | PASS（需硬断言） | 当前定制 `E2ELoss` 明确构造 O2M `topk=10`、O2O `topk=7, topk2=1`；脚本分别调用两个原生 `get_assigned_targets_and_loss(..., return_details=True)`，读取 `fg_mask/target_gt_idx/target_* /pred_bboxes`，没有自行重写 assignment。缺口是运行时未断言实际 criterion/assigner 参数，见 C1。 |
| P3/P4/P5 映射 | PASS | 由 `prediction['feats']` 的实际 H×W 拼接长度构造 level id，并将每锚 stride 与 `make_anchors` 返回的 stride tensor 逐项相等检查；不是猜分段。 |
| `S` 定义 | PASS | occupancy 只使用等权 `(level,dx,dy)` 点；13×13、extent 1.5、sigma .25；base-2 JS；两边有前景时 `exp(-JS)`，任一边无前景时强制 `S=0`。平均 S 包含这些丢失项，符合规范。 |
| `Qretain` 定义 | PASS | 每正锚 `sqrt(sigmoid(p_assigned_class) * IoU(pred_box, assigned_GT))`，每 GT 取最大，随后 `min/max`；没有被用作 occupancy 权重。相关只在 clean/view 均有前景对象上算。 |
| Spearman 方向 | PASS | 平均秩正确处理 ties；常量或不足 3 点返回 `None`；gate 只接受 `rho >= +0.20`，负相关不会因绝对值而通过。 |
| 单调 gate | PASS | `left + 0.01 >= right` 正确表达“允许最多 0.01 的相邻上升”；至少 3/5 checkpoint、跨 checkpoint 平均 weak→strong drop ≥.02、最强 view 至少 2 个 checkpoint 正相关，且 O2M/O2O 都需通过。 |
| 样本门槛 | PASS | 每个 epoch×head×q 同时检查存活可比较对象 ≥100、双边 assignment ≥100、coverage ≥.50；任一 cell 不足即全轮 `INCONCLUSIVE`，优先于 `NO-GO/GO`。 |
| workers/pinned/panel 固定 | PASS（需强化） | loader 显式 workers=4、shuffle=False、pin_memory=True，并检查实际 `num_workers` 与 `pin_memory`；每 checkpoint 重建并比较 panel hash。`build_yolo_dataset()` 内 `rect = cfg.rect or rect`，因此仅传 `rect=False` 不是形式上的硬保证，见 C2。 |
| BF16/no-update | PASS（缺少事后证据） | torch 在设置 `CUDA_VISIBLE_DEVICES` 后才 import；模型 training 输出格式、所有 BN eval、`torch.inference_mode()`、CUDA BF16 autocast，无 optimizer/backward，也没有调用有状态的外层 criterion update。建议输出运行时 dtype/BN/参数摘要，见 C4。 |
| epoch 识别 | CONDITIONAL | 正常未 strip 的 `epoch10/20/30/40.pt` 用 checkpoint 内 epoch+1 正确；strip 后 `last.pt` 用配置总 epoch=50 也符合本轨迹。但当前代码也把 strip 后 `best.pt` 当作 epoch 50，这是错误的，且未验证五个文件确属同一 B50 轨迹，见 C5/C6。 |
| 输出 JSON/CSV/MD | CONDITIONAL | JSON 含参数、checkpoint 哈希、汇总和裁决；CSV/MD 字段与主 gate 一致。但没有代码/API 哈希、实际 criterion 参数、实际 AMP/BN/loader 状态、轨迹签名和 GT 几何存活率，不能独立证明输出执行的是冻结协议，见 C1–C6。 |

## 3. 必须在正式运行前修正/加固

### C1（高）：运行时不验证实际 O2M/O2O assigner，源码漂移可静默测错 head

当前 `forward_assignments()` 只假设 `trainer.model.criterion.one2many/one2one` 存在。旧 `E2EDetectLoss` 也满足这一形状，但其 O2O 是直接 `topk=1`，不是本规范要求的 YOLO26 `topk=7→topk2=1`；脚本会顺利运行并给出看似有效的 O2O 结果。

**修正建议：** trainer 建立后立即硬断言并写入 JSON：criterion 完整类名；O2M assigner `topk==10 && topk2==10`；O2O assigner `topk==7 && topk2==1`；两个 branch 的 stride 均为 `[8,16,32]`。不符时硬失败，而不是 warning。

### C2（高）：`rect=False` 只作为参数传入，未形成有效运行时不变量

当前 loader API 在 `build_yolo_dataset()` 中执行 `rect = cfg.rect or rect`。因此只传 `rect=False` 无法覆盖一个为 true 的 `trainer.args.rect`。当前 B50 默认值预计为 false，但规范将 `rect=False` 定为硬约束，代码没有断言它。

**修正建议：** overrides 显式加入 `rect=False, multi_scale=False, fraction=1.0, cache=False`；dataset 构建后断言 `dataset.rect is False`、`dataset.augment is False`、长度足够 256；loader 断言顺序 sampler，且最终 `images == batch * max_batches == 256`。把这些“actual”值写入 JSON。

### C3（中高）：存活筛选正确，但输出完全丢失被筛对象，无法审计选择偏差

`object_rows` 只写 `common_ids`。报告中的 `object_pairs` 是“最强 view 仍存活的对象数”，`assignment_coverage` 又是其中双边有 foreground 的比例；它不是原始 GT→四视图存活率。若边界/大目标被大量过滤，结果仍可能过 100/50% gate，而输出无法显示这一事实。规范的剩余风险已经要求报告过滤对象的尺寸/位置分布，当前实现没有保存。

**修正建议：** 每 checkpoint（或仅首次 panel）记录 `clean_gt_total`、各 q `geometric_retained`、四视图交集数/比例；至少按 GT area 分位与中心到边界距离分位报告 retained/dropped。主 gate 可保持不变，但任何正式解释必须同时呈现该覆盖。

### C4（中）：输出缺少足以证明“BF16、无更新、BN 冻结”的实际证据

代码路径本身正确，但 JSON 的 method 只写了概述，没有实际 autocast dtype、模型 training flag、BN 数量/全部 eval、参数/BN-buffer 前后摘要，也没有脚本和 loss/tal 源码身份。发生环境漂移时，旧结果无法复核。

**修正建议：** JSON 写入 Python/PyTorch/Ultralytics/CUDA 版本、GPU 名、脚本/loss/tal/build 文件 SHA256、autocast dtype、model.training、BN total/eval count、梯度启用状态；正式 probe 前后比较 `state_dict` 中参数与 BN buffers 的轻量 hash，任何改变硬失败。

### C5（高）：strip 后 `best.pt` 被错误标成配置末轮

`checkpoint_epoch()` 对文件名 `last` 和 `best` 都在 `epoch<0` 时返回 `train_args['epochs']`。`best.pt` 权重来自最佳轮，不等于第 50 轮；把它标为 epoch 50 会直接污染时间曲线并可能翻转 gate。

**修正建议：** 冻结协议只允许末轮文件名 `last.pt`；明确拒绝 `best.pt`。对 `last.pt` 同时核对 `train_results` 已覆盖 50 轮（或保存前元数据），不能只信配置的计划 epoch 数。

### C6（高）：未验证五个 checkpoint 来自同一 B50 轨迹

当前只校验 epoch 集合、panel hash和各文件 hash。来自不同 seed、不同配置甚至不同 run 的五个 checkpoint 也可能通过并得到正式 `GO`。这违反“同一 B50 轨迹”的首要前提。

**修正建议：** 要求五个 resolved path 的共同 `weights/` 父目录；从 checkpoint `train_args` 构建并比较不可变轨迹签名（至少 model/data/epochs/batch/imgsz/seed/nms、增强参数、研究 gain、run name/project）；对模型结构/类别数也做签名。将签名写入 JSON。若为了跨目录归档，改为显式 `--trajectory-id` 加受审计 manifest，而不是放宽检查。

### C7（中）：冻结阈值仍可由命令行任意改写，输出不会标记协议偏离

strength、grid、sigma、样本门槛、单调容差、drop、rho 和 expected epochs 都是 CLI 参数。这样可以不改代码便扫描测量尺度或降低 gate，最终报告仍只写 `GO`，与预注册要求冲突。

**修正建议：** 正式模式将这些值硬编码为 schema v1 并逐项断言；若保留研究/调试 override，则输出 verdict 必须为 `NON_PROTOCOL_DIAGNOSTIC`，不得产生 `GO/NO-GO`。只有 output path/name/device 可自由配置。

## 4. 建议补充的隔离测试

现有 10 个测试覆盖 JS 极值、quality 不参与 occupancy、missing assignment、Spearman、gate 和 panel mismatch，但没有覆盖最容易出真实错误的 runtime boundary。正式上机前至少新增：

1. 纯 CPU 小张量的 affine 图像/框往返测试：中心点、四角点、scale<1、scale>1、正负平移、裁剪与 70% 临界值。
2. 两张图、每图不同 GT 数的伪 assignment，验证 `target_gt_idx`→`local_maps`→原始 GT 的精确配对；并验证删去中间 GT 后映射不串号。
3. 伪造 O2M/O2O prediction dict 的 level block，验证 anchor index、stride、P3/P4/P5 和逆映射坐标。
4. 对真实 remote 环境做 **1 checkpoint × 1 batch × clean+1 view** 的只读 smoke：断言 criterion/topk、tensor shapes/dtypes、objects 非空、JSON 可 round-trip、模型状态 hash 不变。该 smoke 不是正式结果，不占用正式 probe gate。
5. checkpoint provenance 测试：拒绝 `best.pt`、拒绝混 seed/混 run、拒绝重复 epoch、拒绝非冻结 CLI 门槛。

## 5. 最终裁决

**CONDITIONAL。**

核心测量实现没有发现会系统性翻转结论的数学错误：几何正逆映射、共同 GT 生存交集、原生双 head assignment、等权 occupancy、`S=exp(-JS)`/missing→0、对称 `Qretain`、正向 Spearman 以及单调/动态范围/样本 gate 均与规范一致。

但 C1、C2、C5、C6、C7 使脚本目前不能自证“跑的正是冻结的 ASAC-P0”：错误 criterion、rect 模式、`best.pt`、混轨迹或放宽阈值都可能生成外观正常的正式裁决。应先补齐这些硬失败条件，再做一次单 batch 只读 runtime smoke；全部通过后可升级为 **PASS 并授权正式 ASAC-P0**。在此之前，不建议上传后直接启动 80-forward 正式探针，也不应接受其 `GO/NO-GO` 为有效结果。
