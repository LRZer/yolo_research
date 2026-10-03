# ASAC-P0 v2 加固后独立复审

日期：2026-09-19  
裁决：**CONDITIONAL / NO-RUN**

> C1–C5 与 C7 已形成实际硬失败或非预注册裁决；核心几何、索引、稳定性、质量和决策数学未发现新的翻转性错误。  
> **C6 仍有一个阻断漏洞：代码只证明五个 checkpoint 彼此一致，没有完整证明它们就是冻结的 YOLO26n/VOC B50 轨迹。** 在补齐该精确来源约束并回归前，不授权上传后启动正式 80-forward ASAC-P0，也不接受其 `GO/NO-GO` 为有效裁决。

## 1. 复审范围与边界

完整阅读并逐行核对：

- `work/remote_research_source/scripts/probe_asac_assignment_stability.py`
- `work/remote_research_source/scripts/test_probe_asac_assignment_stability.py`
- `work/yolo26_research/ASAC_PROBE_SPEC.md`
- `outputs/yolo26_research/ASAC_P0_INDEPENDENT_AUDIT.md`

同时只读核对了本地留存的实际接口：

- `ultralytics/utils/loss.py` 中 `v8DetectionLoss`、`E2ELoss` 及 assignment details；
- dataset/dataloader build 接口；
- checkpoint 保存接口；
- 同步快照中的 `TaskAlignedAssigner`、`Detect`、`DetectionTrainer` 接口。

本轮没有访问或改动远端，没有使用 GPU，也没有修改 probe、测试、spec 或 Ultralytics 实现。本机 Python 没有 `torch/ultralytics`，因此不能在本地执行真实模型 forward；此限制不影响纯逻辑单测、语法编译和来源校验的反例复现，但仍要求修复后先做一次远端只读单 batch smoke，才能进入正式 80-forward。

审计输入 SHA256：

| 文件 | SHA256 |
|---|---|
| probe | `AEE8BD1B682AB02DB62ACD03EBD918B435C5C3FF6A514214452CC1579948726C` |
| tests | `B70E6A70F76C61B625BA56D5386CB7A5004058C68FEC6A2D9DBBE6AAC08327CD` |
| spec | `DC59F473F507FEAD4F284B42048E40A10A84C8254A30970794CA53DD69CD139B` |
| prior audit | `9BF8AB6FE90DA30052ACD0BB61B5D1B8E52B8A91FDFFCCC09BBDE8D8FBD461A6` |
| local loss snapshot | `E4DD9E41FB4AF4AE085A8D6F686AB158C6B64CE4ADF344258E8679CBD59077DB` |
| local data-build snapshot | `1C3131ADC60DC9763B9EDCDE853D3BC461EC908C2ED9D1965B2610BFD8C27847` |
| local trainer snapshot | `FD69BCACBDF6DA0EE17E6C0F3154F4DD9477D932B249DE0408C362012A7B3708` |

## 2. C1–C7 逐项复核

| 项目 | 复审结论 | 实际执行语义 |
|---|---|---|
| C1：原生 O2M/O2O contract | **PASS** | 第一次 forward 前要求 criterion 完整类名严格为 `ultralytics.utils.loss.E2ELoss`；O2M `topk/topk2=10/10`、O2O `7/1`；两支 stride 为 `8/16/32`；branch assignment 方法可调用；Detect 存在 one-to-one head 且为 end-to-end。旧 direct-top1 O2O 会硬失败。实际 contract 写入每个 checkpoint JSON，并跨 checkpoint 比较。 |
| C2：固定非 rect panel/loader | **PASS** | trainer overrides 显式固定 `rect=False, multi_scale=0, fraction=1, cache=False` 并检查 actual；dataset 检查 `rect=False, augment=False, len>=256`；loader 检查 workers=4、pin memory、顺序 sampler；末端检查 `batch*max_batches=images=paths=256` 且路径唯一。违反任一项硬失败。 |
| C3：几何存活与选择偏差报告 | **PASS** | geometry CSV 覆盖所有 clean GT，逐 q 写存活标志及四视图交集；JSON/MD 写 clean 总数、逐 q 数量/比例、交集比例、retained/dropped 面积与边界分布；geometry 行 SHA256 在五个 checkpoint 间必须一致。 |
| C4：BF16/BN/零更新证明 | **PASS（需远端 smoke 实证）** | 每次 forward 断言 `inference_mode` 与 CUDA BF16 autocast；模型保持 training-format 输出，全部 BN eval；optimizer 必须不存在；全部参数和 BN buffers 前后 SHA256 必须不变。输出 Python/PyTorch/Ultralytics/CUDA/GPU、实际 dtype、criterion contract 及六类关键源码路径/hash。代码路径未调用 optimizer/backward 或外层有状态 criterion update。 |
| C5：epoch/`best.pt`/strip | **PASS** | 仅接受五个精确文件名；`best.pt` 明确拒绝；numbered checkpoint 必须未 strip 且内部 epoch 精确一致；`last.pt` 只允许内部 49 或 -1；每个 checkpoint 的嵌入 `train_results` 必须恰好覆盖对应 epoch，另要求 run `results.csv` 恰好 50 行且末行为 epoch 50。 |
| C6：同一冻结 B50 来源 | **FAIL / BLOCKING** | 同一 `weights/` 父目录、五个 epoch、跨 checkpoint train-args/git/model-structure 一致、run `args.yaml/results.csv` 存在等检查已补上；但精确 B50 值只冻结了部分字段。一个一致但非 YOLO26n、增强已改、TQGB 非零、project 任意的轨迹仍可通过，详见第 3 节。 |
| C7：预注册参数不可静默改写 | **PASS** | v2 的 batch/imgsz/workers/panel、三档 q、几何、occupancy、epoch、样本和全部决策阈值均来自 `FROZEN_PROTOCOL`。任一 CLI 偏离均进入 deviations，并使最终 verdict 优先成为 `NON_PREREGISTERED`，不能产生 `GO/NO-GO`。额外对 22 个 CLI 字段逐一变异，全部被识别。 |

## 3. 阻断项 B1：C6 仍允许“同轨但非冻结 B50”

### 3.1 精确缺口

`TRAJECTORY_FIELDS` 中以下 **19 个字段没有进入 `FROZEN_B50_ARGS` 的精确值校验**：

```text
model
data
optimizer
hsv_h
hsv_s
hsv_v
degrees
shear
perspective
flipud
fliplr
mixup
cutmix
copy_paste
tqgb_cls_m2o_gain
tqgb_cls_o2m_gain
tqgb_direction
project
```

当前逻辑只要求这些值在五个 checkpoint 之间相同；`data` 另只要求等于本次用户传入的 `--data`，`model_structure` 也只要求五次彼此相同。它没有要求：

- `model` 是冻结的 `yolo26n_baseline.yaml`；
- `data` 是冻结 VOC server YAML，而不是任意与 CLI 自洽的 YAML；
- 全部固定增强参数等于 B50 值；
- TQGB 两个 gain 为零/历史 checkpoint 中语义等价的 absent，direction 为 none/absent；
- `project` 是冻结 runs 目录，且实际 `run_dir == project/name`；
- model structure 不仅彼此相同，而且等于预期 YOLO26n/VOC20 结构。

此外，影响训练轨迹但当前甚至未纳入 `TRAJECTORY_FIELDS` 的 B50 配置（例如 `amp`、`cls_remap`、优化器/LR schedule 与 detection loss gains）仍只能依赖人工路径约定。最稳妥的修复不是继续零散补字段，而是对一份规范化 B50 训练 manifest 做精确 hash/逐项校验，或直接 pin 已确认的五个 checkpoint SHA256。

### 3.2 可重复的隔离反例

在不触碰远端的本地隔离测试中，构造五个 epoch 正确、同一 `weights/`、相同 git/structure、完整 `args.yaml` 与 50 行 `results.csv` 的 metadata records，然后统一改为：

```text
model=/tmp/not-yolo26n.yaml
degrees=25.0
tqgb_cls_m2o_gain=0.08
tqgb_cls_o2m_gain=0.15
tqgb_direction=bidirectional
project=/tmp/other-project
```

调用当前 `validate_b50_trajectory()` 的结果是：

```text
UNEXPECTED_PASS /tmp/not-yolo26n.yaml 25.0 0.08 /tmp/other-project
```

这是 validator 级反例；正式 CLI 上游会先加载真实 checkpoint，因此攻击/事故需要真实的五个自洽 checkpoint 文件，但一旦文件存在，当前 C6 gate 不会因上述非 B50 值拒绝它们。它证明问题不是“报告字段少”，而是正式 `GO/NO-GO` 的来源前置条件可被错误轨迹满足。

### 3.3 最小修复要求

正式运行前至少完成以下全部事项：

1. 把上述 19 个字段纳入精确 B50 校验；历史 B50 中缺失的 TQGB 字段先规范化为语义默认值（两个 gain `0.0`、direction `none`），再比较，不能简单把“任意 absent/任意值”视为同一配置。
2. 固定预期 model/data/project/name，并断言规范化后的 `run_dir == Path(project) / name`；对 model YAML/结构/nc/stride 与 VOC 类别签名比较预期值，而不只是五次相等。
3. 将 checkpoint `train_args` 与 run `args.yaml` 的整个规范化冻结字段集合逐项比较；不要只比较当前 `REQUIRED_TRAJECTORY_FIELDS` 子集。
4. 将尚未进入 `TRAJECTORY_FIELDS`、但会改变 B50 权重的训练配置纳入规范 manifest；优先保存并验证 manifest SHA256，避免以后新增字段再次漏检。
5. 新增负测并要求硬失败：wrong model、wrong data、任一 HSV/几何增强偏离、任一 TQGB gain 非零、wrong direction、wrong project/run relation、wrong structure，以及 `args.yaml` 在任一冻结字段上与 checkpoint 不一致。

可选但最强的额外保险：在远端重新只读确认后，pin 既有证据中记录的五个 B50 文件 SHA256；这样复制目录不影响合法性，而任何换轨/重写 checkpoint 都会立即失败。

## 4. 数学、索引与执行语义复核

### 4.1 affine 与 GT 几何

- 标签正变换为 `u_view = 0.5 + s(u_clean-0.5) + t`；`affine_grid` 使用的逆采样变换为 `u_clean = (u_view-0.5-t)/s + 0.5`，二者互逆。
- `align_corners=False` 与归一化平移中的 `2t` 一致；图像外填充通过先减后加 `114/255` 实现。
- 框先做未裁剪的同一正变换，再裁到 `[0,1]`；可见率为 clipped/raw transformed area；`>=0.70` 与宽高 `>=2 px` 均按规范执行。
- 每个 view 的 assigner 接收该 view 全部有效 GT；统计对象集合是 clean 与三个 q 的几何存活交集，避免 q 间对象组成变化伪造单调性。

### 4.2 GT/anchor/level 索引

- `base_local_maps` 与变换后的 `local_maps` 均保持每图 GT 的原始局部序号；变换后的 labels 按 image、再按原始行顺序拼接。
- 当前 TAL 返回的 `target_gt_idx` 是每图局部索引；`get_targets()` 内的 batch offset 只用于 gather，返回值仍是 offset 前的局部索引。脚本按 `local_maps[image][local_gt]` 恢复原始 GT，并对上界硬失败。
- P3/P4/P5 由实际 feature `H*W` block 构造；逐锚 stride 与 `make_anchors` 的 stride tensor 精确交叉验证。anchor center 先转 view 归一坐标，再用已知 affine 逆变换到 clean，最后除以 clean GT 宽高。

### 4.3 `S`、`Qretain`、Spearman 与 gate

- occupancy 只读等权 `(level,dx,dy)`，不读 class/IoU/target-score；三个 level 为互不重叠平面，`13x13`、extent `1.5`、sigma `0.25`。
- JS 使用 base 2 并裁在 `[0,1]`；两边均有 foreground 时 `S=exp(-JS)`，任一边缺失时显式 `S=0`。平均 `S` 包含这些缺失项。
- anchor quality 为 `sqrt(sigmoid(assigned-class logit) * aligned IoU(pred, assigned GT))`；每 GT 取最大；`Qretain=min/max`。相关只在 clean/view 都有 foreground 的对象上计算。
- Spearman 使用平均秩处理 ties，少于 3 点或常量返回 `None`；只接受正向 `rho>=0.20`。
- 单调条件 `left+0.01>=right` 正确表示最多允许 0.01 的相邻上升；两个 head 都要求至少 3/5 checkpoint 单调、平均 weak-to-strong drop 至少 0.02、最强 view 至少两个 checkpoint 达到正相关。
- 每个 `epoch x head x q` 同时要求 object pairs `>=100`、assigned both `>=100`、coverage `>=0.50`；样本不足在正式参数下优先得到 `INCONCLUSIVE`。任何非预注册 CLI 偏离又优先得到 `NON_PREREGISTERED`。

### 4.4 零更新、BF16 与 panel

- torch 在设置 `CUDA_VISIBLE_DEVICES` 后才 import；模型只为取得双头 raw dict 保持 training mode，全部 BatchNorm 随后切为 eval。
- forward 被 `torch.inference_mode()` 包围，且函数内部再次断言；CUDA autocast dtype 必须是 BF16。
- 原生 O2M/O2O assignment 共用一次模型 forward；没有 optimizer、backward 或外层 `E2ELoss.__call__/update`。
- 参数及全部 BN buffers 的前后 hash/count 不变是硬条件；loader、criterion、dtype、source、trajectory 证据进入 JSON。
- 顺序 train panel 恰好是前 256 个唯一图像；panel 和 geometry 指纹五次必须一致。

以上部分未发现新的数学或索引阻断项。现有 20 个合成测试仍未真正执行 torch 版 affine 像素/框往返、两图不等 GT 数的完整 assignment mapping、伪 feature block 的 anchor/level 映射；它们应与修复 C6 的负测一并补入，远端单 batch smoke 再覆盖真实 shape/dtype/API 边界。

## 5. 本地验证结果

执行：

```text
python -m unittest test_probe_asac_assignment_stability.py -v
```

结果：**20/20 PASS**，`Ran 20 tests ... OK`。

执行：

```text
python -m py_compile probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
```

结果：**PASS**，无输出、退出码 0。

另执行了 C7 全字段变异扫描：除派生常量 `panel_images` 外的 22 个冻结 CLI 字段逐一偏离，全部被 `protocol_status()` 捕获。

这些成功结果不消除 B1：现有 provenance 正测和 mixed-seed 负测没有覆盖“所有 checkpoint 一致但整体不是冻结 B50”的情况。

## 6. 最终裁决

**CONDITIONAL / NO-RUN。**

- 核心 ASAC 测量实现和 C1–C5/C7 加固可以保留。
- C6 必须按第 3.3 节修复并增加一致但非 B50 的负测。
- 修复后先重跑全部本地测试/编译，再在远端执行 `1 checkpoint x 1 batch x clean+1 view` 的只读 smoke，确认真实 criterion、dtype、panel、对象非空、JSON round-trip 和 state hash 不变。
- 只有上述步骤全部通过，才可升级为 **GO** 并给出正式 80-forward 远端调用清单。

由于本轮不是 GO，**本报告不提供、也不授权正式远端执行命令**。
