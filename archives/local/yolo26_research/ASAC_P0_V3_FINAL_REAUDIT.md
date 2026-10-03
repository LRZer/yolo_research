# ASAC-P0 v3 最终独立复审

日期：2026-09-19  
裁决：**GO（分阶段授权）**

> v2 复审发现的 C6 阻断漏洞已经实质关闭。旧版可通过的完整恶意轨迹现在被硬拒绝；真实 B50 `args.yaml` 的 124 个历史字段与冻结值逐项、逐类型一致，仅允许补入 3 个有明确历史依据的 TQGB no-op 字段，形成 127 字段 manifest，SHA256 与 spec 完全一致。  
> 本裁决授权同步、远端 CPU provenance preflight、`1 checkpoint × 1 batch × (clean + 1 view)` 只读 GPU smoke；**只有这些门禁全部通过，才授权正式 80-forward ASAC-P0**。任何 manifest、checkpoint hash、模型结构/参数量、runtime contract、BF16、状态完整性或主机健康检查失败，都必须停止，不能放宽冻结值追求通过。

## 1. 范围、边界与审计输入

已完整阅读并复核：

- `outputs/yolo26_research/ASAC_P0_V2_REAUDIT.md`
- `work/remote_research_source/scripts/probe_asac_assignment_stability.py`
- `work/remote_research_source/scripts/test_probe_asac_assignment_stability.py`
- `work/yolo26_research/ASAC_PROBE_SPEC.md`

同时只读交叉核对了真实 B50 `args.yaml`、PBGA provenance JSON、SARC provenance JSON 与既有运行日志。本轮未访问或修改远端，未使用 GPU，也未修改 probe、测试、spec 或 Ultralytics 实现。Windows 本机没有真实远端 checkpoint/runtime，故真实模型反序列化和 forward 仍必须由下述远端只读 preflight/smoke 完成。

审计输入 SHA256：

| 文件 | SHA256 |
|---|---|
| probe v3 | `A0D87624F013D79C69C30ADA1B9295E5BD21304B3DE9778336030C79B0EA6E74` |
| tests v3 | `975D77CD74767B526A5810CD76E58D477865964D28BDE49657A57C0563A6A96A` |
| spec v3 | `CB322330C1D3366FCA03294D5557C81D25B65479C50950F516A1F267099E5C56` |
| v2 re-audit | `D4CE1F84C706CE97172400EA4C531CC327AAE1B0ED54961313E63032A85E3134` |
| real B50 `args.yaml` | `D7A98308306AC9CA7D5691233CD80850CF5555D4452FD9926260B5239A5F46F8` |
| PBGA provenance JSON | `37AAEEC55564E5A1A70CA0FCF9860F8D9974CE6AD7FA61BEDC6D584786CB7DA9` |
| SARC provenance JSON | `C0BE277C260684ECD1D4B073CFC5F457514D53B27C38200022E7C393B8B58912` |

## 2. C1–C7 最终裁决

| 项目 | 最终结论 | 强制语义 |
|---|---|---|
| C1：原生 O2M/O2O contract | **PASS，需 smoke 实证 runtime** | 严格要求 `E2ELoss`；O2M `topk/topk2=10/10`、O2O `7/1`；两支 stride `8/16/32`；原生 assignment 方法可调用；Detect one-to-one/end-to-end 均存在。旧 direct-top1 O2O 硬失败。 |
| C2：固定非 rect panel/loader | **PASS，需 smoke 实证 runtime** | trainer、dataset、loader 三层检查 `rect=False`、无增强、顺序 sampler、workers=4、pinned memory；正式运行必须恰好 `4×64=256` 张唯一 train 图。 |
| C3：几何存活/选择偏差 | **PASS** | 所有 clean GT 均记录逐强度存活、四视图交集、retained/dropped 尺寸和边界分布；五个 checkpoint 的 geometry 行必须同指纹。 |
| C4：BF16、BN、零更新 | **PASS，需 smoke 实证 runtime** | 每次 forward 断言 inference mode 与 CUDA BF16 autocast；训练格式输出但全部 BN eval；不得创建 optimizer；参数和 BN buffer 前后 SHA256 必须不变。 |
| C5：epoch、`best.pt`、strip | **PASS** | 仅接受 `epoch10/20/30/40.pt + last.pt`；numbered checkpoint 必须未 strip 且内部 epoch 精确；`best.pt` 永久拒绝；嵌入结果与 50 行 `results.csv` 双重证明完成度。 |
| C6：精确冻结 B50 来源 | **PASS** | 127 字段完整 manifest、独立 `args.yaml`、固定 model/data/project/name、`run_dir=project/name=save_dir`、五个内容 hash、YOLO26n/VOC20 结构与 2,511,600 参数均为硬门禁。旧恶意轨迹已硬拒绝。 |
| C7：预注册参数 | **PASS** | 任何 CLI protocol 偏离均进入 deviations，并把结果强制降为 `NON_PREREGISTERED`，不得产生正式 `GO/NO-GO`。 |

## 3. C6 阻断回归：已关闭

### 3.1 124 + 3 = 127 manifest 完整性

独立载入真实留存的 B50 `args.yaml` 后得到：

- 历史文件字段数：**124**；
- `FROZEN_B50_ARGS` 字段数：**127**；
- 真实 124 字段相对冻结历史子集：missing `[]`、unexpected `[]`、值/类型 mismatch `{}`；
- 唯一可缺失字段：`tqgb_cls_m2o_gain=0.0`、`tqgb_cls_o2m_gain=0.0`、`tqgb_direction=none`；
- 补入这三个 no-op 字段后，与 `FROZEN_B50_ARGS` 完全相等；
- canonical manifest SHA256：`58054c007178b94fccd87d8c6d1ae265a5487cc3b3609cd58301c7fa68c1fe7a`，与 spec 完全一致。

又做了穷举缺失字段回归：逐一删除每个历史字段，**124/124 全部硬拒绝**；逐一缺失三个 TQGB 字段时只补其指定默认值；三个同时缺失也只补这三个值；加入任意未来/未知 key 会硬拒绝。不存在“通用缺 key 自动补默认”的路径。

真实 POSIX 路径在 Windows `pathlib` 下不会被认作本机绝对路径，因此本轮对真实文件采用逐值/逐类型比较和 canonical hash；完整路径规范化由 Linux CPU provenance preflight 作为正式前置门禁。这是平台验证边界，不是放宽。

### 3.2 旧 `UNEXPECTED_PASS` 复现

完全重放 v2 报告中的最强反例：五个 checkpoint 仍保持相同 signature，`args.yaml` 也与恶意 records 一致，同时统一改为 wrong model、`degrees=25`、两个非零 TQGB gain、`bidirectional` 和 wrong project。

结果从旧版：

```text
UNEXPECTED_PASS /tmp/not-yolo26n.yaml 25.0 0.08 /tmp/other-project
```

变为：

```text
REJECTED: checkpoint is not the frozen B50 manifest
```

异常中逐项列出了 `model`、`project`、`degrees`、`tqgb_cls_m2o_gain`、`tqgb_cls_o2m_gain`、`tqgb_direction` 六个偏差；未依赖“checkpoint 之间恰好不一致”才拒绝。

### 3.3 路径、args 与结构约束

- 即使通过内部测试 seam 把“期望 manifest”也改成另一 project，并令五个 checkpoint 和 `args.yaml` 全部自洽，只要真实 checkpoint 目录不等于 `project/name`，即硬拒绝；`save_dir` 还要单独等于同一目录。
- `args.yaml` 独立读取、严格规范化并先对冻结 manifest、再对 checkpoint metadata 比较；checkpoint 自报 hash 会重新计算，不能伪造。
- 对 `model_class`、`head_class`、`nc`、stride、参数量、VOC 类名、YAML `nc`、YAML scale 的 8 类独立变异全部拒绝。
- 结构要求为 `DetectionModel + Detect`、VOC20 固定名称、`nc=20`、stride `8/16/32`、参数量 `2,511,600`、YAML `nc=20` 且 scale 为 `n` 或历史未记录。既有五次 PBGA/SARC 日志均报告 `2,511,600` 参数；真实 checkpoint 仍须由 smoke 再实证。

### 3.4 checkpoint 内容 pin

probe 中的 pin 与 PBGA、SARC 两份独立留存 JSON 完全一致：

| checkpoint | SHA256 |
|---|---|
| `epoch10.pt` | `ac6833319b8b39509ac4b02cedcdd90994220620fc2559b6d0549eddc94e21d6` |
| `epoch20.pt` | `ea4b17ae9f09c8774cf17970fc0071e95c1cc8abf48b7bdeac4315c79da4e9de` |
| `epoch30.pt` | `cdbb6e3d72b587aba9754a3fd0450a64a50170ea1cdb13e318a046ddd16e6aa9` |
| `epoch40.pt` | `a2f334e6c260790f5ba03702e26746f856dcde20cb31a36229be25f9d62b0765` |
| `last.pt` | `49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb` |

pin 集合还必须与五个固定文件名精确相等；缺 pin、替换文件或只伪造 metadata 均不能通过。

## 4. 数学、索引、几何、状态与裁决复核

- affine 像素采样与框变换互为同一 scale/translation 的 inverse/forward；三档强度共享由路径+seed 决定的方向，幅度线性嵌套。
- 每图 GT 先独立建立 local-to-original map；assigner 的 per-image local GT index 在使用前有越界检查，不会把 batch 内不同图片的 GT 串号。
- P3/P4/P5 block 长度由 feature map 实际形状生成，并以 `make_anchors` 的 stride tensor 逐 anchor 验证；非固定尺寸顺序假设会失败而不是静默错位。
- occupancy 只由 `(level, dx, dy)` 正样本点构造，不以质量加权；JS 使用 base 2，稳定性 `exp(-JS)`；任一侧无 assignment 时稳定性为 0。
- 质量读数与 occupancy 分离，使用 `sqrt(sigmoid(class logit) × aligned IoU)`；Spearman 只对两侧均有 assignment 的对象计算，coverage 和最小样本数另行硬门禁。
- retained/dropped 几何报告揭示最强视图交集造成的选择偏差；没有把该报告误当作偏差已被消除。
- 每个正式 forward 在 inference mode + CUDA BF16 autocast 内执行；BN 保持 eval；参数和 BN buffers 前后 hash 不变且 optimizer 不存在，否则硬失败。
- panel 与 geometry 指纹跨 checkpoint 必须一致；runtime source bundle 与 criterion contract 也必须跨 checkpoint 一致。
- 裁决优先级正确：protocol 偏离先变为 `NON_PREREGISTERED`；样本/coverage 不足为 `INCONCLUSIVE`；正式 GO 需 O2M/O2O 都满足强度单调下降、弱强差和质量相关门槛；否则 NO-GO。没有从 test/val AP 反调阈值的路径。

本轮未发现新的数学、索引、几何、BF16、零更新、panel 或 decision 阻断项。真实 tensor shape/dtype/API 与状态 hash 仍由 smoke 门禁覆盖。

## 5. 本地验证结果

在 `work/remote_research_source/scripts` 执行：

```text
python -m py_compile probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
```

结果：**PASS，exit 0**。

```text
python -m unittest test_probe_asac_assignment_stability.py -v
```

结果：**33/33 PASS**，`Ran 33 tests ... OK`。

此外独立执行了非 unittest 的 manifest 穷举、旧恶意轨迹、path/run relation 和 8 类结构变异复现；结果如第 3 节。

## 6. 精确安全同步与运行清单

以下顺序是本 GO 的组成部分；不得跳过。`<REMOTE>` 仅代表操作方已经核实的 SSH 目标，远端路径和内容 hash 则是精确冻结值。

### 6.1 只同步两份已审计脚本，并先在 staging 校验

本地发送到新 staging 目录，不直接覆盖运行脚本：

```bash
ssh <REMOTE> 'mkdir -p /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts/.asac_v3_stage'
scp work/remote_research_source/scripts/probe_asac_assignment_stability.py \
    work/remote_research_source/scripts/test_probe_asac_assignment_stability.py \
    <REMOTE>:/home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts/.asac_v3_stage/
```

远端 staging 必须得到：

```bash
cd /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts/.asac_v3_stage
sha256sum probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
# 必须逐字为：
# a0d87624f013d79c69c30ada1b9295e5bd21304b3de9778336030c79b0ea6e74  probe_asac_assignment_stability.py
# 975d77cd74767b526a5810cd76e58d477865964d28bde49657a57c0563a6a96a  test_probe_asac_assignment_stability.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research \
  /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python -m py_compile \
  probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research \
  /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python -m unittest \
  test_probe_asac_assignment_stability.py -v
```

必须再次得到 33/33 PASS。然后保留当前远端脚本备份，再从 staging 安装；若当前脚本存在未审查差异，停止并先人工合并，不可盲目覆盖：

```bash
cd /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts
stamp=$(date -u +%Y%m%dT%H%M%SZ)
cp -p probe_asac_assignment_stability.py "probe_asac_assignment_stability.py.before_v3_${stamp}"
cp -p test_probe_asac_assignment_stability.py "test_probe_asac_assignment_stability.py.before_v3_${stamp}"
install -m 0644 .asac_v3_stage/probe_asac_assignment_stability.py probe_asac_assignment_stability.py
install -m 0644 .asac_v3_stage/test_probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
sha256sum probe_asac_assignment_stability.py test_probe_asac_assignment_stability.py
```

### 6.2 GPU 前：CPU-only 五 checkpoint provenance/structure preflight

```bash
cd /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research \
/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python - <<'PY'
import json
import torch
import probe_asac_assignment_stability as p

run = "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42"
data = "/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml"
paths = [f"{run}/weights/{name}" for name in ("epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "last.pt")]
records = []
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    record = p.checkpoint_record_from_payload(path, payload)
    model = payload.get("ema") if payload.get("ema") is not None else payload.get("model")
    record["model_structure"] = p.model_structure_record(model)
    record["checkpoint_sha256"] = p.file_sha256(path)
    records.append(record)
    del model, payload
evidence = p.validate_b50_trajectory(records, data)
assert len(evidence["trajectory_args"]) == 127
assert evidence["production_frozen_manifest_sha256"] == "58054c007178b94fccd87d8c6d1ae265a5487cc3b3609cd58301c7fa68c1fe7a"
assert evidence["validated_model_structure"]["parameter_count"] == 2511600
print(json.dumps({
    "manifest_fields": len(evidence["trajectory_args"]),
    "manifest_sha256": evidence["production_frozen_manifest_sha256"],
    "structure": evidence["validated_model_structure"],
    "run_evidence": evidence["run_evidence"],
}, ensure_ascii=False, indent=2))
PY
```

任何异常立即停止。尤其不能因为真实 checkpoint 不匹配 127 字段、2,511,600 参数或固定 hash 而修改常量。

### 6.3 单 checkpoint、单 batch、clean+1 view 只读 GPU smoke

先确认 GPU 4 无其他任务、NUMA node 1 可用，并记录开始时间；已知故障主机如无法读取 kernel/MCE 健康状态，停止并请求有权限的操作者检查。

```bash
set -euo pipefail
nvidia-smi -i 4
gpu_pids=$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader)
test -z "$gpu_pids"
numactl --hardware
start_utc=$(date --iso-8601=seconds)
cd /home/ubuntu/rz/yolo_study/yolo26_voc_research/scripts
PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_research CUDA_VISIBLE_DEVICES=4 \
numactl --cpunodebind=1 --membind=1 \
/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python - <<'PY'
import json
from types import SimpleNamespace
import probe_asac_assignment_stability as p

p.initialize_runtime(4)
args = SimpleNamespace(
    data="/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml",
    batch=64, imgsz=640, workers=4, seed=42,
    max_scale_delta=0.50, max_translate=0.10,
    min_visibility=0.70, min_box_pixels=2.0,
    expected_strides=[8.0, 16.0, 32.0],
    grid_bins=13, grid_extent=1.5, kernel_sigma=0.25,
)
checkpoint = "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s42/weights/epoch10.pt"
trainer = p.make_trainer(checkpoint, args)
contract = p.validate_criterion_contract(trainer, args.expected_strides)
loader, loader_meta = p.make_fixed_loader(trainer, args)
before = p.integrity_snapshot(trainer)
if not before["model_training"] or not before["bn_all_eval"] or before["optimizer_created"]:
    raise RuntimeError(f"invalid zero-update precondition: {before}")

with p.torch.inference_mode():
    raw = next(iter(loader))
    paths = [str(x) for x in raw["im_file"]]
    batch = trainer.preprocess_batch(raw)
    bs, _, height, width = batch["img"].shape
    if bs != 64 or len(paths) != 64 or len(set(paths)) != 64:
        raise RuntimeError(f"smoke batch is not 64 unique images: bs={bs}, paths={len(paths)}, unique={len(set(paths))}")
    canonical = p.canonical_boxes(batch, bs)
    base_maps = p.base_local_maps(batch, bs)
    identity = [{"scale": 1.0, "tx": 0.0, "ty": 0.0} for _ in range(bs)]
    clean, clean_dtype = p.forward_assignments(trainer, batch)
    params = p.affine_parameters(paths, 0.25, args.seed, args.max_scale_delta, args.max_translate)
    view_batch, view_maps = p.transform_batch(batch, params, args)
    view, view_dtype = p.forward_assignments(trainer, view_batch)
    if view_dtype != clean_dtype:
        raise RuntimeError(f"autocast/runtime dtype evidence changed: clean={clean_dtype}, view={view_dtype}")
    counts = {"clean": {}, "view_q0.25": {}}
    for head, (assignment, prediction, strides) in clean.items():
        counts["clean"][head] = len(p.extract_object_records(
            assignment, prediction, base_maps, canonical, identity, (height, width), strides, args))
    for head, (assignment, prediction, strides) in view.items():
        counts["view_q0.25"][head] = len(p.extract_object_records(
            assignment, prediction, view_maps, canonical, params, (height, width), strides, args))
if any(value <= 0 for group in counts.values() for value in group.values()):
    raise RuntimeError(f"empty assignment object set: {counts}")
after = p.integrity_snapshot(trainer)
p.assert_integrity_unchanged(before, after, checkpoint)
result = {"criterion": contract, "loader": loader_meta, "clean_dtype": clean_dtype,
          "view_dtype": view_dtype, "object_counts": counts,
          "integrity_before": before, "integrity_after": after}
json.loads(json.dumps(result))
print(json.dumps(result, ensure_ascii=False, indent=2))
if hasattr(loader, "close"):
    loader.close()
PY

kernel_log=$(mktemp)
if ! journalctl -k --since "$start_utc" --no-pager >"$kernel_log"; then
  echo 'cannot read kernel health log; smoke is not valid' >&2
  rm -f "$kernel_log"
  exit 1
fi
if grep -Ei 'mce|edac|hardware error|sigbus|bus error' "$kernel_log"; then
  echo 'hardware/runtime fault detected; smoke is not valid' >&2
  rm -f "$kernel_log"
  exit 1
fi
rm -f "$kernel_log"
```

smoke 必须确认：criterion 10/10 与 7/1、loader 固定、两支 clean/view objects 非空、BF16 autocast evidence 稳定、JSON round-trip 成功、参数/BN hash 不变、optimizer 不存在，且期间没有 SIGBUS/MCE/EDAC/hardware error。若 `journalctl` 无权限，不能把空输出解释为健康。

### 6.4 正式 80-forward 命令

先确认输出目录和 log 均不存在，避免覆盖旧证据：

```bash
test ! -e /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2
test ! -e /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2.log
```

随后执行精确冻结命令：

```bash
set -euo pipefail
start_utc=$(date --iso-8601=seconds)
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
  --output-dir /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2 \
  2>&1 | tee /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2.log
status=${PIPESTATUS[0]}
test "$status" -eq 0
! grep -Eiq 'sigbus|bus error' /home/ubuntu/rz/yolo_study/yolo26_voc_research/diagnostics/asac_p0_b50_v2.log
kernel_log=$(mktemp)
if ! journalctl -k --since "$start_utc" --no-pager >"$kernel_log"; then
  echo 'cannot read kernel health log; formal run is not valid' >&2
  rm -f "$kernel_log"
  exit 1
fi
if grep -Ei 'mce|edac|hardware error|sigbus|bus error' "$kernel_log"; then
  echo 'hardware/runtime fault detected; formal run is not valid' >&2
  rm -f "$kernel_log"
  exit 1
fi
rm -f "$kernel_log"
```

只有 exit 0、无 SIGBUS/MCE/EDAC/hardware error，且 JSON 中 `preregistered=true`、`protocol_deviations={}`、五 checkpoint/content hash、manifest/结构、panel/geometry、criterion/runtime source 和前后 state hash 全部通过时，输出的 `GO/NO-GO/INCONCLUSIVE` 才是有效正式结果。故障主机上的结果最多作为探索证据，最终机制结论仍需健康硬件与后续多 seed 复现。

## 7. 最终结论

**GO。** v2 的唯一阻断项 C6 已被完整 manifest、窄化历史默认、内容 hash、目录关系、独立 `args.yaml` 与精确模型结构共同关闭；旧 `UNEXPECTED_PASS` 已转为确定性拒绝。其余 C1–C5/C7、数学与决策语义没有发现新阻断。

GO 不等于跳过实机证据：远端 CPU provenance preflight 与单 batch 只读 GPU smoke 是正式 80-forward 的硬前置条件。任何失败都自动撤销本次运行授权并回到 NO-RUN；不得通过修改冻结常量、删除字段或更换 checkpoint 使其“通过”。
