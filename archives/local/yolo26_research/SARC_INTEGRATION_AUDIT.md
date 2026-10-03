# SARC 真实 YOLO26 集成审计

日期：2026-09-19（Asia/Shanghai）  
状态：**LOCAL CANDIDATE PASS / REMOTE AND CUDA NOT TOUCHED**

## 0. 结论

已把通过 staging 的 SARC 集成到一份从真实研究 checkout 只读取得的、完全独立的 YOLO26 源码快照中：

- 候选源码：`work/sarc_integration/snapshot/ultralytics_research`
- 未改动参照：`work/sarc_integration/reference/ultralytics_research`
- 原始源码归档：`work/sarc_integration/snapshot.tar.gz`
- **唯一建议用于未来源码同步的最小覆盖包**：`work/sarc_integration/sarc_source_overlay.tar.gz`
- 完整 SHA256 清单：`work/sarc_integration/SARC_CANDIDATE_SHA256SUMS.txt`

本轮没有向 `gpu8-team` 写入任何文件、没有启动任何远端进程、没有占用 GPU，也没有触碰正在运行 TQGB/ASAC 的原研究目录。当前结果只授权未来在 TQGB 完全结束后，向一个新建的独立目录部署并做 smoke；不授权覆盖 `/home/ubuntu/rz/yolo_study/ultralytics_research`。

## 1. 上游来源锁定

只读来源：`gpu8-team:/home/ubuntu/rz/yolo_study/ultralytics_research`

- branch：`research/dual-head-diagnostics`
- HEAD：`9d45ee4613e79d7bceef7a39e42c15c278b0d4e8`
- 抓取时 working tree 已有：
  - `M ultralytics/cfg/__init__.py`
  - `M ultralytics/cfg/default.yaml`
  - `M ultralytics/data/build.py`
  - `M ultralytics/utils/loss.py`

因此，本审计中的 R0 是“抓取时真实研究 working tree 的 R0”，不是 clean upstream YOLO 的抽象 R0。候选保留了既有 KD/TQGB 研究代码，但所有 SARC 主配置显式把 KD/TQGB gains 置零，并由 runner fail-fast 防止混跑。

抓取时上游锚点 SHA256：

| 文件 | SHA256 |
|---|---|
| `ultralytics/nn/modules/conv.py` | `548ee1f3fa28542ad46b920e4d4271594efa09bc76c4ab3067a08a966f30648d` |
| `ultralytics/nn/modules/__init__.py` | `1a8102c7f6dc0f736e22ab8afcd0168ac5fc49c8c090139cdcce32aafeebd8ba` |
| `ultralytics/nn/tasks.py` | `c1da24859fd56b9869759bd4bfb753456c9074bfc3238b51a08ea3c6cbeda1ae` |
| `ultralytics/utils/loss.py` | `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9` |
| `ultralytics/cfg/default.yaml` | `1294dd04cdb28c7ecdbaf15dc01c9684b8365ba81be59f6ac47a5939bbd0e63a` |
| `ultralytics/cfg/__init__.py` | `ab0b03b1eb30bfe841fb551566d8b2e16c8cb7f6d38a3f21ef6bc81cc72f553e` |
| `ultralytics/engine/trainer.py` | `2c406877ac037a0b6691be1039f36750df3d7a675935627dafc4735a0b6decc7` |
| `ultralytics/cfg/models/26/yolo26.yaml` | `b1d1fa0c69eced64f9939536038bd8d697c32e72200b34187a767de431af7ef0` |
| 原远端 `scripts/train_research.py` | `d8f39b19c11434f4f322bc4630b02af795528c8618c9bc20e48a58433022f7cd` |

未来部署前若任一锚点变化，必须停止并重新做三方集成；不能把本覆盖包盲目解压到漂移后的源码上。

## 2. 默认关闭与 R0 不变保证

默认参数为：

```yaml
sarc_cal_gain: 0.0
sarc_cal_target: native
sarc_cal_level_permutation: none
```

保证边界如下：

1. 官方 `yolo26.yaml` 字节未改；R0 图中没有 SARC block，没有新增参数或 buffer。
2. `E2ELoss` 在未发现 SARC block 时走原返回路径，不追加 loss 分量，也不增加 `sarc_*` 日志键。
3. 默认 gain 为零；仅选择显式 `yolo26n-sarc-r*.yaml` 才建立 SARC 图。
4. 跨 pristine/candidate 双进程、同 seed CPU 指纹完全一致：
   - graph：exact
   - 参数数：exact，均为 `2,504,970`（`nc=3` 探针）
   - state SHA256：均为 `6a779dbaace92207d18d96125c54e89f114812bfd19bebc365f11c77112a9793`
   - loss 与 loss-items：exact
   - gradient SHA256：均为 `095f7952e2baa2b06a74484b71fac9a12aa86c59ddbad9acc5a383c1e38412de`
5. R1/R2 主配置的 `sarc_cal_gain=0.0`；实现返回与图相连的精确浮点零，测试断言 `sarc_cal_loss.item() == 0.0`。

这证明的是当前锁定 CPU 软件栈上的数学/图等价；CUDA kernel 的 bitwise 等价仍需真机验证。

## 3. 实现与需求对应

### 3.1 图与融合

- `SARCFixedBlock`（R1）：单个 DW7 + BN + mix BN + SiLU + gamma residual。
- `SARCBlock`（R2/R3）：DW7 主分支，加 dilation `1/2/3` 的三个 DW3-BN 分支；静态 trainable logits 经 softmax 混合，再过 mix BN、SiLU、gamma residual。
- 三个真实 YAML 都从原 neck 的 layer `16/19/22` 取 P3/P4/P5，插入 layer `23/24/25`，Detect 直接消费 `[23,24,25]`；实际解析 stride 为 `[8,16,32]`。
- deploy/fuse 把卷积、所有 branch BN、softmax 常量和 mix BN 合成单 DW7。
- `gamma * SiLU(z)` 的 gamma **不穿过 SiLU**；deploy block 保留 `SiLU -> frozen gamma buffer multiply -> residual add`，避免错误变成 `SiLU(gamma*z)`。
- `BaseModel.fuse()` 与 `is_fused()` 已识别 SARC；融合后训练分支、branch logits 与 calibration state 被移除。

### 3.2 原生 O2M assignment 与 R3

- 统计直接来自真实 `v8DetectionLoss` 的 O2M/TAL assignment：逐层使用 `fg_mask` 选中 matched target boxes，并校验 `target_gt_idx`、anchor 数量及 stride-block layout。
- 每个 positive assignment 计算 `u=log2(sqrt(w_gt*h_gt)/stride_level)`；当前定义是 positive-anchor weighted，同一 GT 有多个 matched positives 时会重复计入，不是 unique-GT 去重版。
- soft centers `[2.25,3.00,3.75]`、temperature `0.50`、EMA decay `0.99`、有效更新 warmup `16` 保持 staging 约定。
- 每层先形成 `[count_0,count_1,count_2,mass]`，DDP 对完整张量做 `SUM all_reduce`，之后每个 rank 才更新 EMA；CPU 两 rank Gloo 实测更新后 bitwise 一致。
- R3 calibration 为 `0.01 * sum KL(stopgrad(native_target) || softmax(branch_logits))`；遵循 YOLO 本地 batch-size loss 缩放约定。

### 3.3 独立日志与 checkpoint

- `sarc_cal_loss` 作为独立 loss item 输出，训练 CSV 列为 `train/sarc_cal_loss`。
- 另记录每层 gamma、三个 histogram bins、三个 branch weights、mass、update count。
- checkpoint 的 `train_results` 保留 `train/sarc_cal_loss` 历史。
- `sarc_hist_ema`、`sarc_hist_updates`、`sarc_hist_mass` 是 persistent buffers。
- checkpoint 前只把这三类 live-model buffer 复制到 model EMA：避免 float histogram 被 ModelEMA 二次平滑，也避免整数 counter 保持陈旧。实际 `BaseTrainer.save_model()` 路径已测试。

### 3.4 可选 target override

接口存在但不改变 R1/R2/R3 主矩阵：

| `sarc_cal_target` | 目标 | 约束 |
|---|---|---|
| `native` | 本层原生 assignment EMA | 默认；R3 主配置 |
| `uniform` | 每层 `[1/3,1/3,1/3]` | 仅显式正 gain 控制 |
| `permute` | 固定 level permutation 的 native EMA | 必须同时给出 `sarc_cal_level_permutation`，例如 `2,1,0` |

非法 target、非法/重复 permutation、gain 为零却启用 override、非 SARC 图启用选项、R1 gain 非零、非三层/非 P3-P5 layout 均 fail-fast。启动日志打印 model mode、gain、target、permutation。

已提供的控制配置不属于主 R1/R2/R3：

- `controls/sarc_r3_uniform_target_e50.yaml`
- `controls/sarc_r3_permuted_target_e50.yaml`，固定 `2,1,0`

只有 R3 先稳定胜过 R2 后才允许运行控制。R3>R2 只说明额外 KL 可能有益，不能单独证明 assignment 语义；至少要用等 gain/更新强度的 fixed permutation 阴性控制。GT-only/heuristic-level 控制尚未实现，是后续论文因果声明的阻断项。

## 4. CPU 验证结果

测试环境：Windows CPU，Python `3.11.9`，PyTorch `2.6.0+cu118`，但显式 `CUDA_VISIBLE_DEVICES=-1`；未调用 GPU。

最终命令：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:YOLO_AUTOINSTALL='false'
$env:USE_LIBUV='0'
F:\Anaconda\envs\pytorch\python.exe -m unittest -v .\work\sarc_integration\test_sarc_integration.py
```

结果：`9/9 PASS`，最终复跑 `Ran 9 tests in 5.566s`。覆盖：

- 三份真实 YAML parse、P3/P4/P5 接线、channel、stride 与参数预算；
- R0/R1/R2/R3 真实 `DetectionModel` + 原生 TAL loss forward/backward，finite gradients；
- R0 无 SARC loss/state；R1/R2 calibration exact zero；R3 warmup、EMA 更新与 ready 后非零 KL；
- standalone float64 fusion（阈值 `2e-12`）及真实整模型 fusion（阈值 `2e-4`）；
- deploy gamma 是 buffer 而不是 parameter；
- synthetic SUM→EMA 与真实两 rank CPU/Gloo collective；
- native/uniform/permute 接口与 fail-fast；
- calibration state_dict round trip；
- 完整 checkpoint 使用 live calibration buffers，并保存独立 `train/sarc_cal_loss` 历史。

此外：全源码 `compileall` 通过；259 个 YAML 均可解析；R0 pristine/candidate 指纹测试通过。

## 5. 参数与部署口径

以原 `nc=80` YAML 解析：

| 模型 | 训练态参数 | 相对 R0 | 融合部署相对 R0 |
|---|---:|---:|---:|
| R0 | `2,572,280` | `0` | `0` |
| R1 | `2,596,027` | `+23,747` | `+22,400` |
| R2 | `2,610,820` | `+38,540` | `+22,400` |
| R3 | `2,610,820` | `+38,540` | `+22,400` |

R1/R2/R3 融合后 topology 与可训练卷积参数增量相同，并各保留三个 frozen gamma 标量 buffers。这里没有把 gamma 错误折入 SiLU 前卷积。

## 6. R1/R2/R3 冻结配置

| 组 | model YAML | calibration | 正式配置 |
|---|---|---|---|
| R1 | `yolo26n-sarc-r1.yaml` | exact `0` | `sarc_r1_fixed_dw7_e50.yaml` |
| R2 | `yolo26n-sarc-r2.yaml` | exact `0` | `sarc_r2_reparam_no_cal_e50.yaml` |
| R3 | `yolo26n-sarc-r3.yaml` | native, gain `0.01` | `sarc_r3_native_cal_e50.yaml` |

三者共同锁定：50 epochs、batch 64、imgsz 640、physical GPU `4,5,6,7`、BF16、seed 42、deterministic、pretrained false、KD/TQGB gains zero。R1→R2→R3 必须顺序跑，不能与 TQGB seed runs 并发。

## 7. 哈希与最小安全同步面

### 7.1 最小源码覆盖包

`sarc_source_overlay.tar.gz` SHA256：

```text
590826dd9295e79109433ece11388ddb99757bbba9166031b7bcb63b6a888926
```

包内严格只有以下 10 个路径：

1. `ultralytics/nn/modules/conv.py`
2. `ultralytics/nn/modules/__init__.py`
3. `ultralytics/nn/tasks.py`
4. `ultralytics/utils/loss.py`
5. `ultralytics/cfg/default.yaml`
6. `ultralytics/cfg/__init__.py`
7. `ultralytics/engine/trainer.py`
8. `ultralytics/cfg/models/26/yolo26n-sarc-r1.yaml`
9. `ultralytics/cfg/models/26/yolo26n-sarc-r2.yaml`
10. `ultralytics/cfg/models/26/yolo26n-sarc-r3.yaml`

不得同步整个 `snapshot/`：CPU 测试产生的 `__pycache__` 不属于候选补丁，且整树覆盖会扩大污染面。

### 7.2 候选源码 SHA256

| 路径（相对 candidate source） | SHA256 |
|---|---|
| `ultralytics/nn/modules/conv.py` | `124db7ace068d44bd8f740b4a219b9439fa2fed02e2220ce637be4d2cbaa1e76` |
| `ultralytics/nn/modules/__init__.py` | `e02ea7f1ae54c80547423258dc3e5718ca4ed73b4eb72ba31402f03759c145d6` |
| `ultralytics/nn/tasks.py` | `678989b7648d27f8eee880c33d1e14984933eb717e4daeb4f150600676578db5` |
| `ultralytics/utils/loss.py` | `c8184807469e7226798117b1aa5145736e7f256a7704f146f87150df13fa70c4` |
| `ultralytics/cfg/default.yaml` | `68f79c22d22546e91cf7d99f05cb3c0d83cb009cc0a42ab187199ddfd40ed2aa` |
| `ultralytics/cfg/__init__.py` | `944d9895e22b69f4fcac57b798eb2bd2192d1c783873aa8991e1f8ee198852a8` |
| `ultralytics/engine/trainer.py` | `78b547346066c7ace10d855a230afaad1b23ac4acb49fbad0730dcd74f02c566` |
| `ultralytics/cfg/models/26/yolo26n-sarc-r1.yaml` | `cdd5a47e0fb62868958edf5958ddecd6131185dbd8a4a02fd8baac107cfd81ef` |
| `ultralytics/cfg/models/26/yolo26n-sarc-r2.yaml` | `1cc4d5c35b9d5ddcfdf57f44c4587fcf1d707a114d15da3151b6ac8146c6bd64` |
| `ultralytics/cfg/models/26/yolo26n-sarc-r3.yaml` | `5c7b3188935ce629fc8691f432e5ab40ea124169bc441b80a77e1956cdcd4de7` |

### 7.3 实验与验证文件 SHA256

| 文件（相对 `work/sarc_integration`） | SHA256 |
|---|---|
| `external_scripts/train_research.py` | `91a238221c7cc3bba930cb80989a97ca2468510c6905a91919cb94a016367558` |
| `research_configs/sarc_r1_fixed_dw7_e50.yaml` | `37576bef13a2c32c019ddd68c9245b46d1c0771ad1b4c7a7a136a942b5f9086f` |
| `research_configs/sarc_r2_reparam_no_cal_e50.yaml` | `e2275354bcad578ca529259dc92fd041f0e031be8287d4c4b68e05c23254000f` |
| `research_configs/sarc_r3_native_cal_e50.yaml` | `3f7cccc430ecd1afe05971b75a6b23b831c148aa85351e14005c1c404fe21636` |
| `research_configs/controls/sarc_r3_uniform_target_e50.yaml` | `aeaa4a2d8b391c9c2950e5362196151e5fe6292626726c2d134490f715ef9fbe` |
| `research_configs/controls/sarc_r3_permuted_target_e50.yaml` | `bf526c6029b0beeffca5c142bb2e85a6fb884cbb3255440ee7fed3e90e8d5225` |
| `research_configs/smoke/sarc_r1_cuda_smoke_e1.yaml` | `b10bf24b17063375f130767b2a3f359e337bf4f4bbeaf8d5f0977b613eb381ab` |
| `research_configs/smoke/sarc_r2_cuda_smoke_e1.yaml` | `e4afd861dfdfe86eac2c5526f4a6b2049a63f62167353dabe9dd944d96109ac1` |
| `research_configs/smoke/sarc_r3_cuda_smoke_e1.yaml` | `25f2745adccf90b7d80c9d43f095c95c76af89ef9681435c7bfa0ef5901ee3a7` |
| `test_sarc_integration.py` | `7e3077c77e568d4e2c62c82a320fb3e2d4ea90c487c9a98316d621f4edb8f189` |
| `probe_r0_equivalence.py` | `fadaee345245caedfcf6f6f776121d6563b039ed6fac1af1244d5562904370c5` |
| `snapshot.tar.gz` | `ce664ee5fe0c86aa1b97eeeffb5851fcee5196de9191631a3f18f7819d0eff0f` |

实验侧未来只需额外同步 runner、3 个主配置、3 个 smoke 配置；两个 controls 仅在 R3>R2 后同步/运行。完整机器可校验格式清单在 `SARC_CANDIDATE_SHA256SUMS.txt`。

## 8. 未来隔离部署步骤（本轮未执行）

### 8.1 硬阻断项

满足全部条件前不得执行下面写操作：

1. TQGB/ASAC 与 seed runs 已完全退出；`nvidia-smi` 和进程表确认 GPU 4–7 空闲。
2. 原目录 HEAD、dirty-file 列表和第 1 节 SHA256 全部仍匹配。
3. `/home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate` 不存在；若已存在，停止人工审计，绝不覆盖。
4. 新实验根 `/home/ubuntu/rz/yolo_study/yolo26_voc_sarc` 与旧 `yolo26_voc_research` 分离。
5. overlay 上传后的 SHA256 必须为 `590826...a888926`，且 `tar -tzf` 仍严格是上述 10 文件。
6. 先 CPU import/parse/fusion，再单卡 CUDA forward/backward，再 4 卡 BF16/DDP smoke；任一步失败都不得进入 50 epoch。

### 8.2 创建独立源码副本

下面是未来命令示例，**未执行**。先人工核对 hash gate：

```bash
ssh gpu8-team
cd /home/ubuntu/rz/yolo_study/ultralytics_research
git rev-parse HEAD
git status --short
sha256sum \
  ultralytics/nn/modules/conv.py \
  ultralytics/nn/modules/__init__.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/cfg/default.yaml \
  ultralytics/cfg/__init__.py \
  ultralytics/engine/trainer.py \
  ultralytics/cfg/models/26/yolo26.yaml
```

只有输出匹配第 1 节，才创建独立副本：

```bash
set -euo pipefail
test ! -e /home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
cp -a /home/ubuntu/rz/yolo_study/ultralytics_research \
  /home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
mkdir -p /home/ubuntu/rz/yolo_study/yolo26_voc_sarc/{scripts,configs/datasets,configs/research/smoke,configs/research/controls,runs,smoke_runs}
cp /home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml \
  /home/ubuntu/rz/yolo_study/yolo26_voc_sarc/configs/datasets/voc_server.yaml
```

从本机只上传最小包与批准的实验文件，绝不 rsync 整个 snapshot：

```powershell
$sarc = 'C:\Users\Lenovo\Documents\Codex\2026-09-17\referenced-chatgpt-conversation-this-is-an\work\sarc_integration'
scp "$sarc\sarc_source_overlay.tar.gz" gpu8-team:/home/ubuntu/rz/yolo_study/_sarc_source_overlay.tar.gz
scp "$sarc\external_scripts\train_research.py" gpu8-team:/home/ubuntu/rz/yolo_study/yolo26_voc_sarc/scripts/train_research.py
scp "$sarc\research_configs\sarc_r1_fixed_dw7_e50.yaml" "$sarc\research_configs\sarc_r2_reparam_no_cal_e50.yaml" "$sarc\research_configs\sarc_r3_native_cal_e50.yaml" gpu8-team:/home/ubuntu/rz/yolo_study/yolo26_voc_sarc/configs/research/
scp "$sarc\research_configs\smoke\sarc_r1_cuda_smoke_e1.yaml" "$sarc\research_configs\smoke\sarc_r2_cuda_smoke_e1.yaml" "$sarc\research_configs\smoke\sarc_r3_cuda_smoke_e1.yaml" gpu8-team:/home/ubuntu/rz/yolo_study/yolo26_voc_sarc/configs/research/smoke/
```

远端校验后只向新副本解压：

```bash
set -euo pipefail
echo '590826dd9295e79109433ece11388ddb99757bbba9166031b7bcb63b6a888926  /home/ubuntu/rz/yolo_study/_sarc_source_overlay.tar.gz' | sha256sum -c -
tar -tzf /home/ubuntu/rz/yolo_study/_sarc_source_overlay.tar.gz
tar -xzf /home/ubuntu/rz/yolo_study/_sarc_source_overlay.tar.gz \
  -C /home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
```

### 8.3 CPU 与单卡 CUDA 阻断 smoke

先在新副本上用 CPU parse/fuse；推荐把本地测试目录按相同布局复制到临时验证目录，或至少运行以下最小检查：

```bash
cd /home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
CUDA_VISIBLE_DEVICES=-1 YOLO_AUTOINSTALL=false PYTHONPATH=$PWD python - <<'PY'
import torch
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel

d = Path('ultralytics/cfg/models/26')
for name in ('r1', 'r2', 'r3'):
    m = DetectionModel(str(d / f'yolo26n-sarc-{name}.yaml'), verbose=False).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 64, 64))
    m.fuse(verbose=False)
    with torch.no_grad():
        m(torch.randn(1, 3, 64, 64))
    print(name, m.stride.tolist(), sum(p.numel() for p in m.parameters()))
PY
```

随后只用一张空闲卡做真 CUDA F+B；不通过则不得 4 卡：

```bash
cd /home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
CUDA_VISIBLE_DEVICES=7 YOLO_AUTOINSTALL=false PYTHONPATH=$PWD python - <<'PY'
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

model = DetectionModel('ultralytics/cfg/models/26/yolo26n-sarc-r3.yaml', nc=3, verbose=False).cuda().train()
model.args = get_cfg(overrides={'epochs': 1, 'sarc_cal_gain': 0.01, 'sarc_cal_target': 'native', 'sarc_cal_level_permutation': 'none'})
batch = {'img': torch.rand(2,3,64,64,device='cuda'), 'batch_idx': torch.tensor([0,1],device='cuda'), 'cls': torch.tensor([[0.],[1.]],device='cuda'), 'bboxes': torch.tensor([[.3,.35,.22,.18],[.68,.62,.25,.2]],device='cuda')}
loss, items = model(batch)
loss.sum().backward()
assert torch.isfinite(loss).all() and 'sarc_cal_loss' in items
print(loss.detach(), items['sarc_cal_loss'], [b.sarc_hist_updates.item() for b in model.model[23:26]])
PY
```

注意：这里进程内只暴露 physical GPU 7，所以模型看到的是 logical `cuda:0`；正式 runner 则不要预设 `CUDA_VISIBLE_DEVICES`，由配置的 `device: "4,5,6,7"` 选择物理卡。

### 8.4 真机 4 卡 BF16/DDP smoke

确认 GPU 4–7 空闲后，在独立项目根按 R1→R2→R3 顺序执行，禁止并发：

```bash
cd /home/ubuntu/rz/yolo_study/yolo26_voc_sarc
export PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
export YOLO_AUTOINSTALL=false
python scripts/train_research.py --config configs/research/smoke/sarc_r1_cuda_smoke_e1.yaml
python scripts/train_research.py --config configs/research/smoke/sarc_r2_cuda_smoke_e1.yaml
python scripts/train_research.py --config configs/research/smoke/sarc_r3_cuda_smoke_e1.yaml
```

smoke 硬验收：

- 三组均完成 1 epoch，无 NCCL/BF16/shape/NaN 错误；
- R1/R2 `train/sarc_cal_loss` 全程精确 `0.0`；
- R3 update count、mass 增长，三层 rank 间一致，warmup 前 cal 为零；
- `last.pt` 的 `train_results` 有独立 `train/sarc_cal_loss`；
- `last.pt` 的 `ema.state_dict()` 有三层 `sarc_hist_ema/updates/mass`，值与结束时 live 统计一致；
- 从 checkpoint resume 一个极短 smoke 后 counter 连续增长而非归零；
- fuse 前后同一输入误差满足预注册阈值，再尝试 exporter。

checkpoint 快查示例：

```bash
PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate python - <<'PY'
import torch
p='/home/ubuntu/rz/yolo_study/yolo26_voc_sarc/smoke_runs/sarc_r3_cuda_smoke_e1/weights/last.pt'
c=torch.load(p,map_location='cpu',weights_only=False)
print('cal history:', c['train_results'].get('train/sarc_cal_loss'))
for k,v in c['ema'].state_dict().items():
    if k.endswith(('sarc_hist_ema','sarc_hist_updates','sarc_hist_mass')):
        print(k,v)
PY
```

### 8.5 正式主配置命令

只在全部 smoke 通过后顺序执行：

```bash
cd /home/ubuntu/rz/yolo_study/yolo26_voc_sarc
export PYTHONPATH=/home/ubuntu/rz/yolo_study/ultralytics_sarc_candidate
export YOLO_AUTOINSTALL=false
python scripts/train_research.py --config configs/research/sarc_r1_fixed_dw7_e50.yaml
python scripts/train_research.py --config configs/research/sarc_r2_reparam_no_cal_e50.yaml
python scripts/train_research.py --config configs/research/sarc_r3_native_cal_e50.yaml
```

若 R3 没有按预注册标准胜过 R2，停止 SARC 主线。若 R3 胜过 R2，才同步并运行 fixed-permutation（优先）与 uniform controls；不得把 R3>R2 直接解释成 assignment 语义成立。

## 9. 尚未验证的风险

1. 未在 CUDA、BF16、NCCL 或真实 4×3090 上运行；CPU/Gloo 不能替代 NCCL smoke。
2. 未测完整 VOC 640/batch64 的显存、吞吐、稳定性与 AP。
3. 未跑真实 ONNX/TensorRT/exporter；目前只验证 PyTorch `.fuse()`，gamma 常数乘法的 exporter 保留仍是阻断项。
4. 未验证真实训练 checkpoint 的中断恢复后 EMA/count/mass 连续性；只覆盖了实际保存路径与 state round trip。
5. R3 gain `0.01` 的效果和数值范围未经真训练验证；当前只证明接口和梯度成立。
6. 只在 CPU 两 rank Gloo 验证全局 SUM 后 EMA；四 rank NCCL 的 collective/order 尚未验证。
7. `torch.compile`/图捕获未验证；loss 内 collective 可能引入 graph break。
8. 外部 logger 对新增大量逐层标量列的兼容性/容量尚未验证。
9. 当前 native histogram 是 positive-anchor weighted；unique-GT、GT-only、frozen-prior、实际 folded-kernel contribution/ERF 尚未验证。
10. fixed permutation 和 uniform 接口已实现，但只有 R3>R2 后才应运行；fixed permutation 是最低等强度语义阴性控制，uniform 不能替代它。
11. 远端 working tree 可能在 TQGB 收尾期间漂移；部署前 hash gate 是硬条件，漂移时必须重新集成。

## 10. 最终裁决

本地候选已满足进入“隔离真机 smoke”的代码条件，但尚未满足进入 50-epoch 正式训练的硬件/部署条件。下一步的唯一安全路径是：等待现有工作完全结束 → 上游 hash gate → 新建独立 source/project → 只解压 10 文件 overlay → CPU → 单卡 CUDA → 4 卡 BF16/DDP smoke → checkpoint/resume/export 审计 → 顺序 R1/R2/R3。任何情况下都不应覆盖现有 `ultralytics_research` 或与 TQGB seed runs 共用输出目录。
