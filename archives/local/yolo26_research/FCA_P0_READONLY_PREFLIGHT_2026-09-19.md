# FCA-P0 只读 provenance 预检

日期：2026-09-19 17:27 CST  
状态：**PASS FOR PLANNING ONLY；未实现/未运行 FCA-P0，未修改远端文件**

## 1. 预检目的

在用户决定是否批准 FCA-P0 前，只确认未来探针会读取正确的服务器、数据、源码和 baseline checkpoints。本轮没有：

- 创建或覆盖远端文件；
- 修改 Ultralytics 源码；
- 启动训练、验证或 assignment probe；
- 访问 VOC2007 test 指标；
- 安装软件包。

## 2. SSH 身份与路径

| 项目 | 只读实测 |
|---|---|
| SSH 用户 | `ubuntu` |
| HOME | `/home/ubuntu` |
| 主机名 | `ubuntu` |
| 项目目录 | `/home/ubuntu/rz/yolo_study/yolo26_voc_research`：存在 |
| 源码目录 | `/home/ubuntu/rz/yolo_study/ultralytics_research`：存在 |
| 数据目录 | `/home/ubuntu/rz/yolo_study/datasets/VOC`：存在 |
| baseline run | `b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1`：存在 |

这再次确认当前仍是共享 Linux 用户 `ubuntu`；本项目不触碰 `~/.codex` 或共享 Codex 登录状态。

## 3. 冻结 checkpoint：远端与本地一致

| checkpoint | SHA256 |
|---|---|
| `epoch0.pt` | `dff866ec3fa3087731de178064e40350b3d9625bb3679d427e9e3a954a79c346` |
| `epoch10.pt` | `fe96652a133d9f7f13bb586ed06eb517608227f30ea523bc0d717d77762cafbe` |
| `epoch20.pt` | `17d1b6f0d4089e607899560446ea6102f89427c205906c83bb691d6c9b82acda` |
| `epoch30.pt` | `cec52cee8bfe0aafb3e10fbb94022ac4e0c98897541b0e440d9557f5846d623a` |
| `epoch40.pt` | `bfa1467eb89b38124d3a0f7ee98d4b79d7608ac4e513edb608ebe1e672e6965a` |
| `last.pt` | `a0bab3ef6b42b5cf9b295fbbbe9a322ebd39bc5cf568883aecb9ba4ddf547748` |

六个远端 SHA256 与已归档本地文件和 `SHA256SUMS.txt` 逐项一致。`best.pt` 不进入 FCA-P0。

## 4. 源码与运行时 contract

| 项目 | 实测值 |
|---|---|
| Git branch | `research/dual-head-diagnostics` |
| Git HEAD | `9d45ee4613e79d7bceef7a39e42c15c278b0d4e8` |
| `head.py` | `1dc4a4a90401149edeb55cd679e7f33be2c36eae1317b6f67f13c7a8b18d60b8` |
| `tal.py` | `9468fb73df7805a4d297302aa221e0d7a6e46e224d0a0768e5abf3f4d2cf2861` |
| `loss.py` | `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9` |
| `voc_server.yaml` | `5650b70651a70021b0a58fdb228db71cdc3324972352ef406cd1bd977d321d40` |

源码 worktree 不是 clean：`cfg/__init__.py`、`cfg/default.yaml`、`data/build.py`、`utils/loss.py` 有研究修改。这是现有研究分支的预期状态，因此 FCA-P0 不得仅凭 `git status clean` 判断 provenance，而应硬校验上述文件及实际用到的完整源码 bundle。

用冻结 `last.pt` 在 CPU 上初始化真实 criterion，得到：

```text
criterion = ultralytics.utils.loss.E2ELoss
O2M topk/topk2 = 10/10
O2O topk/topk2 = 7/1
head strides = 8/16/32
end2end = true
one2one head = present
nc = 20
```

因此 FCA-P0 预注册所针对的接口与当前远端运行时完全一致。

## 5. 数据清单

`voc_server.yaml` 的 train split 仍严格为：

- `images/train2012`
- `images/train2007`
- `images/val2012`
- `images/val2007`

`images/test2007` 只在 val/test 字段，本探针不得读取。

| split | images | labels | GT lines |
|---|---:|---:|---:|
| train2012 | 5,717 | 5,717 | 13,609 |
| train2007 | 2,501 | 2,501 | 6,301 |
| val2012 | 5,823 | 5,823 | 13,841 |
| val2007 | 2,510 | 2,510 | 6,307 |
| **合计** | **16,551** | **16,551** | **40,058** |

当前远端 manifest 指纹：

- 训练图像相对路径+字节数：`4a91fffa66e61125f414751d5bef2937457a5aeeb76a4e09fd6497c2be910cc0`；
- 有序 label 文件逐文件 SHA256 bundle：`39c8623c143706f2caede28dae7b5a248d9d2dd703aca12de043de942508c28a`。

正式运行仍需生成 loader 实际有序 image/GT-geometry 指纹；文件清单指纹不能替代 letterbox 后几何一致性证明。

## 6. Python/GPU/故障状态

| 项目 | 实测值 |
|---|---|
| Ultralytics | `8.4.146` |
| PyTorch | `2.6.0+cu118` |
| CUDA runtime | `11.8` |
| CUDA available | `true` |
| SciPy | 未安装 |
| NetworkX | 已安装 |

物理 GPU 4–7 均为 RTX 3090，NVML 未发现 compute process。每卡约 `474,742,784` bytes 的基础占用不是训练进程。

服务器时间 `2026-09-19T17:27:41+08:00` 时检查当日内核日志：存在 11:17 开机时的 EDAC 控制器初始化，但按以下故障口径没有新记录：

- uncorrected hardware memory error；
- sending SIGBUS；
- memory failure；
- EDAC corrected/uncorrected/error event。

历史 RAM 故障事实不变。任何正式 FCA-P0 执行窗口出现新故障都必须整轮作废，不能 resume 拼接；最终论文证据仍需健康硬件复现。

## 7. 就绪判断与剩余边界

只读 provenance 预检通过，说明未来可以在不重新训练 baseline 的情况下，安全读取冻结 checkpoints 做 FCA-P0。

尚未得到用户批准，因此当前仍不允许：

- 编写/部署远端 instrumentation；
- 运行一批或全量 forward；
- 根据潜在结果改变预注册阈值；
- 进入 Phase 1 matching 或任何训练。

若用户批准，第一步仍应是隔离实现 + CPU 合成/穷举单测 + `1 checkpoint × 1 batch` bitwise smoke；smoke 通过后才允许六 checkpoint 全量扫描。
