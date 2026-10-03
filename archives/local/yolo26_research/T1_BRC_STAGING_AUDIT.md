# TQGB-T1 BRC 本地 staging 实现审计

日期：2026-09-19  
状态：**本地 CPU 合成验证通过；未同步远端、未运行 YOLO/BF16/NCCL smoke、未正式训练**

## 1. 已实现的控制

新增默认关闭的 `tqgb_clip_reference`：

- `combined`：现有官方行为，仍按 `||g_base + g_route||` 计算全模型裁剪系数；
- `base`：T1-BRC 控制，先用 DDP-SUM 后的 route gradient 从 combined gradient 重建 `g_base`，只按 `||g_base||` 计算系数，再把该系数施加到未移除 route 的 combined gradient。

因此 BRC 保留原 T1 的 route mask、route count、common-anchor normalization、projection、gain 和 route gradient，只移除 route 对其它参数统一裁剪系数的瞬时间接影响。

首版 fail-fast 限定为：T1 `m2o`、正向 gain>0、反向 gain=0、旧 KD=0、普通 detector、`one2one_cv3` 存在、`compile=False`、`accumulate=1` 且 `nbs==global batch`。它不会静默用于 T2、外部 KD 或梯度累积训练；BRC 遇到 OOM 也会直接失败，不允许自动减小 batch 后破坏这一不变量。

## 2. 改动文件与 SHA-256

| 文件 | SHA-256 |
|---|---|
| `ultralytics/utils/tqgb_clip.py` | `cc05d32937c0d1c0c2cc5c596310a4cc3ea245184f4a9dd37ba9ecb9db2a4606` |
| `ultralytics/engine/trainer.py` | `fd69bcacbdf6da0ee17e6c0f3154f4dd9477d932b249de0408c362012a7b3708` |
| `ultralytics/utils/loss.py` | `e4dd9e41fb4af4ae085a8d6f686ab158c6b64ce4adf344258e8679cbd59077db` |
| `ultralytics/cfg/default.yaml` | `19c64e970d2bdc148dc592f146b40280a9c413c710c79829011ea988196eb1c0` |
| `ultralytics/cfg/__init__.py` | `acfe4e6d977b90327a088252fb94819ff178c7da637c2447e1e32340ec91dcc5` |
| `scripts/verify_tqgb_brc.py` | `9765c50f20cbeb9a7dca7da194e67c006ac6628c41a693c84695e39862b965e0` |
| `configs/research/tqgb_t1_brc_cls_m2o_proj_g008_e50_w4_pin_numa1.yaml` | `04b01461b0acfebef7f0f81fc86a170d5fbb97f53eee06e5257ab8f62a9e78fa` |

以上路径均以本地 `work/remote_research_source/` 为根。没有修改或覆盖 ASAC/PBGA/SARC 脚本。

## 3. 训练接口和审计量

研究 loss 显式暴露 `tqgb_loss_index=4`，训练器不再无断言地猜测 route 分量。BRC 在标准 combined backward 前只对 `one2one_cv3` 捕获 local route gradient；每个 rank 对每个 route 参数都执行一次相同顺序的 `SUM all_reduce`，本地无 route 时发送显式零张量，避免 rank 间 route coverage 不同导致 collective 不一致。

在 AMP unscale 后：

1. `g_base = g_combined - g_route`（只在 O2O cls 参数上减）；
2. 计算 FP32 全局 base norm；
3. 所有 combined gradients 乘 `min(1, 10/base_norm)`；
4. 不执行第二次 global clipping；
5. 清空 route cache 后再 optimizer/EMA。

正式运行会把以下全局口径的 epoch mean 写入 `results.csv`：

- `train/grad_base_reference_norm`；
- `train/grad_combined_norm`；
- `train/grad_tqgb_route_o2o_cls_norm`；
- `train/grad_clip_scale_base_applied`；
- `train/grad_clip_scale_combined_counterfactual`；
- `train/grad_shared_scale_ratio_counterfactual`；
- `train/grad_postscale_combined_norm`；
- `train/grad_route_parameter_coverage`。

## 4. 已执行测试

本机 `PyTorch 2.6.0+cu118`，强制 `CUDA_VISIBLE_DEVICES=''`，所有测试仅用 CPU：

```text
{"status": "PASS", "device": "cpu", "tests": [
  "default_off_config",
  "route_off_equivalence",
  "route_active_reconstruction",
  "two_rank_gloo_autograd_and_sum"
]}
```

覆盖内容：

1. 默认配置为 `combined`，只有正式 BRC 配置显式选择 `base`；
2. route=0 时，BRC 与 `torch.nn.utils.clip_grad_norm_` 的参数梯度和系数一致；
3. route-active 时，base norm 被正确重建、非 route 参数使用 `alpha_base*g_base`，接收头保留 `alpha_base*(g_base+g_route)`；
4. 两进程 Gloo + `find_unused_parameters=True` 下，真实 `DistributedDataParallel` forward、`autograd.grad(route)`、手工 route `SUM`、combined backward 和 BRC 后梯度均与闭式全局批结果一致；
5. 修改过的 Python 文件 `py_compile` 通过。

Windows Gloo 输出了 localhost/libuv 和 `find_unused_parameters=True` 的提示性 warning，但进程正常完成、所有断言通过；这不是测试失败。

## 5. 尚未验证的风险

1. **NCCL/CUDA reducer**：CPU Gloo 等价通过不等于四卡 NCCL 已通过；必须先做 TQGB 专属四卡 smoke。
2. **真实 YOLO 图与 BF16**：尚未在 YOLO26n 的真实 route loss 上验证显存增量、吞吐、BF16 route capture 误差和每个 `one2one_cv3` 参数的覆盖。
3. **显存/速度**：额外 `autograd.grad(..., retain_graph=True)` 和 route-gradient clone 会增加 O2O cls 的计算/显存；虽只捕获一个 head，仍需 smoke 实测。
4. **路由拓扑假设**：首版只捕获 O2O classification 参数，依赖已通过的 T1 stop-gradient 隔离测试；若未来修改 loss 使 route 进入 shared/O2M，当前实现必须 fail/re审计，不能直接复用。
5. **硬件故障**：服务器有既往不可纠正内存故障；任何 smoke/正式 run 出现新 MCE/SIGBUS 都必须作废并从头重跑。
6. **科学范围**：BRC 只排除 global clipping shared-scale coupling，不证明逐 anchor O2M teacher semantics；若 BRC 通过，仍需 T1-CRG 语义阴性对照与多 seed。

## 6. 下一安全动作

远端训练完全空闲后，再按固定顺序执行：源码差异与 hash 核对 → CPU/单卡真模型梯度等价 → 两进程 CUDA DDP → GPU 4–7、workers=4、pinned memory、NUMA1 的 3-epoch BF16 smoke → 无故障后才允许 seed42 50-epoch 正式 BRC。当前 staging 不能直接视为可运行的正式结果。
