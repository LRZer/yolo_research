# ASAC A2 live state 独立复算（epoch 1–17）

审计时间：2026-09-19 22:33:16+08:00。对象为 faulty-host exploratory run `asac_a2_live_e50_g4567_s42_w4_pin_numa1_reboot_r1`。本审计只读，不修改文件或训练进程。

## 结论

无阻断问题。逐轮独立重算得到：

- `Sref=median(S3,S4,S5)=0.9106082852556012`。
- epoch>=6 的 `EMA=.8*prev+.2*S`、指数映射 q_target、`Δlog(q)` 的 ±.05 限制、[.25,1] 边界及上一轮 `q_next` 到下一轮 `train_q` 全部与 JSONL 逐位一致，最大绝对误差为 0。
- epoch17：train_q=`0.7104696703374558`、S=`0.9252236744296913`、EMA=`0.9226073731541369`、q_target=`0.7578343269544915`、q_next=`0.7468962292775695`。
- 已观测 q 范围 `[0.48952523182446545,0.7468962292775695]`；epoch 8–11、15–17 正确触发 ±.05 log-rate clip，无越界。
- 每轮 O2M/O2O object_pairs=624；最低 coverage=`623/624=0.9983974358974359`，高于 .95 门槛；OOM fallback 全为 false。
- BF16/inference/dtype 证据一致；RNG hash 固定为 `afa008...`；panel、population、config、trainer、measurement、bundle hash 与 protocol 和真实文件全部一致。
- 参数与注册 buffer hash 随模型训练逐轮变化；每轮测量前后完整性检查均通过；`asac_error.txt` 不存在。

Detect 普通 tensor 属性 hash 在 epoch1→2 变化一次，随后固定。它发生在 epoch1 panel 测量之后的首次验证阶段，符合 anchors/strides 推理 cache 初始化；每轮 panel 内部仍严格满足 pre/post 相等，因此不构成测量修改模型或协议漂移。

本报告只确认控制器和证据链按预注册执行，不评价 AP，也不改变 Phase-1a 判定门槛。
