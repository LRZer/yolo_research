# ASAC A2 首次正式启动中断记录（2026-09-19 22:12）

## 裁决

该 run **INVALID / 不可恢复 / 禁止 resume**。它没有产生任何 epoch 或性能结果，只保留为启动与主机重启证据。后续若继续，必须使用全新输出名从随机初始化重新开始。

## 时间线

- 22:11:42：启动前 direct NVML 显示物理 GPU4–7 均为 0% 且无 compute PID；目标 run、sidecar、log、start 均不存在。
- 约 22:11:59：写入 start 文件并启动 formal launcher。
- launcher 成功完成 provenance/golden 门禁，创建外部 launch sidecar；四卡 DDP 已初始化，尚未进入 epoch 1。
- 约 22:12 后 SSH 连接中断；目标 `127.0.0.1:9000` 的反向通道拒绝连接。
- 22:14：服务器重新启动；22:16 恢复 SSH，`uptime` 约 1 分钟。

## 远端残留与哈希

- run：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/asac_a2_live_e50_g4567_s42_w4_pin_numa1`
- 外部 sidecar SHA-256：`7e36ab5167c8e73294b329ebff5b53d2ab62088f8f06a2f1f32ea508c2c50604`。
- run 中只有 0 字节 `args.yaml`；无 `results.csv`、checkpoint、`asac_protocol.json` 或 `asac_state.jsonl`。
- start/log 也因突然重启留下 0 字节文件；mtime 分别为 22:11:59 和 22:12:06。交互端在断线前已经看到 launcher 和 DDP 初始化输出，但这些输出不得替代缺失的持久化训练证据。
- 旧目录、sidecar 和空文件全部保留，不删除、不覆盖。

## 当前启动的硬件事件

当前启动在 22:14:57 记录 `ProcessorError` / L1 cache data-read Machine Check：severity=`recoverable`，但记录字段 `Uncorrected: true`；CPU303、socket1、Bank0、地址 `0x27b85e3b40`。这是新的硬件风险证据，不能将该次重启归因于 ASAC 代码，也不能证明与 ASAC 无关。

服务器所有者此前已经明确允许在已知故障主机上继续使用，因此后续运行仍可作为 faulty-host exploratory；但每个 run 必须从头开始、逐 run 检查新增 MCE/SIGBUS，任何运行窗口内事件都使该 run 无效，最终正结果仍必须在健康硬件复现。
