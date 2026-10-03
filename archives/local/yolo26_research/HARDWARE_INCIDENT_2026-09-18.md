# YOLO26 研究服务器硬件内存故障记录

## 结论

2026-09-18 的 D2 训练未能开始。连续多次 `SIGBUS` 不是模型代码、GPU 显存、磁盘或 `/dev/shm` 容量导致，而是 Linux 内核确认的不可纠正主机内存错误。为避免静默破坏 checkpoint、缓存或数据，远程训练已暂停；管理员处理硬件前不应继续强行重试。

## 已确认事实

- D1 和 B50 均完整训练 50 epochs，结果文件分别在 14:00 和 14:32 写完。
- 当前能查到的首批内核 MCE 记录从 17:01 开始，晚于 D1/B50 完成时间。
- D2 的三次原配置启动和一次 `workers=2` recovery 均在第一个有效 epoch 前失败。
- 主机仍有约 226 GiB 可用内存，`/dev/shm` 约 126 GiB 可用，根分区约 437 GiB 可用；因此不是容量不足。
- 内核原文包含：
  - `Uncorrected hardware memory error in user-access`
  - `Memory failure: ... Sending SIGBUS to python ... due to hardware memory corruption`
  - `Sending SIGBUS to pt_data_worker` / `pt_data_pin`
  - 多个不同物理页地址、CPU 核和两个 PPIN 均出现记录。

### 24 小时日志的进一步定位

用户导出的本次启动完整相关日志共 757 行。按明确事件计数：

- `Uncorrected hardware memory error in user-access`：47 次；
- `Sending SIGBUS`：47 次；
- `EDAC MC*: 1 UE`：至少 14 次；
- 启动阶段 `event severity: fatal`：1 次。

所有带 FRU/EDAC 源定位的信息均一致指向：

- `FRU Text: Locator: P0MC0`；
- `EDAC MC0`；
- `mc#0 csrow#1 channel#0`；
- `Unified Memory Controller`；
- `UECC`，即不可纠正 ECC 错误。

因此最优先怀疑 Processor/Socket 0 的 Memory Controller 0 所连接的 `csrow 1 / channel 0` 内存路径，包括对应 DIMM、插槽、通道或内存控制器。后续记录中的两个 PPIN 和多个 CPU 编号主要表示不同 CPU 核消费了已污染内存页，不能仅凭消费者编号认定两个处理器都发生相同 DIMM 故障；应由管理员使用 EDAC、BMC 与主板插槽映射确认具体 FRU。

日志还显示服务器启动后约 10 秒即报告一个 `fatal`、`Uncorrected`、`Processor Context Corrupt: true` 的 processor/micro-architectural error（Socket 1）。该记录可能由固件/APEI 在启动时上报，但至少证明平台在本轮研究训练前就已有严重硬件健康告警，不能归因于 YOLO 或 D2。

## 已采取措施

- 没有把失败启动当作实验结果；D2 尚无有效 epoch。
- 失败目录移动到远程 `runs/_failed_stubs/`，保留参数与审计痕迹。
- 没有重启共享服务器、删除其他用户进程或清理系统级共享内存。
- 在用户要求谨慎再试后，只运行了一个数秒、限定物理 GPU 4–7 的四进程 CUDA/NCCL 卷积与 all-reduce smoke test；它在 rank 3 再次以 `SIGBUS` 退出。
- smoke test 失败的同一时刻（18:21:43），内核新增 `Uncorrected hardware memory error`，并记录 `Sending SIGBUS to python:347321 due to hardware memory corruption`。这确认故障仍在活动，绝非旧日志残留。
- 用户随后在自己的终端只读查询内核日志，发现 18:22:35–18:24:17 仍连续产生新的不可纠正内存错误；受害进程已从 Python 扩展到 `sed`、`fgrep`、`modinfo` 和 `tokio-rt-worker`。这表明故障影响整台主机的一般用户态进程，不是 YOLO、Conda、DataLoader 或特定 GPU 进程问题。
- 随即停止进一步 GPU 训练和压力测试，防止扩大数据损坏风险。

## 前两日历史记录

当前 journal 只保存本次启动，但轮转的 `/var/log/kern.log` 仍覆盖 9 月 16–17 日。只读复制后在本地统计得到：

- 2026-09-16：启动时已有 1 组 `fatal` 硬件事件，包含 `Storage error in DRAM memory`、`P0MC0`、`Processor Context Corrupt: true` 和 `Uncorrected: true`；保留日志中未见当天用户进程 SIGBUS。
- 2026-09-17：42 次 EDAC UE、20 次不可纠正用户态内存访问、20 次 SIGBUS、2 组启动阶段 fatal 记录。
- 9 月 17 日故障并非单一路径：
  - `P0MC0 / EDAC MC0 / csrow 1 / channel 0`：5 次 UE；
  - `P0MA0 / EDAC MC0 / csrow 1 / channel 3`：9 次 UE；
  - `P1MI0 / EDAC MC1 / csrow 0 / channel 6`：16 次 UE；
  - `P1MG0 / EDAC MC1 / csrow 1 / channel 9`：12 次 UE。
- 受害进程包含 `postgres`、`node`、`claude`、`conda`、`rg`、`libuv-worker`、`python3` 等普通服务和工具。
- `last -x reboot` 显示 9 月 17 日发生多次重新开机，说明平台不稳定性早于本研究本轮训练。

因此，今天短窗口内集中出现的 `P0MC0` 不能被解释为唯一故障 DIMM；历史证据表明两侧 memory controller 的多条路径均出现 UE。必须由管理员排查整个平台的 DIMM、插槽、供电、主板、CPU memory controller、BIOS/固件和内存配置，不能通过用户态 NUMA 绑定可靠规避。

## 建议管理员处理

1. 检查完整 `journalctl -k` / MCE / EDAC / `rasdaemon` 记录，定位出错 NUMA socket、内存通道和 DIMM。
2. 对故障内存下线或更换；如需重启、BIOS/BMC 诊断或离线内存测试，应由服务器管理员安排。
3. 检查关键项目、数据缓存和最近 checkpoint 的文件系统完整性。
4. 修复后先确认一段时间内不再产生新的 uncorrected MCE，再运行短四卡通信/数据加载 smoke test。
5. smoke test 通过后，重新从随机初始化启动 D2；不要续接任何失败 stub。

## 恢复门槛

只有同时满足以下条件才继续正式实验：

- 管理员确认硬件故障已处理；
- 内核日志无新的不可纠正内存错误；
- 四卡 4–7 的 CUDA/NCCL 与数据加载 smoke test 通过；
- D2 输出目录为空且从头训练；
- 研究日志记录维修时间、验证证据与恢复后的精确配置。
