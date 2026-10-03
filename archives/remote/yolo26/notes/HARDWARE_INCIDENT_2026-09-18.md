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

## 已采取措施

- 没有把失败启动当作实验结果；D2 尚无有效 epoch。
- 失败目录移动到远程 `runs/_failed_stubs/`，保留参数与审计痕迹。
- 没有重启共享服务器、删除其他用户进程或清理系统级共享内存。
- 停止进一步 GPU 训练和压力测试，防止扩大数据损坏风险。

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
