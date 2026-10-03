# CUB 原图组 ResNet50 单卡长训记录

日期：2026-09-25（北京时间）。研究目录：`/home/ubuntu/rz/foveated_classification_research`。

## 冻结设置

- A/Original 原图组；`torchvision.resnet50(weights=None, num_classes=200)`，不使用预训练、不使用框/掩膜、不缩放或裁切原图。原生图像居中置于500×500画布，空白使用官方训练图有效像素计算的 RGB 均值 `(124,128,111)`；按相同训练集RGB均值和标准差归一化。
- 官方train 5,994张参与训练；官方test 5,794张每轮完整评估。用户明确选择不另划开发集。因每轮查看test，后者不再是盲测；第200轮 `last.pt` 为预定主结果，不能依test曲线搜索超参。
- seed 2005，200 epoch，单物理GPU0，有效batch64 = microbatch16×梯度累积4，workers4。注意BatchNorm实际看到的是microbatch16，不等同于物理batch64。
- 训练期仅做确定性随机水平翻转和轻度亮度/对比度/饱和度扰动；test无随机增强。SGD momentum0.9、weight decay0.0005，LR依有效batch64从0.025起，5轮线性warmup，随后余弦下降至0.00001；AMP FP16。所有对照组须保持等同训练预算与公共设置。
- 物理GPU0映射为唯一可见CUDA设备；`numactl --cpunodebind=0 --membind=0`。不占用其他卡、不改全局环境。

## 启动前验收与硬件边界

- 训练集RGB统计由全部5,994张官方train有效像素计算，元数据SHA见 `CUB_OFFICIAL_TRAIN_RGB_STATS_2026-09-25.json`；远端CPU核对官方train/test数、200类及元数据SHA通过。单GPU前后向短测32图、1次优化更新通过，实际峰值显存3.7069 GiB。
- 启动前物理GPU0 4 MiB/0%，NUMA0空闲约50,308 MiB，当前启动周期EDAC mc0不可纠正错误计数为0；GPU1–7均有其他任务，不触碰。
- 此轮内核日志在2026-09-24 12:14启动时记录一条带前一周期时间戳的socket0 CPU缓存MCE；不是此次训练启动后新出现的EDAC内存错误，但不能据此宣称机器硬件完全健康。若出现新的MCE、EDAC不可纠正错误或SIGBUS，需重新评估训练有效性。

## 执行状态

- 2026-09-25 04:02:14 CST：任务专用启动程序验收后启动训练，PID 962802，日志 `runs/resnet50_A_native500_s2005_e200/train.log`，独立配置SHA-256 `e9c4703dbde50beab3ca3c8c26936c059fe787ce2103433054d8b6382869e1c7`、统计文件SHA-256 `e1caf32dd617b8d91720d117ff8527c75a5dbc1b82c71fbb1eabaecd88a7b309`。04:02:32复查进程仍在、GPU0约4630 MiB/98%。
- 每轮日志、`results.csv`、`state.json`及原子更新的`last.pt`写入独立run目录。至04:04:53完成3/200轮；第1–3轮完整单轮分别50.39、49.38、49.70秒，平均49.82秒/轮，其中训练约37.6–38.1秒、完整官方test约11.8–12.3秒。按照当前测得速度和第3轮时刻粗估约06:48 CST完成200轮，仍会受其他任务的共享CPU/磁盘负载、硬件和后续速度变化影响。3轮时mc0本boot不可纠正错误计数仍为0。
- 第3轮临时test Top-1 1.691%、Top-5 6.093%。正处于随机初始化与5轮warmup早期，不是最终准确率，也不能用它调参或做A/D结论。
