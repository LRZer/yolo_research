# ASAC Phase-1 工程门禁报告（2026-09-19）

## 当前研究方向

ASAC（Assignment-Stability Adaptive Curriculum）研究的是：在不改变推理网络的前提下，使用固定训练图像 panel 上的 O2M/O2O 分配稳定性，闭环调节下一轮 scale/translate 增强强度。其目标不是再堆一个注意力模块，而是检验“模型当前的分配稳定性是否可以作为训练增强课程的反馈信号”。

## 门禁结论

**PASS，可以进入 Phase-1a 的正式 A2/A0 50 轮筛选。** 这只是工程实现和测量一致性通过，不是精度提升结论。

### 1. 测量金标准

- 固定 checkpoint：B50 seed42 epoch50 `last.pt`，SHA-256 `49076ee6878d1478423bf714cdbe257fcd08e7dc67b5a9afa701b5e8e25e96fb`。
- 固定 panel：256 张 train 图；跨 clean/q=.25/.50/.75 共同存活 624 个 GT。
- panel 内容 SHA-256：`8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`。
- population SHA-256：`f57cffdeb8bc29ef66a21c384ac0c3cd7ceff9552c48245263cb6c7c51b14926`。
- q=.50 的新实现测量：O2M `0.9545479413122414`，O2O `0.9192160195274237`。
- 相对 P0 冻结期望的绝对差：O2M `0.0`，O2O `0.0`。
- 原生 criterion 合约：`E2ELoss`；O2M topk/topk2=`10/10`，O2O=`7/1`；stride=`8/16/32`；训练格式 checkpoint 中双分支存在，`head.end2end=False`。
- BF16/inference、模型参数/全部 buffer/Detect 额外 tensor、RNG 与 assigner OOM fallback 门禁全部通过。
- golden artifact SHA-256：`aabeb03b02da7e8e5c7d8f4b26efe8314b933816beb4364fc08c8e20ef1d0d41`。

早先 v2 因把“存在 O2M/O2O 双分支”误写成 `head.end2end=True` 而在测量前被门禁拒绝；该失败证据保留且未覆盖。v3 按 P0 原始事实修正为 `False`，不是修改测量结果。

### 2. 四卡 DDP 两轮联调

- 物理 GPU：4、5、6、7；NUMA1；batch=64；BF16；seed42；fraction=.02。
- 训练参数为 `workers=4`（每个 DDP rank 4 个，总显示 16 个）且 panel loader `pin_memory=True`。
- 两轮均使用 `train_q=.5`、有效 scale=.25、translate=.05。
- epoch1：O2M=`0.9560010067767226`，O2O=`0.9168933188544697`，联合稳定性=`0.9364471628155961`。
- epoch2：O2M=`0.9560010067767226`，O2O=`0.9169170632456619`，联合稳定性=`0.9364590350111923`。
- 两轮 O2M/O2O coverage 均为 1.0；对象数均为 624；OOM fallback 均为 false。
- panel、population 和 RNG hash 两轮一致；每轮模型完整性 hash 在测量前后自洽。
- run 外部 sidecar 与 run 内 `launch_record.json` SHA-256 均为 `df5d3369a2a22c66205e50a42e5118b3a527320e2ac0a32936d051db9b59b46e`。
- 运行返回码 0；运行窗口无新增 MCE/SIGBUS/Memory failure；结束后 GPU4–7 均无 compute PID。

短训从随机初始化且只看 2% 数据，因此末轮 AP=0 没有研究含义，禁止用于方法比较。此次只验证四卡通信、数据加载、固定 panel、反馈记录和证据链。

## 冻结代码与配置

- isolated source tree SHA-256：`135f65e164acc72d99f225b966a2da8c2bc754ec287e848b2db6a9ed7e4b8eee`（369 files）。
- ASAC measurement/runtime SHA-256：`88a204beed3b18742bf4b03cd201252d499ef2da19827a7d31f8c1a78b5bc7aa`。
- ASAC trainer SHA-256：`e997d4fc3852203f487732ddc086d982231bf87044885265a33c9e5633f4fdf6`。
- golden generator SHA-256：`2897c908f23d5e630e8d998c1084ef56813389a7e91f865d1ff86f557f513402`。
- launcher SHA-256：`b1e3aba1ddfe81ec9ab8c6054723d4057c11edbde9e9ea8e76d068c5769415d9`。
- formal A0 config SHA-256：`ddb4bcc7dcd9ded5a6b4cb8ba0cd45411fcb1fd5eacd2affd149e5ff91bdefcb`。
- formal A2 config SHA-256：`6a82c3f5b1af4f2ba96695dd2dd9783f6dc5963ceb453ac9648401c329bc0c65`。
- 最终可移植代码包 `asac_integration_bundle_v3.tgz` SHA-256：`31144ad7a8932925b143640155154dd9bb933e7e50ea0f7ecaf0d979f01210d6`；已验证 smoke fraction=.02，且不含 `__pycache__`、`.pyc` 或 pytest cache。旧 v2 包保留为历史证据，不再用于启动。

本地 launcher/protocol 测试为 43 PASS、1 项仅因 Windows symlink 权限跳过；远端对应测试为 44 PASS、24 subtests PASS。更早完成的 measurement、static、loader/prefetch/reset 与真实 CUDA 门禁继续有效。

## 下一步与解释边界

Phase-1a 按预注册先跑 seed42 A2，再跑同 seed A0，固定 50 轮 VOC trainval，主读数为固定 epoch50 mAP50-95；last-5 只作稳健性辅助。继续线为 A2-A0 至少 +0.003、last-5 同号、q 动态不退化、额外墙钟低于 12%。不允许根据 test AP 修改控制器、panel 或超参数。

当前服务器存在已知硬件内存故障，因此所有结果继续标记为 faulty-host exploratory；任何正结果必须在健康硬件从头复现后才能作为论文证据。
