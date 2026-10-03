# 08:04 新内存硬件错误：暂停后续 GPU 试验

记录：2026-09-24 08:07 CST。对服务器仅做了任务范围内的文件上传、预检和冻结模型推理诊断；没有触碰其他用户作业/GPU7，也没有修改共享环境。下面区分已经完成的证据与尚未完成的试验。

## 故障前已完成且有落盘证据

- VisDrone 4×4 只读 GT 密度上界已完成 train6471/val548，见 `FOVEATED_ATTENTION_FOUR_PAPERS_REVIEW_2026-09-24.md`。
- 冻结原图 YOLOv8n/960/125ep checkpoint、只用物理 GPU0/NUMA0 的无标签语义选区覆盖试验已于 **07:59:51 CST** 完成并写出 `diagnostics/foveated_region_coverage_20260924/full548_order_checked.json`。文件包含548张/38,759框、模型SHA `e352a00f…491cc7`、manifest SHA `dc54c818…94b048`、预测与原图路径顺序断言。粗检未被任意同类 IoU≥0.5 框覆盖的10,303个GT中，预测热度top4格覆盖框中心75.67%、top8格93.87%；这是区域覆盖率，不是 AP/实际高分辨率再检测成功率。
- 运行前 EDAC 为 `mc0=0, mc1=52`，物理GPU0–6空闲、GPU7有他人PID30205。此前的覆盖程序正常退出并落盘。该时间与下述新硬件错误相隔约4分钟。

## 故障事实与处置

- 08:04:07 至08:04:31，`journalctl -k` 记录同一 MC1/channel0 的多次不可纠正内存错误，EDAC `mc1` 从52升到58，`mc0` 仍0。内核出现 `MCE: Killing python:195841`，随后08:04:29 `Sending SIGBUS to python:195931 due to hardware memory corruption`；另有 `teamviewerd` 与 `apport` 受到SIGBUS。两次刚启动的裁剪推理脚本**CPU自测**均退出码1且无正常输出，时序上高度吻合这两个 python PID；没有进一步进程审计记录能绝对证明PID对应，故表述为“很可能命中本任务自测”。
- 按此前安全约束，**暂停所有新的 GPU 训练/推理及自测**，不机械杀任何正常进程，也不重试或更改 NUMA/系统配置。尚未运行固定中央4格 vs 自适应4格的裁剪推理 AP 试验；`visdrone_foveated_crop_pilot.py` 仅完成本地语法检查并上传，未通过远端自测，不能作为已验证代码或结果。
- 08:12 CST 用本机 `pytorch` conda 环境（PyTorch 2.6.0、Ultralytics 8.4.146）运行该脚本的**纯合成几何/NMS/匹配自测**，返回 `SELF_TEST_OK`。这是本机单元级验证，不代表远端版本兼容、真实图像推理或完整AP验收；远端两次自测仍为受硬件错误影响的失败。
- 服务器所有者先前允许继续使用有历史问题的硬件，不代表可在新的不可纠正错误爆发后忽略现场证据；本轮继续做文献与既有结果的离线分析，实验恢复需重新预检和明确判断。用户最晚10:00的研究截止仍有效；到时停止自动复查但不终止正常作业。

## 08:09 重启后的状态（不是修复证明）

- 服务器 `uptime -s` 显示约08:09:38重新启动。08:15读取 EDAC 为 `mc0=0, mc1=0`，这是**新boot计数重置**，不能覆盖上一boot的52→58错误事实；新boot日志暂只有EDAC设备初始化，无新的MCE/SIGBUS记录。
- NUMA拓扑仍为GPU0–3对应node0的既有配置；node0本地free约120GB，磁盘余约363GB，GPU当前无计算进程。GPU7先前他人进程已因服务器重启消失，与本任务无关且未操作。
- 故障前落盘的完整548图无标签选区JSON已通过 SCP备份至本地 `FOVEATED_VISDRONE_SEMANTIC_REGION_COVERAGE_VAL548.json`，本地与远端SHA256均为 `241afba9…36d2a5`。待新boot保持一段无错静默，再考虑只在node0做纯自测和受控推理；**尚未恢复GPU作业**。

## 08:39 后续更新：受控恢复完成，历史风险仍在

新boot观察后，在严格绑定NUMA0/物理GPU0且GPU空闲条件下，远端CPU自测通过、16图推理冒烟通过；随后完成一次548图三组冻结推理，并因普通predict与validator的 `multi_label` 口径差异，统一为验证器设置后又重算一次全量。详细结果见 `FOVEATED_VISDRONE_CROP_PILOT_RESULT_2026-09-24.md`。本轮结束时新boot EDAC mc0/mc1均为0，近30分钟未见新增MCE/SIGBUS；**这只表示观察窗内没有新记录，不是硬件修复证明**。未启动新训练，未使用GPU4–7，未动其他进程。
