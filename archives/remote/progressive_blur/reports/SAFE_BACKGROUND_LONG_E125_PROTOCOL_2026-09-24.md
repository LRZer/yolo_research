# 安全背景模糊 v0：125 轮长训预注册方案

记录：2026-09-24 01:04 CST，训练前。目标：检验先前 10%/3 轮联合训练负结果是否仅由训练过短、数据量过少造成；不是事后追逐 VOC test 数值。

## 资源决策

- `gpu8-team` 物理 GPU0–3 均空闲，拓扑均属 NUMA0；GPU7 有他人进程，NUMA1 最近出现不可纠正内存错误且仅剩约 5 GB 本地空闲内存。选 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `numactl --cpunodebind=0 --membind=0`，不跨 NUMA 凑卡。
- 开始前 EDAC `mc0 UE=0`、`mc1 UE=52`；最近内核错误记录在 NUMA1 所属页，NUMA0 有更早历史故障。此节点**不能认定健康**，仅供探索；新 MCE/SIGBUS 要记录并复核，不应干预其他用户进程。
- SSD 可用约 365 GB、NUMA0 本地可用约 59 GB。四卡增强烟雾试验 `sbg_aug_4gpu_numa0_e1_s2027` 已成功：1% train90、1ep、真实 Mosaic/平移/缩放/颜色增强、mix=0.5、batch64、workers4/rank、四卡 DDP，权重与单轮 CSV 可读，EDAC 未增。它只是工程检查，其 AP 没有研究意义。

## 配对实验

数据：固定 VOC train90/dev10 seed2027；train90 全量训练，dev10 1,655 张用于每轮验证；VOC2007 test 不用于调参。检测器共同起点是已从随机初始化在 VOC 上训练 20 轮的 `runs/voc_p1/a0_yolov8n_scratch_e20_b64_s2027/weights/last.pt`，**不是 COCO 官方预训练**。模糊组加载门控 map-only 全 train90 5 轮的 `runs/sbg_pilot/voc_gate_full_e5_g123_s2027/weights/last.pt`；先前审计证明其中 YOLO 检测器权重与共同起点逐 tensor 相同，因此门控的 5 轮没有额外检测器更新。两组将各做 **125 轮新的完整检测器训练**，相当于检测器累计约 145 轮，不能称为“从随机初始化只训练 125 轮”。

两组共享：seed2027、imgsz640、总 batch64、四卡、workers4/rank、AdamW `lr0=0.0005`、`lrf=0.01`、cosine 日程、warmup3、mosaic1 并于末10轮关闭、平移0.1、缩放0.5、水平翻转0.5、相同 HSV、BF16、每轮严格 dev 验证、每10轮额外保存检查点、早停耐心200。39 个共同配置字段已逐项比对一致。

模糊组：`sigma=8`、最大混合系数0.5、前5轮混合0，接下来10轮线性升至0.5；框监督辅助损失权重0.25，推理端仅用图像预测门控，无真实框。原图对照：完全不加载门控，以同一检测器起点训练同样125轮。两个作业只能串行占用 GPU0–3；先模糊组，后原图对照，不能同时占用或妨碍他人使用的 GPU7。

配置文件：`work/voc_pgb_yolov8n/configs/safe_background_voc_long_e125_s2027.json` 与 `baseline_voc_long_e125_s2027.json`，同名文件同步远端项目 `configs/`。输出隔离在远端 `/home/ubuntu/rz/progressive_blur_detection_research/runs/sbg_long_e125/`。日志和恢复状态另放该项目 `_management/sbg_long_e125_s2027/`。

## 验收与判断

1. 每组 `results.csv` 恰好包含 epoch1–125 的连续有限记录，`best.pt`/`last.pt` 可加载，checkpoint 的类别、输入、训练参数与配置一致。只看到“125/125”进度条不能算通过。
2. 主比较固定为两组第125轮权重的相同独立 `YOLO.val` 设置下 dev mAP50–95 与 mAP50；`best.pt` 是次要结果，须说明按 dev 选择可能乐观。之后看 20 类 AP、门控目标保护和 RTX3090 同批量延迟。不能拿旧 10%/3ep 结果作直接因果对照。
3. 只跑 seed2027 是探索性判断。若模糊组具有实际 AP/速度权衡价值，再考虑第二种子、另一数据集、健康机器复现；若无优势，保留负结果，不追加调参追逐 dev。
4. 监测应低频；服务器离线或重启先核实，不能把断线当作训练完成。只有在本任务进程确实不存在、GPU0–3 再次空闲、NUMA0/内核错误和 checkpoint 完整性都过关后，才考虑续训。不得从损坏权重强行恢复，不得终止他人的进程。

本协议为启动前约束。后续阶段、错误和结论均追加到研究日志，不修改上述比较规则以迎合结果。
