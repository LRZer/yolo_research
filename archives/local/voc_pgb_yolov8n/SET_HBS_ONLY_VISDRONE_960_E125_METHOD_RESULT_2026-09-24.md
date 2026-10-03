# VisDrone HBS-only 首组 125 轮结果（等待配对复评）

记录：2026-09-24 约 06:57 CST。本文件仅归档 **HBS-only 首组**完成事实；identity-extra-loss 对照仍在训练，不能据此宣称方法优于对照、标准 YOLO 或已发表 SET。

## 完成与验收

- 任务专属管理器事件显示 HBS 子进程 PID 169129 于 06:52:57 CST 退出码 0，随后发出 `job_verified_complete`。管理器审计代码核对 `results.csv` 恰好连续 125 个有限 epoch，以及 `best.pt`/`last.pt` 可加载、模型类型、10 类、HBS 模式和训练参数。
- 远端训练路径：`/home/ubuntu/rz/progressive_blur_detection_research/runs/set_hbs_visdrone_960_e125/hbs_only_s2027_g0123_numa0/`。CSV 为表头加 125 行；两份完训 checkpoint 均约 8 MB。该组未使用 COCO 预训练，数据为 VisDrone official train6471/val548，seed2027、imgsz960、batch64、GPU0–3、NUMA0。
- 第 125 轮训练日志的 val P/R/mAP50/mAP50–95 分别为 `0.50634 / 0.37930 / 0.38069 / 0.21474`。这是训练时逐轮验证值，**不是**统一独立复评结果，也不能与不同实验的历史基线直接做因果比较。CSV 第 125 轮 `time` 为 3523.28 秒，约 58.7 分钟累计训练时间。
- 本次检查时 EDAC `mc0=0, mc1=52` 与启动前一致；近 10 分钟内核日志未见新 MCE/SIGBUS。未据此宣称服务器硬件整体健康。

## 配对进度与待办

- 管理器在 06:52:57 CST 对 identity 组重新通过资源/硬件预检，随即启动子进程 PID 178814；四个 DDP 进程占物理 GPU0–3，NUMA 策略 `bind:0`。GPU7 的其他用户 PID 30205 未被操作。
- identity 需同样完成并审计 125 轮；之后用同一套 `YOLO.val` 配置独立复评两组第 125 轮 `last.pt`，再报告 P/R/AP、逐类及小目标诊断与成本。`best.pt` 只能作次要结果。
- 本研究只移植已发表 SET 的 HBS 子模块，未实现 API，不属于完整 SET 或已证实的原创方法。单种子、转换后 YOLO 标签与官方 VisDrone 评测也限制外推。
