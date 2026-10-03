# YOLO26n/VOC 150轮基线：真实框保护式推理模糊预注册协议

冻结时间：2026-09-23 20:37 CST。用户要求将先前 VisDrone 的“真实标注框保护、框外三档模糊”诊断迁移到早期 YOLO26n/VOC 150轮成熟基线。仅验证，不重新训练，不修改既有权重、VOC原始图像和标注。

## 固定模型与数据

- checkpoint 固定为 `/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42/weights/last.pt`，第150轮；训练记录显示最佳轮为150。它是随机初始化训练的 YOLO26n，不是官方 COCO 预训练模型。
- 数据固定为 `/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml`，VOC07+12 trainval 共16,551张训练，VOC2007 test 4,952张、YOLO转换后12,032个评估框。本轮不读取训练图像做测试。
- 主结果采用 `nms=false` 的端到端 O2O 部署头，历史同权重独立复评 mAP50–95≈0.57689；另做 `nms=true` 的 O2M+NMS，历史训练末轮≈0.58705。二者分表报告，不混为同一个“基线”。
- 图像 `imgsz=640`，`batch=16`（仅推理吞吐，不改变结果预期），`workers=4`、`conf=.001`、IoU `.7`、`max_det=300`、FP32、rect=True。原图重算值若与历史记录不一致，先检查评估路径，不以旧值硬充新结果。

## 五个预定条件 × 两个检测头

每个头单独运行 raw、sham、light、medium、heavy，全部基于同一权重和评估样本。raw 原图不处理；sham 读取框并做数据往返但不改像素；三档模糊在当前 LetterBox 输入像素上分别使用高斯 `sigma=3/8/16`。来自评估 batch 的真实 YOLO 框转成当前输入 xyxy，**框内保留原像素**；不扩边（margin 0、minimum margin 0），框外边缘以羽化 sigma=8 渐变到模糊图。与刚完成的 VisDrone GT Oracle 使用相同图像变换参数。记录 mAP50、mAP50–95、逐类 AP、真实框保护比例、像素变化和耗时。

VOC 的 YOLO 转换标签遵循现有评估协议，会跳过标注为 `difficult` 的原始 XML 目标；本轮“真实框”特指用于本评估的 YOLO GT 框，不宣称保护了原始 XML 中所有 difficult 区域。不能因此改变评估标签。

## 可信度和资源门禁

本轮将评估真值用于**处理待预测图像**，属于信息泄露的 Oracle；任何 AP 提升都不是无标注可部署提升。仅比较同一个 YOLO26/VOC 头的 raw 与三档差值；YOLOv8n/VisDrone 是另一模型与数据集，绝对 AP 不横向比较。VOC07 test 已在此前研究中反复查看，本轮也不是论文的盲测证据。

先做程序单测、少量图 smoke，再完成全量 raw/sham，检查二者 AP 完全一致。使用进程级 `CUDA_VISIBLE_DEVICES=4`、`numactl --cpunodebind=0 --membind=0`，仅在物理 GPU4 空闲时运行，不碰 GPU0–3或其他进程。开始时服务器当前 boot EDAC `mc0=0, mc1=52`；NUMA0 绑定不是硬件健康证明。前后检查 EDAC/MCE/SIGBUS；若 task 失败或权重/数据不符，停止并记录，不伪造完成结果。
