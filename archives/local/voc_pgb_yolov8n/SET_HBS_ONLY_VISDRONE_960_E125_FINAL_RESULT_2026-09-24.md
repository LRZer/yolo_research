# VisDrone/YOLOv8n：SET-HBS-only 移植的 125 轮配对结论

记录：2026-09-24 约 07:35 CST。**本次 HBS-only 移植未超过同预算 identity-extra-loss 对照，预注册的探索性 go 条件未达到，判定该版本 no-go。** 这是单种子、VisDrone 转换标签、Ultralytics 评价器下的结果；不能推出完整 SET 或所有背景特征平滑方法无效，也不能将本次试验写成原创提升。

## 方法边界与可比性

- 来源：SET/CVPR 2025 的 Hierarchical Background Smoothing (HBS)；原论文完整方法还包括 API。本项目只把 HBS 训练期分支移到 YOLOv8n 三个检测特征尺度，没有实现 API。[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf) · [官方代码](https://github.com/HuixinSun/SET/blob/master/mmdet/models/detectors/fcos_set.py)
- 两组均为随机初始化 YOLOv8n、VisDrone official train6471/val548、10类、imgsz960、seed2027、125轮、batch64、物理 GPU0–3/NUMA0。配置除运行名和 `hbs_mode` 外一致，两个冻结 JSON 的 SHA256 与预注册文档一致；起始检测器参数逐 tensor 一致。
- HBS 组损失为原始检测损失加 HBS 特征分支检测损失；identity 组用未经平滑的原始特征计算同一份第二损失，控制“额外检测监督/损失尺度”因素。两组推理均只走原 YOLO 路径，不读取框或运行 HBS 分支；identity 是本轮**主因果对照**。旧的普通 YOLOv8n/VisDrone 结果仅供背景参考，不能代替这一配对组。
- 管理器事件依次记录 HBS 和 identity 子进程退出码0、各自 `job_verified_complete` 和最终 `pair_verified_complete`。两组 CSV 各为表头加连续125个有限 epoch；`best.pt`、`last.pt` 可加载，检查点类别、模式及训练参数一致。末轮权重是主比较，best仅次要。

## 同一验证集独立复评：第125轮 last.pt

两组均在同一 VisDrone official val 548 张、同一数据 YAML SHA256 `715bfbfc…a2e84e334` 上，以同一 GPU0、FP32、`YOLO.val`、imgsz960、batch8、workers4、conf0.001、IoU0.7、max_det1000、rect=True 评估。完整数据 YAML SHA256、权重 SHA256、参数和逐类指标见两份原始 JSON；不是官方 VisDrone challenge 评价器。

| 指标 | HBS-only | identity 对照 | HBS − 对照 |
|---|---:|---:|---:|
| Precision（类别均值） | 0.508415 | 0.492073 | +0.016342 |
| Recall（类别均值） | 0.378393 | 0.403812 | −0.025420 |
| mAP50 | 0.380833 | 0.396232 | −0.015399 |
| mAP50–95 | **0.214733** | **0.229070** | **−0.014337** |

主指标低约 **1.434 AP 点**，且 Recall 和 mAP50 同时下降。训练 CSV 末轮 mAP50–95 为 HBS 0.21474、identity 0.22907，与独立复评相符。旧标准原图 YOLOv8n/960/125ep 的约0.23255只能做非配对上下文，不以其与本轮数字进行严格因果判断。

| 类别 | HBS AP50–95 | identity AP50–95 | 差值 |
|---|---:|---:|---:|
| pedestrian | 0.1966 | 0.2091 | −0.0125 |
| people | 0.1220 | 0.1363 | −0.0143 |
| bicycle | 0.0477 | 0.0617 | −0.0140 |
| car | 0.5438 | 0.5629 | −0.0191 |
| van | 0.3001 | 0.3182 | −0.0180 |
| truck | 0.1906 | 0.2124 | −0.0218 |
| tricycle | 0.1355 | 0.1400 | −0.0046 |
| awning-tricycle | 0.0676 | 0.0894 | −0.0218 |
| bus | 0.3547 | 0.3521 | +0.0026 |
| motor | 0.1887 | 0.2085 | −0.0198 |

九类下降、一类小幅上升；负结果不是仅由某一类别造成。HBS 的 Precision 略高但 Recall 更低，表现为更保守的输出；不能仅凭这两个均值定位具体的训练机制。

## 按目标尺寸的诊断性召回

使用相同 `YOLO.predict` 设置、同一548张/38,759个转换后标注，对每类预测框做置信度排序的一对一匹配。尺寸桶按“若缩放到640方形 letterbox 时的框面积平方根”定义，**不是标准 AP_small，也不是官方 VisDrone 评价**；`predict` 预处理也可能与上述 `val` 略有不同，因此只在两组之间比较本诊断。

| 桶 | 标注框数 | HBS R@IoU.5 | identity R@IoU.5 | HBS R@IoU.75 | identity R@IoU.75 |
|---|---:|---:|---:|---:|---:|
| 全部 | 38,759 | 0.7078 | 0.7277 | 0.4063 | 0.4327 |
| tiny，<16px | 27,103 | 0.6283 | 0.6570 | 0.2655 | 0.2987 |
| small，16–32px | 8,724 | 0.8806 | 0.8824 | 0.6936 | 0.7059 |
| medium，32–96px | 2,897 | 0.9299 | 0.9213 | 0.8564 | 0.8595 |
| large，≥96px | 35 | 0.8286 | 0.9143 | 0.6000 | 0.7429 |

`tiny` 占标注约70%，且 HBS 在 IoU.5 / IoU.75 分别低2.87 / 3.32个百分点，是值得进一步核查的主要失效群；large 仅35框，不能作稳定结论。此诊断仍不能区分“掩码漏保护”“特征平滑破坏上下文”“训练期两损失冲突”或“超参数不适配”。

## 成本、安全与下一步判断

- CSV 累计训练时间 HBS 3523.28 秒、identity 2150.44 秒，HBS 约 **1.64倍**。两者推理的原 YOLO 路径相同；同一验证器单次测得模型前向约1.384/1.382 ms/图，差异可忽略。但 HBS 检查点保留训练期 denoiser 参数，`YOLO.val` 模型摘要约4.05M vs3.01M 参数；不应声称权重文件或加载内存零成本。两组后处理耗时不同，单次 validator 计时不足以作稳定部署延迟结论。
- GPU 作业固定物理 GPU0–3，NUMA0；GPU7其他用户进程未操作。完训及评估前 EDAC `mc0=0, mc1=52`，最近10分钟无新 MCE/SIGBUS；历史硬件问题仍在，不将其表述为机器完全健康。
- 预注册探索性 go 线是 HBS 比 identity 至少 +0.005 mAP50–95，并保持小目标。实际 −0.014337 且 tiny 召回下降，故本版 **no-go**。不基于同一个 val 集反复搜索卷积核或权重使数字变好。
- 后续优先做只读/CPU 机制审计：检查三个特征尺度的框保护覆盖、真实训练增强后的 tiny 框是否在粗层 nearest 掩码中消失，并审视原论文 HBS+API 与本轮 HBS-only 的差异。若没有清晰、可复验的失效证据，就不继续训练这个分支。另选已发表方法也必须先有同预算控制、多个种子和独立数据集才能谈论文贡献。

原始证据：`SET_HBS_VISDRONE_HBS_LAST_VAL548.json`、`SET_HBS_VISDRONE_IDENTITY_LAST_VAL548.json`、`SET_HBS_VISDRONE_HBS_LAST_SIZE_RECALL.json`、`SET_HBS_VISDRONE_IDENTITY_LAST_SIZE_RECALL.json`，以及远端任务目录 `_management/set_hbs_visdrone_960_e125_s2027/events.jsonl` 和两组 `results.csv` / checkpoint。
