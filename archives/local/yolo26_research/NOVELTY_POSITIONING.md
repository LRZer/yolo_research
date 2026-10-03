# YOLO26 双分配协同：创新边界与最近工作定位

日期：2026-09-18  
用途：决定哪些表述不能主张、哪些机制值得继续用实验验证。本文档是持续更新的检索审计，不替代投稿前的系统综述。

## 1. 最近工作带来的硬约束

| 最近工作 | 已覆盖内容 | 对本项目的约束 |
|---|---|---|
| [YOLOv10, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf) | 一致双分配联合训练 O2M/O2O，实现 NMS-free YOLO | 不能把“双头联合训练”或“一致匹配”当作贡献 |
| [One-to-Few, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_One-to-Few_Label_Assignment_for_End-to-End_Dense_Detection_CVPR_2023_paper.html) | 软 anchors 早期侧重表征学习、后期侧重去重 | 不能泛称“首次按训练阶段从密集监督转向稀疏监督” |
| [Group DETR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Group_DETR_Fast_DETR_Training_with_Group-Wise_One-to-Many_Assignment_ICCV_2023_paper.html) 与 [Mr. DETR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Mr._DETR_Instructive_Multi-Route_Training_for_Detection_Transformers_CVPR_2025_paper.html) | 用辅助 O2M 查询/路由强化 O2O 检测器，推理删除辅助路线 | 不能把“训练期辅助 O2M、零推理开销”本身当作贡献 |
| [Label Assignment Distillation, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf) | Co-learning LAD 允许共同训练网络动态交换 teacher/student 角色 | 不能声称首次动态切换教师和学生 |
| [CrossKD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html) | 将学生 head 中间特征送入教师 head，缓解标注监督与教师预测冲突 | “跨头蒸馏”“避免冲突监督”都不是足够窄的创新表述 |
| [BTKD++, IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02924-x) | 依据分类正确性和 IoU 区分检测教师的 easy/hard 样本，并做偏差过滤、纠正和动态课程 | 不能声称首次做 GT/IoU 质量门控或教师可靠性课程 |
| [Dual-head self-distillation via an auxiliary lightweight neck, Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S0925231226011471) | 单代 YOLO 双头自蒸馏，分类和定位都迁移，辅助 head 仅训练使用 | 不能把“YOLO 双头、单代、无外部教师、自蒸馏”作为核心新颖性 |

## 2. 当前可防守的研究问题

现有工作大多处理外部 teacher/student、额外辅助 head/query，或预先规定单向辅助关系。本项目关注的是同一 NMS-free detector 内由两种 assignment 产生的原生 O2M/O2O heads：二者共享数据、结构相近，却因正样本密度、ProgLoss 权重与部署角色不同而随训练发生相对能力反转。

较安全的论文问题是：

> 在原生双分配检测器中，O2M 是否始终适合作为 O2O 的教师？若不是，能否仅利用训练期、同 anchor/同 GT 的可观测证据，控制知识流的方向与退休时机，同时保持推理图完全不变？

这个问题与“更大模型教更小模型”不同，也比泛化的选择性 KD 更具体。当前 D0/B50 诊断已经观察到 O2M 整体更优率随训练下降并跨过 0.5；TQGB 探针进一步显示分类与定位的较优 head 经常不同。

## 3. 候选贡献的强弱排序

### A. S-QGAT：低复杂度、可解释，但新颖性中等

1. 只在两头分配到同一 GT 的同一 anchor 上传递，避免 query/anchor 对齐歧义；
2. 局部用分类概率与 IoU 的联合质量选择 O2M→O2O 候选；
3. 全局用上一完整训练 epoch 的 O2M 胜率控制下一 epoch 强度，并在胜率达到 0.5 后单调退休；
4. 统计只来自 train，不读取 val/test，不改变推理结构。

风险：BTKD++ 已覆盖教师质量过滤和动态课程，因此只有“原生双分配角色反转 + 因果全局状态 + 同 anchor/GT”整体成立，并在多种子和 COCO 上稳定改善时，才有足够辨识度。若只有 VOC 单 seed 的小幅增益，更适合作为技术报告或更强方法的消融。

### B. TQGB：新颖性更强，但必须有完整消融

1. 分类和定位分别比较 O2M/O2O 的任务质量，而不是给整个 anchor 指定单一教师；
2. 每个任务允许 O2M→O2O 或 O2O→O2M，响应训练中的角色反转；
3. 将教师建议投影到“学生预测—原监督目标”的闭区间，知识只能缩短监督残差，不能越过目标制造反向梯度；
4. O2O→O2M 路径可把部署 head 的稀疏高质量信息送回 O2M head 和共享 trunk，形成 deployment-to-trunk bridge；
5. 不增加推理参数或 FLOPs。

TQGB 只有在以下比较同时成立时才可作为核心贡献：dynamic > forward-only、dynamic > reverse-only、dynamic > no-projection、dynamic > no-trunk-bridge，并优于简单对称一致性。否则“复杂但无必要”的审稿风险很高。

## 4. 目前不能写进摘要的说法

- 首个 YOLO 自蒸馏或首个双头自蒸馏；
- 首个 O2M/O2O 协同训练；
- 首个动态教师/学生交换；
- 首个质量感知、IoU 感知或 GT 感知蒸馏；
- 单凭 VOC 单 seed 提升即声称通用、显著、SOTA；
- 没有梯度/归一化对照就把 D2 的提升归因于质量选择；
- 没有健康硬件复现就把当前数值作为最终论文结论。

## 5. 投稿级证据最低要求

1. 固定末轮、配对 seeds `{0,1,42}`，报告每个 seed 和 paired mean/std；
2. baseline、强度匹配、状态控制、关键模块消融；
3. VOC 用于机制验证，COCO2017 YOLO26n 是“通用检测”主证据；
4. 至少再验证一个模型尺度（优先 YOLO26s）；
5. 报告 AP/AP50/AP75/APS/APM/APL、训练时间、峰值显存、推理参数/FLOPs/延迟；
6. 最终结果在健康硬件从头复现，当前故障主机仅保留探索记录。
