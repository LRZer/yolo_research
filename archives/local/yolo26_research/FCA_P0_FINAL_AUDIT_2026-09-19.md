# FCA-P0 全量零训练审计最终报告

时间：2026-09-19 19:11–19:18（Asia/Shanghai）  
正式产物：`fca_p0_formal_A1_20260919_1912/`  
机器裁决：**NO-GO_ON_CLEAN_VOC**

## 1. 一句话结论

在冻结的 YOLO26n seed-0 轨迹与完整 VOC07+12 trainval 上，O2O 最终 GT coverage 的理论可避免缺口只有 `31 / 240,348 = 0.0129%`，image-cluster bootstrap 95% CI 为 `[0.0087%, 0.0176%]`，远低于预注册的 `0.25%` 继续门槛；中后期 checkpoint 几乎没有该失效。因此，不实现、不训练 clean-VOC coverage-first matcher，关闭该主线。

这不是程序失败，也不是“方法训练后无提升”。它是更靠前的机制筛选：真实数据中的可干预事件太少，不值得投入训练预算。

## 2. 审计范围与执行协议

- 数据：VOC2007 trainval + VOC2012 trainval，共 `16,551` 张图、`40,058` 个 GT；没有读取 VOC2007 test 指标。
- checkpoint：`epoch0/10/20/30/40/last.pt` 六个 BF16 全量扫描；`last.pt` 另做 FP32 敏感性复核。
- 每次扫描：259 batches，顺序、无增强、`imgsz=640`、batch 64、workers 4、pinned memory、NUMA node 1。
- 设备：只使用物理 GPU 4；GPU 5–7 未被本任务占用。
- 零训练约束：inference mode；没有 optimizer、backward、gradient 或参数更新；114/114 个 BN 均为 eval。
- 六个 BF16 与 FP32 扫描均得到相同数据几何 SHA256：`05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`。
- 正式运行耗时 `458.54 s`；运行窗口新增 MCE/SIGBUS/EDAC 事件为 0。
- 21 项 CPU/合成回归测试全部通过；首批 O2M/O2O 的原生与插桩输出逐位一致。
- 远端正式目录和本地归档的全部 16 个受检产物 SHA256 一致。

## 3. 协议修正 A1 与意外发现

第一次正式尝试在产生全量统计前由保护断言终止。原因不是 trace 错位，而是冻结版 TAL 的真实 conflict 语义：共享 anchor 的 winner 使用所有 GT 的 `overlaps.argmax(1)`，未以 S1/top-k 掩码限制，因此 S2 可能新增 S1 外边。

在揭示任何完整 checkpoint 结果前冻结 A1：

- 原样保留 raw S2/S3，绝不修改原生 TAL，也不把新增边静默投影回 S1；
- 主 matching 图仍严格使用预注册的 `E_topk=S1`；
- 对原生已覆盖 GT identity 先在 S1 中做无损 surrogate rematch，再求最大匹配；若任一图不可无损 rematch，则自动阻断 GO；
- 同时报告 S1→S2 的删边、新增边、coverage loss/gain 与 S3 off-graph edge。

正式数据中确实观察到非单调 rewiring：六个 BF16 checkpoint 合计，O2O 在 S2 新增 `152` 条 S1 外边，O2M 新增 `138` 条。其性质是：

- 所有新增边都仍在 S0 几何池内；没有出现 S0 外新增边；
- O2O 的新增边均未保留到 S3，且没有造成 S2→S3 GT coverage loss；
- O2M 的 138 条新增边保留到 final，分布于 95 个 GT；off-graph edge 的 raw alignment 中位数仅约 `2.36e-4`，虽然 standard IoU 中位数约 `0.442`；
- 两个 head 的 S1→S2 coverage gain 均为 0。换言之，新增边改变了 pair，却没有让原来未覆盖的 GT 重新获得 coverage。

这是一个真实实现细节和可复现实验观察，但在本数据上的效应规模不足以单独形成强论文路线。

## 4. O2O 主结果

| checkpoint | precision | avoidable gap | rate | r≥0.50 gap | native final coverage |
|---|---:|---:|---:|---:|---:|
| epoch0 | BF16 | 28 | 0.0699% | 26 | 40,030/40,058 |
| epoch10 | BF16 | 1 | 0.0025% | 1 | 40,057/40,058 |
| epoch20 | BF16 | 0 | 0 | 0 | 40,058/40,058 |
| epoch30 | BF16 | 0 | 0 | 0 | 40,058/40,058 |
| epoch40 | BF16 | 0 | 0 | 0 | 40,058/40,058 |
| last | BF16 | 2 | 0.0050% | 2 | 40,056/40,058 |
| last | FP32 | 2 | 0.0050% | 2 | 40,056/40,058 |

六个 BF16 checkpoint 合计 `31/240,348 = 0.0129%`。预注册门槛需要至少约 0.25%，即当前点估计的约 19.4 倍。即使使用 bootstrap 95% 上界 `0.0176%`，仍约低 14.2 倍。

所有 31 个事件都发生在 S1→S2 conflict 阶段；S2→S3 的 GT coverage loss 为 0。因此，`topk2=1` 会删除边，但本次没有额外删除任何 GT 的最终 coverage。原命题中“conflict→topk2 联合造成显著最终 extinction”未被支持。

## 5. 时间稳定性与受影响样本

- 31 个事件对应 31 个不同 GT，没有任何 GT 在两个 checkpoint 重复出现。
- `28/31` 来自 epoch0，epoch10 只有 1 个，epoch20/30/40 都是 0，last 只有 2 个。
- 尺寸：`28/31` 为 `>=64 px`，另外仅 1 个 32–64、1 个 16–32、1 个 <16；它不是 clean VOC 上的 tiny-object 主导失效。
- 重叠：29/31 的 GT 与其他 GT 最大 IoU 至少 0.3，说明它确实集中于重叠/竞争，但事件量极小。
- competition degree：23 个为 1，8 个为 2–3，没有 ≥4。
- 类别计数：person 15、sofa 6、chair 6，其余 car/train/bottle/sheep 各 1；这些是事件计数，不应解释为经类别频率校正后的显著类别效应。
- 没有任何预注册 size×competition cell 达到 `>=50` 事件和跨至少 4 个 checkpoint 的 dense-tiny 触发条件。

epoch0 的 28 个事件中，精确 `G_r` 匹配在 `r>=0.50` 下仍可保留 26 个，说明早期事件未被 zero-quality 边主导；但是高质量并不能弥补事件总量和时间稳定性不足。last 的 BF16/FP32 均为 2 个事件且落在阈值同侧，裁决对精度口径不敏感。

## 6. O2M sidecar

O2M 六个 BF16 checkpoint 合计 gap 为 `169/240,348 = 0.0703%`，逐 checkpoint 为 `29/7/26/31/39/37`。它高于 O2O 且在后期持续，因此不能称为“O2O 特有的 coverage failure”；更准确的表述是一般 TAL conflict-order 现象。

但 O2M 仍远低于 0.25% 主效应门槛，单 checkpoint 最大也只有 39 个事件，且 O2M 是训练辅助头而非部署 O2O 头。该 sidecar 不足以推翻 clean-VOC NO-GO，也不授权改题后直接训练。

## 7. 自动门槛

通过：

- 几何/数据顺序完全一致；
- 原生覆盖 identity 在所有图上均可在 S1 中无损 rematch；
- zero-quality gap share 为 0；
- last BF16/FP32 位于 0.25% 阈值同侧。

失败：

- pooled bootstrap 下界未达到 0.25%；
- 没有 4 个 checkpoint 达到效应量与 100-event 门槛；
- 因前项不成立，4-checkpoint 高质量保留门也不成立；
- 没有 late checkpoint 达标。

由于 pooled bootstrap 95% 上界已经低于 0.25%，满足预注册 `NO-GO_ON_CLEAN_VOC` 条件。

## 8. 研究决策

1. **停止 clean-VOC FCA matching 主线。** 不进入 Phase 1 离线 matcher 对照，更不启动 coverage-first 训练；否则是在只有几十个事件的机制上继续投入并容易过拟合。
2. **不把 A1 rewiring 直接包装成创新。** 它是真实发现，但新增边没有产生 coverage gain，O2O final 又把它们全部消除；还存在 IEICE 2025 point competition/label reassignment、OTA/POTO/O2F 等强邻近工作。
3. **不把结果外推为“遥感无效”。** clean VOC 缺少高密度微小目标；本轮只否定 clean-VOC 主张。若以后明确转向遥感，可针对 AI-TOD-v2/VisDrone 等另写预注册审计，但当前 dense-tiny 自动触发器没有通过，不能把它当作本轮自然延伸。
4. **下一路线应重新讨论，而不是继续调 FCA 阈值。** 优先比较新的、可测量的问题，例如 shift/aliasing、定位噪声或领域特定失效；先做零训练因果诊断，再决定训练。

## 9. 产物索引

- `fca_p0_formal_A1_20260919_1912/decision.json`：机器裁决和全部门槛。
- `fca_p0_formal_A1_20260919_1912/report.md`：自动人读报告。
- `fca_p0_formal_A1_20260919_1912/objects.csv`：逐 GT 记录。
- `fca_p0_formal_A1_20260919_1912/images.csv`：逐图 matching 统计。
- `fca_p0_formal_A1_20260919_1912/components.csv`：有效 S1 连通分量统计。
- `fca_p0_formal_A1_20260919_1912/protocol.json` 与 `provenance.json`：冻结协议、A1、源码/checkpoint/data 指纹。
- `fca_p0_formal_A1_20260919_1912/scans.json`：loader、精度、criterion、模型完整性和逐 checkpoint 运行证据。
- `fca_p0_formal_A1_20260919_1912/SHA256SUMS.txt`：完整产物校验。

服务器原始目录：`/home/ubuntu/rz/yolo_study/yolo26_voc_research/audits/fca_p0_formal_A1_20260919_1912`。本地归档与其 SHA256 全部一致。
