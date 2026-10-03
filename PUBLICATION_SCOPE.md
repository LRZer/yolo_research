# Publication scope / 公开归档范围

This is an evidence-backed research archive, not a pretrained-model release. The two current narrative editions cover the entire recorded project, including early YOLO26 work, the cross-domain baseline suite, VOC/VisDrone blur and foveation, CUB representations, global/local models and the user-stopped BN experiment.

本仓库是基于原始证据的研究档案，不是预训练模型发行版。中英文当前报告覆盖全部已记录项目：早期 YOLO26、跨域基线、VOC/VisDrone 模糊与凹视、CUB 图像表示、全局/局部模型及用户停止的 BN 实验。

## Included / 已公开

- **1,812 evidence files** from local research outputs, server reports/runs and project-owned source. The evidence manifest stores original and published SHA256, file size and normalization/redaction/compression history. Unrelated upstream framework documentation was excluded from research counts.
- Original CSV/JSON/JSONL metrics, configuration and data/model fingerprints, full per-class and per-epoch values, diagnostic panels, failure/stop/recovery/audit records and recorded dataset profiles.
- **96 canonical server training CSVs / 6,056 epoch rows**. These include smoke, pilots, invalid starts and incomplete runs; the run registry distinguishes them. They are not 96 completed proposed-method experiments.
- **374 historical research Markdown records**, in their original languages. The curated Chinese and English chapters both synthesize the full research history. Historical notes are evidence, not current instructions; early live status and plans can be superseded.
- Research trainers, transforms, modules, tests, launch/audit scripts, configurations, source indices, scientific figures and scripts to regenerate the publication outputs.
- Four large numerical records preserved by lossless gzip: two frozen-detector prediction JSON files, and the FCA image/object tables. Their uncompressed SHA256 is recorded as `source_sha256`. Gzip can be read using Python's standard `gzip` module without expanding the files on disk.

公开内容包括 1,812 个证据文件、96 份服务器规范训练 CSV 的 6,056 行逐轮结果、374 份历史研究 Markdown、362 份历史 Python 实现、配置、逐类 AP、TP/FP、保护率和数据集全量统计。无关上游框架文档不计入本项目研究数量。规范训练记录包含短测、试验、失败启动和未完成组，**不能将其统称为 96 项成功实验**。历史档案保留原语言；当前完整研究报告提供中英文双版本。

## Not redistributed / 未重新分发

- Raw dataset images, segmentation masks and complete third-party dataset archives. Dataset-derived input previews are not republished as original image collections; transformation code, preview dimensions and manifests are retained. Synthetic geometry figures are explicitly illustrations, not experimental samples.
- Model/checkpoint binaries. Their paths, hashes, loading audits and measured results remain available, but the checkpoints themselves are not claimed to be public downloads.
- Third-party paper PDFs, full vendored framework trees, unrelated project/user files, SSH configurations/private keys, authentication tokens and private network endpoint values.
- Unsaved outputs cannot be invented: the current CUB experiments did not persist prediction-level confusion/calibration/per-class accuracy files. No such measurements are fabricated here.

不公开原始数据图像、分割掩膜、模型权重、第三方论文 PDF、完整框架副本、无关用户文件及认证信息。保留处理代码、预览尺寸清单、模型哈希和审计结果；合成几何图不冒充真实实验图。未保存的 CUB 逐样本预测、混淆矩阵和校准数据不补造。

## Provenance and interpretation / 来源与解释规则

Current chapters take precedence over older status snapshots. All collected original numeric values remain unchanged except lossless text normalization; publication redactions do not change scientific values. Duplicate local/server snapshots keep their provenance but are not counted twice as training runs. A source hash identifies historical inputs; it does not assert the public archive contains the original dataset or weight bytes.

当前章节及最终审计/停止记录优先于旧状态快照。原始科学数值不改写，文本标准化与公开脱敏另行标注；本地/远端重复快照不重复计算训练。来源哈希不等同于已公开原数据或权重。

Oracle interventions used evaluation GT boxes. Best results were selected from inspected evaluation sets. G6 stopped at 50 epochs. These limitations are part of the findings, not missing success claims. Nothing in this publication launches or resumes training.

Oracle 使用评估真值框，best 来自已查看评估集，G6 在 50 轮被用户停止。这些限制属于正式结论。发布操作不启动或恢复训练。

See [evidence manifest](results/evidence_manifest.csv), [reproducibility (English)](docs/en/07_reproducibility.md), [复现说明](docs/zh-CN/07_reproducibility.md) and [licensing notice](LICENSING.md).
