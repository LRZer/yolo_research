# Licensing and attribution / 许可与归属

This historical publication does **not** assign a blanket MIT/Apache license to every archived file. Preserve and follow the license/copyright headers of the originating implementation and the terms of its dependencies. Publication of an archive is not a claim that all underlying data, frameworks or methods were created by this project.

本次归档**不对全部历史文件统一追加 MIT/Apache 许可**。应保留并遵守原实现的许可、版权头及依赖条款。公开档案不意味着原始数据、框架或既有方法由本项目原创。

## Frameworks and adopted methods / 框架与移植方法

- Ultralytics YOLO code and derivatives remain subject to the applicable [Ultralytics license](LICENSES/Ultralytics-AGPL-3.0.txt), including its AGPL-3.0 or separately obtained commercial terms as appropriate. The original license text is included with the narrowly selected research source overlays. Custom integrations do not cancel those requirements. This repository is not an official Ultralytics release.
- PyTorch/torchvision dependencies retain their upstream licenses. Their complete source distributions are not vendored here.
- SET-HBS-only is an explicitly attributed component adaptation, not a complete SET reproduction or an original spectral-enhancement invention. The [authors' repository](https://github.com/HuixinSun/SET) and official paper are cited in both narrative editions. Other attention, assignment, normalization and crop-related work is similarly attributed in the literature chapter.
- Project-specific source with no explicit grant in the original file is archived without inventing a new permission grant. Consult the repository owner and relevant upstream rights holders for reuse conditions.

Ultralytics 及派生实现遵循相应 AGPL-3.0/另行商业许可，PyTorch/torchvision 保持原许可。SET-HBS-only 明确标注移植来源；不能将移植、复现或简单组合表述为原创。没有原许可声明的项目代码不在本次整理中擅自重新授权。

## Data and papers / 数据与论文

CUB images are restricted to the uses specified by the [official dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/), including non-commercial research/education restrictions and underlying image copyrights. VOC, VisDrone and every other dataset retain their own licenses and attribution requirements. Data acquisition/conversion code and aggregate statistics do not confer ownership of the original images. Obtain data directly from its authorized source.

CUB、VOC、VisDrone 及其他数据集均遵循原始许可与引用要求；下载/转换脚本及汇总统计不赋予原图版权。请从授权来源自行获取数据。第三方论文仅引用，不重新分发 PDF。

Archived configs/logs may reference original local/server paths and upstream URLs as provenance. Those references do not grant access to machines, data, credentials or weights.

历史路径和上游链接仅作为来源，不提供服务器、数据、凭据或权重访问权限。
