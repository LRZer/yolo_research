# JODD 下载状态（2026-09-20）

## 统一数据目录

- 服务器：gpu8-team；SSH身份：ubuntu。
- 用户指定的统一目录：`/home/ubuntu/rz/datasets`。
- JODD独立目录：`/home/ubuntu/rz/datasets/JODD`。
- 两目录本轮新建，realpath已核对，所在分区可用空间约443 GiB。
- 未移动或覆盖 `/home/ubuntu/rz/yolo_study/datasets` 中既有数据，也未修改训练配置。

## 当前状态：尚未下载成功

官方数据页：<https://www.godac.jamstec.go.jp/jedi/e/dataset/jedi_organism_detection_dataset.html>

官方DOI元数据页：<https://www.jamstec.go.jp/datadoi/doi/10.17596/0004035.html>

预期压缩包：`JediOrganismDetectionDataset.zip`，官网标称343 MB。许可CC BY-NC-SA 4.0，仅科研/教育用途，使用时引用DOI 10.17596/0004035。

服务器和本机访问官方数据域名均连接失败；服务器HTTPS重复尝试、HTTP诊断也失败，curl报告错误7。域名可解析到192.244.228.101，但本次未能建立连接。尚不能据此断言服务器故障或官网全球停机。

官方DOI页可正常访问（HTTP200），但只列出同一数据域名下的英日入口，没有备用下载地址。没有获得可验证的ZIP直链，也没有生成伪装成数据包的占位ZIP。

本轮没有下载到图片或标注，没有解压、格式转换、训练或使用GPU。没有修改系统网络、代理、凭据或共享软件环境。

## 后续

官网下载站恢复后，从官方页面提取真实ZIP链接，下载到本目录的临时文件，校验ZIP完整性并记录SHA256后再标记下载完成。若用户能在浏览器成功下载官方ZIP，也可以上传到本目录并进行同样校验。不得把第三方转换集冒充官方原包。

本记录不代表已安排自动重试。

## 后续复查

用户自行尝试英日官方入口也无法打开。本轮再次访问报TLS连接失败；官方DOI元数据页仍可达并继续指向原域名，未发现迁址证据或可核验的备用镜像。搜索引擎缓存仍能列出文件名，但不代表实时下载可用。官网公开咨询邮箱为jedi-admin@jamstec.go.jp；没有代用户发送邮件。此后已随8项数据集任务安排有限自动复查，详见DATASET_DOWNLOAD_HANDOFF_2026-09-20.md；JODD持续标为未下载，不影响其他数据集队列。
