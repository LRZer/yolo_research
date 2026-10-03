# CUB-200-2011 官方数据下载与初验

时间：2026-09-24 22:20（北京时间）。范围：仅本机 `E:\datasets\CUB-200-2011`；未下载 Oxford-IIIT Pet，未上传服务器，未启动训练。

## 来源、用途和文件

[Caltech 官方 CUB 页面](https://www.vision.caltech.edu/datasets/cub_200_2011/)列出两份包。第一份“Images and annotations”是当前分类及框中心实验的**必需主包**；第二份“Segmentations”是可选的鸟体掩膜，第一阶段不要求使用，但可用于后续检查目标本体被模糊的比例。官方页面将图像用途限制在非商业研究和教育。

| 文件 | 官方记录 | 大小（字节） | 官方 MD5 | 本地 SHA-256 | 结果 |
|---|---|---:|---|---|---|
| `CUB_200_2011.tgz` | [CaltechDATA 20098](https://data.caltech.edu/records/65de6-vp158) | 1,150,585,339 | `97eceeb196236b17998738112f37df78` | `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081` | MD5 与发布者一致 |
| `segmentations.tgz` | [CaltechDATA 20097](https://data.caltech.edu/records/w9d68-gec53) | 39,272,883 | `4d47ba1228eae64f2fa547c47bc65255` | `dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f` | MD5 与发布者一致 |

原始包保存在 `E:\datasets\CUB-200-2011\original_archives`，解压结果保存在 `E:\datasets\CUB-200-2011\extracted`，两份原始包均保留。下载使用本机既有的 `127.0.0.1:7897` 代理，仅设置在下载进程中；主包按四段续传，合并并校验 MD5 后清理了四份临时分段文件。发布者包和标注未修改。

## 结构、全量解码与配对

- 主包解压后有 **11,788 张 JPG、200 个类别**。`images.txt`、`image_class_labels.txt`、`bounding_boxes.txt`、`train_test_split.txt` 均有相同的 11,788 个唯一图像 ID；磁盘图像与清单一一对应。
- 官方划分为 train **5,994** 张（split=1）、test **5,794** 张（split=0）。后续开发验证应只从官方 train 中再划分，不提前用 test 选参数。
- 分割包解压后有 **11,788 张 PNG**，与 JPG 相对路径一一匹配且尺寸完全一致。全部 11,788 张 JPG 和 11,788 张 PNG 均做完整像素解码，无缺失或解码错误。
- 图片尺寸不是统一的，发现 1,152 种尺寸。JPG 中 11,780 张为 RGB、8 张为灰度 `L`；掩膜中 11,786 张为灰度 `L`、2 张为 `LA`。后续数据加载器应显式规范通道，并测试特殊模式，不能假定所有源文件都是 RGB/L。
- 发布者原始框有 **30 张**超出图片边界（最大超出 32 像素；另有 26 像素一例），例如 ID 1218 的框 `(92,51,440,303)` 对应 `500×410` 图像、ID 5007 的框 `(34,33,438,326)` 对应 `500×333` 图像。原始标注保持不动；若实验需要裁边界，必须只在派生的数据加载/坐标变换中明确处理，并记录这 30 张的策略。两张最大越界案例的掩膜模式也为 `LA`，值得在后续视觉审计中单独复查。

结论：**两份官方包已下载、完整解压，校验和与图像/掩膜配对通过；不能称原始框完全无异常。** 当前仅完成数据准备和静态验收，未开展任何分类或模糊实验。审计脚本为 `outputs/foveated_classification/audit_cub_official.py`，下载脚本为 `outputs/foveated_classification/download_cub_official.py`。
