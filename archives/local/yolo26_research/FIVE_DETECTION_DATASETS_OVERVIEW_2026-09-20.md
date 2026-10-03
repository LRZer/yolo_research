# 五个目标检测数据集概况

核对日期：2026-09-20。根据用户截图，仅检索汇报；未下载新数据、转换标注、训练或连接服务器。体积是指定版本的发布说明近似值，不是本轮全包实测；解压及保留原包需要额外空间。

| 数据集/版本 | 领域 | 检测类别 | 图片与划分 | 下载体积口径 |
|---|---|---|---|---|
| PASCAL VOC 2007 | 通用自然场景 | 20 | 9963；train2501/val2510/test4952，trainval5011 | 官方trainval450MB+带标注test430MB，约0.88GB |
| VisDrone2019-DET | 无人机航拍 | 10 | 常用有标注部分8629：train6471/val548/test-dev1610；另test-challenge1580，全DET共10209 | 作者三包1.44+0.07+0.28≈1.79GB；含challenge≈2.07GB，不是视频/跟踪全包 |
| KITTI Object Detection（2D左目彩色） | 自动驾驶道路 | 主要榜单评测Car/Pedestrian/Cyclist三类，原标签不止三种类型 | train7481/test7518，共14999；test标签不公开，训练内验证划分需固定说明 | 左目彩色包约12GB+训练标签5MB；不包含右目、点云、连续帧 |
| SKU-110K，SKU110K_fixed/Ultralytics口径 | 零售货架密集商品 | 1类object，不是11万检测类 | 11743；train8219/val588/test2936 | Ultralytics标示约13.6GB，不与旧论文或其他清理版数量混用 |
| MS COCO 2017检测 | 通用复杂场景 | 80 | train118287/val5000；完整test40670，其中test-dev20288 | train+val约20GB量级，Ultralytics训练集配置标示20.3GB；全test还需约6–7GB |

## 各自特点与使用限制

- VOC2007：原生XML水平框，官方总24640框。体积小、调试快，但只在此集小幅提高AP不足以支撑广泛性能主张。传统VOC07 AP@0.5与COCO式AP50:95不是同一指标。我们此前VOC07+12训练口径不是VOC2007单独5011张trainval。
- VisDrone：小目标、密集、尺度差和遮挡突出；10类为pedestrian/people/bicycle/car/van/truck/tricycle/awning-tricycle/bus/motor。ignored regions、other与遮挡/截断字段需按官方协议处理。test-dev有标注可作保留评测，test-challenge标注不公开；不要把全任务260万框归到静态DET子集。作者仓库提供百度及Google下载。
- KITTI：2D图像检测与3D/BEV任务不能混比。官方Car IoU0.7、Pedestrian/Cyclist0.5，按Easy/Moderate/Hard评价，2019后使用40个recall位置。不要拿通用YOLO mAP50:95直接替代官方榜单。当前官方限制学生项目/小改动向测试榜单提交，应在训练集划分上评估。
- SKU：原标注CSV、水平框；超过170万标注商品，平均约147目标/图，是密集定位而非商品型号分类。图像数不多但目标多，不等于训练轻量；缩图、重复预测、相邻商品边界、每图检测数量上限都需要核查。不能未经机制审计就认为NMS相关旧方法适配端到端部署头。作者限定学术非商业用途。
- COCO：水平框/实例分割等JSON标注，通用性检验价值高但计算量大。使用train2017训练、val2017本地评估与官方test-dev提交是不同报告口径。20GB严格上限下它在边界附近或略超，不能算小数据；原包+解压+缓存所需空间更大。

## 对当前研究的判断（建议，不是实验结论）

如果希望从通用检测转向明确问题：VisDrone适合航拍小目标/尺度与密集问题，SKU适合密集定位/漏检/重复预测问题；KITTI适合道路场景专项验证；VOC保留快速实验作用，COCO用于后续通用性外验。这些都是成熟常用基准，不因领域专用就可断言竞争少或容易发文，选题仍需测量具体失效。

## 主要来源

1. [VOC2007官网与下载大小](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/)
2. [VOC2007训练/验证统计](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html)
3. [VisDrone作者仓库、子任务包及GT政策](https://github.com/VisDrone/VisDrone-Dataset)
4. [Ultralytics VisDrone具体DET划分](https://docs.ultralytics.com/datasets/detect/visdrone/)
5. [KITTI官方2D benchmark、下载及评测规则](https://www.cvlibs.net/datasets/kitti/eval_object.php)
6. [SKU作者仓库与学术用途](https://github.com/eg4000/SKU110K_CVPR19)
7. [SKU当前fixed版划分与单类说明](https://docs.ultralytics.com/datasets/detect/sku-110k/)
8. [SKU CVPR2019论文入口](https://openaccess.thecvf.com/content_CVPR_2019/html/Goldman_Precise_Detection_in_Densely_Packed_Scenes_CVPR_2019_paper.html)
9. [COCO官方检测任务](https://cocodataset.org/dataset/detection-2017.htm)
10. [Ultralytics COCO2017配置、大小与test-dev口径](https://docs.ultralytics.com/datasets/detect/coco/)

补充：本轮尝试只读HEAD核对COCO源包大小时遇到HTTPS证书域名不匹配，没有关闭证书验证或下载包；故体积采用发布说明近似值，不虚称实测字节数。
