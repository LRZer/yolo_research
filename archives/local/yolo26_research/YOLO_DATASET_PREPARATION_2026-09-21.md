# YOLO数据集训练前准备（2026-09-21）

## 最终状态

- 服务器派生目录：`/home/ubuntu/rz/object_detection_dataset_test/processed_yolo`
- 状态：`ready_for_yolo_training_test_used_as_val`
- 共11个YOLO派生版本、22个train/test分区；当前Ultralytics 8.4.146全量扫描通过，错误0、损坏图0、最终重复标签警告0。
- 每个`dataset.yaml`的`val`和`test`都指向`images/test`，因此训练每轮使用test评测。
- 有原val的数据已合并进train；未新建val；按用户决定不处理相邻视频帧数据泄漏。
- 原图使用绝对软链接，原始图像和标签未修改；本轮未启动训练、未使用GPU。

## 规模与派生处理

| 派生数据集 | train图/框 | test图/框 | 类别 | 主要处理 |
|---|---:|---:|---:|---|
| AP-10K | 8018/10394 | 1997/2634 | 54 | 使用官方split1，train+val合并 |
| DUO | 6671/63997 | 1111/10517 | 4 | 裁剪953个越界框，跳过1个裁剪后无正面积框 |
| ExDark | 4800/16593 | 2563/7117 | 12 | 官方train+val合并；裁剪50个越界框；处理1处扩展名大小写映射 |
| HIT-UAV | 2319/19971 | 579/4780 | 4 | 使用水平热红外版本；148个DontCare框排除 |
| HRSID-JPG | 3642/11047 | 1962/5922 | 1 | 使用JPG版train/test，不使用不一致的合并JSON或PNG标签 |
| LLVIP-infrared | 12025/34135 | 3463/8302 | 1 | 红外版本；VOC闭区间坐标转换保留5个一像素宽框 |
| LLVIP-visible | 12025/34135 | 3463/8302 | 1 | 可见光版本；与红外共享标注但独立训练 |
| ORCA-class-level-seen | 9290/27320 | 2744/8954 | 24 | 使用闭集class-level seen协议；移除357个派生重复框 |
| SeaClear | 6888/25349 | 1722/6200 | 40 | 固定SHA256排序80/20自定义划分；移除6个派生重复框 |
| VisDrone2019-DET | 7019/381959 | 1610/75102 | 10 | train+val合并；跳过14200条ignore/others、1个非正面积有效类框；去重4框 |
| ZeroWaste-f | 3574/21689 | 929/5055 | 4 | train+val合并；裁剪138个越界框；test去重22框 |

派生YOLO标签共显式移除389个完全重复行：ORCA 357、SeaClear 6、VisDrone 4、ZeroWaste-f 22。原始标注全部保留不变。

## 验收

1. 图片软链接与标签逐一对应，无断链。
2. 所有标签均为5列YOLO检测格式，类别编号合法，归一化坐标在`[0,1]`，宽高为正。
3. 当前Ultralytics对22个分区全量建立缓存并读取通过。
4. 每个分区完成真实数据管线取样，均生成`3×640×640`图像张量和检测标签。
5. 最终复验未再出现自动去重、损坏图片或解析错误。

## 服务器证据与入口

- `processed_yolo/FINAL_READINESS_REPORT.md`：最终人类可读报告。
- `processed_yolo/FINAL_READINESS.json`：最终机器可读状态。
- `processed_yolo/ULTRALYTICS_VALIDATION.json`：Ultralytics全量扫描结果。
- `processed_yolo/PREPARATION_REPORT.json`：完整转换规模及规则。
- `processed_yolo/TRAINING_COMMANDS.md`：限定后四张GPU的训练命令模板；尚未执行。
- 每个数据集目录含`dataset.yaml`、`conversion_report.json`、`source_map.tsv`和README。

最终文件SHA256：

- `FINAL_READINESS.json`: `fdcb3e995b4320a7d488b2ec17fbd14b09406cd8dba7c10c038d5b02ba6999b4`
- `FINAL_READINESS_REPORT.md`: `3247d9b245eea435851d8c52d266babd9e50cf7ce07df5dfd284b42de292bd9c`
- `PREPARATION_REPORT.json`: `128e51ef7af361454680e234b15a74e2836163e027a2a71f52c001ff6422209c`
- `ULTRALYTICS_VALIDATION.json`: `a72c704b9cc1854f206f3f2449075f70006644b4bcea5adfd7ca7b152368c375`

JODD仍无实际数据包，不在本次转换范围内。
