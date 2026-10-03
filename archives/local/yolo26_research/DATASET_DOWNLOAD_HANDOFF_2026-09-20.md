# 数据集下载与验收接续记录

## 2026-09-20 21:37 用户要求删除LLVIP半包及JODD网盘检索

- 用户明确要求“LLVIP直接删除，等我自己下”。检查archives仅有protocol文档目录和LLVIP.zip.part，确认无相关下载进程、目标为预期普通文件后，删除 `E:/datasets/LLVIP/archives/LLVIP.zip.part`，2383264080 bytes（约2.38GB），复核已不存在。非回收站删除，原断点不能继续；需用户重新下载完整包。保留体积很小的作者说明和历史校验记录，不碰其它数据或服务器。
- **覆盖所有历史“保留2.38GB断点”记录：当前LLVIP已无任何数据包或part。** 自动复查维持暂停，用户自行网盘下载，不启动替代下载。
- 按用户要求检索JODD全名、压缩包名、DOI及百度/夸克/HuggingFace镜像关键词，未找到可核实的百度或夸克分享。官方DOI https://www.jamstec.go.jp/datadoi/doi/10.17596/0004035.html 和论文数据说明仍指向JAMSTEC官方343MB ZIP，不能由搜索无结果断言网盘绝对不存在。没有新增下载或联系作者。
- 用户询问YOLO转换含义，本轮仅解释：原JSON/XML/其它文本框标注转换为每图TXT的零起始类别+归一化中心xywh，并生成数据路径/类别配置，保留官方划分与原始数据；不是缩图/重新标注或自动改分辨率。不授权也未执行数据格式转换或训练。说明核对Ultralytics官方文档 https://docs.ultralytics.com/datasets/detect/ 。

## 用户最新决定：百度/夸克网盘由用户手动下载

- 用户明确表示：有百度网盘或夸克网盘的交由用户自行下载。后续提供核实的作者链接与提取码，不自动操作网盘账号、客户端或启动同一包的替代下载。
- 当前确认LLVIP作者百度主数据链接 https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg ，提取码14lc；不混淆额外未配准原视频或训练权重的网盘链接。建议用户把完整原包保留原名放入E:/datasets/LLVIP/archives，完成后告知再核验包版本、完整性与配对。
- LLVIP已有Google断点文件保留，不重启续传；自动复查维持PAUSED。百度包若与Google包格式/字节数不同，不直接判损坏或拼接，按实际官方版本重新核对，尤其确认2023-02-21更新标签。
- 已下载的DUO/AP-10K/HIT-UAV/SeaClear/ExDark/ORCA无需重下。JODD尚无已核实的作者百度/夸克备用链接；不能推荐来源不明镜像冒充官方。

## 2026-09-20 20:48 外部配额阻塞，自动复查已暂停（最高优先级）

- LLVIP Range进程28896在2383264080 bytes停止，连续5次请求8MiB得到HTTP200而非206，拒绝追加；原part完好保留（59.5%，总4003858172 bytes）。小32字节探测仍206，但相同断点的实际8MiB请求返回HTTP200 text/html，不能将小探测成功当可续传证据。
- 本轮有限重试PID28516（日志LLVIP_range_retry_v2.log）同样停止；已读实际错误网页，明确Google Drive **Quota exceeded / Too many users have viewed or downloaded this file recently**，页面提示可能需最多24小时。不是磁盘不足或本地文件损坏。不得用不断缩小片段、换IP/身份等规避限额；不继续自动重复请求。
- 作者冻结版download_dataset.md提供备用百度网盘 https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg ，提取码14lc。本轮web打开失败，浏览器只读入口尝试30秒超时，未进入提取或登录，无法验证能否匿名下载；**不声称已经证实需要账号**，交由用户选择手动尝试或等待配额恢复。未打开额外原视频包链接，不用不明镜像。
- 20:48检查无dataset_run_local/dataset_retry_local/dataset_verify_local的Python进程在运行；未杀任何无关进程。原数据、LLVIP part与所有日志全部保留，无服务器连接、中转、上传或全局网络修改。
- 按用户既有外部阻塞处理指令，调用产品工具将自动复查IDautomation更新为**PAUSED**，保留原name/prompt/rrule/targetThreadId；工具确认成功。等待用户指示，不擅自自动恢复或另建监控。没有声称8套全成功。
- 最终暂存状态：5套原格式验收通过（DUO/AP-10K/HIT-UAV/SeaClear/ExDark）；ORCA全文件/图片/框结构核验完成，保留读取语义警告及REVIEW.md；LLVIP 2.383/4.004GB受官方Drive配额阻塞；JODD未下载且官方源TLS持续失败。全部在E:/datasets。
- 用户若手动获得官方LLVIP完整ZIP，可放E:/datasets/LLVIP/archives/LLVIP.zip（当前不存在同名完整包；不要把part当完整包）。后续须重新核对4003858172 bytes、ZIPCRC、全量解码、15488可见光/红外对及新版XML配对后才能完成验收；保留part直至完整包验证成功。若用户选择等待，由其明确指示后再恢复有限检查。

## 2026-09-20 20:32例行复查（无新异常）

- LLVIP已保存2215491920/4003858172 bytes（约55.3%），较20:18增长约285MB；专用PID28896命令行核对一致，stderr为空，checks仍running，分块续传正常。无需重启或并发重下。
- DUO/AP-10K/HIT-UAV/SeaClear/ExDark五套验收已通过；ORCA文件完整、全量解码与框结构核验通过，但保留原始数据使用警告，先前已向用户解释不是下载损坏，不能在未确认全部任务语义时改成无条件通过。
- JODD上次20:15有限重试失败，仅过去约17分钟，本轮不重复请求。未连接服务器，无GPU/训练或全局配置操作。没有实质新结果，不另发进度通知；保持自动复查，待LLVIP整包通过再启动全量验收。

## 2026-09-20 20:16 ExDark验证器修正（当前入口）

- **20:18最终结果：ExDark v3已完成并通过**（verified_usable_original_format，errors/warnings为空），PID28712结束。7363数据图+1说明图全部解码，7363标签/23710框/12类、官方划分3000/1800/2563、唯一大小写映射已核对。通过整套验收更新为**5/8**：DUO/AP-10K/HIT-UAV/SeaClear/ExDark。ORCA仍needs_review；LLVIP约1930/4004MB续传中；JODD阻塞。不再重验已通过ExDark，保留READING_NOTES供Linux迁移读取。
- **20:17后续修正优先**：v2已完成，说明图问题解除后仍有1个精确名称不匹配：图片2015_00391.JPG vs标签2015_00391.jpg.txt。7363全集合casefold后唯一无碰撞，其他名称精确对应。验证器记录`exdark_label_filename_case_map`并严格检查唯一匹配，不改原文件。Linux读取/转换必须使用该映射，说明落盘 `E:/datasets/ExDark/READING_NOTES.md`。已启动v3全量重验**PID28712**，日志 `_management/ExDark_validation_v3.log`及`.err.log`；下轮读新结果，不再跟踪已结束PID22264。这个是原文件命名差异，不再概括为“所有误报都来自说明图”。
- ExDark第一遍全量解码7364图片均成功，7363标注文本数和官方train3000/val1800/test2563计数正确，但报图像集合不一致。精确集合审计发现唯一额外文件是 `raw/groundtruth.tar.gz/ExDark_Annno/annotations.png`，官方README明确将其作为bbox说明插图引用，不是数据样本；缺图0。
- 已修正验证器：该**确切路径**仍全量解码并列入auxiliary_images，但不计入7363数据样本或划分集合。不是泛泛排除所有无标注图片，不删除文件，不修改原始标签。添加exdark_dataset_images、各类bbox计数和合法类检查。
- 20:14启动修正版完整重验**PID22264**，日志 `_management/ExDark_validation_v2.log`及`.err.log`；原PID6868已完成，旧VALIDATION仍needs_review直到新进程结束。下轮读新结果finished及日志，不重启。解压结果用已有SHA256完成标记安全复用。
- 独立只读文本审计12类共23710框：People7460、Car2927、Chair2377、Cup1656、Bottle1593、Table1483、Boat1389、Bicycle1120、Motorbike1072、Dog1018、Cat909、Bus706。不存在未知类名。此项不替代待完成的修正版验收。
- LLVIP独立Range续传PID28896继续，20:15约1897/4004MB，stderr无错误；其余6套包下载完整。通过整套验收仍为DUO/AP-10K/HIT-UAV/SeaClear四套，ORCA的文件/结构通过但两项源数据警告保留。
- 距上次官方JODD访问超过一小时，本轮仅再试一次本机既有7897代理连接官方落地页，curl35 TLS握手失败（不是有效网页响应；HTTP200仅是代理CONNECT）。不密集重试、不使用不明镜像；无服务器操作。下载/验证仍有进展，自动复查继续。

## 2026-09-20 20:03全量核验与语义审阅（当前入口）

- 本轮末补充：ExDark最后motorbike已download_verified，16/16清单项齐全，20:02已启动ExDark全量验收PID6868（日志 `_management/ExDark_validation.log`及`.err.log`）。下轮读该日志/VALIDATION，不重复启动；下面“仍下载”是本轮稍早快照。
- **SeaClear通过整套验收**，VALIDATION为verified_usable_original_format，errors/warnings空；8610图全部解码、31555框/40类引用与尺寸匹配通过。PID3432已完成，不重跑。
- **ORCA全量核验完成但保留needs_review**，PID27800已完成。14646图片全部解码，主JSON+10份split引用/尺寸/框检查错误0，仅前述未引用1图、README未解释-1两条警告。不是文件坏或少下载。
- 查阅官方项目页与论文HTML§3.1：https://orca.hkustvgd.com/ ，https://arxiv.org/html/2512.21150v1#S3.SS1 。作者只对面积>1024的框生成描述；实测7465条label=-1全部bbox面积≤1024，7464空caption、1非空。这支持-1与小框未生成描述相关的推断，尚不是作者明确枚举定义。**label描述正负不代表bbox正负，不得因label=1/-1删除检测框。** 原数据不改。审阅全文在 `E:/datasets/ORCA/REVIEW.md`，含精确标签分布、例外和来源。
- 作者项目页本身出现14647/14645两个图像数，冻结包实际14646/引用14645；不能改本地数据迎合网页。后续检测读取可按官方标注索引，不把未引用图当背景；caption/grounding协议另定。目前不擅自将needs_review改成无条件完成。
- 原下载PID24976仍处理ExDark最后motorbike.tar.gz，20:01为164044837/177428605 bytes，其他15/16均完成；完整后运行本地ExDark验证器并核对官方split。LLVIP独立Range PID28896约1670/4004MB继续增长，stderr无异常。SeaClear/ORCA的旧failed日志不能代表现态。
- 已通过4/8：DUO、AP-10K、HIT-UAV、SeaClear；ORCA文件核验已完成但语义待审；ExDark/LLVIP下载中；JODD源阻塞。未连接服务器，本轮JODD不密集重试，自动复查继续。

## 2026-09-20 19:48修复与验收启动（当前入口）

- SeaClear全包1711829309 bytes，发布者MD5一致，本地SHA256 `2a053f748d6bdc5df8d776e87b72832f2908316f0f419f6a8d562df6df086c13`。原Windows bsdtar对零字节目录Bistrina/Bluerobotics HD尝试解压，报Can't decompress an entry marked as a directory；这不是MD5不匹配或下载缺失。
- 已修正RAR测试：rarfile只读列出8612个普通成员，任务临时目录保存成员清单，用bsdtar -T显式选择全部文件做一次性流式解压到空设备（8610 JPG+dataset.json+TXT），全部通过。未忽略文件解码错误，未修改原RAR，未新增系统工具。标准输入-T -在此Windows版本不支持，所以实际使用任务范围临时清单，结束自动清理。
- SeaClear独立retry复用现有包完成校验，checks已为download_verified；原runner日志/最终汇总的failed属于修复前历史。原始RAR为non-solid。验证器RAR解压改成安全检查所有成员路径/链接/容量且确认目标空后，用相同文件清单一次性解压，并核对每个输出字节数，避免逐成员重扫1.7GB。19:46启动全量验收**PID3432**；日志 `_management/SeaClear_validation.log`及`.err.log`。下次检查完成标记/VALIDATION；若失败先审阅，不删除部分raw或盲重启。
- SeaClear主JSON实际8610图、31555标注、40类；包内8610 JPG basename全唯一，适用当前匹配器。以上只是包/元数据核验，还没有宣布逐图验收完成。
- ORCA清单14/14已download_verified。最后一份intra_class_val_unseen.json已补齐（与原worker有文件锁协调；不重复改原文件）。19:47启动全量验收**PID27800**，日志 `_management/ORCA_validation.log`及`.err.log`。需审阅完整输出，预计之前记录的额外图片和未定义caption label会留下needs_review；不得为通过检查删除原标注。
- 已通过整套验收仍为3/8：DUO、AP-10K、HIT-UAV。ExDark剩余motorbike/people/table持续下载，LLVIP已约1502/4004MB且增长。原24976及LLVIP28896保持，不连接服务器、不恢复中转。JODD本轮未复试；持续不可达信息未变。
- 下载/验证代码编译检查通过。下轮优先读SeaClear/ORCA验收结果，ExDark全16文件通过后启动全量验收及官方split核对。完成之后再统一处理ORCA语义审阅和JODD阻塞，自动复查继续。

## 2026-09-20 19:34复查（当前入口）

- **HIT-UAV全量验收通过**：VALIDATION状态verified_usable_original_format，errors/warnings均空。11600个图片文件全部解码（2898源图×4格式副本+8示意图）；normal/rotate JSON及各2898份XML图像、尺寸、对象有限/正尺寸与划分检查通过。划分train2029/val290/test579；每套24899标注，其中普通目标24751（Person12312、Car7311、Bicycle4980、OtherVehicle148），DontCare148另计，不把它当普通类别。旋转框检查范围详见VALIDATION注释，未承诺未实施的旋转角点越界检查。已完成DUO/AP-10K/HIT-UAV三套原始格式验收，未做YOLO转换。
- ORCA images.zip已完整下载并通过发布者SHA256及ZIP全CRC。**14647是压缩包条目数，不是图片数**：1目录+14646图片；主data.json引用14645图片，缺图0，多出 `images/pygoscelis_adeliae_Pygoscelis adeliae36.jpg`，保留原文件。不能按14647图片做验收期望值。
- ORCA label=-1共7465行，其中7464行caption为空、1行非空；README未解释-1。仅记录观察，不擅自断定其语义、不删标注。验证器新增主JSON与实际图片集合审计、caption标签分布/未知值审计；未引用图片和未知语义保留needs_review警告，待全包审阅。label1是负描述，不当新的正检测目标。
- 19:31载荷快照：ExDark872/1518MB（11/16文件）、LLVIP1318/4004MB、ORCA1072/1098MB（11/14文件）、SeaClear1566/1712MB；均无当前failed。ORCA仍缺intra_class三份划分文件，因此**未启动整套验收**。待14/14全部download_verified后运行本地验证器ORCA，不因图片zip完成而跳过划分文件。SeaClear、ExDark同理。
- 验证脚本已编译通过；原下载PID24976、LLVIP独立Range PID28896继续，HIT验收PID26744已完成，不再重复启动。尚未进行本轮JODD重试，官方源阻塞无新信息；无需密集重试。未连接服务器或改全局环境。

## 2026-09-20 19:20验收更新（优先入口）

- **DUO全量验收通过**：修正版VALIDATION.json为verified_usable_original_format，errors/warnings均空；7782张全量解码、逐split精确图像/尺寸/框匹配通过。4类、74515框、train6671/test1111；不把重复basename认定为泄漏。
- **AP-10K全量验收通过**：10015张全量解码，9份官方split JSON全部图像/尺寸/框匹配通过；54类，每个split train7023/val995/test1997。额外只读schema审计9文件图片ID/框ID重复均0，缺bbox均0，三套train与test文件名交集均0。是原始格式可用，不代表已转为YOLO，也不证明姿态数据的检测标注穷尽性。
- **HIT-UAV下载完成**：814751578 bytes，发布者MD5 bb6c5a33659b7507eae83458729c6109一致，ZIP全CRC通过。SHA256 5fed15d2ff895afb16c0ca328ea19353f416d7543e6f39ba54e44ca26c538555。19:17开始全量验收PID26744，日志 `_management/HIT-UAV_validation.log`及`.err.log`。不要重复启动；下次读取VALIDATION审阅。
- HIT真实包结构：normal_json/normal_xml/rotate_json/rotate_xml各2898张副本，另README示意图。水平JSON键名是annotation（单数）和filename，不是标准COCO键；旋转JSON为5维中心/宽高/角度。已按真实格式适配验证器：全部副本图片解码、JSON精确variant/split目录匹配、XML对象与图片匹配、类别及有限/正尺寸校验、划分集合一致性检查；旋转框不错误套用水平xywh越界公式。DontCare保留为忽略标记，不当普通第5有效类。原数据未修改。
- 通用COCO核验新增重复image/category/annotation ID及缺bbox检查；AP-10K额外只读审计已覆盖新增项。若后续数据出现真实问题保留needs_review，不改标注迎合检查。
- 19:18下载快照：ExDark634/1518MB（7/16文件verified）、LLVIP1058/4004MB、ORCA789/1098MB（5/14文件verified）、SeaClear1398/1712MB；当前checks无failed，原runner历史LLVIP failed日志已由独立Range续传修复。原24976和LLVIP28896继续运行，未碰服务器。
- ORCA data.json已读到**14645 images、42217 annotation行、670 categories**，不是先前待核的14647。label分布为2:12873、1:12431、0:9448、-1:7465；README只解释0/1/2，-1含义尚未核实。42217行不能直接作为独立正样本目标数，负描述和重复框需区别处理。此为元数据初审，等待图片zip及全split完成再做全量核对，不擅自删除-1或label1原标注。
- JODD仍未下载，源阻塞不变；本轮不重复无效访问。已完成2/8，其余继续；自动复查保持启用。原格式验收与后续研究质量审计/YOLO转换分开报告。

## 2026-09-20 19:05实质进展（当前入口）

- 仍仅操作本机E:/datasets；没有服务器连接、上传或SSH中转。原四worker下载PID24976继续运行。逐文件checks为准：原runner最终汇总可能保留已被单独重试修复的LLVIP失败状态。
- DUO完整包2940459407 bytes，ZIP全部CRC通过，SHA256 `5de58f03a09c152607a99722f453100a0de7c37fb1a6bc3b767a2e4293ee1992`；发布者未提供可对照的哈希，记录的SHA256仅是本地指纹。已安全解压并完成第一遍7782张图片全解码，零解码错误。
- DUO实际train6671/test1111，共74515框（63998/10517），4类海参/海胆/扇贝/海星。独立schema审计 `DUO/checks/annotation_schema_audit.json`：各split图片/框/类别ID无重复、无缺bbox。两split有1111个同名basename，但目录不同，不能据此认定图片重复或数据泄漏。
- 第一遍DUO验收因验证器只按basename去重而产生6671而非7782的假警告。已修正为按images/train或images/test准确匹配，未改任何原数据。修正版全量重验仍运行，exec会话82036；最近输出decoded2000/7782。下次读取DUO/VALIDATION.json的finished和status确认新结果，不把旧needs_review当新失败，也不提前宣告整套通过。
- AP-10K主包3403294327 bytes，ZIP全部CRC通过，SHA256 `420980abb135d6f66bcc8e29f289a46081214016192ae197ad24bc1525c8e62c`（本地指纹，非发布者对照）。仅标注主包，无额外未标注包。19:02启动全量验收PID20808，日志 `_management/AP-10K_validation.log` 和 `.err.log`；仍在解压/验证，不重复启动。
- LLVIP在613267792 bytes发生curl33，拒绝开放式断点续传。实测相同官方URL的明确闭区间Range正常206且Content-Range/总大小正确。保留prefix，停止本任务旧retry PID13904后改用每块8MiB的严格Range续传：必须206、范围/总大小/块长度全一致才追加并fsync；每块最多5次、总6小时预算。未改变系统代理，使用既有进程级7897。当前PID28896，入口 `dataset_retry_local.py LLVIP --resume-interrupted`，日志 `_management/LLVIP_range_retry.log` 与`.err.log`，runner为`LLVIP.retry_runner.json`。已实测从613MB增长至764MB，恢复有效。禁止再并发启动该文件；文件锁保护仍在。
- 下载器代码已加入curl33有限重试及可选range_chunked，但原PID24976加载的是旧版，其后其它包若curl33失败需使用新版独立重试入口，不必重启健康worker。脚本均已语法编译检查。
- 19:01快照：DUO2940MB/AP3403MB下载完成；HIT794/815MB、Sea1168/1712MB、ExDark197/1518MB；ORCA主包尚排队；LLVIP上述持续续传。尚未把任何整套判为完成，等待全量检查。JODD保持官方源访问阻塞，不密集复试。
- 下一步：先审阅DUO修正版及AP-10K验收结果；HIT-UAV全包download_verified后再验收。若HIT使用YOLO而非COCO或包含normal/rotated，应按实际格式修正验证器，保留原标签；不能为消除警告改数据。不要只因下载完成就宣布可训练。所有7项完成前自动复查保持启用；JODD持续不可达时按用户指示汇报阻塞并停用等待。

## 2026-09-20 18:40例行状态

本机PID24976正常，stderr为空，failed=0；载荷约6604MB。DUO2858/2940MB（约97%），接近完成但尚未完成ZIP验证，不提前解压或宣布可用；AP-10K2181MB、HIT-UAV630MB、SeaClear812MB，均持续增长。下次优先检查DUO包状态，只有download_verified后才启动 `python dataset_verify_local.py DUO` 并审阅真实标注。当前其他已验证文件无变化；不干扰正常下载，不连接服务器，不重复18:28已失败的JODD请求。

## 2026-09-20 18:28状态

本机PID24976正常，stderr为空，清单failed=0。载荷约5512MB：AP-10K1946MB（57%）、DUO2406MB（82%）、HIT-UAV570MB（70%）、SeaClear466MB（27%）、ExDark123.5MB（8%）。LLVIP主包/ORCA影像仍排队，尚无完整数据集可验收。间隔超过一小时后对JODD官网仅作一次有限复查，仍TLS连接失败；不加密集重试，不更改数据源。用户询问报告，按此最新实测汇报，不承诺完成时间。无服务器操作。

## 2026-09-20 18:17例行状态（无新阻塞）

本机PID24976正常，stderr为空，清单failed=0。下载载荷约4252MB：AP-10K1616MB、DUO1797MB、HIT-UAV308MB、SeaClear407MB、ExDark123.5MB。活动文件继续增长，既有已验证单包/文档状态不变，尚无完整集可验收。保留四并发下载，未操作服务器。例行状态不另通知用户。

## 2026-09-20 18:01例行状态（无新阻塞）

本机PID24976继续运行，stderr为空，清单failed=0。下载载荷约2549MB：AP-10K1077MB、DUO717MB、HIT-UAV277MB、SeaClear355MB、ExDark123.5MB；其余主包仍排队。全部活动下载相较17:49均增长，未观察到需要干预的停滞。尚无新的完整包或数据集可验收；不重启任务、不连接服务器、不重复JODD失败请求。本轮无实质新结果，不通知用户。

## 2026-09-20 17:49例行状态（无新阻塞）

本机PID24976正常，错误日志为空，各清单项failed=0。下载载荷约1691MB：AP-10K535MB、DUO610MB、HIT-UAV196MB、SeaClear227MB、ExDark123.5MB。ExDark首包与4份协议文档/ORCA小文件状态不变，尚无完整数据集可做整套验收。LLVIP主包和ORCA影像仍在四worker队列等待，非失联或失败。保留当前下载，不重启、不连接服务器；未重复JODD无效尝试。此次仅例行进度，无需用户操作。

## 2026-09-20 17:38本机复查

- 本机PID24976仍正常运行，无失败文件状态/无stderr异常。各包字节数持续增长；约961MB下载载荷（不含本轮预检解压），其余仍排队；没有任何完整数据集通过验收。
- ExDark的bicycle.tar.gz首个图片包完成：123330805 bytes，发布者MD5一致，SHA256=07ee437623679a43aca9c1e3de0adf4f28d8f7cb54f73126646d98bd9fc3bd52，tar1306成员含辅助资源。
- 已用本机验证器安全解压该包，652张有效图像全部PIL verify/load通过，错误0。预检记录E:/datasets/ExDark/checks/bicycle.image_preflight.json。此项仅是单包预检，不是ExDark整套验收；解压完成标记允许后续整套核验复用而不覆盖原文件。
- 补齐本机ExDark协议文档及imageclasslist、LLVIP新版下载说明。作者ExDark划分清单实际为train3000/val1800/test2563，共7363；已在验证器加入官方清单与图片集合一致性检查，不自行重划。
- 本轮没有连接服务器，没有恢复代理转发或上传文件。JODD状态未变，不重复无效尝试。

## 最新决定：2026-09-20 17:22起仅本机下载（覆盖下面所有远端下载安排）

用户要求：服务器进度少则改在电脑下载，随后明确要求删除服务器本次所有下载内容，之后由用户自行把电脑数据转到服务器。不得自动重新启动服务器下载或上传数据。

- 本机目标 `E:\datasets`。C盘仅剩15.35GB，不够稳妥容纳全部压缩包与解压文件；E盘136.34GB可用，目录原本不存在，本轮创建。
- 已停止远端v3下载并检查无dataset_acquire/curl数据进程，验证所有目标均在 `/home/ubuntu/rz/datasets` 且无符号链接后，删除DUO、SeaClear、ORCA、HIT-UAV、LLVIP、AP-10K、ExDark的archives目录。合计约918MiB，含part、已完成小型文档。JODD没有数据包。保留根目录、各集空目录、脚本/元数据/日志约1.5MiB；旧checks的download_verified属于删除前历史，不再代表文件存在。
- 删除为服务器直接删除，不在回收站；可重新下载。未碰 `/home/ubuntu/rz/yolo_study/datasets` 的VOC等旧数据。
- 原SSH中转已终止，远端127.0.0.1:17997无监听；不恢复。
- 本机后台PID24976，启动器 `work/dataset_run_local.py`，通过Start-Process -WindowStyle Hidden启动，4下载worker。进程级HTTPS_PROXY/HTTP_PROXY使用已存在本机127.0.0.1:7897，不修改系统代理。
- 本机清单 `E:\datasets\_management\download_manifest.json`；runner状态local_runner.json；日志local_download_20260920.log与.err.log；各集checks及archives布局保持不变。新下载重新开始，服务器part没有转到本机。
- 本机下载已实测开始：约一分钟DUO32MiB、SeaClear8MiB，DUO当时平均约0.65MiB/s、SeaClear最近5秒约1.4MiB/s，但HIT-UAV/ExDark约60KiB/s，不能保证所有来源同样提速。
- Python为 `C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`。跨平台下载器已加Windows文件锁、目标路径约束、HTTP1.1有限续传、ZIPCRC/JSON检查；本机RAR用Windows自带bsdtar3.8.8流式全解码到空设备校验，不落盘提取。
- 本机仅在 `E:\datasets\_management\vendor` 安装rarfile4.2，无全局pip环境修改。验证器Windows rarfile.BSDTAR_TOOL='tar'，安全解压和全量图片/标注验证仍待完整数据实测。
- 完整集下载后，在本地work目录运行 `python dataset_verify_local.py DATASET`，它明确将DATASET_ROOT设为E:/datasets。不要在本机直接用默认Linux路径启动dataset_verify.py。RAR若逐成员解码固实包太慢，先检查所有成员路径及链接属性后再设计一次性安全提取，不能跳过完整性验证。
- 自动复查IDautomation已更新为仅本机，保持每10分钟。后续只看本段入口。JODD仍阻塞，待其余可达项完成后汇总，必要时报告阻塞而非无限重试。

## 用户授权与完成条件

用户要求在gpu8-team的 `/home/ubuntu/rz/datasets` 统一存放数据，下载JODD，以及DUO、ORCA、AP-10K、LLVIP、HIT-UAV、ExDark、SeaClear；确认完整可用后结束。下载完成不等于验收完成，不启动训练、不改变研究协议或既有VOC路径。服务器共享，禁止系统/全局网络改动或影响他人进程。

## 2026-09-20 17:15复查更新（优先于下方历史状态）

- 总落盘约835 MiB，尚无完整数据集通过验收，只有协议文档/小型元数据文件download_verified。
- v2出现ExDark bicycle/boat和ORCA data.json的curl92（HTTP/2 PROTOCOL_ERROR），HIT-UAV出现curl18（传输提前结束）。保留所有.part，未发生发布者校验和不匹配。
- 已仅停止本任务v2并确认无旧下载进程后，切换v3：HTTP/1.1，每次连接最多300秒，遇到连接/提前结束/超时/HTTP2等网络错误，从现有part重新续传，最多48次有限重试。按文件加flock锁防止并发重复写。同名ZIP必须通过真实ZIP格式及CRC，JSON必须能解析，不能只凭扩展名当完成。
- 当前tmux：`rz_dataset_download_v3_20260920`；当前日志：`acquisition_v3_20260920.log`。v2已停止。专用中转保持不变。
- Google包通过真实bytes=0-0响应Content-Range冻结字节数：DUO2940459407、AP-10K3403294327、LLVIP4003858172。清单已更新，必须核对总大小。
- 已实际下载并验证ExDark固定commit的 `protocol/imageclasslist.txt`、`protocol/README.md`，以及LLVIP固定commit的 `protocol/README.md`、`protocol/download_dataset.md`；它们在各集archives/protocol下。仍需在验收逻辑中使用ExDark划分清单。
- 更新验证器对tar根目录 `./` 的处理，避免空路径元组造成假阳性；仍未以完整包实测。
- 17:08:41–17:11:42服务器内核记录14次明确发送SIGBUS的硬件内存错误，涉及MainThread、conda、GUI/运行时等进程；未匹配本任务当时下载父进程32884及其curl子PID。不得断言硬件健康，也不能把HTTP中断归因为MCE。延续用户/所有者此前允许继续的授权，仅NUMA1下载并保留严格完整性核验，不动共享系统和别人进程。此事件已向用户简要通知。
- 本次不重复无效JODD联网尝试，已有多路径失败且用户也打不开；JODD保持阻塞，不影响其余队列。全部可达项完成前保持自动复查。

## 首次启动时的历史状态

- 8个独立目录已建立，开始时磁盘可用约443 GiB。
- 当前7个可达数据集约14.6 GB下载材料，来自作者仓库指向的Google Drive、Zenodo、4TU、UM Dataverse及作者Hugging Face。Google Drive包DUO真实总长2940459407 bytes（此前2.7 GB为粗略口径）。
- 当前总落盘约307 MiB，全部仍未完成验收。ORCA仅小型.gitattributes文件完成，不能说ORCA完成。
- 速度很不稳定：Google Drive包曾约0.5 MiB/s，SeaClear短时约1 MiB/s；最后观察平均DUO约137 KiB/s、AP-10K约104 KiB/s、LLVIP约194 KiB/s，最近数秒约30–70 KiB/s。早期1–3小时估计不再可靠，当前单包日志估计5–9小时，不能承诺结束时间。
- JODD官网www.godac.jamstec.go.jp本机、服务器、代理路径均失败（连接失败/TLS中断/HTTP502）。DOI元数据页可达，仅指向同域名，无ZIP直链。尚未下载任何JODD图片，保持阻塞。没有使用第三方冒充官方。

## 正在运行的进程与网络

- 当前下载tmux会话：`rz_dataset_download_v3_20260920`。
- 日志：`/home/ubuntu/rz/datasets/acquisition_v3_20260920.log`。
- 旧direct和proxy会话已仅对本任务发送Ctrl-C停止，不应再启用。部分文件留存并续传。
- 临时SSH反向转发：本机 `ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -R 127.0.0.1:17997:127.0.0.1:7897 gpu8-team`，本轮exec会话ID7916。
- 本机7897是从Windows现有默认代理配置核实的地址，未修改其配置。远端端口仅绑定127.0.0.1，只有任务下载进程设置 `HTTPS_PROXY=http://127.0.0.1:17997`。任务结束须关闭仅本任务转发。如失效先确认现存专用进程，不重复抢占端口，不杀无关ssh进程。
- 下载python绑定NUMA1，OMP/OPENBLAS线程1，不导入GPU计算框架、不使用GPU。6个下载worker，每文件curl限制12 MiB/s（实际远低于上限），支持.part续传；v3每次max-time300秒、最多48次有限网络重试。最终失败需检查后从part重新续传，不直接删除。

## 文件与操作入口

远端脚本：

- `/home/ubuntu/rz/datasets/dataset_acquire.py`
- `/home/ubuntu/rz/datasets/dataset_verify.py`

本地对应 `work/dataset_acquire.py`、`work/dataset_verify.py`。本地编辑必须apply_patch，再scp到明确远端路径；不修改共享环境。

下载清单：`_management/download_manifest.json`。逐文件下载状态：`DATASET/checks/*.json`与对应download.log；最终结果：`_management/download_results.json`。archives保留原包；raw由验证脚本安全解压；VALIDATION.json/MD是每数据集验收记录。

只有某数据集所有清单项状态为download_verified之后，才运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 numactl --cpunodebind=1 --membind=1 /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python /home/ubuntu/rz/datasets/dataset_verify.py DATASET
```

脚本完成了语法检查，尚未以完整数据实测。必须审阅结果、修正解析逻辑产生的假阳性，不得修改原标签去迎合验收。脚本使用路径安全检查、ZIP/RAR校验、安全流式解压、PIL全量verify和load、COCO图像尺寸/引用/框检查、LLVIP双模态与XML匹配、ExDark文本标注检查。临时.rar解析依赖rarfile4.2仅安装到 `_management/vendor`，没有改conda环境或系统软件；系统unrar已存在。

解压若中断会保留raw部分内容，脚本不覆写而拒绝继续；需在限定目录下审阅恢复，不做不加检查的递归删除。原始JSON/图片必须保留。

## 来源与特别核验事项

- HIT-UAV Zenodo7633134 v1.2.1，814751578 bytes，MD5 bb6c5a33659b7507eae83458729c6109，包含normal/rotated两套数据，物理图像可能重复，独立图像数2898。DontCare不是普通有效类。
- SeaClear 4TU UUID4f1dff25-e157-4399-a5d4-478055461689，RAR1711829309 bytes，MD5 1cfcf0c2fa3ef0dc219a66f063c2fe99，8610图/40类。
- ExDark官方Dataverse DOI10.22452/RD/JUSQEK，14个文件（12类tar、groundtruth.tar.gz、展示collage.png），均保存发布者MD5。另需保存官方GitHubGroundtruth/imageclasslist.txt与README（固定划分，不能自行重划）；仓库commit810e4a0c5d6eddebd7d2280d371a090decc719bb。检查文本格式头可能需要调整验证器，7363图与标注匹配。展示collage不要算数据图片。
- ORCA作者HF WongYukKwan/ORCA，source_metadata.json已固定commit及文件树，LFS文件有SHA256。包括images.zip、data.json、全部split_annotations。此前14645/14647计数差须用实际JSON/图片核实；caption label0/1/2含负描述，不能随意当成新增检测对象或直接复制标注。
- DUO官方Drive文件1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs，7782图，作者6671/1111 train/test；使用条款另核。
- AP-10K官方Drive1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad，仅10015张标注主包，不下载可选约50k未标注数据；9个COCO split文件，54物种，姿态任务不等于穷尽性检测标签。
- LLVIP官方新版Drive1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv，15488可见光/红外对；2023-02-21作者更新标签。作者仓库commit c1a655cce437ebfd990a97b04fc48fbb99f4c47b。核对XML和图像同名匹配，保留原始双模态数据；不下载额外未配准原视频。

## 自动复查

已创建本任务heartbeat，ID `automation`，名称“领域数据集下载与完整性验收”，每10分钟复查。只在实质进展/失败/完成/需用户操作时通知，不对未变化状态刷屏。完成后停用并关闭专用中转；若官方访问持续不可用且安全替代穷尽，诚实报告已完成项和阻塞并停用等待用户，不虚构全完成。
