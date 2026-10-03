# 数据集上传目标与接续记录（2026-09-21）

## 04:55 68/68传输完成，独立最终审计运行中（最新）

- ZeroWaste-f主ZIP 7,518,242,799 bytes的远端SHA256=`9ce00fe12e0e4163d94f83e167e7d358211702d58fd6b82b9b9021615a80f117`一致；公开密码解压完成，9009成员/7,533,613,802解压bytes逐项CRC和大小核对通过，9006张图像文件全部解码，无AppleDouble。其余4个审计/来源文件也已完成。
- 04:53:44上传状态68/68、FailedEntries为空；复验接续进程正常输出`CONTINUATION_FINISHED_FINAL_AUDIT_STILL_REQUIRED`并退出。当前没有上传worker，不应重启。
- finalizer30304已复核本地68个原始/说明文件，并于04:54进入`remote_independent_final_audit`。最新日志已重新验证HIT-UAV三个清单项。该独立审计会重新计算所有远端SHA、逐压缩成员CRC或SHA、枚举解压成员集合并重新解码图片；尚未覆盖全部清单，不能宣称目标完成。

## 03:34 首批结束、ExDark复验成功，仅剩ZeroWaste-f（最新）

- LLVIP原ZIP 4,008,227,348 bytes完成远端SHA256=`75c308c2c44de441fe854cfb9aa24259dcaac5a83b4f79c668e548f95f37af42`核对；46,465成员/4,054,277,998解压bytes逐项CRC/大小通过，30,976张图片全部解码，无AppleDouble。LLVIP 5/5完成。
- HRSID两份SUPPLEMENTAL也完成，HRSID 5/5。初批worker正常以batch_finished结束，共60/68；复验守候器PID36576随后取得同一worker锁，无并发状态写入。
- ExDark bicycle/boat/bottle三个旧误判包均以`verification_only_recovery=true`通过：完整对照包内成员大小与SHA256、不覆盖已解压内容；真实图片652/679/547张全解码，AppleDouble辅助文件653/680/548个保留并核对。ExDark恢复为19/19，FailedEntries清零。
- 03:34:23总进度63/68，只剩ZeroWaste-f五项。7,518,242,799-byte主ZIP已经开始上传；之后仍需公开密码解压、9009成员CRC、9006图片文件解码及最终独立全清单审计。当前不可宣称全部完成。

## 02:52 VisDrone整套完成，LLVIP开始（最新）

- VisDrone 5/5清单文件完成。train：12,942成员=6,471图+6,471标签；val：1,096成员=548图+548标签；test-dev：3,220成员=1,610图+1,610标签，三个原ZIP远端哈希、成员CRC/大小和全部图片解码通过。
- 用户补充的包确认是有标注的test-dev；未上传此前不合适的challenge包。3条train零高度原始行及ignore读取要求继续保留在MANUAL_VALIDATION，未改标签。
- 当前完整交付六套：HIT-UAV、DUO、AP-10K、SeaClear、ORCA、VisDrone；总进度53/68。02:51开始LLVIP原ZIP 4,008,227,348 bytes，完成后需验证30,976张图片、15,488对可见光/红外配对和15,488份XML。

## 02:47 VisDrone train远端验收通过（最新）

- `VisDrone2019-DET-train.zip` 1,549,875,511 bytes上传完成，远端SHA256=`c3176397ef9c859600e632d91f7f4881ee75cffd0cad768e3dca1388ed3586d3`与冻结清单一致。
- 安全解压后12,942成员/1,556,080,035 bytes逐项CRC及大小检查通过：6,471张训练图片全部Pillow verify+load成功，6,471个标签文件保留，无AppleDouble辅助文件。
- 原3条零/负尺寸行（其中2条为ignore/other）仍只在validation_note提示读取协议，未修改原标注。当前49/68文件，开始上传81,625,797-byte的val包；之后为正确的test-dev。

## 02:32 HRSID原包验收通过，VisDrone train开始（最新）

- HRSID JPG版RAR 608,910,165 bytes的远端SHA256=`743be02ed6df0a303fecd15db237ad84ba6fbf1b09b5111f171196e92ec2444d`一致；5607个成员/659,517,399解压bytes逐成员校验通过，其中5604张JPG全部解码、3个JSON保留。MANUAL_VALIDATION两份也已完成。
- HRSID的两份`SUPPLEMENTAL_SPLIT_CONSISTENCY`位于冻结清单索引61/62，排在LLVIP后面，当前未漏清单、只是尚未轮到；因此HRSID目前3/5，暂不计整套完成。
- 当前48/68文件，已进入VisDrone train原ZIP 1,549,875,511 bytes。train完成后还需val、用户补充的正确test-dev和两份MANUAL_VALIDATION；原始3条零高度框保留并记录。

## 02:25 ORCA整套清单完成，HRSID开始（最新）

- ORCA 17/17清单文件均已完成远端哈希核对；原图片包安全解压和14,646张全图检查通过，主标注、9份原始划分JSON、README、REVIEW和VALIDATION材料均保留。
- 当前五套完整交付：HIT-UAV、DUO、AP-10K、SeaClear、ORCA。总体45/68文件；ExDark为16/19，三个旧AppleDouble误判包待当前批次结束复验。
- 已开始HRSID 608,910,165 bytes的JPG版原RAR，02:25约22MB。完成上传后仍需远端RAR成员CRC/大小、5604张图像解码；split与合并JSON对象集合差异只记录、不修改。

## 02:22 ORCA图片大包完成远端验收（最新）

- ORCA `images.zip` 987,885,669 bytes上传完成，远端SHA256=`3b19aa41dbfd3eb98d8b58e7abea873551297710d6a9f6b423c9f923adf35cbd`与冻结清单一致。
- 远端解压14,646个文件/14,646张图片全部逐成员CRC及Pillow verify+load通过，解压内容1,005,467,962 bytes；未发现AppleDouble辅助文件。本次不是只凭ZIP能打开就判成功。
- 主`data.json`未引用的1张图片和caption `-1`语义警告原样保留在验收说明，未删除或改标注。02:22快照ORCA已5/17，worker继续传其余划分/说明文件；总体最终独立审计尚未运行。

## 02:10 ExDark后续包持续通过，空间充足（最新）

- 02:09:17实际状态26/68文件完成；ExDark table及READING_NOTES通过，正在上传VALIDATION.json。旧bicycle/boat/bottle仍是已审阅待复验项，没有新增失败类型。
- 02:09只读核查服务器任务目录约21,248,525,742 bytes（含原包与解压内容），所在文件系统仍有453,405,798,400 bytes可用；空间不是当前风险。未修改服务器或其他目录。
- 监控脚本现同时显示主worker、已审阅复验守候器、finalizer和最终报告守候器的实际进程身份与状态，防止只依据旧状态文件误判存活。

## 02:07 最终报告守候器已启动（最新）

- `work/dataset_delivery_closer.py`已编译并启动，PID11976，创建02:06:38；状态`E:/datasets/_management/DELIVERY_CLOSER_STATE_20260921.json`，日志`DELIVERY_CLOSER_20260921.log/.err.log`。最多等待8小时，不参与当前传输。
- 它只接受finalizer状态`final_audit_passed_agent_review_and_readme_pending`，并重新核对：清单无pending、UPLOAD_STATE精确覆盖68文件、本地原始文件审计精确覆盖、远端最终审计状态/错误/数据集/文件映射全部一致。任一条件不符即停止，不发布完成声明。
- 条件全部满足后，才生成`SERVER_DATASET_INDEX_FINAL_20260921.md`与`FINAL_COMPLETION_REPORT_20260921.md`；远端README仅在仍等于本任务已知进行中版本时原子替换，避免覆盖意外改动，最终报告写入任务`_management`并核对SHA256。它不会自行把active goal标完成，仍需最终agent审阅证据。
- 02:06:39当前22/68文件完成，ExDark people通过，进入协议/说明小文件；三个旧AppleDouble误判仍待当前批次结束后的只读复验。主worker、复验守候器、finalizer与delivery closer均已实际查证存活。

## 02:05 传输继续，最终独立验收加严（最新）

- 02:04:52主worker14528仍存活，ExDark motorbike完成，正在上传people；共21/68文件、9,760,948,467/25,564,671,681 bytes完成，剩余约15.804GB。未出现新的失败类型。
- `dataset_final_remote_audit.py`进一步加严：所有压缩包成员除逐文件CRC/SHA256外，会在最终阶段再次解码实际图片；仅对文件名及AppleDouble文件头都匹配的辅助元数据跳过图片解码。还会枚举解压目录，拒绝缺少成员、多余文件或符号链接，并核对首次验收记录中的图片计数。
- 加严后的脚本已通过本机编译检查，但尚未部署和执行；finalizer会在68个清单文件全部完成后上传它并运行。因此目前仍不能声称最终验收通过。

## 01:58 已安排有限自动复验接续（最新）

- 已启动`work/retry_reviewed_exdark.py`，PID36576，创建01:57:56；日志`E:/datasets/_management/REVIEWED_EXDARK_RETRY.log`及`.err.log`。最多等待4小时，只在主worker锁释放且状态batch_finished后执行一次接续。执行前仅接受已审阅的bicycle/boat/bottle AppleDouble误解码错误；有其他失败则停止供检查。
- 接续过程持有同一工作锁，调用原worker完整哈希/解压检查流程，跳过已经完成文件、复验三个失败包，并纳入清单中的ZeroWaste-f等剩余项；不并发写UPLOAD_STATE，不改原始数据。现有supervisor可能在接续前因失败记录自行结束，这是预期；不要据此再启动重复worker。
- 更新只读监控脚本，使其也识别接续进程担任worker时的真实命令行。最终独立验收仍由finalizer等待全部68文件后启动。目标尚未完成。

## 01:56 ExDark辅助文件误判已修正，三个旧失败待重新核验（最新）

- 用户确认VS Code仍可以连接。先前跳转连接拒绝已恢复，当前worker14528（创建00:51:23）持续上传，非服务器整体离线。没有修改服务器网络或共享配置。
- ExDark的bicycle、boat、bottle在图片验收时报错，指向`._*.jpg/png`。已读取包内文件头`0005160700020000`，确认是macOS AppleDouble元数据，不是真实图片损坏的证据。
- 验证助手仅对名称以`._`开头且文件头匹配的辅助文件跳过图片解码；仍保留文件、逐成员核对内容哈希。实际图片继续全部解码。增加完整解压目录的只读复验恢复，不覆盖原始文件。助手已原子部署，SHA256=`5a96ee6677622a953947e6fea7b6ac1ffe0427162caee74bf4eaf0b1981e3624`。
- 修正后bus通过：527真实图片、528辅助文件；car通过：638真实图片、639辅助文件。01:56状态14/68文件完成、cat开始上传。前三个失败尚未重新验收，不得声称整个ExDark完成。
- 后续：当前批次正常继续，不改其正在写入的UPLOAD_STATE。批次结束/工作锁释放后，再运行worker重试三个失败（已上传完整包可复用，解压目录仅复验），并处理ZeroWaste剩余批次。现supervisor会遇failed后退出，需要届时核实状态后接续；不要并发启动第二worker。finalizer30304仍等待所有文件完成。

## 01:50 SeaClear完成，ExDark开始（最新）

- SeaClear原RAR1711829309 bytes与本地SHA256=`2a053f748d6bdc5df8d776e87b72832f2908316f0f419f6a8d562df6df086c13`一致，远端8612文件/8610图片解压与全图检查通过；两份VALIDATION说明也已传完。
- 当前12/68文件完成，对应HIT-UAV、DUO、AP-10K、SeaClear四套。正在传ExDark的分类别tar.gz包，后续还会带上groundtruth、官方划分列表及大小写映射说明。
- 01:49:37实际查证worker14528、finalizer30304存活，FailedEntries为空。远端SeaClear解码临近尾声时pgrep未再找到helper，是因它已完成退出；随后本地结果和下一包开始证实正常结束，不是任务掉线。
- 最终远端独立检查脚本增加对非压缩包的独立图片（例如ExDark的collage.png）的完整解码，避免只检查了压缩包内的图片而漏掉附带说明图。该最终脚本尚待所有文件传完后由finalizer部署运行，不能提前声称通过。

## 01:32 AP-10K完成，SeaClear开始上传（最新）

- AP-10K原包3403294327 bytes已完整传输，SHA256=`420980abb135d6f66bcc8e29f289a46081214016192ae197ad24bc1525c8e62c`一致；远端10024文件、10015图片全部解压/CRC/图片解码检查通过，两个VALIDATION文件也已传完。
- 当前9/68文件完成，对应HIT-UAV、DUO、AP-10K三套完整交付。SeaClear的1.712GB原RAR刚开始上传。状态中此前SSH断连留下的failed条目均已被成功重试消除，当前FailedEntries为空。
- 01:32实际查证worker14528（创建00:51:23）和finalizer30304仍存活，最终验收守候状态正常；不得重新启动活跃队列。进展MD已刷新，仍未达到全部上传完成条件。

## 01:05 目录说明已交付、传输仍在推进

- `work/publish_dataset_index.py`已实际将本机SERVER_DATASET_INDEX_20260921.md写入远端根目录README.md，并核对SHA256=`8f3cc63d20ec516b82433aeea20739fea3be616292bec208b780894fdd4f00dc`。以独占创建方式写入，已有同名不同内容会拒绝覆盖。README明确标为“上传进行中”，最终完成后需在确认现有内容仍是本任务版本后更新完成说明；不能让进行中标题作为最终交付状态。
- worker14528存活（已查命令行），AP-10K从此前0.44GB推进到0.672/3.403GB，约1.82MB/s。supervisor38728及finalizer30304本轮先前已查证存活，错误日志均为空。不重启正常传输，不将正常等待判为阻塞。
- 本轮属于实质进展（目录说明交付）及已验证等待（上传进程存活且传输字节增长）。剩余任务仍是把整套队列传完、远端独立最终验收、审阅并更新最终README/报告；目标保持active。

## 01:02 当前状态与收尾守候

- DUO已经完成远端解压/逐图核验及两份说明传输，HIT-UAV、DUO两套完成；现上传AP-10K。worker14528及supervisor38728仍已查证存活。
- 新增最终验收守候器PID30304：`work/dataset_upload_finalizer.py`，状态`FINALIZER_STATE_20260921.json`，日志`FINALIZER_20260921.log/.err.log`。已实际启动。每45秒检查全部68文件状态与sha，最多6小时。全部传完后复核全部本地原始文件SHA、上传最终manifest/验收脚本、运行远端独立全量检查并取回结果。不自动宣称目标完成，最后仍需agent审阅和交付目录说明。
- 最后输出预计为本机`FINAL_LOCAL_ORIGINALS_AUDIT.json`、本机及远端`FINAL_TRANSFER_AUDIT.json`；当前这些尚未生成，不能说最终验收已过。
- 已编写本机`E:/datasets/_management/SERVER_DATASET_INDEX_20260921.md`，列出10套数据布局、样本数和原标签警告。此文件仍是进行中说明，待传到远端README并在最终完成时更新，不能作为已交付证据。

## 00:56 连接恢复，10套已全部入队（最高优先级）

- 00:51只读SSH探测重新返回reachable，未改服务器或中转配置。用户也回复VS Code可以连接。网络阻塞已解除，不要沿用00:49状态判blocked。
- 重新启动worker：PID14528 **被Windows重用**，新CreationDate=2026-09-21 00:51:23；命令仍`work/dataset_upload_worker.py`，日志改为`UPLOAD_20260921_resume1.log/.err.log`。只信当前进程身份和创建时间，不凭旧PID判断。DUO远端保留前缀1,724,907,520 bytes，与本地前缀SHA256一致，已从该偏移续传。00:55末约2.62/2.94GB、3.66MB/s，尚未完成。
- 新接续器PID38728、CreationDate00:51:44，日志`UPLOAD_SUPERVISOR_20260921_resume1.log/.err.log`；旧32168已因failed记录自行退出。原failed记录会随本轮逐项成功被替换，不要因为它们仍在状态表就并发重启活着的worker。
- ZeroWaste-f已完整通过官方MD5/ZIPCRC、原生安全解压、9009文件解压CRC及9006图片文件全解码、COCO与掩码配对检查：4503原图+4503掩码，train3002/18002框、val572/3687框、test929/5077框，总26766框，4类。无结构错误；138个原始bbox越界（100/21/17）作为使用警告保留。
- ZeroWaste原验证器曾把跨split同名图片当错误。逐个文件及RGB像素SHA256核查仅3处同名，全部内容不同；验证器改为按split路径识别并记录证据，不改数据。不将“像素不完全相同”夸大成所有相邻视频帧相互独立。
- ZeroWaste审计入口`prepare_zerowaste_audit.py`、`audit_zerowaste_annotations.py`；最后结果`MANUAL_VALIDATION.json/.md`为verified_with_usage_notes。原下载35768和验收25568/11804/9460均已结束；新的后续以实际进程核对为准。
- 接续器已将ZeroWaste 5文件加入清单。**总10套、68文件、25,564,671,681 bytes，pending为空**。JODD无包明确排除成功计数。当前worker读取的是本批启动时清单（不含后来加的ZeroWaste）；完成后接续器会自动启动剩余5文件的新批次。
- 远端ingest助手已重新部署，新增对公开密码ZIP用现有unzip解压，之后仍逐文件CRC/逐图decode。未安装任何全局软件。
- 全新最终验收脚本`work/dataset_final_remote_audit.py`已编译检查，**尚未部署/运行**；所有上传完成后需将最终manifest及脚本传入任务_management，重新逐文件SHA256/逐个解压成员CRC或tar内容SHA校验，并审阅FINAL_TRANSFER_AUDIT.json才可完成目标。
- 可运行`work/render_dataset_upload_progress.py`生成本地`_management/UPLOAD_PROGRESS_20260921.md`和workspace `DATASET_UPLOAD_PROGRESS_2026-09-21.md`。只输出状态快照，不等于进程实时存活证明。终端中文乱码是输出编码，落盘UTF-8正常。

## 00:49 连接中断与本地验收（优先入口）

- 第一批worker PID14528在DUO传到约1,723,858,944 bytes时SSH断开，随后新连接均出现`channel 0: open failed: connect failed: Connection refused / stdio forwarding failed`。这是跳转到目标端口失败，不是证明压缩包坏或GPU/内存报错。服务器实际断点字节尚无法联网核对。
- 独立只读SSH探测同样拒绝连接。已严格核对PID及命令行后只停止本任务worker14528，保留所有本地包和远端upload-part；不重启服务器、不改FRP/SSH配置、不碰其他进程。状态已改为interrupted_after_verified_ssh_connection_refused，不能把残留running状态当进程还活着。
- HIT-UAV之前的完成结果仍有效；DUO未完成，AP-10K未传成功。其余任务还在清单待办。接续器32168将在读取已有failed条目后自行退出，后续必须重新确认进程/连接再启动，不能并发重复传。
- upload_worker已修正：未来遇到明确的共用SSH通道不可达会在有限重试后结束批次，避免将每个后续包都重复试一遍。恢复后需核对远端partial前缀SHA才能续传。
- 新ZeroWaste-f包已到齐7,518,242,799 bytes，MD5与发布者一致，SHA256为`9ce00fe12e0e4163d94f83e167e7d358211702d58fd6b82b9b9021615a80f117`。本地原生解压/全CRC/逐图验收进程PID25568运行，日志`ZeroWaste_native_audit.log/.err.log`，状态`ZeroWaste-f/EXTRACTION_VALIDATION.json`。只是文件到齐/MD5通过，尚不宣称标注可用性验收结束。
- HRSID补充完整标注及检测框对比均得到split-only22条、combined-only4条；详情`SUPPLEMENTAL_SPLIT_CONSISTENCY.json/.md`已加入上传清单。官方GitHub固定commit仅提供3573张PNG图/9870标注的train2017，和本地JPG版本不同，不把哈希差异误判为下载损坏、不替换标签。

## 00:43 后续进展（优先）

- HIT-UAV已完整上传、SHA256一致、远端解压检查通过：17441个文件，11600图片文件全部verify/load成功。当前在传DUO，最新观察约806MB/2940MB，约3.70MB/s。
- 已启动串行批次接续器 PID32168：`work/dataset_upload_supervisor.py`，日志`E:/datasets/_management/UPLOAD_SUPERVISOR_20260921.log`及`.err.log`。每30秒读取审计结果/任务锁；初批worker退出后处理新加入清单，不与其并发上传。若有上传/解压失败停止交给agent审阅，最多运行6小时；不自动删除或覆盖失败文件。
- HRSID全量通过：5604图，train3642/11047框、test1962/5922框，共16969框（以审计JSON各split精确数为准；合并JSON实际16951框，说明不同JSON之间需要进一步核对对象集合，不能简单假定完全一致）。原文件均保留；已入队。
- VisDrone新test-dev已完成ZIP CRC及全量解码，1610图/1610标签。train6471、val548、test-dev1610三包和审计MD/JSON已加入上传清单。train原标签中发现3条零高度框：2条score=0/category=0的忽略区域，1条score=1/category=4的普通目标。报告保留`file_integrity_passed_annotation_review_required`，不修改标签、不称所有标注无瑕疵。不能为了通过验收把普通零高度目标当忽略区域。
- LLVIP全量ZIPCRC、30976图片解码、15488对及XML名称/尺寸配对通过，train12025/test3463，42437目标。5个XML框xmin==xmax（020118、020236、090437、091180、170350）；VOC像素闭区间和连续坐标解释可能不同，训练读取协议需再审阅。报告保持标注待审，不改原标签。接续器将依据已完成审计加入原包和说明。
- 新包注册器：`work/register_manual_uploads.py`。清单被冻结的源文件/报告若改变则拒绝覆盖，需人工审阅。注册器不应在接续器运行时并发启动。
- ZeroWaste-f仍待完整下载与验收；JODD仍没有实际包。目标仍active，不能标为全部上传完成。

## 当前授权覆盖旧下载交接记录

用户明确要求上传所有已下载数据集，在服务器 rz 下建立英文“目标检测数据集测试”目录，全部解压并检查；2026-09-21 又明确要求建立目标并持续上传到完成。旧下载记录中的“不连接/不上传服务器”不再适用于本任务。

- 目标：`gpu8-team:/home/ubuntu/rz/object_detection_dataset_test`。
- 只操作该新建目录。所有者标记：`_management/OWNER.txt`，内容 `rz-dataset-ingest-20260921`。
- 保留本地和远端原始包，不改原图/标注、不转 YOLO、不训练、不用 GPU、不改共享环境或别人的进程。
- 服务器此前存在硬件内存错误，用户/所有者已同意继续使用；上传前后完整 SHA256、解压 CRC/大小及逐图解码必须保留，不能声称硬件健康。
- 上传前检查目标不存在；服务器目标所在盘约 443 GiB 可用。目标已实际创建。

## 已启动进程与入口

- 上传 worker PID 14528，入口 `work/dataset_upload_worker.py`；这是本轮实际启动的上传进程，不是此前下载进程。
- 状态：`E:/datasets/_management/UPLOAD_STATE_20260921.json`；清单：`UPLOAD_MANIFEST_20260921.json`；日志：`UPLOAD_20260921.log`、`UPLOAD_20260921.err.log`、`upload_transport.stderr.log`。
- 初批 6 套、48 个文件、11,486,627,186 bytes：HIT-UAV、DUO、AP-10K、SeaClear、ExDark、ORCA。逐文件本地 SHA256 已重新对照旧验收指纹。
- worker 本轮启动时读取一次清单；后来加入的数据需要本批结束后再运行同一入口（它复用成功记录，不重复上传）。锁防并发。不要同时启动第二个 worker。
- 传输通过 SSH 标准输入写任务专用 `.upload-part`。续传前比较完整已传前缀 SHA256，完成后远端 SHA256 一致才改成最终包名；不同内容不覆盖。
- 每个压缩包远端解压到 `<dataset>/raw/<archive-name>/`，先审查路径/链接/空间。完成后逐文件大小、CRC（ZIP/RAR）、图片 verify+load，成功写 `_INGEST_COMPLETE.json`。如果解压中断留下非空目录，脚本拒绝覆盖，需检查后有针对性恢复，不能盲目删除。
- 远端辅助脚本 `dataset_remote_ingest.py` 与纯 Python `rarfile.py` 位于任务 `_management`。使用已有 conda Python/Pillow、已有 unrar，没有安装/更改全局环境。
- 初批从 HIT-UAV 开始，00:36 左右已实际传输约696 MB，约3.7 MB/s（非最终完成状态）。

## 新数据审计

- `work/audit_manual_datasets.py` 全量读取 ZIP/CRC/图片解码/配对和框结构，HRSID 使用已有安全 RAR 解压器。只写新的 `MANUAL_VALIDATION.json/.md`，不改源数据。
- LLVIP审计 PID29072；VisDrone审计 PID28368；HRSID审计 PID33924。日志在 `_management/<dataset>_manual_audit.log` 及 `.err.log`。
- LLVIP选 `E:/datasets/LLVIP/LLVIP/LLVIP.zip`，4,008,227,348 bytes。与 `E:/BaiduNetdiskDownload/LLVIP.zip` 全部成员路径、大小和 CRC 一致，但这不独立证明2023标签版本；保留版本说明。
- HRSID原包 `E:/BaiduNetdiskDownload/HRSID_JPG.rar`，608,910,165 bytes，JPEG版本，5604图+3个JSON。不能把文件名当完整验收。
- 用户刚补充 `E:/datasets/VisDrone2019-DET/VisDrone2019-DET-test-dev.zip`，311,045,829 bytes，ZIP内部 **1610 JPG +1610 TXT**，根目录为 images/annotations。现在本地旧 challenge ZIP 已不在列表，不由助手删除、不自动恢复。test-dev加入审计，后续上传 train/val/test-dev 三包。
- 新审计报告无 errors 并审阅 warnings 后，冻结 SHA256 加入上传清单。ORCA语义警告、VisDrone ignore类别、ExDark大小写映射必须原样带到服务器，不通过改标签消除警告。

## 尚未完成

- ZeroWaste-f 原下载 PID35768 仍运行；日志/状态位于自身目录。总7,518,242,799 bytes，官方MD5 `e26e31a58080bca6782dca0e56074c5d`。公开解压密码 `UP#1VuX409z4`。需完整包MD5、全量解码及标注核验，再上传并解压。当前下载器ZIP纯Python解密CRC可能较慢，不能把仍在验证当卡死。
- JODD没有实际已下载数据包；不纳入“已上传完成”的虚假计数，不在此任务恢复不明镜像下载。
- 自动复查 `automation` 仍是 PAUSED，旧prompt禁止上传，不能直接恢复。当前使用用户明确启动的 active goal 持续推进；如需调度必须重写合规新指令。

## 完成标准

当前已下载的每个数据集全部原始包及必要划分/说明文件远端哈希一致，安全解压和全量图片解码通过，原标注完整且本地配对/结构审计通过（已知语义警告如实记录），汇总本地/远端 MD。不能仅凭传输进程退出就完成目标；未下载或外部阻塞的内容单独说明。

## 2026-09-21 05:06 最终交付

- 已完成 10 套数据集、68 个清单文件、25,564,671,681 bytes 的上传与验收：HIT-UAV、DUO、AP-10K、SeaClear、ExDark、ORCA、HRSID、VisDrone2019-DET、LLVIP、ZeroWaste-f。
- 本地 68 个原始/说明文件已重新核对大小与 SHA256；远端 68 个文件逐项匹配。独立最终审计状态为 `all_manifest_files_and_extracted_content_verified`，错误列表为空。
- 对压缩包重新核验成员 CRC 或 SHA256、解压成员集合与逐图解码；原始标签未为通过验收而修改。ExDark 的 AppleDouble 条目只作识别和验证期恢复，不覆盖原包；ZeroWaste-f 的公开密码包已完整核验。
- 最终服务器 README SHA256：`728f51fa25fc5ae5b1d827b30774225f351c10d884d4d252016fbb5d7e0c26cc`；最终完成报告 SHA256：`592e4d5341bebc8ce2402dd4c0bbefb186565213a97a975eb6db2a5d9363a95b`。
- JODD仍无实际下载包，不在上述完成计数中，也没有用来源不明的镜像替代。训练未启动、GPU未使用、共享环境及其他进程未修改。
