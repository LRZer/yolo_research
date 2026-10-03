# CUB ResNet50 E组：黑白无模糊，125轮顺序训练

北京时间2026-09-25 16:38，D组第125轮独立审计返回`verified_revised_e125`后，重新检查GPU0空闲4MiB、NUMA0可用约108GiB、当前boot mc0/mc1不可纠正错误计数0、无新增明确MCE/SIGBUS，也无其他CUB训练进程。启动时仍保留服务器重启BERT未解码硬件记录的风险，不据此宣称硬件修复。

E组CPU配置/数据检查通过（官方train 5,994、test 5,794、200类），配置SHA-256为`9d731cacbaaa17724ecfc486bbbfa7dad3d6b7cb573bcf5c5376687b71c38a74`；物理GPU0、NUMA0上的32图/1次优化步短测`smoke_passed`，峰值分配显存3.7069GiB。16:38:20顺序启动E训练PID**525246**，只用物理GPU0和NUMA0。针对该真实PID的唯一125轮精确停止监视器PID**526355**于16:38:38返回`watching`；16:38:50两进程仍在运行，GPU0约4630MiB/98%，mc0 UE=0。E尚无正式完成轮次和准确率，不能宣称结果。

E在共同RGB增强后将有效原生图以Pillow `RGB→L→RGB`变灰，train/test相同，不用GT框、不模糊；仍是scratch ResNet50、seed2005、有效batch64、workers4、原200轮余弦计划前125项。训练目录名保留`e200`反映原配置时间尺度，实际只将在第125轮完整保存后停止。后续须严格审计才可称完成，五组最终比较仍须固定末轮且承认官方test已逐轮查看、单种子与E后追加等限制。
