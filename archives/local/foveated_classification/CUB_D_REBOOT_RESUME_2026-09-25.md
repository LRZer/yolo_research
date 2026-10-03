# CUB D组服务器重启与安全续训记录

北京时间2026-09-25 11:45:34，gpu8-team重启；原D训练PID1189729及停止监视器PID1190369随系统消失。重启前最后完整记录为第15轮，`results.csv`、`state.json`、`last.pt`均停在第15轮；未将未完成的后续轮次计入结果。

12:05–12:08 CST检查：新boot ID `eacf4add-8b2b-4283-a1dc-abfb4abd2e2b`，物理GPU0空闲4MiB/0%，NUMA0空闲约112GiB，当前boot EDAC mc0/mc1 UE和CE均为0，未见当前boot明确不可纠正MCE或SIGBUS。内核启动时显示`BERT: [Hardware Error]: Skipped 1 error records`和`Total records found: 1`；其来源/严重程度未被该日志解析，不能据计数清零宣称硬件已修复。本次未对其他用户进程或GPU1–7做任何操作。

专用只读CPU恢复审计`audit_cub_resume.py`返回`verified_resumable_checkpoint`：CSV连续1–15且有限、每轮5994/5794图和94更新、状态/CSV/checkpoint均为15、配置SHA `3240c52c2971bd1748af2c031f90ea8e3b1b3851fa87bab15b82a2bf79fd1873`、训练统计SHA一致、200类模型权重有限、优化器/AMP状态可加载，checkpoint SHA `2dac6eb6155f389a73f9a884ddf3cedac8b30b51df2ac51bfeefb99bd7540497`。未从损坏或不一致检查点强行恢复。

经再次确认没有本任务训练进程、GPU0空闲及NUMA0资源足，由一次性任务专属启动器`resume_cub_d_after_reboot.py`于12:08 CST启动新D训练PID40937，绑定物理GPU0/NUMA0。训练日志明确`RESUME verified from epoch 15`，将从第16轮继续。重建针对新PID的唯一125轮停止监视器PID42017，日志`revised_e125_watcher_resume.log`返回`watching`。原始启动与旧监视器日志保留，续训元数据记录于远端D run的`resume_launch_epoch15.json`。后续仍需最终125轮严格审计才可称D完成。
