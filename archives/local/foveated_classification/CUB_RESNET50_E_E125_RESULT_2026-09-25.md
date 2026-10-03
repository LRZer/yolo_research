# CUB ResNet50 E组：黑白无模糊，实际125轮结果

日期：2026-09-25（北京时间）。E组在物理GPU0、NUMA0上完成第125轮并保存`last.pt`，任务专属监视器在确认CSV完整125行及配置哈希后，仅对E训练PID525246发出停止信号。主进程退出后，GPU0仍由任务专属DataLoader残留PID585983占用；核对PPID=1、PGID=525246、完整命令行和标准输出均指向E运行目录后，仅向该残留PID发SIGTERM，GPU0恢复4MiB。未触碰其他用户进程。

独立CPU审计返回`verified_revised_e125`：125个连续有限epoch、每轮train5,994/test5,794图与94优化步、原200轮余弦计划前125项、可读取且第125轮的`last.pt`、配置/统计哈希与200类权重均符合要求。固定第125轮官方test：Top1 **27.096997%**、Top5 **55.799103%**、test loss **3.247952**；平均每轮49.8415秒，其中训练37.9728秒、完整test11.8684秒。相对A原图Top1 **低22.868485百分点**。仅说明本次CUB/ResNet50 scratch/单种子设置下，去掉颜色信息损害很大；不能外推到所有分类数据或预训练模型。

E的输入是在共同RGB训练增强后`RGB→L→RGB`，训练和测试均灰度、不模糊、不使用GT框。`results.csv` SHA-256：`24137bad94a0db7e6f244c5e2cc4d3ce90d5589d8f48f2786c27b34ea4cff5a3`；`last.pt` SHA-256：`df03bc7caad7320078b5a396c47b0a2e08ba0e931e3c21946e4e035042d8258b`。官方test逐轮查看、E后追加、单种子，结果仅为探索性对照。五组总体解释见`CUB_RESNET50_FIVE_GROUP_E125_FINAL_REPORT_2026-09-25.md`。
