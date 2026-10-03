# TQGB 分类路由独立代码审计

日期：2026-09-19  
对象：T1/T2 分类路由实现与正在运行的 T1  
结论：**未发现足以使当前 T1 作废的致命问题；允许继续。**

## 1. 核心实现核验

- 匹配集合同时要求 O2M/O2O foreground、相同 GT 索引和相同离散类别；
- O2M→O2O 与 O2O→O2M 路由在正式 `margin=0.05` 下互斥；
- 教师概率与用于构造投影边界的学生当前概率均 detach；
- 投影准确实现 `clamp(c_teacher, min(c_student,t_student), max(...))`；
- Bernoulli KL 的 target detach，梯度只进入接收 logit；
- 每图除以所有 common anchors，不使用 active-count 放大；
- 正反方向独立增益，T1 配置为 `0.08/0`、`m2o`、projection on；
- 当前 T1 至审计时已产生非零 route/scaled loss，无 NaN/Inf，epoch 1 与 B50 完全一致。

## 2. 当前只能称“分类版 TQGB”

规范中的定位双向路由和 GT 残差空间投影尚未进入代码。T1/T2 是合法的分类因果实验，但在实现和测试 T3/T4 前，不能声称“完整 TQGB 已实现”，也不能把分类结果外推为定位机制成立。

## 3. stop-gradient 不等于其它参数完全不变

单独反传 T1 路由项时，直接梯度只进入 O2O classification head；反向路线则进入 O2M classification head 与 shared trunk。实际训练把基础损失和路由损失相加后执行全模型 global clipping，因此新增头部梯度仍会通过统一裁剪系数间接缩放其它分支更新。后续若 O2M/shared checkpoint 发生变化，不能把它误判成 teacher detach 失效。

梯度校准显示缩放后的双向路线使 shared update 的 global-clip scale 最低约 0.994；这把耦合控制得较小，但没有将其严格消除。若 T2 获得正收益，仍需 no-global-clip 或分组裁剪控制来排除该混杂。

## 4. 校准数值的准确口径

现有探针保存的是每个 checkpoint 上 8 个 batch 指标的均值。所谓 9.48%/8.82% 是“五个 checkpoint 的最大 batch-mean”，不是任意单 batch 的硬上界。正式报告必须使用这个口径；后续补充 per-batch max/P95 分布，但这不影响增益在 test 前冻结，也不使 T1 无效。

## 5. DDP 日志口径

`tqgb_*` count/rate/keep/loss 由 rank 0 的本地 batch 记录，没有跨四卡汇总；epoch CSV 又是 batch ratio 的算术平均，不等于 `sum(route)/sum(common)`。训练梯度语义本身正确，因为每图 common-normalization 后由 DDP 平均。论文机制统计应增加 logger 侧 numerator/denominator 汇总，不能把当前 CSV 标作四卡全局比例。

## 6. 验证侧解释

验证也会计算 `val/tqgb_*`，但不会反传，不影响 AP。它只是诊断量，不是普通 supervised validation loss，也不能参与超参数或最佳轮次选择。当前快速协议仍只报告固定 epoch 50；后续论文级实验需使用未污染开发集并在锁定方法后一次评测 test。

## 7. 测试缺口与补强

当前已覆盖投影、双向 stop-gradient、空 correspondence、gain=0、真模型 CUDA 与四卡 BF16 smoke。审计后已新增并通过显式 synthetic case：即使 O2M/O2O foreground 指向相同 GT，只要离散类别不同，也必须得到 `tqgb_common_count=0` 和零路由损失。仍需补充：

1. TQGB 专属多进程 DDP 数值等价；
2. BF16 下空候选、投影归零和 stop-gradient 断言；
3. 定位路线实现后的四方向隔离测试。

梯度校准探针也已补写逐 batch `mean/P95/max/min` 输出；待正式训练释放 GPU 后在原固定 checkpoints 重跑，用来替换“最大 checkpoint 批均值”这一较弱口径。该补写尚未产生新数值，不能提前把旧校准百分比解释成硬上界。

后续 SARC loader 审计发现：当前框架的 `trainer.get_dataloader(..., mode="val")` 会把配置 workers 数翻倍。旧 TQGB 梯度/覆盖探针虽传入 `workers=4`，实际诊断 loader 很可能为 8 workers；正式 T1/T2 训练使用 train loader，仍是 4 workers，不受影响。worker 数只改变预取并不改变冻结 batch 上的数学定义，但旧探针的协议记录不准确。新版探针已改为直接构建无增强 loader，并强制断言实际 `num_workers=4`、pinned memory；旧校准值在新版重跑一致前仅作为增益预注册来源，不作为精确可复现实验清单。

这些是工程加固项，不是当前已发现的实现错误。
