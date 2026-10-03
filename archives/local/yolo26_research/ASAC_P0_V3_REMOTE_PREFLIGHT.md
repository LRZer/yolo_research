# ASAC-P0 v3 远端预检记录

日期：2026-09-19（Asia/Shanghai）  
状态：**STOP / NO GPU RUN**

## 已通过

- 本地独立复审裁决：分阶段 GO。
- 远端 staging 文件哈希与审计冻结值一致：
  - probe：`a0d87624f013d79c69c30ada1b9295e5bd21304b3de9778336030c79b0ea6e74`
  - tests：`975d77cd74767b526a5810cd76e58d477865964d28bde49657a57c0563a6a96a`
- 远端 CPU `py_compile` 通过。
- 远端 CPU 单元测试 33/33 通过。
- 安装前旧 v2 文件已按时间戳 `20260918T203334Z` 留存备份；安装后的运行文件哈希再次匹配上述 v3 冻结值。

## 阻断结果

在 GPU 前的五 checkpoint CPU provenance/structure preflight 中，`epoch10.pt`、`epoch20.pt`、`epoch30.pt`、`epoch40.pt` 依次读取后，处理历史 B50 `last.pt` 时失败：

```text
RuntimeError: checkpoint .../runs/b50_no_kd_e50_g4567_s42/weights/last.pt
train_args lacks frozen B50 manifest fields ['save_dir']
```

该失败发生在任何 GPU forward 之前。按 v3 复审协议，不能为了使探针通过而删除字段、修改冻结 manifest 或换 checkpoint；因此没有执行单 batch GPU smoke，也没有执行正式 80-forward ASAC-P0。

## 解释边界与下一步

- 随后的只读核对确认：四个 numbered checkpoint 的 `train_args` 均为 124 字段、含完整 `save_dir`，且 `epoch` 分别为 10/20/30/40；最终 `last.pt` 为 123 字段、只缺 `save_dir`，并被标记为 `epoch=-1`。独立 `args.yaml` 仍为 124 字段且含正确 `save_dir`。
- 源码 `strip_optimizer()` 会在训练结束时令 `epoch=-1`，并把 `train_args` 重建为 `DEFAULT_CFG_KEYS` 内的字段；`final_eval()` 明确对 `last.pt` 调用它。由于 `save_dir` 不在该默认键集合，观察到的单字段缺失与确定性终态 strip 路径吻合。
- 这解释了失败来源，但**不自动授权放宽 v3**：独立 `args.yaml`、固定目录关系、内容哈希和 stripped-state 必须共同组成 v4 的窄化替代证明，且仍需重新跑恶意轨迹回归。
- 若继续 ASAC，需形成 v4：numbered checkpoint 继续要求完整 127 字段；仅对已固定内容哈希、`epoch=-1` 且来自精确 `.../weights/last.pt` 的终态 checkpoint，允许从已独立验证的同 run `args.yaml` 补回唯一的 `save_dir`。除此之外的任意缺失仍硬拒绝，并重新接受完整恶意轨迹审计。
- 在 v4 重新获准前，ASAC 状态为 `NO-RUN`；这不影响 TQGB seed 配对与其他独立方法实验。
