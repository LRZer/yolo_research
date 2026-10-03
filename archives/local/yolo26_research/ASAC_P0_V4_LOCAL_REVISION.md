# ASAC-P0 provenance v4 本地修订记录

日期：2026-09-19  
状态：**本地实现与 37/37 CPU 测试通过；远端真实 preflight 尚待正式训练释放资源**

## 修订原因

v3 在真实 B50 五 checkpoint CPU preflight 中处理 `last.pt` 时停止，因为其 `train_args` 缺少 `save_dir`。只读源码核对确认：

- `epoch10/20/30/40.pt` 均为未 strip checkpoint，`train_args` 有 124 字段并包含 `save_dir`；
- `last.pt` 被标记为 `epoch=-1`，`train_args` 有 123 字段且只缺 `save_dir`；
- Ultralytics `final_eval()` 对 `last.pt` 调用 `strip_optimizer()`；该函数把 `train_args` 重建为 `DEFAULT_CFG_KEYS`，会确定性移除运行期 `save_dir`。

## v4 的唯一例外

仅当以下条件全部成立时，provenance verifier 才恢复 `save_dir`：

1. 文件名精确为 `last.pt`；
2. checkpoint 内 `epoch == -1`，证明是终态 stripped checkpoint；
3. 除三个历史 TQGB no-op defaults 外，唯一缺失的必需字段恰好是 `save_dir`；
4. 恢复值不是通用默认，而是内容固定 checkpoint 的精确路径 `checkpoint_path.parent.parent`；
5. 后续仍必须同时通过固定 checkpoint SHA256、固定 `project/name`、真实 run 目录、独立 `args.yaml` 完整 manifest、50 行 `results.csv` 和模型结构验证。

任意第二个缺失字段、未 strip 的 `last.pt` 缺字段、stripped numbered checkpoint、错误 run path、错误 args.yaml 或错误内容哈希仍硬失败。完整 127 字段冻结 manifest 与其 SHA256 不变。

## 验证

- `py_compile`：通过。
- ASAC 单元/恶意轨迹回归：37/37 通过。
- 新增测试覆盖：
  - 终态 `last.pt` 仅缺 `save_dir` 时从精确 run path 恢复；
  - 再缺任意第二字段时拒绝；
  - 未 strip `last.pt` 缺 `save_dir` 时拒绝；
  - 五 checkpoint 真实形态的端到端 synthetic panel 通过，且只有 `last.pt` 带 recovery evidence。

冻结文件哈希：

| 文件 | SHA256 |
|---|---|
| `probe_asac_assignment_stability.py` | `b0d764fdb00c54a926e27a0a5b8ffd8f016c626998c75321fc1697f4ed444bf9` |
| `test_probe_asac_assignment_stability.py` | `9f18d0526285a1e400b6842a5d26db33f9e007161cedf429492698733ecce8d6` |

## 运行门禁

这次修订不授权立即占用 GPU。先在 TQGB seed0 正式训练结束后：

1. 远端 staging 哈希与 37/37 CPU tests；
2. 真实五 checkpoint CPU provenance/structure preflight；
3. 只在上述通过后，才考虑单 batch只读 BF16 smoke；
4. 任一步失败均停止，不能继续扩展例外集合。
