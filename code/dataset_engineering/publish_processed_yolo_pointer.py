#!/usr/bin/env python3
"""Append an idempotent processed-YOLO readiness pointer to the dataset root README."""

import os
from pathlib import Path


root = Path("/home/ubuntu/rz/object_detection_dataset_test")
path = root / "README.md"
marker = "<!-- PROCESSED_YOLO_READY_20260921 -->"
block = f"""

{marker}
## YOLO训练派生版本

- 目录：`{root}/processed_yolo`
- 最终状态：`ready_for_yolo_training_test_used_as_val`
- 11个派生版本、22个train/test分区均由当前Ultralytics全量读取通过。
- 每个`dataset.yaml`的`val`和`test`都指向`images/test`；有原val的数据已并入train。
- 原始数据未修改；训练尚未启动。详见`processed_yolo/FINAL_READINESS_REPORT.md`。
"""
text = path.read_text(encoding="utf-8")
if marker not in text:
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text.rstrip() + block + "\n", encoding="utf-8")
    os.replace(tmp, path)
print("pointer_present", marker in path.read_text(encoding="utf-8"))
