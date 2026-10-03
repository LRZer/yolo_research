#!/usr/bin/env python3
"""Cold-check and publish the final YOLO training-readiness report."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import ultralytics
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test/processed_yolo")
LOGGER.setLevel(logging.ERROR)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    prep_path = ROOT / "PREPARATION_REPORT.json"
    validation_path = ROOT / "ULTRALYTICS_VALIDATION.json"
    prep = load(prep_path)
    uv = load(validation_path)
    errors = []
    if uv.get("status") != "all_datasets_loaded_by_ultralytics" or uv.get("errors"):
        errors.append("final Ultralytics validation did not pass cleanly")
    prep_by = {d["dataset"]: d for d in prep["datasets"]}
    uv_by = {d["dataset"]: d for d in uv["datasets"]}
    if set(prep_by) != set(uv_by) or len(prep_by) != 11:
        errors.append(f"dataset coverage mismatch: prep={sorted(prep_by)} uv={sorted(uv_by)}")
    pipeline = []
    for name in sorted(prep_by):
        d = ROOT / name
        yaml_path = d / "dataset.yaml"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if cfg.get("val") != "images/test" or cfg.get("test") != "images/test":
            errors.append(f"{name}: val/test policy mismatch")
        data = check_det_dataset(str(yaml_path), autodownload=False)
        item = {"dataset": name, "splits": {}}
        for split in ("train", "test"):
            expected = prep_by[name]["stats"][split]
            actual = uv_by[name]["splits"][split]
            for key in ("images", "boxes"):
                actual_key = f"{key}_loaded"
                if int(expected[key]) != int(actual[actual_key]):
                    errors.append(f"{name}/{split}: {key} report={expected[key]} ultralytics={actual[actual_key]}")
            ds = YOLODataset(img_path=data[split], data=data, task="detect", imgsz=640,
                             augment=False, cache=False, prefix="", batch_size=1, stride=32)
            sample = ds[0]
            shape = list(sample["img"].shape)
            if shape != [3, 640, 640]:
                errors.append(f"{name}/{split}: unexpected sample tensor shape {shape}")
            item["splits"][split] = {
                "image_tensor_shape": shape,
                "boxes_in_sample": int(len(sample["cls"])),
                "resized_shape": list(sample["resized_shape"]),
            }
        pipeline.append(item)
    duplicate_warnings = 0
    final_log = Path("/home/ubuntu/rz/object_detection_dataset_test/_management/ULTRALYTICS_VALIDATION_FINAL_20260921.log")
    if final_log.is_file():
        duplicate_warnings = final_log.read_text(encoding="utf-8", errors="replace").count("duplicate labels removed")
    if duplicate_warnings:
        errors.append(f"final Ultralytics scan emitted {duplicate_warnings} duplicate-label warnings")
    report = {
        "status": "ready_for_yolo_training_test_used_as_val" if not errors else "failed",
        "finished": time.time(),
        "root": str(ROOT),
        "policy": "train+val -> train when available; val and test YAML entries both point to test",
        "datasets": len(prep_by),
        "partitions_loaded": 2 * len(prep_by),
        "duplicate_yolo_rows_removed": prep.get("duplicate_yolo_rows_removed"),
        "duplicate_warnings_in_final_scan": duplicate_warnings,
        "pipeline_samples": pipeline,
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "gpu_used": False,
        "training_started": False,
        "raw_files_modified": False,
        "errors": errors,
    }
    save(ROOT / "FINAL_READINESS.json", report)
    if errors:
        raise SystemExit("; ".join(errors))
    for name, d in prep_by.items():
        cp = ROOT / name / "conversion_report.json"
        conv = load(cp)
        conv["status"] = "ready_for_yolo_training_test_used_as_val"
        conv["final_ultralytics_validation"] = "passed"
        save(cp, conv)
    prep["status"] = "ready_for_yolo_training_test_used_as_val"
    prep["final_ultralytics_validation"] = {
        "status": uv["status"], "datasets": len(uv["datasets"]),
        "partitions": 2 * len(uv["datasets"]), "duplicate_warnings": duplicate_warnings,
        "pipeline_sample_status": "all_22_splits_pipeline_sample_passed",
    }
    prep["finished_readiness_audit"] = time.time()
    save(prep_path, prep)
    lines = [
        "# YOLO训练就绪最终报告", "",
        "## 状态", "",
        "- `ready_for_yolo_training_test_used_as_val`",
        "- 11个派生版本、22个train/test分区均由当前Ultralytics全量读取通过。",
        "- 每个YAML的`val`与`test`均指向`images/test`，因此训练每轮评测test。",
        "- 有原val的数据已并入train；未另建val；未处理相邻帧泄漏。",
        "- 原始数据未修改；原图以绝对软链接复用；本次未启动训练、未使用GPU。",
        f"- 派生标签显式移除完全重复行：{prep.get('duplicate_yolo_rows_removed', 0)}；最终扫描重复警告：{duplicate_warnings}。",
        "", "## 规模", "",
        "| 数据集 | train图/框 | test图/框 | 类别 | 主要派生处理 |", "|---|---:|---:|---:|---|",
    ]
    for name in sorted(prep_by):
        d = prep_by[name]; a=d["stats"]["train"]; b=d["stats"]["test"]
        corr = json.dumps(d.get("corrections", {}), ensure_ascii=False)
        lines.append(f"| {name} | {a['images']}/{a['boxes']} | {b['images']}/{b['boxes']} | {d['class_count']} | `{corr}` |")
    lines += [
        "", "## 使用入口", "",
        "- 每个派生目录中的`dataset.yaml`可直接用于YOLO检测训练。",
        "- 训练命令模板见`TRAINING_COMMANDS.md`；模板限定后四张GPU，但本次没有执行。",
        "- 每个目录的`source_map.tsv`可将派生文件追溯到原图，`conversion_report.json`记录转换政策。",
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "FINAL_READINESS_REPORT.md").write_text(text, encoding="utf-8")
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "datasets": len(prep_by),
                      "partitions": 2 * len(prep_by), "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
