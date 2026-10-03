#!/usr/bin/env python3
"""Remove exact duplicate rows from generated YOLO labels and reconcile reports."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test/processed_yolo")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def render_global(global_report):
    lines = [
        "# YOLO训练前准备报告", "",
        f"- 状态：`{global_report['status']}`",
        "- 所有YAML的`val`和`test`均指向各自`images/test`；训练每轮评测test。",
        "- 有原val的数据已合并到train；没有创建val划分。",
        "- 原图使用绝对软链接，原标签不改；YOLO标签及修正规则位于各派生目录。",
        "- 派生YOLO标签中的完全重复行已显式去重，避免依赖运行时临时去重。",
        "", "| 派生数据集 | train图/框 | test图/框 | 类别 |", "|---|---:|---:|---:|",
    ]
    for r in global_report["datasets"]:
        a, b = r["stats"]["train"], r["stats"]["test"]
        lines.append(f"| {r['dataset']} | {a['images']}/{a['boxes']} | {b['images']}/{b['boxes']} | {r['class_count']} |")
    lines += ["", "详细的来源映射、派生修正和注意事项见各目录的`conversion_report.json`与`README.md`。"]
    (ROOT / "PREPARATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global_report = load(ROOT / "PREPARATION_REPORT.json")
    by_name = {r["dataset"]: r for r in global_report["datasets"]}
    audit_by_name = {r["dataset"]: r for r in global_report["structural_audits"]}
    report = {"status": "running", "started": time.time(), "datasets": {}, "errors": []}
    total = 0
    for dataset_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        conversion_path = dataset_dir / "conversion_report.json"
        if not conversion_path.is_file():
            continue
        split_counts = {}
        examples = []
        for split in ("train", "test"):
            removed = 0
            for label_path in sorted((dataset_dir / "labels" / split).glob("*.txt")):
                original = label_path.read_text(encoding="utf-8").splitlines()
                unique = list(dict.fromkeys(original))
                delta = len(original) - len(unique)
                if delta:
                    tmp = label_path.with_suffix(".txt.tmp")
                    tmp.write_text(("\n".join(unique) + "\n") if unique else "", encoding="utf-8")
                    os.replace(tmp, label_path)
                    removed += delta
                    if len(examples) < 30:
                        examples.append({"split": split, "label": label_path.name, "removed": delta})
            split_counts[split] = removed
            total += removed
        conversion = load(conversion_path)
        old_total = int(conversion.get("corrections", {}).get("duplicate_yolo_rows_removed", 0))
        newly_removed = sum(split_counts.values())
        conversion.setdefault("corrections", {})["duplicate_yolo_rows_removed"] = old_total + newly_removed
        conversion["corrections"]["duplicate_yolo_rows_removed_by_split"] = split_counts
        for split, removed in split_counts.items():
            conversion["stats"][split]["boxes"] -= removed
        conversion["status"] = "converted_deduplicated_and_structurally_verified_pending_final_ultralytics_audit"
        save(conversion_path, conversion)
        by_name[dataset_dir.name] = conversion
        if dataset_dir.name in audit_by_name:
            for split, removed in split_counts.items():
                audit_by_name[dataset_dir.name]["splits"][split]["boxes"] -= removed
        readme = (dataset_dir / "README.md").read_text(encoding="utf-8")
        readme = re.sub(r"- train：\d+图，\d+框。", f"- train：{conversion['stats']['train']['images']}图，{conversion['stats']['train']['boxes']}框。", readme)
        readme = re.sub(r"- test：\d+图，\d+框。", f"- test：{conversion['stats']['test']['images']}图，{conversion['stats']['test']['boxes']}框。", readme)
        readme = re.sub(r"- 派生修正：`.*?`。", f"- 派生修正：`{json.dumps(conversion['corrections'], ensure_ascii=False)}`。", readme)
        (dataset_dir / "README.md").write_text(readme, encoding="utf-8")
        report["datasets"][dataset_dir.name] = {"removed_by_split": split_counts, "examples": examples}
        for cache in (dataset_dir / "labels").glob("*.cache"):
            cache.unlink()
    global_report["datasets"] = [by_name[r["dataset"]] for r in global_report["datasets"]]
    global_report["structural_audits"] = [audit_by_name[r["dataset"]] for r in global_report["structural_audits"]]
    global_report["status"] = "all_yolo_derivatives_deduplicated_pending_final_ultralytics_audit"
    global_report["duplicate_yolo_rows_removed"] = total
    global_report["updated"] = time.time()
    save(ROOT / "PREPARATION_REPORT.json", global_report)
    render_global(global_report)
    report["removed_total"] = total
    report["finished"] = time.time()
    report["status"] = "duplicate_rows_removed_and_reports_reconciled"
    save(ROOT / "DEDUPLICATION_REPORT.json", report)
    print(json.dumps({"status": report["status"], "removed_total": total,
                      "datasets": {k: v["removed_by_split"] for k, v in report["datasets"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
