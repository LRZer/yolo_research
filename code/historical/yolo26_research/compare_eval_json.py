#!/usr/bin/env python3
"""Compare aggregate and per-class metrics from locked checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("Box-P", "Box-R", "mAP50", "mAP75", "mAP50-95")


def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_class = {row["Class"]: row for row in payload.get("per_class", [])}
    if not by_class:
        raise ValueError(f"{path} has no per_class metrics")
    return payload, by_class


def fmt(value, signed=False):
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, base_class = load(args.baseline)
    method, method_class = load(args.method)
    if base_class.keys() != method_class.keys():
        raise ValueError("baseline and method class sets differ")

    rows = []
    for name in base_class:
        b, m = base_class[name], method_class[name]
        rows.append(
            {
                "class": name,
                "instances": int(b["Instances"]),
                **{f"baseline_{key}": float(b[key]) for key in METRICS},
                **{f"method_{key}": float(m[key]) for key in METRICS},
                **{f"delta_{key}": float(m[key]) - float(b[key]) for key in METRICS},
            }
        )

    key = "delta_mAP50-95"
    ranked = sorted(rows, key=lambda row: row[key], reverse=True)
    positive = sum(row[key] > 0 for row in rows)
    negative = sum(row[key] < 0 for row in rows)
    weighted_delta = sum(row[key] * row["instances"] for row in rows) / sum(row["instances"] for row in rows)
    payload = {
        "baseline": base,
        "method": method,
        "aggregate_delta": {
            "precision": method["precision"] - base["precision"],
            "recall": method["recall"] - base["recall"],
            "map50": method["map50"] - base["map50"],
            "map50_95": method["map50_95"] - base["map50_95"],
        },
        "positive_classes": positive,
        "negative_classes": negative,
        "instance_weighted_class_delta_map50_95": weighted_delta,
        "per_class": ranked,
    }

    lines = [
        "# 固定 checkpoint 逐类别比较",
        "",
        f"> baseline: `{args.baseline}`  ",
        f"> method: `{args.method}`",
        "",
        "## 汇总",
        "",
        f"- ΔmAP50-95: {fmt(payload['aggregate_delta']['map50_95'], signed=True)}",
        f"- ΔmAP50: {fmt(payload['aggregate_delta']['map50'], signed=True)}",
        f"- 改善/下降类别数: {positive}/{negative}",
        f"- 按实例数加权的类别 ΔmAP50-95: {fmt(weighted_delta, signed=True)}",
        "",
        "| class | instances | baseline AP | method AP | ΔAP | ΔAP50 | ΔAP75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['class']} | {row['instances']} | {fmt(row['baseline_mAP50-95'])} | "
            f"{fmt(row['method_mAP50-95'])} | {fmt(row['delta_mAP50-95'], signed=True)} | "
            f"{fmt(row['delta_mAP50'], signed=True)} | {fmt(row['delta_mAP75'], signed=True)} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
