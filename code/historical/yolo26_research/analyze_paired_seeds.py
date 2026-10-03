#!/usr/bin/env python3
"""Aggregate fixed-epoch paired baseline/method results across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def read_csv(path: Path, expected_epochs: int):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError(f"{path}: expected contiguous epochs 1..{expected_epochs}, got {epochs[:3]}..{epochs[-3:]}")
    return rows


def mean(values):
    return sum(values) / len(values)


def fmt(value, signed=False):
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, help="SEED=BASELINE_CSV,METHOD_CSV")
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for spec in args.pair:
        seed, paths = spec.split("=", 1)
        baseline_raw, method_raw = paths.split(",", 1)
        baseline_path, method_path = Path(baseline_raw).resolve(), Path(method_raw).resolve()
        baseline = read_csv(baseline_path, args.expected_epochs)
        method = read_csv(method_path, args.expected_epochs)
        final_base = {name: float(baseline[-1][column]) for name, column in METRICS.items()}
        final_method = {name: float(method[-1][column]) for name, column in METRICS.items()}
        last5_delta = mean(
            [float(method[i][METRICS["map50_95"]]) - float(baseline[i][METRICS["map50_95"]]) for i in range(-5, 0)]
        )
        last10_delta = mean(
            [float(method[i][METRICS["map50_95"]]) - float(baseline[i][METRICS["map50_95"]]) for i in range(-10, 0)]
        )
        records.append(
            {
                "seed": int(seed) if seed.lstrip("-").isdigit() else seed,
                "baseline_csv": str(baseline_path),
                "method_csv": str(method_path),
                "baseline_final": final_base,
                "method_final": final_method,
                "delta_final": {name: final_method[name] - final_base[name] for name in METRICS},
                "delta_last5_map50_95": last5_delta,
                "delta_last10_map50_95": last10_delta,
            }
        )

    deltas = [record["delta_final"]["map50_95"] for record in records]
    summary = {
        "protocol": "paired fixed final epoch; last-5/10 diagnostic only",
        "pairs": records,
        "n": len(records),
        "positive_pairs": sum(value > 0 for value in deltas),
        "mean_delta_map50_95": mean(deltas),
        "sample_std_delta_map50_95": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "min_delta_map50_95": min(deltas),
        "max_delta_map50_95": max(deltas),
        "screening_pass": len(records) >= 3 and sum(value > 0 for value in deltas) >= 2 and mean(deltas) >= 0.003,
        "warning": "Faulty-host exploratory unless every pair is reproduced from scratch on healthy hardware.",
    }
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("non-finite paired delta")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# TQGB 配对多种子固定末轮汇总",
        "",
        "> 只用每个 seed 的固定第 50 轮作为主比较；last-5/10 仅作稳定性诊断。",
        "",
        "| seed | baseline | method | ΔmAP50-95 | Δlast-5 | Δlast-10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {fmt(record['baseline_final']['map50_95'])} | "
            f"{fmt(record['method_final']['map50_95'])} | {fmt(record['delta_final']['map50_95'], True)} | "
            f"{fmt(record['delta_last5_map50_95'], True)} | {fmt(record['delta_last10_map50_95'], True)} |"
        )
    std = summary["sample_std_delta_map50_95"]
    lines += [
        "",
        f"- 正向配对：{summary['positive_pairs']}/{summary['n']}",
        f"- 平均 paired ΔmAP50-95：{fmt(summary['mean_delta_map50_95'], True)}",
        f"- paired Δ 样本标准差：{'—' if std is None else fmt(std)}",
        f"- 预注册筛选结论：{'PASS' if summary['screening_pass'] else 'NOT YET / NO-GO'}",
    ]
    (args.output_dir / "PAIRED_SEED_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
