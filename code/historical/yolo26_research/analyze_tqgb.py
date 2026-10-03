#!/usr/bin/env python3
"""Analyze fixed-epoch TQGB ablations without best-epoch selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
DIAGNOSTICS = {
    "m2o_rate": "train/tqgb_cls_m2o_rate",
    "o2m_rate": "train/tqgb_cls_o2m_rate",
    "m2o_keep": "train/tqgb_cls_m2o_projection_keep",
    "o2m_keep": "train/tqgb_cls_o2m_projection_keep",
    "m2o_raw_loss": "train/tqgb_cls_m2o_raw_loss",
    "o2m_raw_loss": "train/tqgb_cls_o2m_raw_loss",
    "m2o_scaled_loss": "train/tqgb_cls_m2o_scaled_loss",
    "o2m_scaled_loss": "train/tqgb_cls_o2m_scaled_loss",
    "route_loss": "train/tqgb_cls_loss",
}
EQUIVALENCE_FIELDS = (
    "train/o2o_box_loss",
    "train/o2o_cls_loss",
    "train/o2o_l1_loss",
    "train/o2m_box_loss",
    "train/o2m_cls_loss",
    "train/o2m_l1_loss",
    *METRICS.values(),
)
PHASES = ((1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50))


def number(value):
    try:
        parsed = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def mean(values):
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def fmt(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def read_run(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=CSV, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    if not rows or epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: epochs are not contiguous 1..N")
    return name, path, rows


def phase(rows, baseline, start, end):
    stop = min(end, len(rows), len(baseline))
    if start > stop:
        return None
    indices = range(start - 1, stop)
    method = [number(rows[i].get(METRICS["map50_95"])) for i in indices]
    base = [number(baseline[i].get(METRICS["map50_95"])) for i in indices]
    return {
        "epochs": f"{start}-{stop}",
        "map50_95_mean": mean(method),
        "delta_vs_baseline_mean": mean(
            [left - right for left, right in zip(method, base) if left is not None and right is not None]
        ),
        **{
            f"{label}_mean": mean([number(rows[i].get(column)) for i in indices])
            for label, column in DIAGNOSTICS.items()
        },
    }


def summarize(name, path, rows, baseline):
    final = {label: number(rows[-1].get(column)) for label, column in METRICS.items()}
    base_final = {label: number(baseline[len(rows) - 1].get(column)) for label, column in METRICS.items()}
    phases = [value for start, end in PHASES if (value := phase(rows, baseline, start, end)) is not None]
    rolling = {
        f"last_{window}": phase(rows, baseline, max(1, len(rows) - window + 1), len(rows))
        for window in (5, 10)
    }
    prefix = {}
    count = min(5, len(rows), len(baseline))
    for column in EQUIVALENCE_FIELDS:
        prefix[column] = max(
            abs(number(rows[i].get(column)) - number(baseline[i].get(column))) for i in range(count)
        )
    return {
        "name": name,
        "source": str(path),
        "epochs": len(rows),
        "final": final,
        "delta_vs_baseline": {label: final[label] - base_final[label] for label in METRICS},
        "first_five_max_abs_diff_vs_baseline": prefix,
        "phases": phases,
        "rolling": rolling,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/results.csv")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    loaded = [read_run(spec) for spec in args.run]
    by_name = {name: rows for name, _, rows in loaded}
    if args.baseline not in by_name:
        raise ValueError(f"Baseline {args.baseline!r} was not supplied")
    baseline = by_name[args.baseline]
    if len(baseline) < args.expected_epochs:
        raise ValueError("Baseline is shorter than expected")
    summaries = []
    for name, path, rows in loaded:
        if len(rows) != args.expected_epochs:
            raise ValueError(f"{name} has {len(rows)} epochs, expected {args.expected_epochs}")
        summaries.append(summarize(name, path, rows, baseline))

    payload = {
        "primary_rule": "fixed final epoch only",
        "warning": "Single-seed faulty-host exploratory screening; not publication-level evidence.",
        "baseline": args.baseline,
        "runs": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tqgb_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# TQGB 固定末轮消融汇总",
        "",
        "> 主比较只使用固定末轮；阶段与末段均值只作机制和稳定性诊断。",
        "",
        "## 固定末轮",
        "",
        "| 实验 | P | R | mAP50 | mAP50-95 | ΔmAP50-95 vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        final, delta = item["final"], item["delta_vs_baseline"]
        lines.append(
            f"| {item['name']} | {fmt(final['precision'])} | {fmt(final['recall'])} | "
            f"{fmt(final['map50'])} | {fmt(final['map50_95'])} | {fmt(delta['map50_95'], signed=True)} |"
        )

    for item in summaries:
        if item["name"] == args.baseline:
            continue
        lines += [
            "",
            f"## {item['name']} 阶段诊断",
            "",
            "| epochs | mAP50-95 | Δ vs baseline | m2o rate | o2m rate | m2o keep | o2m keep | route loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in item["phases"]:
            lines.append(
                f"| {row['epochs']} | {fmt(row['map50_95_mean'])} | "
                f"{fmt(row['delta_vs_baseline_mean'], signed=True)} | {fmt(row['m2o_rate_mean'])} | "
                f"{fmt(row['o2m_rate_mean'])} | {fmt(row['m2o_keep_mean'])} | "
                f"{fmt(row['o2m_keep_mean'])} | {fmt(row['route_loss_mean'])} |"
            )
        lines += ["", "| 窗口 | epochs | mAP50-95 | Δ vs baseline |", "|---|---:|---:|---:|"]
        for label, row in item["rolling"].items():
            lines.append(
                f"| {label.replace('_', '-')} | {row['epochs']} | {fmt(row['map50_95_mean'])} | "
                f"{fmt(row['delta_vs_baseline_mean'], signed=True)} |"
            )
    (args.output_dir / "TQGB_FINAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
