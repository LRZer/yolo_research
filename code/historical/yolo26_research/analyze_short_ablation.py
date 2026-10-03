#!/usr/bin/env python3
"""Summarize the preregistered 50-epoch B50/D1/D2 screen without best-epoch cherry-picking."""

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
    "candidate_rate": "train/kd_candidate_rate",
    "effective_rate": "train/kd_effective_rate",
    "grad_direction_agree": "train/kd_grad_direction_agree",
    "teacher_closer": "train/kd_teacher_closer",
    "teacher_better": "train/teacher_better",
    "kd_scale": "train/kd_scale",
    "kd_loss": "train/o2o_kd_loss",
}
KEY_EPOCHS = (1, 5, 10, 15, 20, 30, 40, 50)


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def read_run(spec: str) -> tuple[str, Path, list[dict[str, str]]]:
    if "=" not in spec:
        raise ValueError(f"Run must be NAME=CSV, got: {spec}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"Epoch sequence is not contiguous 1..N in {path}: {epochs[:3]}...{epochs[-3:]}")
    return name, path, rows


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def fmt(value: float | None, digits: int = 5, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def summarize(name: str, path: Path, rows: list[dict[str, str]]) -> dict:
    final = rows[-1]
    final_metrics = {label: number(final.get(column)) for label, column in METRICS.items()}
    best = {}
    for label, column in METRICS.items():
        candidates = [(number(row.get(column)), int(float(row["epoch"]))) for row in rows]
        candidates = [(value, epoch) for value, epoch in candidates if value is not None]
        value, epoch = max(candidates) if candidates else (None, None)
        best[label] = {"value": value, "epoch": epoch}

    scheduled_rows = [row for row in rows if (number(row.get(DIAGNOSTICS["kd_scale"])) or 0.0) > 0]
    scheduled_means = {
        label: mean([number(row.get(column)) for row in scheduled_rows])
        for label, column in DIAGNOSTICS.items()
    }
    curve = {}
    for epoch in KEY_EPOCHS:
        if epoch <= len(rows):
            row = rows[epoch - 1]
            curve[str(epoch)] = {
                **{label: number(row.get(column)) for label, column in METRICS.items()},
                **{label: number(row.get(column)) for label, column in DIAGNOSTICS.items()},
            }
    return {
        "name": name,
        "source": str(path),
        "epochs": len(rows),
        "elapsed_seconds": number(final.get("time")),
        "final": final_metrics,
        "best_exploratory_only": best,
        "scheduled_kd_window_mean": scheduled_means,
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/results.csv")
    parser.add_argument("--baseline", required=True, help="Name supplied in --run")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = [summarize(*read_run(spec)) for spec in args.run]
    by_name = {item["name"]: item for item in summaries}
    if args.baseline not in by_name:
        raise ValueError(f"Baseline {args.baseline!r} was not supplied")
    baseline = by_name[args.baseline]
    for item in summaries:
        item["delta_vs_baseline"] = {
            label: (item["final"][label] - baseline["final"][label])
            if item["final"][label] is not None and baseline["final"][label] is not None
            else None
            for label in METRICS
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_rule": "fixed final epoch only",
        "warning": "Single-seed 50-epoch screen; not a publication-level conclusion.",
        "baseline": args.baseline,
        "runs": summaries,
    }
    (args.output_dir / "short_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = args.output_dir / "short_ablation_final.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["run", "epochs", "elapsed_seconds"]
        fields += list(METRICS)
        fields += [f"delta_{name}" for name in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "run": item["name"],
                    "epochs": item["epochs"],
                    "elapsed_seconds": item["elapsed_seconds"],
                    **item["final"],
                    **{f"delta_{key}": value for key, value in item["delta_vs_baseline"].items()},
                }
            )

    lines = [
        "# B50 / D1 / D2 预注册短训汇总",
        "",
        "> 主比较严格采用固定第 50 轮；最佳轮次仅保存在 JSON 中供诊断，不用于选择结论。单种子 50 轮只是方向筛选，不是论文级证据。",
        "",
        "| 实验 | epochs | P | R | mAP50 | mAP50-95 | ΔmAP50 | ΔmAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        final, delta = item["final"], item["delta_vs_baseline"]
        lines.append(
            f"| {item['name']} | {item['epochs']} | {fmt(final['precision'])} | {fmt(final['recall'])} | "
            f"{fmt(final['map50'])} | {fmt(final['map50_95'])} | {fmt(delta['map50'], signed=True)} | "
            f"{fmt(delta['map50_95'], signed=True)} |"
        )

    lines += ["", "## 预定 KD 时间窗内的机制均值", ""]
    lines += [
        "| 实验 | 候选率 | 有效率 | 与原监督方向一致 | 教师更接近 target | 教师更优 | KD scale | KD loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        diag = item["scheduled_kd_window_mean"]
        lines.append(
            f"| {item['name']} | {fmt(diag['candidate_rate'])} | {fmt(diag['effective_rate'])} | "
            f"{fmt(diag['grad_direction_agree'])} | {fmt(diag['teacher_closer'])} | "
            f"{fmt(diag['teacher_better'])} | {fmt(diag['kd_scale'])} | {fmt(diag['kd_loss'])} |"
        )
    (args.output_dir / "short_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
