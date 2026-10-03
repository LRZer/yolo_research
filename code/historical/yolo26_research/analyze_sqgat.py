#!/usr/bin/env python3
"""Analyze a fixed-epoch S-QGAT run against preregistered controls.

The primary comparison is always the final epoch.  Intermediate epochs and
best values are diagnostic only and are never used to select the conclusion.
"""

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
CONTROLLER = {
    "schedule": "train/kd_schedule_scale",
    "state": "train/kd_state_scale",
    "reliability": "train/kd_reliability",
    "reliability_ref": "train/kd_reliability_ref",
    "combined_scale": "train/kd_scale",
    "kd_loss": "train/o2o_kd_loss",
    "candidate_rate": "train/kd_candidate_rate",
    "effective_rate": "train/kd_effective_rate",
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


def finite(value: str | float | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if not rows or epochs != expected:
        raise ValueError(f"{path}: expected contiguous epochs 1..N, got {epochs[:3]}...{epochs[-3:]}")
    return rows


def values(row: dict[str, str], columns: dict[str, str]) -> dict[str, float | None]:
    return {name: finite(row.get(column)) for name, column in columns.items()}


def delta(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def controller_value(row: dict[str, str], name: str, column: str) -> float | None:
    """Treat negative logging sentinels as missing controller measurements."""
    value = finite(row.get(column))
    if name in {"reliability", "reliability_ref"} and value is not None and value < 0:
        return None
    return value


def phase_summary(
    start: int,
    end: int,
    method: list[dict[str, str]],
    baseline: list[dict[str, str]],
    control: list[dict[str, str]] | None,
) -> dict:
    stop = min(end, len(method), len(baseline), len(control) if control else len(method))
    if start > stop:
        return {"epochs": f"{start}-{end}", "count": 0}
    indices = range(start - 1, stop)
    method_map = [finite(method[i].get(METRICS["map50_95"])) for i in indices]
    base_map = [finite(baseline[i].get(METRICS["map50_95"])) for i in indices]
    control_map = (
        [finite(control[i].get(METRICS["map50_95"])) for i in indices] if control else None
    )
    return {
        "epochs": f"{start}-{stop}",
        "count": stop - start + 1,
        "method_map50_95_mean": mean(method_map),
        "delta_vs_baseline_mean": mean([delta(x, y) for x, y in zip(method_map, base_map)]),
        "delta_vs_control_mean": (
            mean([delta(x, y) for x, y in zip(method_map, control_map)]) if control_map else None
        ),
        **{
            f"{name}_mean": mean([controller_value(method[i], name, column) for i in indices])
            for name, column in CONTROLLER.items()
        },
    }


def final_summary(rows: list[dict[str, str]]) -> dict[str, float | None]:
    return values(rows[-1], METRICS)


def max_prefix_difference(
    left: list[dict[str, str]], right: list[dict[str, str]], epochs: int
) -> dict[str, float | None]:
    count = min(epochs, len(left), len(right))
    result: dict[str, float | None] = {}
    for column in EQUIVALENCE_FIELDS:
        differences = [
            abs(delta(finite(left[i].get(column)), finite(right[i].get(column))) or 0.0)
            for i in range(count)
        ]
        result[column] = max(differences) if differences else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--control-name", default="strength-matched control")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=50)
    args = parser.parse_args()

    method = read_csv(args.method.resolve())
    baseline = read_csv(args.baseline.resolve())
    control = read_csv(args.control.resolve()) if args.control else None
    if len(method) != args.expected_epochs:
        raise ValueError(f"Method has {len(method)} epochs; expected {args.expected_epochs}")
    if len(baseline) < len(method) or (control is not None and len(control) < len(method)):
        raise ValueError("Controls must contain at least as many epochs as the method")

    method_final = final_summary(method)
    baseline_final = final_summary(baseline[: len(method)])
    control_final = final_summary(control[: len(method)]) if control else None
    phases = [phase_summary(start, end, method, baseline, control) for start, end in PHASES]
    rolling = {
        f"last_{window}": phase_summary(max(1, len(method) - window + 1), len(method), method, baseline, control)
        for window in (5, 10)
    }
    state_values = [finite(row.get(CONTROLLER["state"])) for row in method]
    reliability_values = [
        controller_value(row, "reliability", CONTROLLER["reliability"]) for row in method
    ]
    first_state_below_one = next(
        (i + 1 for i, value in enumerate(state_values) if value is not None and value < 1.0 - 1e-12),
        None,
    )
    first_retired = next(
        (i + 1 for i, value in enumerate(state_values) if value is not None and value <= 1e-12), None
    )

    payload = {
        "decision_rule": "fixed final epoch only; intermediate phases are diagnostic",
        "warning": "Single-seed faulty-host exploratory result; publication claims require healthy-hardware paired seeds.",
        "sources": {
            "method": str(args.method.resolve()),
            "baseline": str(args.baseline.resolve()),
            "control": str(args.control.resolve()) if args.control else None,
        },
        "epochs": len(method),
        "final": {
            "method": method_final,
            "baseline": baseline_final,
            "delta_vs_baseline": {
                name: delta(method_final[name], baseline_final[name]) for name in METRICS
            },
            "control": control_final,
            "delta_vs_control": (
                {name: delta(method_final[name], control_final[name]) for name in METRICS}
                if control_final
                else None
            ),
        },
        "controller": {
            "first_state_below_one_epoch": first_state_below_one,
            "first_retired_epoch": first_retired,
            "minimum_state": min(value for value in state_values if value is not None),
            "final_state": state_values[-1],
            "final_reliability": reliability_values[-1],
            "final_reliability_ref": finite(method[-1].get(CONTROLLER["reliability_ref"])),
        },
        "warmup_prefix_max_abs_diff_vs_baseline": max_prefix_difference(method, baseline, 5),
        "phases": phases,
        "rolling": rolling,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sqgat_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "sqgat_phases.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(dict.fromkeys(key for row in phases for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(phases)

    final = payload["final"]
    lines = [
        "# S-QGAT 固定第 50 轮分析",
        "",
        "> 主结论只看固定末轮；阶段曲线仅用于机制诊断。该结果来自单种子故障主机探索，不能直接作为论文最终数值。",
        "",
        "## 固定末轮",
        "",
        "| 实验 | P | R | mAP50 | mAP50-95 | ΔmAP50-95 vs B50 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| B50 | {fmt(baseline_final['precision'])} | {fmt(baseline_final['recall'])} | {fmt(baseline_final['map50'])} | {fmt(baseline_final['map50_95'])} | +0.00000 |",
        f"| S-QGAT | {fmt(method_final['precision'])} | {fmt(method_final['recall'])} | {fmt(method_final['map50'])} | {fmt(method_final['map50_95'])} | {fmt(final['delta_vs_baseline']['map50_95'], signed=True)} |",
    ]
    if control_final:
        lines.append(
            f"| {args.control_name} | {fmt(control_final['precision'])} | {fmt(control_final['recall'])} | "
            f"{fmt(control_final['map50'])} | {fmt(control_final['map50_95'])} | "
            f"{fmt(delta(control_final['map50_95'], baseline_final['map50_95']), signed=True)} |"
        )
    lines += [
        "",
        "## 状态控制器",
        "",
        f"- 首次收缩：epoch {first_state_below_one if first_state_below_one else '未发生'}",
        f"- 首次完全退休：epoch {first_retired if first_retired else '未发生'}",
        f"- 最小/末轮 state scale：{fmt(payload['controller']['minimum_state'])} / {fmt(payload['controller']['final_state'])}",
        f"- 末轮 reliability / reference：{fmt(payload['controller']['final_reliability'])} / {fmt(payload['controller']['final_reliability_ref'])}",
        "",
        "## 固定阶段均值",
        "",
        "| epochs | S-QGAT mAP50-95 | Δ vs B50 | Δ vs control | state | reliability | KD scale | KD loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in phases:
        if not phase.get("count"):
            continue
        lines.append(
            f"| {phase['epochs']} | {fmt(phase['method_map50_95_mean'])} | "
            f"{fmt(phase['delta_vs_baseline_mean'], signed=True)} | "
            f"{fmt(phase['delta_vs_control_mean'], signed=True)} | "
            f"{fmt(phase['state_mean'])} | {fmt(phase['reliability_mean'])} | "
            f"{fmt(phase['combined_scale_mean'])} | {fmt(phase['kd_loss_mean'])} |"
        )
    lines += [
        "",
        "## 末段稳定性（补充，不替代固定末轮）",
        "",
        "| 窗口 | epochs | S-QGAT mAP50-95 | Δ vs B50 | Δ vs control |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, window in rolling.items():
        lines.append(
            f"| {label.replace('_', '-')} | {window['epochs']} | {fmt(window['method_map50_95_mean'])} | "
            f"{fmt(window['delta_vs_baseline_mean'], signed=True)} | "
            f"{fmt(window['delta_vs_control_mean'], signed=True)} |"
        )
    (args.output_dir / "S_QGAT_FINAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
