#!/usr/bin/env python3
"""Analyze a growing Ultralytics YOLO26 diagnostic results.csv.

The script intentionally uses only the Python standard library. It reads a byte
snapshot of results.csv, ignores an incomplete trailing row (possible while the
trainer is appending), and writes a compact Markdown report plus machine-readable
JSON and checkpoint CSV files.

Example:
    python analyze_d0_diagnostics.py \
        --run-dir /path/to/d0_run \
        --output-dir /path/to/outputs/yolo26_research \
        --prefix D0_DIAGNOSTIC
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


CHECKPOINTS = (1, 5, 10, 25, 50, 100, 150)
METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
}
DIAGNOSTIC_NAMES = (
    "same_gt_rate",
    "teacher_better",
    "teacher_cls_better",
    "kd_active",
    "quality_gap",
    "head_box_iou",
    "prog_o2m_w",
    "prog_o2o_w",
    "kd_scale",
)
PERCENT_LIKE = {
    "same_gt_rate",
    "teacher_better",
    "teacher_cls_better",
    "kd_active",
    "head_box_iou",
    "prog_o2m_w",
    "prog_o2o_w",
}


@dataclass(frozen=True)
class CsvSnapshot:
    headers: list[str]
    rows: list[dict[str, float]]
    skipped_rows: int
    duplicate_epochs: int


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    number = as_float(text)
    if number is not None:
        return int(number) if number.is_integer() else number
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    return text


def read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read top-level scalar YAML keys without requiring PyYAML."""
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z0-9_./()-]+", key):
            result[key] = parse_scalar(value.split(" #", 1)[0])
    return result


def read_csv_snapshot(path: Path) -> CsvSnapshot:
    """Read one stable-enough snapshot, tolerating a concurrently appended tail."""
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found: {path}")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        headers = [item.strip() for item in next(reader)]
    except StopIteration as exc:
        raise ValueError(f"results.csv is empty: {path}") from exc
    if not headers or "epoch" not in headers:
        raise ValueError(f"results.csv has no epoch column: {path}")

    by_epoch: dict[int, dict[str, float]] = {}
    skipped_rows = 0
    duplicate_epochs = 0
    for fields in reader:
        if not fields or all(not item.strip() for item in fields):
            continue
        if len(fields) != len(headers):
            skipped_rows += 1
            continue
        raw_row = dict(zip(headers, fields))
        epoch_value = as_float(raw_row.get("epoch"))
        if epoch_value is None or epoch_value < 0:
            skipped_rows += 1
            continue
        epoch = int(round(epoch_value))
        parsed: dict[str, float] = {"epoch": float(epoch)}
        for key, value in raw_row.items():
            number = as_float(value)
            if number is not None:
                parsed[key] = number
        if epoch in by_epoch:
            duplicate_epochs += 1
        by_epoch[epoch] = parsed
    rows = [by_epoch[key] for key in sorted(by_epoch)]
    if not rows:
        raise ValueError(f"results.csv contains no complete numeric epoch rows: {path}")
    return CsvSnapshot(headers, rows, skipped_rows, duplicate_epochs)


def choose_column(headers: Sequence[str], aliases: Sequence[str]) -> str | None:
    available = set(headers)
    return next((name for name in aliases if name in available), None)


def paired_values(rows: Iterable[dict[str, float]], x_col: str, y_col: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x, y = row.get(x_col), row.get(y_col)
        if finite(x) and finite(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 1e-15:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def regression_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-15:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def median_for(rows: Sequence[dict[str, float]], col: str) -> float | None:
    values = [float(row[col]) for row in rows if finite(row.get(col))]
    return statistics.median(values) if values else None


def last_n_valid(rows: Sequence[dict[str, float]], col: str, count: int = 10) -> list[dict[str, float]]:
    valid = [row for row in rows if finite(row.get(col))]
    return valid[-count:]


def metric_summary(rows: Sequence[dict[str, float]], col: str) -> dict[str, Any] | None:
    valid = [row for row in rows if finite(row.get(col))]
    if not valid:
        return None
    latest = valid[-1]
    best = max(valid, key=lambda row: (float(row[col]), float(row["epoch"])))
    return {
        "column": col,
        "latest": float(latest[col]),
        "latest_epoch": int(latest["epoch"]),
        "best": float(best[col]),
        "best_epoch": int(best["epoch"]),
    }


def diagnostic_trend(
    rows: Sequence[dict[str, float]], col: str, map_col: str | None
) -> dict[str, Any] | None:
    valid = [row for row in rows if finite(row.get(col))]
    if not valid:
        return None
    first, latest = valid[0], valid[-1]
    epochs = [float(row["epoch"]) for row in valid]
    values = [float(row[col]) for row in valid]
    recent = valid[-min(10, len(valid)) :]
    recent_epochs = [float(row["epoch"]) for row in recent]
    recent_values = [float(row[col]) for row in recent]
    corr_map = None
    if map_col:
        map_values, diag_values = paired_values(valid, map_col, col)
        corr_map = pearson(map_values, diag_values)
    return {
        "column": col,
        "count": len(valid),
        "first_epoch": int(first["epoch"]),
        "first": float(first[col]),
        "latest_epoch": int(latest["epoch"]),
        "latest": float(latest[col]),
        "delta": float(latest[col]) - float(first[col]),
        "slope_per_epoch": regression_slope(epochs, values),
        "recent_window": len(recent),
        "recent_median": statistics.median(recent_values),
        "recent_slope_per_epoch": regression_slope(recent_epochs, recent_values),
        "corr_epoch": pearson(epochs, values),
        "corr_map50_95": corr_map,
    }


def find_mature_rows(rows: Sequence[dict[str, float]], args: dict[str, Any]) -> list[dict[str, float]]:
    warmup = as_float(args.get("o2o_kd_warmup")) or 0.0
    ramp = as_float(args.get("o2o_kd_ramp")) or 0.0
    planned = as_float(args.get("epochs"))
    fade = as_float(args.get("o2o_kd_fade")) or 0.0
    start = max(1, int(math.ceil(warmup + ramp)))
    end = int(planned - fade) if planned is not None and planned > fade else math.inf
    mature = [row for row in rows if start <= row["epoch"] <= end]
    return mature or list(rows[-min(10, len(rows)) :])


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def threshold_diagnostics(rows: Sequence[dict[str, float]], args: dict[str, Any]) -> dict[str, Any]:
    q0 = as_float(args.get("o2o_kd_min_quality"))
    margin0 = as_float(args.get("o2o_kd_margin"))
    gate = args.get("o2o_kd_gate")
    mature = find_mature_rows(rows, args)
    active_train = median_for(mature, "train/kd_active")
    active_val = median_for(mature, "val/kd_active")
    gap_train = median_for(mature, "train/quality_gap")
    teacher_train = median_for(mature, "train/teacher_better")
    cls_train = median_for(mature, "train/teacher_cls_better")

    result: dict[str, Any] = {
        "scope": "diagnostic screening only; aggregate epoch logs cannot identify optimal thresholds",
        "gate_enabled": gate,
        "observed_epoch_start": int(mature[0]["epoch"]) if mature else None,
        "observed_epoch_end": int(mature[-1]["epoch"]) if mature else None,
        "observed_epoch_count": len(mature),
        "mature_train_kd_active_median": active_train,
        "mature_val_kd_active_median": active_val,
        "mature_train_quality_gap_median": gap_train,
        "mature_train_teacher_better_median": teacher_train,
        "mature_train_teacher_cls_better_median": cls_train,
        "candidates": [],
        "screening_order": [],
    }
    if gate is False:
        result["note"] = "门控已关闭，因此本次运行不能诊断 quality/margin 阈值。"
        return result
    if q0 is None or margin0 is None:
        result["note"] = "args.yaml 中没有找到当前 quality/margin 阈值。"
        return result

    q_step = 0.05
    m_step = 0.025
    # Change one threshold at a time. A diagonal relaxed/strict pair would confound
    # min_quality with margin and make a small ablation hard to interpret.
    candidates = {
        "anchor": {"min_quality": round(q0, 4), "margin": round(margin0, 4)},
        "quality_relaxed": {
            "min_quality": round(clamp(q0 - q_step, 0.0, 0.95), 4),
            "margin": round(margin0, 4),
        },
        "quality_strict": {
            "min_quality": round(clamp(q0 + q_step, 0.0, 0.95), 4),
            "margin": round(margin0, 4),
        },
        "margin_relaxed": {
            "min_quality": round(q0, 4),
            "margin": round(clamp(margin0 - m_step, 0.0, 0.5), 4),
        },
        "margin_strict": {
            "min_quality": round(q0, 4),
            "margin": round(clamp(margin0 + m_step, 0.0, 0.5), 4),
        },
    }
    if active_train is None:
        order = ["anchor", "margin_relaxed", "margin_strict", "quality_relaxed", "quality_strict"]
        rationale = "还没有成熟阶段的 train/kd_active；仅保留围绕锚点的对称单变量候选。"
    elif active_train < 0.05:
        order = ["margin_relaxed", "quality_relaxed", "anchor", "margin_strict", "quality_strict"]
        rationale = "成熟阶段激活率低于 5%，优先诊断放宽 margin，其次放宽 min_quality。"
    elif active_train > 0.40:
        order = ["margin_strict", "quality_strict", "anchor", "margin_relaxed", "quality_relaxed"]
        rationale = "成熟阶段激活率高于 40%，优先诊断收紧 margin，其次收紧 min_quality。"
    else:
        order = ["anchor", "margin_strict", "margin_relaxed", "quality_strict", "quality_relaxed"]
        rationale = "成熟阶段激活率处于 5–40% 的诊断带，先保留锚点，再对 margin 和 min_quality 分别做单变量包围。"
    candidate_order = ("quality_relaxed", "anchor", "quality_strict", "margin_relaxed", "margin_strict")
    result["candidates"] = [{"label": label, **candidates[label]} for label in candidate_order]
    result["screening_order"] = order
    result["rationale"] = ("当前成熟窗口少于 5 个 epoch，以下顺序仅属早期预判；" if len(mature) < 5 else "") + rationale
    result["note"] = (
        "这些数值只是局部包围，不是估计出的最优值。正式选择前仍需候选级 quality/advantage 分布和受控实验；"
        "一次只改变一个阈值，避免归因混乱。"
    )
    return result


def fmt(value: Any, digits: int = 4) -> str:
    return "—" if not finite(value) else f"{float(value):.{digits}f}"


def fmt_percent(value: Any, digits: int = 1) -> str:
    return "—" if not finite(value) else f"{100 * float(value):.{digits}f}%"


def fmt_diag(name: str, value: Any) -> str:
    return fmt_percent(value) if name in PERCENT_LIKE else fmt(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rendered.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(rendered)


def compact_config(args: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "epochs",
        "batch",
        "device",
        "seed",
        "pretrained",
        "nms",
        "o2o_kd_gain",
        "o2o_kd_temp",
        "o2o_kd_min_quality",
        "o2o_kd_margin",
        "o2o_kd_warmup",
        "o2o_kd_ramp",
        "o2o_kd_fade",
        "o2o_kd_gate",
    )
    return {key: args[key] for key in keys if key in args}


def make_markdown(summary: dict[str, Any]) -> str:
    meta = summary["metadata"]
    metrics = summary["metrics"]
    complete = meta["run_complete"]
    latest_word = "最终" if complete else "当前快照"
    lines = [
        "# D0 YOLO26 双头诊断报告",
        "",
        f"> 快照时间：{meta['generated_at']}；已记录 {meta['row_count']} 个完整 epoch，最新为 epoch {meta['latest_epoch']}。",
        "",
        "## 结论边界",
        "",
        (
            "本报告分析的是 `nms=false` 的 O2O 无 NMS 指标。"
            if summary["config"].get("nms") is False
            else "请先核对本次验证是否设置为 `nms=false`；否则指标可能不是 O2O 无 NMS结果。"
        ),
        (
            "`o2o_kd_gain=0`，因此门控统计只是观察量，没有改变模型优化。"
            if as_float(summary["config"].get("o2o_kd_gain")) == 0
            else "本次 `o2o_kd_gain` 非零，诊断量和训练优化相互影响，不能视为纯观察基线。"
        ),
    ]
    if not complete:
        if meta["planned_epochs"] is None:
            lines.extend(["", "args.yaml 中没有计划轮数；“最好值”仅指当前快照，不能自动判定为最终结果。"])
        else:
            lines.extend(["", f"训练尚未达到计划的 {meta['planned_epochs']} 轮；“最好值”仅指当前已完成轮次，不能当作最终结果。"])

    metric_rows = []
    labels = {"precision": "Precision", "recall": "Recall", "map50": "mAP50", "map50_95": "mAP50-95"}
    for key in ("precision", "recall", "map50", "map50_95"):
        item = metrics.get(key)
        if item:
            metric_rows.append([labels[key], fmt(item["latest"]), item["latest_epoch"], fmt(item["best"]), item["best_epoch"]])
    lines.extend(["", f"## {latest_word}与当前最好 O2O 指标", "", md_table(["指标", latest_word, "轮次", "当前最好", "轮次"], metric_rows)])

    cp_rows = []
    for row in summary["checkpoints"]:
        cp_rows.append(
            [
                row["epoch"],
                fmt(row.get("precision")),
                fmt(row.get("recall")),
                fmt(row.get("map50")),
                fmt(row.get("map50_95")),
                fmt_percent(row.get("train_same_gt_rate")),
                fmt_percent(row.get("train_teacher_better")),
                fmt_percent(row.get("train_teacher_cls_better")),
                fmt_percent(row.get("train_kd_active")),
                fmt(row.get("train_quality_gap")),
                fmt_percent(row.get("train_head_box_iou")),
                f"{fmt(row.get('train_prog_o2m_w'), 3)}/{fmt(row.get('train_prog_o2o_w'), 3)}",
            ]
        )
    lines.extend(
        [
            "",
            "## 关键轮次",
            "",
            "只列出已经真实出现的预设轮次；未完成的轮次不会用邻近值代替。",
            "",
            md_table(
                ["epoch", "P", "R", "AP50", "AP50-95", "同 GT", "教师更优", "教师分类更优", "门控激活", "质量差", "双头框 IoU", "O2M/O2O 权重"],
                cp_rows,
            ),
        ]
    )

    trend_rows = []
    for item in summary["diagnostic_trends"]:
        name = item["name"]
        trend_rows.append(
            [
                item["split"],
                name,
                fmt_diag(name, item["first"]),
                fmt_diag(name, item["latest"]),
                fmt_diag(name, item["delta"]),
                fmt(item["slope_per_epoch"], 6),
                fmt_diag(name, item["recent_median"]),
                fmt(item["corr_epoch"], 3),
                fmt(item["corr_map50_95"], 3),
            ]
        )
    lines.extend(
        [
            "",
            "## 双头诊断趋势",
            "",
            md_table(["split", "量", "首值", "最新", "变化", "斜率/epoch", "最近中位数", "r(epoch)", "r(AP50-95)"], trend_rows),
            "",
            "相关系数只用于发现伴随变化；训练时间同时推动损失、双头关系与 AP，不能据此宣称因果。",
        ]
    )

    checks = summary["checks"]
    if checks.get("val_schedule_appears_stale"):
        lines.extend(
            [
                "",
                "### 验证侧调度列告警",
                "",
                f"`val/prog_o2m_w` 与 `val/prog_o2o_w` 分别固定在约 {fmt(checks.get('val_prog_o2m_latest'), 4)} / {fmt(checks.get('val_prog_o2o_latest'), 4)}，`val/kd_scale` 固定为 {fmt(checks.get('val_kd_scale_latest'), 4)}，与此同时训练侧权重正在按 epoch 变化。",
                "",
                "这表明 EMA 验证副本的 criterion epoch/schedule 状态似乎没有同步。不要把 `val/*` 的 ProgLoss 权重或 KD scale 当作当前轮次的调度值，调度分析只采用 `train/*`。本次 O2O Precision/Recall/mAP 检测指标不受这组日志字段影响。",
            ]
        )

    threshold = summary["threshold_diagnostics"]
    candidate_rows = [
        [item["label"], fmt(item["min_quality"], 3), fmt(item["margin"], 3)] for item in threshold.get("candidates", [])
    ]
    lines.extend(
        [
            "",
            "## 门控阈值的诊断性筛选建议",
            "",
            f"成熟窗口：epoch {threshold.get('observed_epoch_start', '—')}–{threshold.get('observed_epoch_end', '—')}（{threshold.get('observed_epoch_count', 0)} 个 epoch）；训练门控激活中位数 {fmt_percent(threshold.get('mature_train_kd_active_median'))}，质量差中位数 {fmt(threshold.get('mature_train_quality_gap_median'))}。",
            "",
        ]
    )
    if candidate_rows:
        lines.extend(
            [
                md_table(["候选", "min_quality", "margin"], candidate_rows),
                "",
                "建议诊断顺序：" + " → ".join(threshold.get("screening_order", [])) + "。" + threshold.get("rationale", ""),
            ]
        )
    lines.extend(["", threshold.get("note", "")])

    lines.extend(
        [
            "",
            "## 完整性检查",
            "",
            f"- 读取完整行：{meta['row_count']}；跳过不完整/无效行：{meta['skipped_rows']}；重复 epoch（保留最后一行）：{meta['duplicate_epochs']}。",
            f"- ProgLoss 训练权重和最大偏离 1：{fmt(checks.get('max_train_prog_weight_sum_error'), 6)}。",
            f"- O2M 权重单调不增：{checks.get('train_o2m_weight_nonincreasing', '—')}；O2O 权重单调不减：{checks.get('train_o2o_weight_nondecreasing', '—')}。",
            f"- 验证侧调度状态疑似未同步：{checks.get('val_schedule_appears_stale', '—')}（只解释日志字段，不影响本次 O2O 检测指标）。",
            "- JSON 保留原始小数；Markdown 中百分比仅为易读展示。",
        ]
    )
    display_warnings = [warning for warning in summary["warnings"] if not warning.startswith("val/prog_o2m_w")]
    if display_warnings:
        lines.extend(["", "### 其他警告", ""] + [f"- {warning}" for warning in display_warnings])
    return "\n".join(lines).rstrip() + "\n"


def monotonic(values: Sequence[float], increasing: bool, tolerance: float = 1e-7) -> bool | None:
    if len(values) < 2:
        return None
    pairs = zip(values, values[1:])
    return all(b >= a - tolerance for a, b in pairs) if increasing else all(b <= a + tolerance for a, b in pairs)


def numeric_range(values: Sequence[float]) -> float | None:
    return max(values) - min(values) if values else None


def build_summary(run_dir: Path, snapshot: CsvSnapshot, args: dict[str, Any]) -> dict[str, Any]:
    rows = snapshot.rows
    headers = snapshot.headers
    columns = {key: choose_column(headers, aliases) for key, aliases in METRIC_ALIASES.items()}
    planned_value = as_float(args.get("epochs"))
    planned_epochs = int(planned_value) if planned_value else None
    latest_epoch = int(rows[-1]["epoch"])
    run_complete = planned_epochs is not None and latest_epoch >= planned_epochs

    warnings: list[str] = []
    if snapshot.skipped_rows:
        warnings.append("检测到不完整或非数值行；这通常是训练器正在追加 CSV，报告已安全忽略这些行。")
    for key, col in columns.items():
        if col is None:
            warnings.append(f"缺少指标列 {key}。")
    if args.get("nms") is not False:
        warnings.append("args.yaml 未确认 nms=false，不能自动保证这些指标是 O2O 无 NMS结果。")

    metrics = {key: metric_summary(rows, col) for key, col in columns.items() if col}
    checkpoint_rows: list[dict[str, Any]] = []
    row_by_epoch = {int(row["epoch"]): row for row in rows}
    for epoch in CHECKPOINTS:
        if epoch not in row_by_epoch:
            continue
        row = row_by_epoch[epoch]
        item: dict[str, Any] = {"epoch": epoch}
        for key, col in columns.items():
            item[key] = row.get(col) if col else None
        for name in DIAGNOSTIC_NAMES:
            item[f"train_{name}"] = row.get(f"train/{name}")
            item[f"val_{name}"] = row.get(f"val/{name}")
        checkpoint_rows.append(item)

    trends: list[dict[str, Any]] = []
    map_col = columns.get("map50_95")
    for split in ("train", "val"):
        for name in DIAGNOSTIC_NAMES:
            col = f"{split}/{name}"
            if col not in headers:
                continue
            trend = diagnostic_trend(rows, col, map_col)
            if trend:
                trends.append({"split": split, "name": name, **trend})

    o2m_values = [float(row["train/prog_o2m_w"]) for row in rows if finite(row.get("train/prog_o2m_w"))]
    o2o_values = [float(row["train/prog_o2o_w"]) for row in rows if finite(row.get("train/prog_o2o_w"))]
    val_o2m_values = [float(row["val/prog_o2m_w"]) for row in rows if finite(row.get("val/prog_o2m_w"))]
    val_o2o_values = [float(row["val/prog_o2o_w"]) for row in rows if finite(row.get("val/prog_o2o_w"))]
    train_kd_scales = [float(row["train/kd_scale"]) for row in rows if finite(row.get("train/kd_scale"))]
    val_kd_scales = [float(row["val/kd_scale"]) for row in rows if finite(row.get("val/kd_scale"))]
    sums = [
        abs(float(row["train/prog_o2m_w"]) + float(row["train/prog_o2o_w"]) - 1.0)
        for row in rows
        if finite(row.get("train/prog_o2m_w")) and finite(row.get("train/prog_o2o_w"))
    ]
    checks = {
        "max_train_prog_weight_sum_error": max(sums) if sums else None,
        "train_o2m_weight_nonincreasing": monotonic(o2m_values, increasing=False),
        "train_o2o_weight_nondecreasing": monotonic(o2o_values, increasing=True),
        "train_prog_o2m_range": numeric_range(o2m_values),
        "train_prog_o2o_range": numeric_range(o2o_values),
        "val_prog_o2m_range": numeric_range(val_o2m_values),
        "val_prog_o2o_range": numeric_range(val_o2o_values),
        "val_prog_o2m_latest": val_o2m_values[-1] if val_o2m_values else None,
        "val_prog_o2o_latest": val_o2o_values[-1] if val_o2o_values else None,
        "train_kd_scale_range": numeric_range(train_kd_scales),
        "val_kd_scale_range": numeric_range(val_kd_scales),
        "val_kd_scale_latest": val_kd_scales[-1] if val_kd_scales else None,
        "val_kd_scale_all_zero": bool(val_kd_scales) and all(abs(value) <= 1e-9 for value in val_kd_scales),
    }
    checks["val_schedule_appears_stale"] = bool(
        numeric_range(o2m_values)
        and numeric_range(o2m_values) > 1e-4
        and numeric_range(val_o2m_values) is not None
        and numeric_range(val_o2m_values) <= 1e-9
        and checks["val_kd_scale_all_zero"]
    )
    if checks["val_schedule_appears_stale"]:
        warnings.append(
            "val/prog_o2m_w 与 val/prog_o2o_w 固定在约 0.8/0.2，且 val/kd_scale 始终为 0；"
            "EMA 验证 criterion 的 epoch/schedule 状态似乎没有同步。不要把 val 侧调度列解释为当前 epoch，"
            "调度轨迹以 train/* 为准；本次 O2O 检测指标不受此日志状态影响。"
        )

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "run_dir": str(run_dir.resolve()),
            "results_csv": str((run_dir / "results.csv").resolve()),
            "row_count": len(rows),
            "latest_epoch": latest_epoch,
            "planned_epochs": planned_epochs,
            "run_complete": run_complete,
            "skipped_rows": snapshot.skipped_rows,
            "duplicate_epochs": snapshot.duplicate_epochs,
        },
        "config": compact_config(args),
        "metric_columns": columns,
        "metrics": metrics,
        "checkpoints": checkpoint_rows,
        "diagnostic_trends": trends,
        "threshold_diagnostics": threshold_diagnostics(rows, args),
        "checks": checks,
        "warnings": warnings,
    }


def write_checkpoint_csv(path: Path, checkpoints: Sequence[dict[str, Any]]) -> None:
    base = ["epoch", "precision", "recall", "map50", "map50_95"]
    diagnostics = [f"{split}_{name}" for split in ("train", "val") for name in DIAGNOSTIC_NAMES]
    fields = base + diagnostics
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in checkpoints:
            writer.writerow({key: row.get(key) for key in fields})


def sanitize_prefix(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return result or "YOLO26_DIAGNOSTIC"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Ultralytics run directory containing results.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Markdown/CSV/JSON outputs")
    parser.add_argument("--prefix", default="D0_DIAGNOSTIC", help="Output filename prefix")
    parsed = parser.parse_args()

    run_dir = parsed.run_dir.resolve()
    output_dir = parsed.output_dir.resolve()
    prefix = sanitize_prefix(parsed.prefix)
    snapshot = read_csv_snapshot(run_dir / "results.csv")
    train_args = read_simple_yaml(run_dir / "args.yaml")
    summary = build_summary(run_dir, snapshot, train_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{prefix}_REPORT.md"
    json_path = output_dir / f"{prefix}_SUMMARY.json"
    csv_path = output_dir / f"{prefix}_CHECKPOINTS.csv"
    report_path.write_text(make_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_checkpoint_csv(csv_path, summary["checkpoints"])
    print(json.dumps({"report": str(report_path), "json": str(json_path), "csv": str(csv_path), **summary["metadata"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
