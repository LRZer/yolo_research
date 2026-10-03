#!/usr/bin/env python3
"""Deterministic ASAC Phase-1a comparison from frozen run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            {key.strip(): float(value) for key, value in row.items()}
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]
    if len(rows) != 50 or [int(row["epoch"]) for row in rows] != list(range(1, 51)):
        raise ValueError(f"{path}: expected exact epochs 1..50, got {len(rows)} rows")
    for column in ("time", *METRICS.values()):
        if any(column not in row or not math.isfinite(row[column]) for row in rows):
            raise ValueError(f"{path}: missing or non-finite {column}")
    return rows


def read_state(path: Path, expected_variant: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50 or [row["epoch"] for row in rows] != list(range(1, 51)):
        raise ValueError(f"{path}: expected exact state epochs 1..50, got {len(rows)} rows")
    if {row["variant"] for row in rows} != {expected_variant}:
        raise ValueError(f"{path}: unexpected variant")
    return rows


def window(rows: list[dict[str, float]], column: str, count: int) -> float:
    return mean(row[column] for row in rows[-count:])


def curve_mean(rows: list[dict[str, float]], column: str) -> float:
    return mean(row[column] for row in rows)


def common_constant(rows: list[dict], key: str):
    values = {json.dumps(row[key], sort_keys=True) for row in rows}
    if len(values) != 1:
        raise ValueError(f"state field {key} changed: {values}")
    return rows[0][key]


def state_summary(rows: list[dict]) -> dict:
    q = [float(row["train_q"]) for row in rows]
    coverages = [float(row[key]) for row in rows for key in ("o2m_coverage", "o2o_coverage")]
    oom = [
        bool(flag)
        for row in rows
        for flag in row["panel_assigner_oom_fallback"].values()
    ]
    for key in (
        "panel_content_sha256",
        "population_sha256",
        "population_count",
        "measurement_source_sha256",
        "trainer_source_sha256",
        "source_bundle_sha256",
        "config_sha256",
    ):
        common_constant(rows, key)
    return {
        "q_min": min(q),
        "q_max": max(q),
        "q_range": max(q) - min(q),
        "q_mean": mean(q),
        "q_at_upper_bound_epochs": sum(math.isclose(value, 1.0, abs_tol=1e-12) for value in q),
        "q_at_lower_bound_epochs": sum(math.isclose(value, 0.25, abs_tol=1e-12) for value in q),
        "first_q_at_upper_bound_epoch": next(
            (row["epoch"] for row in rows if math.isclose(float(row["train_q"]), 1.0, abs_tol=1e-12)),
            None,
        ),
        "min_coverage": min(coverages),
        "oom_fallback_any": any(oom),
        "panel_content_sha256": rows[0]["panel_content_sha256"],
        "population_sha256": rows[0]["population_sha256"],
        "population_count": rows[0]["population_count"],
        "config_sha256": rows[0]["config_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-results", type=Path, required=True)
    parser.add_argument("--a2-results", type=Path, required=True)
    parser.add_argument("--b50-results", type=Path, required=True)
    parser.add_argument("--a0-state", type=Path, required=True)
    parser.add_argument("--a2-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_paths = {"A0": args.a0_results, "A2": args.a2_results, "official_B50": args.b50_results}
    runs = {name: read_csv(path) for name, path in result_paths.items()}
    states = {
        "A0": read_state(args.a0_state, "A0"),
        "A2": read_state(args.a2_state, "A2"),
    }

    summaries: dict[str, dict] = {}
    for name, rows in runs.items():
        summaries[name] = {
            "epoch50": {label: rows[-1][column] for label, column in METRICS.items()},
            "last5_mean": {label: window(rows, column, 5) for label, column in METRICS.items()},
            "last10_mean": {label: window(rows, column, 10) for label, column in METRICS.items()},
            "curve_mean_50": {label: curve_mean(rows, column) for label, column in METRICS.items()},
            "wall_seconds": rows[-1]["time"],
            "sha256": sha256(result_paths[name]),
        }

    def delta(left: str, right: str, section: str, metric: str = "mAP50-95") -> float:
        return summaries[left][section][metric] - summaries[right][section][metric]

    comparisons = {}
    for reference in ("A0", "official_B50"):
        comparisons[f"A2_minus_{reference}"] = {
            "epoch50": {label: delta("A2", reference, "epoch50", label) for label in METRICS},
            "last5_mean_mAP50-95": delta("A2", reference, "last5_mean"),
            "last10_mean_mAP50-95": delta("A2", reference, "last10_mean"),
            "curve_mean_50_mAP50-95": delta("A2", reference, "curve_mean_50"),
            "epoch_wins_mAP50-95": sum(
                a[METRICS["mAP50-95"]] > b[METRICS["mAP50-95"]]
                for a, b in zip(runs["A2"], runs[reference], strict=True)
            ),
        }

    a0_overhead = summaries["A0"]["wall_seconds"] / summaries["official_B50"]["wall_seconds"] - 1.0
    a2_overhead_paired = summaries["A2"]["wall_seconds"] / summaries["A0"]["wall_seconds"] - 1.0
    state_stats = {name: state_summary(rows) for name, rows in states.items()}
    paired = comparisons["A2_minus_A0"]
    gates = {
        "epoch50_delta_ge_0.003": paired["epoch50"]["mAP50-95"] >= 0.003,
        "last5_delta_positive": paired["last5_mean_mAP50-95"] > 0.0,
        "q_dynamic_range_positive": state_stats["A2"]["q_range"] > 0.0,
        "paired_wall_overhead_lt_12pct": a2_overhead_paired < 0.12,
        "integrity_state_rows_and_coverage": (
            state_stats["A0"]["min_coverage"] >= 0.95
            and state_stats["A2"]["min_coverage"] >= 0.95
            and not state_stats["A0"]["oom_fallback_any"]
            and not state_stats["A2"]["oom_fallback_any"]
        ),
        "a0_q_fixed_at_1": all(math.isclose(float(row["train_q"]), 1.0, abs_tol=1e-12) for row in states["A0"]),
    }
    gates["phase1a_go"] = all(gates.values())

    output = {
        "schema": "ASAC-Phase1a-final-comparison-v1",
        "primary_metric": "fixed epoch-50 VOC mAP50-95",
        "summaries": summaries,
        "comparisons": comparisons,
        "overhead": {
            "A2_vs_paired_A0": a2_overhead_paired,
            "paired_A0_vs_official_B50": a0_overhead,
        },
        "state": state_stats,
        "gates": gates,
        "artifact_sha256": {
            "A0_state": sha256(args.a0_state),
            "A2_state": sha256(args.a2_state),
        },
        "interpretation_boundary": (
            "A0 is the paired ASAC-infrastructure fixed-q=1 control; official_B50 is an external reference, "
            "because A0 diverges numerically from official_B50 after epoch 1."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
