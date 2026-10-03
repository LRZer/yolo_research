"""Audit deterministic CSV reproduction while ignoring wall-clock time.

The script compares the candidate prefix against the matching baseline epochs,
reports exact/non-exact numeric fields, and preserves schema additions separately.
It never treats a shorter candidate as a completed reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def numeric(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--ignore", nargs="*", default=("time",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_fields, base = read_rows(args.baseline)
    cand_fields, cand = read_rows(args.candidate)
    if not cand or len(cand) > len(base):
        raise ValueError("candidate must contain a non-empty prefix no longer than baseline")

    ignore = set(args.ignore)
    common = [field for field in base_fields if field in cand_fields and field not in ignore]
    differences: list[dict[str, object]] = []
    comparable_values = 0
    max_abs = 0.0
    for row_index, (left, right) in enumerate(zip(base, cand), start=1):
        for field in common:
            a, b = numeric(left[field]), numeric(right[field])
            if a is None or b is None:
                if left[field] != right[field]:
                    differences.append(
                        {"epoch_row": row_index, "field": field, "baseline": left[field], "candidate": right[field]}
                    )
                continue
            comparable_values += 1
            delta = abs(a - b)
            max_abs = max(max_abs, delta)
            if delta != 0.0:
                differences.append(
                    {
                        "epoch_row": row_index,
                        "field": field,
                        "baseline": a,
                        "candidate": b,
                        "abs_difference": delta,
                    }
                )

    report = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "expected_epochs": args.expected_epochs,
        "baseline_rows": len(base),
        "candidate_rows": len(cand),
        "complete": len(cand) == args.expected_epochs,
        "ignored_fields": sorted(ignore),
        "common_compared_fields": common,
        "candidate_only_fields": sorted(set(cand_fields) - set(base_fields)),
        "baseline_only_fields": sorted(set(base_fields) - set(cand_fields)),
        "comparable_numeric_values": comparable_values,
        "nonzero_difference_count": len(differences),
        "max_abs_difference": max_abs,
        "exact_prefix_reproduction": not differences,
        "exact_complete_reproduction": len(cand) == args.expected_epochs and not differences,
        "differences": differences[:100],
        "candidate_final": cand[-1],
        "baseline_matching_epoch": base[len(cand) - 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "candidate_rows", "complete", "nonzero_difference_count", "max_abs_difference",
        "exact_prefix_reproduction", "exact_complete_reproduction"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
