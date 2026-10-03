"""Paired, CPU-result-only breakdown by VisDrone GT crowding quartile.

This summarizes existing diagnostic JSONs; it performs no model inference
and never modifies images or labels. The per-image matcher and its caveats
are inherited from analyze_visdrone_prediction_quality.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if any(item.get("images") != 548 or item.get("ground_truth") != 38759 for item in (reference, candidate)):
        raise ValueError("expected matched 548-image / 38,759-GT diagnostics")
    ref_rows = {row["stem"]: row for row in reference["per_image"]}
    cand_rows = {row["stem"]: row for row in candidate["per_image"]}
    if len(ref_rows) != 548 or ref_rows.keys() != cand_rows.keys():
        raise ValueError("image sets do not match")
    for stem in ref_rows:
        if ref_rows[stem]["ground_truth"] != cand_rows[stem]["ground_truth"]:
            raise ValueError(f"GT mismatch in {stem}")
    ordered = sorted(ref_rows, key=lambda stem: (ref_rows[stem]["ground_truth"], stem))
    groups = []
    for index in range(4):
        stems = ordered[index * 137:(index + 1) * 137]
        gt = sum(ref_rows[stem]["ground_truth"] for stem in stems)
        stats = {}
        for name, rows in (("reference", ref_rows), ("candidate", cand_rows)):
            pred = sum(rows[stem]["predictions"] for stem in stems)
            tp50 = sum(rows[stem]["tp_at_0.5"] for stem in stems)
            tp75 = sum(rows[stem]["tp_at_0.75"] for stem in stems)
            stats[name] = {
                "predictions_conf_ge_0.001": pred,
                "tp_iou50": tp50,
                "tp_iou75": tp75,
                "fp_iou50": pred - tp50,
                "recall_iou50": tp50 / gt,
                "recall_iou75": tp75 / gt,
                "images_at_max_det_1000": sum(rows[stem]["predictions"] >= 1000 for stem in stems),
            }
        groups.append({
            "quartile": index + 1,
            "images": 137,
            "min_gt_per_image": ref_rows[stems[0]]["ground_truth"],
            "max_gt_per_image": ref_rows[stems[-1]]["ground_truth"],
            "ground_truth": gt,
            **stats,
            "candidate_minus_reference_tp_iou50": stats["candidate"]["tp_iou50"] - stats["reference"]["tp_iou50"],
            "candidate_minus_reference_fp_iou50": stats["candidate"]["fp_iou50"] - stats["reference"]["fp_iou50"],
        })
    output = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "method": "equal-count GT-crowding quartiles, 137 images each, conf floor 0.001",
        "groups": groups,
        "caveat": "Converted labels omit ignored regions; greedy per-image TP/FP is not official AP. P2/640 versus standard960 differs in architecture and resolution, so these group differences are descriptive, not causal.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
