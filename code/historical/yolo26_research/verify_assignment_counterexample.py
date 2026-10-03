"""Verify that candidate coverage need not survive conflict resolution and topk2 filtering.

This is a synthetic logic check against the exact TaskAlignedAssigner implementation
used by the research checkout. It does not estimate how often the case occurs in data.
"""

from __future__ import annotations

import json

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def main() -> None:
    assigner = TaskAlignedAssigner(topk=7, topk2=1)

    # Both GTs initially accept all seven anchors. GT-A has higher overlap at
    # every shared anchor, even though an injective A->0, B->1 assignment exists.
    initial = torch.ones((1, 2, 7), dtype=torch.float32)
    overlaps = torch.tensor(
        [[[0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
          [0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38]]],
        dtype=torch.float32,
    )
    align_metric = torch.tensor(
        [[[0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65],
          [0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68]]],
        dtype=torch.float32,
    )

    _, foreground, final = assigner.select_highest_overlaps(
        initial.clone(), overlaps, n_max_boxes=2, align_metric=align_metric
    )
    initial_coverage = initial.bool().any(dim=-1)[0].tolist()
    final_coverage = final.bool().any(dim=-1)[0].tolist()
    final_anchor_counts = final.sum(dim=-1)[0].to(torch.int64).tolist()

    report = {
        "topk": assigner.topk,
        "topk2": assigner.topk2,
        "initial_gt_coverage": initial_coverage,
        "final_gt_coverage": final_coverage,
        "final_anchor_counts_per_gt": final_anchor_counts,
        "foreground_anchor_count": int(foreground.sum()),
        "legal_two_gt_injective_assignment_exists": bool(initial[0, 0, 0] and initial[0, 1, 1]),
    }
    assert report["initial_gt_coverage"] == [True, True]
    assert report["final_gt_coverage"] == [True, False]
    assert report["final_anchor_counts_per_gt"] == [1, 0]
    assert report["legal_two_gt_injective_assignment_exists"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
