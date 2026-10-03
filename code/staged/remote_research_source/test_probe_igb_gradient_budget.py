"""Synthetic CPU tests for the isolated IGB-P0 probe.

These tests deliberately do not import Ultralytics or touch a checkpoint.  They
exercise the mathematical seams used by the future CUDA read-only run.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


SCRIPT = Path(__file__).with_name("probe_igb_gradient_budget.py")
SPEC = importlib.util.spec_from_file_location("probe_igb_gradient_budget", SCRIPT)
igb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(igb)


class FakeBBoxLoss:
    """Small additive differentiable analogue of the native bbox-loss interface."""

    def __call__(
        self,
        pred_dist,
        pred_bboxes,
        _anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        _imgsz,
        _stride,
    ):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        box = ((pred_bboxes[fg_mask] - target_bboxes[fg_mask]).square().mean(-1, keepdim=True) * weight).sum()
        distribution = pred_dist[fg_mask].square().mean(-1, keepdim=True)
        distribution = (distribution * weight).sum()
        return box / target_scores_sum, distribution / target_scores_sum


class FakeCriterion:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = FakeBBoxLoss()
        self.hyp = SimpleNamespace(box=2.0, cls=0.5, dfl=1.5)
        self.class_weights = None


class TestPerGTDecomposition(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = [torch.tensor(value, requires_grad=True) for value in (0.2, -0.3, 0.4)]
        f3, f4, f5 = self.features
        pred_scores = torch.stack((torch.stack((f3, f4)), torch.stack((f5, f3 + f4)))).unsqueeze(0)
        pred_distri = torch.stack(
            (
                torch.stack((f3, f4, f5, f3 + f5)),
                torch.stack((f4, f5, f3, f4 + f5)),
            )
        ).unsqueeze(0)
        pred_bboxes = torch.stack(
            (torch.stack((f3, f4, f3 + 1, f4 + 1)), torch.stack((f4, f5, f4 + 1, f5 + 1)))
        ).unsqueeze(0)
        self.tensors = {
            "pred_scores": pred_scores,
            "pred_distri": pred_distri,
            "pred_bboxes": pred_bboxes,
            "anchor_points": torch.zeros(2, 2),
            "stride_tensor": torch.ones(2, 1),
            "imgsz": torch.tensor((10.0, 10.0)),
        }
        self.assignment = {
            "fg_mask": torch.tensor([[True, True]]),
            "target_gt_idx": torch.tensor([[0, 1]]),
            "target_labels": torch.tensor([[0, 1]]),
            "target_scores": torch.tensor([[[0.5, 0.0], [0.0, 0.8]]]),
            "target_bboxes": torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.2, 0.1, 1.2, 1.1]]]),
        }
        self.criterion = FakeCriterion()

    def test_per_gt_terms_sum_to_union_and_have_exact_individual_vjps(self):
        masks = [igb.mask_for_gt(self.assignment, 0, index) for index in (0, 1)]
        terms = [
            igb.native_positive_terms_for_mask(self.criterion, self.tensors, self.assignment, mask)
            for mask in masks
        ]
        union = igb.native_positive_terms_for_mask(
            self.criterion, self.tensors, self.assignment, masks[0] | masks[1]
        )
        for component in ("box", "cls", "dfl_or_l1", "total"):
            self.assertTrue(torch.allclose(terms[0][component] + terms[1][component], union[component], atol=1e-7))

        norms0, joint0 = igb.feature_gradient_norms(terms[0]["total"], self.features, retain_graph=True)
        norms1, joint1 = igb.feature_gradient_norms(terms[1]["total"], self.features, retain_graph=True)
        self.assertAlmostEqual(joint0, math.sqrt(sum(value * value for value in norms0)), places=10)
        self.assertAlmostEqual(joint1, math.sqrt(sum(value * value for value in norms1)), places=10)
        self.assertNotAlmostEqual(joint0, joint1)
        self.assertTrue(all(feature.grad is None for feature in self.features))

    def test_level_mass_count_area_quality_diagnostics(self):
        mask = igb.mask_for_gt(self.assignment, 0, 0)
        values = igb.gt_diagnostics(
            self.tensors, self.assignment, mask, torch.tensor([0, 2], dtype=torch.long)
        )
        self.assertEqual(values["positive_count"], 1)
        self.assertAlmostEqual(values["target_mass"], 0.5)
        self.assertEqual(values["p3_positive_count"], 1)
        self.assertEqual(values["p4_positive_count"], 0)
        self.assertEqual(values["p5_positive_count"], 0)
        self.assertAlmostEqual(values["area"], 0.01)
        self.assertGreaterEqual(values["matched_quality_mean"], 0.0)
        self.assertLessEqual(values["matched_quality_mean"], 1.0)


class TinyHead(nn.Module):
    def forward(self, features):
        return sum((index + 2) * feature.square().sum() for index, feature in enumerate(features))


class TestBoundarySemantics(unittest.TestCase):
    def test_equal_value_detach_preserves_local_feature_vjp(self):
        head = TinyHead()
        source = torch.tensor(0.7, requires_grad=True)
        original = [source * 3, source * -2, source + 1]
        reference_loss = head(original)
        reference = torch.autograd.grad(reference_loss, original)

        capture = igb.DetectBoundaryCapture(head)
        try:
            cut_loss = head(original)
            self.assertIsNotNone(capture.features)
            captured = list(capture.features)
            cut = torch.autograd.grad(cut_loss, capture.features)
        finally:
            capture.close()
        for expected, actual, value, leaf in zip(reference, cut, original, captured):
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(value.detach(), leaf.detach()))
            self.assertIsNot(value, leaf)
        self.assertIsNone(source.grad)


class TestStatisticsAndDecision(unittest.TestCase):
    def test_statistics_with_ties(self):
        self.assertAlmostEqual(igb.gini([0, 0, 10, 10]), 0.5)
        self.assertAlmostEqual(igb.top_fraction_share([1, 2, 3, 4, 10]), 0.5)
        effective, fraction = igb.effective_number([1, 1, 1, 1])
        self.assertEqual(effective, 4.0)
        self.assertEqual(fraction, 1.0)
        self.assertAlmostEqual(igb.spearman([1, 1, 2, 3], [2, 2, 4, 6]), 1.0)
        self.assertIsNone(igb.spearman([1, 1], [2, 3]))

    @staticmethod
    def passing_checkpoint(epoch):
        return {
            "completed_epoch": epoch,
            "processed_batches": 24,
            "selected_gt": 128,
            "matched_selected_gt": 120,
            "assignment_coverage": 120 / 128,
            "batches_with_two_matched_gt": 24,
            "batch_ginis": [
                {"batch_index": index, "matched_gt": 4, "gini": 0.40} for index in range(24)
            ],
            "top20_share": 0.50,
            "spearman_gradient_vs": {"target_mass": 0.50},
        }

    def test_all_four_gates_are_conjunctive(self):
        checkpoints = [self.passing_checkpoint(epoch) for epoch in (11, 21, 31, 41, 50)]
        decision = igb.make_decision(checkpoints)
        self.assertEqual(decision["verdict"], "GO")
        self.assertTrue(all(decision["checks"].values()))

        checkpoints[2] = {**checkpoints[2], "top20_share": 0.10}
        checkpoints[3] = {**checkpoints[3], "top20_share": 0.10}
        checkpoints[4] = {**checkpoints[4], "top20_share": 0.10}
        decision = igb.make_decision(checkpoints)
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertFalse(decision["checks"]["median_checkpoint_top20_share"])

    def test_missing_or_nonfinite_mass_correlation_is_no_go(self):
        checkpoints = [self.passing_checkpoint(epoch) for epoch in (11, 21, 31, 41, 50)]
        for checkpoint in checkpoints[2:]:
            checkpoint["spearman_gradient_vs"]["target_mass"] = None
        decision = igb.make_decision(checkpoints)
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertFalse(decision["checks"]["positive_rho_checkpoint_count"])

    def test_incomplete_batches_or_gt_coverage_cannot_go(self):
        checkpoints = [self.passing_checkpoint(epoch) for epoch in (11, 21, 31, 41, 50)]
        checkpoints[0]["processed_batches"] = 1
        checkpoints[1]["matched_selected_gt"] = 20
        checkpoints[1]["assignment_coverage"] = 20 / 128
        decision = igb.make_decision(checkpoints)
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertFalse(decision["protocol_checks"]["all_24_batches_processed"])
        self.assertFalse(decision["protocol_checks"]["matched_gt_sufficient"])


if __name__ == "__main__":
    unittest.main()
