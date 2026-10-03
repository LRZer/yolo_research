from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # Local lightweight review environments may omit torch.
    torch = None

import probe_final_coverage_audit as fca


class FakeNativeAssigner:
    def __init__(self, s0):
        self.s0 = s0
        self.topk = 2
        self.topk2 = 2

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        return self.s0.clone()

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


def metric(checkpoint, epoch, precision="BF16", gap=120, valid=40000, q50=80, zero=0):
    return {
        "checkpoint": checkpoint,
        "checkpoint_epoch": epoch,
        "precision": precision,
        "gap_r000": gap,
        "gap_r050": q50,
        "valid_gt": valid,
        "gap_rate": gap / valid,
        "r050_retention": q50 / gap if gap else None,
        "zero_quality_gap_units": zero,
        "native_coverage_rematch_infeasible_images": 0,
        "native_final_offgraph_edges": 0,
        "s2_added_edges": 0,
        "s2_added_outside_s0_edges": 0,
    }


class DecisionTests(unittest.TestCase):
    def setUp(self):
        fca.torch = torch

    def _capture_transition(self, s0_values, s1_values, overlap_values):
        s0 = torch.tensor(s0_values, dtype=torch.int8).reshape(1, 3, 1)
        s1 = torch.tensor(s1_values, dtype=torch.int8).reshape(1, 3, 1)
        overlaps = torch.tensor(overlap_values, dtype=torch.float32).reshape(1, 3, 1)
        align = torch.zeros_like(overlaps)
        assigner = FakeNativeAssigner(s0)
        with fca.BoundaryCapture(assigner) as capture:
            assigner.select_candidates_in_gts(None, None, None)
            assigner.select_highest_overlaps(s1, overlaps, 3, align)
        return capture.result()

    @unittest.skipIf(torch is None, "torch is required for native boundary regression")
    def test_native_conflict_can_add_positive_overlap_edge_outside_s1(self):
        capture = self._capture_transition([1, 1, 1], [1, 1, 0], [0.2, 0.3, 0.9])
        self.assertEqual(int(capture["s2_added_counts"].sum()), 1)
        self.assertEqual(int(capture["s1_removed_counts"].sum()), 2)
        self.assertEqual(capture["s2_indices"].tolist(), [[0, 2, 0]])

    @unittest.skipIf(torch is None, "torch is required for native boundary regression")
    def test_zero_overlap_tie_can_add_edge_outside_s0(self):
        capture = self._capture_transition([0, 1, 1], [0, 1, 1], [0.0, 0.0, 0.0])
        self.assertEqual(int(capture["s2_added_counts"].sum()), 1)
        self.assertEqual(int(capture["s2_added_outside_s0_counts"].sum()), 1)
        self.assertEqual(capture["s2_indices"].tolist(), [[0, 0, 0]])

    @unittest.skipIf(torch is None, "torch is required for native boundary regression")
    def test_conflict_winner_already_in_s1_adds_no_edge(self):
        capture = self._capture_transition([1, 1, 1], [1, 1, 0], [0.9, 0.3, 0.8])
        self.assertEqual(int(capture["s2_added_counts"].sum()), 0)
        self.assertEqual(capture["s2_indices"].tolist(), [[0, 0, 0]])

    def test_cluster_bootstrap_is_paired_and_reproducible(self):
        clusters = np.asarray([[0, 6], [1, 12], [2, 18], [0, 24]], dtype=np.int64)
        first = fca.cluster_bootstrap_rates(clusters, 250, 77)
        second = fca.cluster_bootstrap_rates(clusters, 250, 77)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all((first >= 0) & (first <= 1)))

    def test_exact_go_boundaries_pass(self):
        bf16 = [
            metric("epoch0.pt", 1, gap=100, valid=40000, q50=50),
            metric("epoch10.pt", 11, gap=100, valid=40000, q50=50),
            metric("epoch20.pt", 21, gap=100, valid=40000, q50=50),
            metric("epoch30.pt", 31, gap=100, valid=40000, q50=50),
            metric("epoch40.pt", 41, gap=99, valid=40000, q50=99),
            metric("last.pt", 50, gap=100, valid=40000, q50=50),
        ]
        metrics = bf16 + [metric("last.pt", 50, "FP32", gap=100, valid=40000, q50=50)]
        bootstrap = {"ci95_lower": 0.0025, "ci95_upper": 0.0030, "gap": 599, "valid_gt": 240000}
        with mock.patch.object(fca, "bootstrap_pooled", return_value=bootstrap), mock.patch.object(
            fca, "checkpoint_metrics", return_value=metrics
        ):
            decision, _ = fca.make_decision([], [], ["same"] * 7)
        self.assertEqual(decision["verdict"], "GO_TO_OFFLINE_PHASE1")
        self.assertTrue(all(decision["gates"].values()))
        self.assertFalse(decision["dense_tiny_audit_trigger"]["triggered"])
        self.assertIn("primary GO", decision["dense_tiny_audit_trigger"]["quality_note"])

    def test_precision_threshold_crossing_is_inconclusive(self):
        bf16 = [metric(name, epoch) for name, epoch in zip(
            ("epoch0.pt", "epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "last.pt"),
            (1, 11, 21, 31, 41, 50),
        )]
        metrics = bf16 + [metric("last.pt", 50, "FP32", gap=99, valid=40000, q50=70)]
        bootstrap = {"ci95_lower": 0.0025, "ci95_upper": 0.004, "gap": 720, "valid_gt": 240000}
        with mock.patch.object(fca, "bootstrap_pooled", return_value=bootstrap), mock.patch.object(
            fca, "checkpoint_metrics", return_value=metrics
        ):
            decision, _ = fca.make_decision([], [], ["same"] * 7)
        self.assertEqual(decision["verdict"], "PRECISION_SENSITIVE / INCONCLUSIVE")

    def test_clean_no_go_upper_bound(self):
        bf16 = [metric(name, epoch, gap=1, q50=0) for name, epoch in zip(
            ("epoch0.pt", "epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "last.pt"),
            (1, 11, 21, 31, 41, 50),
        )]
        metrics = bf16 + [metric("last.pt", 50, "FP32", gap=1, q50=0)]
        bootstrap = {"ci95_lower": 0.0, "ci95_upper": 0.0001, "gap": 6, "valid_gt": 240000}
        with mock.patch.object(fca, "bootstrap_pooled", return_value=bootstrap), mock.patch.object(
            fca, "checkpoint_metrics", return_value=metrics
        ):
            decision, _ = fca.make_decision([], [], ["same"] * 7)
        self.assertEqual(decision["verdict"], "NO-GO_ON_CLEAN_VOC")

    def test_partial_quality_pass_is_low_quality_no_go(self):
        bf16 = [
            metric("epoch0.pt", 1, q50=80),
            metric("epoch10.pt", 11, q50=80),
            metric("epoch20.pt", 21, q50=80),
            metric("epoch30.pt", 31, q50=40),
            metric("epoch40.pt", 41, q50=40),
            metric("last.pt", 50, q50=40),
        ]
        metrics = bf16 + [metric("last.pt", 50, "FP32", q50=40)]
        bootstrap = {"ci95_lower": 0.0025, "ci95_upper": 0.004, "gap": 720, "valid_gt": 240000}
        with mock.patch.object(fca, "bootstrap_pooled", return_value=bootstrap), mock.patch.object(
            fca, "checkpoint_metrics", return_value=metrics
        ):
            decision, _ = fca.make_decision([], [], ["same"] * 7)
        self.assertEqual(decision["verdict"], "LOW_QUALITY / NO-GO")
        self.assertEqual(len(decision["quality_passes"]), 3)

    def test_native_identity_rematch_failure_blocks_go(self):
        bf16 = [metric(name, epoch) for name, epoch in zip(
            ("epoch0.pt", "epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "last.pt"),
            (1, 11, 21, 31, 41, 50),
        )]
        bf16[2]["native_coverage_rematch_infeasible_images"] = 1
        metrics = bf16 + [metric("last.pt", 50, "FP32")]
        bootstrap = {"ci95_lower": 0.003, "ci95_upper": 0.004, "gap": 720, "valid_gt": 240000}
        with mock.patch.object(fca, "bootstrap_pooled", return_value=bootstrap), mock.patch.object(
            fca, "checkpoint_metrics", return_value=metrics
        ):
            decision, _ = fca.make_decision([], [], ["same"] * 7)
        self.assertFalse(decision["go"])
        self.assertEqual(decision["verdict"], "INCONCLUSIVE_NATIVE_COVERAGE_NOT_REMATCHABLE")
        self.assertIsNone(decision["pooled_zero_quality_gap_share"])

    def test_dense_trigger_rejects_cross_witness_ratio_above_one(self):
        malformed = [{
            "checkpoint": "epoch0.pt",
            "precision": "BF16",
            "head": "O2O",
            "group_type": "size_x_competition",
            "group_value": "tiny|dense",
            "recoverable_gap_gt": 1,
            "valid_gt": 100,
            "recoverable_r_ge_050_share": 2.0,
        }]
        with self.assertRaises(RuntimeError):
            fca.dense_tiny_trigger(malformed)


if __name__ == "__main__":
    unittest.main()
