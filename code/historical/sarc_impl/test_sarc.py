"""CPU synthetic verification for the independent SARC staging module."""

from __future__ import annotations

import json
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarc import (
    AssignmentHistogramEMA,
    SARCPyramidStaging,
    SARCStagingBlock,
    SingleDW7StagingBlock,
    apply_bn_to_fused_conv,
    assignment_relative_scale,
    count_trainable_parameters,
    embed_dilated_3x3_in_7x7,
    fuse_conv_bn_eval,
    sarc_budget,
)


torch.set_num_threads(1)


def randomize_bn(bn: nn.BatchNorm2d) -> None:
    with torch.no_grad():
        bn.running_mean.uniform_(-0.7, 0.7)
        bn.running_var.uniform_(0.2, 2.0)
        bn.weight.uniform_(0.4, 1.4)
        bn.bias.uniform_(-0.3, 0.3)


class TestSARCFusion(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19092026)

    def test_dilated_kernel_embedding_exact(self) -> None:
        x = torch.randn(2, 4, 17, 19, dtype=torch.float64)
        kernel = torch.randn(4, 1, 3, 3, dtype=torch.float64)
        for dilation in (1, 2, 3):
            expected = F.conv2d(x, kernel, padding=dilation, dilation=dilation, groups=4)
            embedded = embed_dilated_3x3_in_7x7(kernel, dilation)
            actual = F.conv2d(x, embedded, padding=3, groups=4)
            self.assertLess((expected - actual).abs().max().item(), 1e-12)

    def test_conv_bn_fusion_exact(self) -> None:
        conv = nn.Conv2d(5, 5, 3, padding=1, groups=5, bias=False).double().eval()
        bn = nn.BatchNorm2d(5).double().eval()
        randomize_bn(bn)
        x = torch.randn(3, 5, 11, 13, dtype=torch.float64)
        kernel, bias = fuse_conv_bn_eval(conv, bn)
        expected = bn(conv(x))
        actual = F.conv2d(x, kernel, bias, padding=1, groups=5)
        self.assertLess((expected - actual).abs().max().item(), 1e-12)

    def test_full_sarc_deploy_equivalence(self) -> None:
        block = SARCStagingBlock(7, gamma_init=0.23).double().eval()
        for bn in [block.main_bn, *block.branch_bns, block.mix_bn]:
            randomize_bn(bn)
        with torch.no_grad():
            block.branch_logits.copy_(torch.tensor([-0.7, 0.4, 1.1], dtype=torch.float64))
        x = torch.randn(2, 7, 23, 21, dtype=torch.float64)
        deploy = block.to_deploy()
        expected = block(x)
        actual = deploy(x)
        self.assertLess((expected - actual).abs().max().item(), 2e-11)

    def test_r1_deploy_equivalence(self) -> None:
        block = SingleDW7StagingBlock(6, gamma_init=-0.17).double().eval()
        randomize_bn(block.main_bn)
        randomize_bn(block.mix_bn)
        x = torch.randn(2, 6, 15, 17, dtype=torch.float64)
        self.assertLess((block(x) - block.to_deploy()(x)).abs().max().item(), 2e-11)

    def test_gamma_cannot_cross_silu(self) -> None:
        y = torch.linspace(-4, 4, 257, dtype=torch.float64)
        gamma = torch.tensor(0.2, dtype=torch.float64)
        correct = gamma * F.silu(y)
        invalid_fold = F.silu(gamma * y)
        self.assertGreater((correct - invalid_fold).abs().max().item(), 0.1)

    def test_zero_gamma_is_identity(self) -> None:
        block = SARCStagingBlock(3, gamma_init=0.0).eval()
        x = torch.randn(1, 3, 9, 9)
        self.assertTrue(torch.equal(block(x), x))


class TestSARCHistogram(unittest.TestCase):
    def test_relative_scale(self) -> None:
        widths = torch.tensor([8.0, 32.0, 128.0])
        heights = widths.clone()
        u = assignment_relative_scale(widths, heights, stride=8)
        self.assertTrue(torch.allclose(u, torch.tensor([0.0, 2.0, 4.0])))

    def test_soft_histogram_and_ema(self) -> None:
        histogram = AssignmentHistogramEMA(decay=0.5, min_updates=2)
        logits = torch.zeros(3, requires_grad=True)
        self.assertEqual(histogram.calibration_loss(logits).item(), 0.0)
        histogram.update(torch.tensor([2.1, 2.2, 2.3]))
        histogram.update(torch.tensor([3.6, 3.7, 3.8, float("nan")]))
        target = histogram.ema_hist.clone()
        self.assertAlmostEqual(target.sum().item(), 1.0, places=12)
        self.assertFalse(target.requires_grad)
        loss = histogram.calibration_loss(logits)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertLess(abs(logits.grad.sum().item()), 1e-7)

    def test_softmax_branch_weights(self) -> None:
        block = SARCStagingBlock(4)
        with torch.no_grad():
            block.branch_logits.copy_(torch.tensor([-100.0, 0.0, 100.0]))
        weights = block.branch_weights
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=7)

    def test_ddp_counts_interface_matches_direct_update(self) -> None:
        direct = AssignmentHistogramEMA(min_updates=1)
        reduced = AssignmentHistogramEMA(min_updates=1)
        u = torch.tensor([1.9, 2.7, 3.1, 4.0])
        direct.update(u)
        counts, mass = reduced.soft_counts(u)
        reduced.update_from_counts(counts, mass)
        self.assertTrue(torch.allclose(direct.ema_hist, reduced.ema_hist, atol=0, rtol=0))


class TestSARCControlsAndBudget(unittest.TestCase):
    def test_modes_and_calibration_interface(self) -> None:
        features = [
            torch.randn(1, 4, 12, 12),
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 16, 3, 3),
        ]
        r1 = SARCPyramidStaging(channels=(4, 8, 16), mode="R1")
        r2 = SARCPyramidStaging(channels=(4, 8, 16), mode="R2")
        r3 = SARCPyramidStaging(
            channels=(4, 8, 16), mode="R3", calibration_gain=0.01, histogram_min_updates=1
        )
        self.assertEqual(len(r1(features)), 3)
        self.assertEqual(len(r2(features)), 3)
        self.assertEqual(r1.calibration_loss().item(), 0.0)
        self.assertEqual(r2.calibration_loss().item(), 0.0)
        r3.update_assignment_histograms(
            [torch.tensor([2.0, 2.2]), torch.tensor([2.8, 3.0]), torch.tensor([3.6, 3.8])]
        )
        loss = r3.calibration_loss()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(len(r3.assignment_state()), 3)

    def test_budget_exact(self) -> None:
        budget = sarc_budget()
        self.assertEqual(budget.deploy_conv_weights, 21_952)
        self.assertEqual(budget.deploy_biases, 448)
        self.assertEqual(budget.deploy_gamma_constants, 3)
        self.assertEqual(budget.deploy_total_parameters, 22_400)
        self.assertEqual(budget.deploy_total_stored_scalars, 22_403)
        self.assertEqual(budget.deploy_macs, 35_123_200)
        self.assertEqual(budget.deploy_flops_two_per_mac, 70_246_400)
        self.assertEqual(budget.training_branch_parameters_r1, 23_747)
        self.assertEqual(budget.training_branch_parameters_r2_r3, 38_540)
        self.assertEqual(budget.training_conv_macs_r2_r3, 54_476_800)

    def test_parameter_formula_matches_modules(self) -> None:
        r1 = SARCPyramidStaging(mode="R1")
        r2 = SARCPyramidStaging(mode="R2")
        budget = sarc_budget()
        self.assertEqual(count_trainable_parameters(r1), budget.training_branch_parameters_r1)
        self.assertEqual(count_trainable_parameters(r2), budget.training_branch_parameters_r2_r3)

    def test_deploy_parameter_formula_and_frozen_gamma(self) -> None:
        deploy_blocks = [SARCStagingBlock(c).eval().to_deploy() for c in (64, 128, 256)]
        actual_parameters = sum(count_trainable_parameters(block) for block in deploy_blocks)
        self.assertEqual(actual_parameters, sarc_budget().deploy_total_parameters)
        for block in deploy_blocks:
            self.assertNotIn("gamma", dict(block.named_parameters()))
            self.assertIn("gamma", dict(block.named_buffers()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
