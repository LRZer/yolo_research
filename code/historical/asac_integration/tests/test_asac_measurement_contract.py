"""Focused tests for ASAC panel numerical scope and EMA side-effect guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snapshot" / "ultralytics_research"))

from ultralytics.utils.asac import ASACPanelEvaluator, _model_integrity  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("non_bn_state", torch.tensor([2.0]))
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.criterion = None
        self.init_calls = 0

    def init_criterion(self):
        self.init_calls += 1
        return object()


class FakeBranch:
    stride = torch.tensor([8.0, 16.0, 32.0])

    @staticmethod
    def parse_output(_):
        prediction = {
            "boxes": torch.zeros(1, 4, 1),
            "scores": torch.zeros(1, 1, 1),
        }
        return {"one2many": prediction, "one2one": prediction}

    @staticmethod
    def get_assigned_targets_and_loss(prediction, batch, return_details=False):
        del batch
        if not return_details:
            raise AssertionError("ASAC must request native assignment details")
        return {"fg_mask": torch.zeros(1, 1, dtype=torch.bool)}, prediction, None


class FakeCriterion:
    def __init__(self):
        self.one2many = FakeBranch()
        self.one2one = FakeBranch()


class FakeForwardModel:
    @staticmethod
    def predict(image):
        return image


class TestMeasurementContract(unittest.TestCase):
    def test_integrity_covers_non_batchnorm_buffers(self):
        model = TinyModel()
        before = _model_integrity(model)
        model.non_bn_state.add_(1)
        after = _model_integrity(model)
        self.assertEqual(before["parameters_sha256"], after["parameters_sha256"])
        self.assertNotEqual(before["registered_buffers_sha256"], after["registered_buffers_sha256"])
        before = _model_integrity(model)
        model.stride.add_(1)
        after = _model_integrity(model)
        self.assertNotEqual(
            before["extra_tensor_attributes_sha256"], after["extra_tensor_attributes_sha256"]
        )

    def test_evaluator_owns_criterion_without_attaching_to_ema(self):
        model = TinyModel()
        trainer = type("Trainer", (), {"ema": type("EMA", (), {"ema": model})()})()
        panel = ([], [], {"sha256": "panel"}, {})
        with mock.patch.object(ASACPanelEvaluator, "_build_panel", return_value=panel):
            evaluator = ASACPanelEvaluator(trainer)
        self.assertEqual(model.init_calls, 1)
        self.assertIsNone(model.criterion)
        self.assertIsNotNone(evaluator.criterion)

    def test_forward_requires_inference_mode_and_limits_autocast_to_forward(self):
        batch = {"img": torch.zeros(1, 3, 8, 8)}
        with self.assertRaisesRegex(RuntimeError, "inference_mode"):
            ASACPanelEvaluator._forward_assignments(FakeForwardModel(), FakeCriterion(), batch)
        with torch.inference_mode():
            outputs, evidence = ASACPanelEvaluator._forward_assignments(
                FakeForwardModel(), FakeCriterion(), batch
            )
        self.assertEqual(set(outputs), {"O2M", "O2O"})
        self.assertTrue(evidence["inference_mode"])
        self.assertFalse(evidence["autocast_enabled"])
        self.assertIsNone(evidence["autocast_dtype"])

    def test_prior_assigner_oom_fallback_is_fail_closed(self):
        model = TinyModel()
        assigner = type("Assigner", (), {"_oom_warned": True})()
        branch = type("Branch", (), {"assigner": assigner})()
        evaluator = object.__new__(ASACPanelEvaluator)
        evaluator.protocol = type("Protocol", (), {"probe_strength": 0.5})()
        evaluator.trainer = type("Trainer", (), {"ema": type("EMA", (), {"ema": model})()})()
        evaluator.criterion = type("Criterion", (), {"one2many": branch, "one2one": branch})()
        with self.assertRaisesRegex(RuntimeError, "OOM fallback"):
            evaluator.measure()


if __name__ == "__main__":
    unittest.main(verbosity=2)
