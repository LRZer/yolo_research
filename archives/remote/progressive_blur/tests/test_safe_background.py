from __future__ import annotations

import copy
import unittest

import torch

from pgb.core import make_targets
from pgb.safe_background import SafeBackgroundFrontEnd, SafeBackgroundGate, safe_gate_loss


class SafeBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(2027)

    def test_parameter_budget_and_odd_image_shape(self) -> None:
        gate = SafeBackgroundGate().eval()
        count = sum(parameter.numel() for parameter in gate.parameters())
        self.assertGreaterEqual(count, 500_000)
        self.assertLessEqual(count, 1_500_000)
        with torch.no_grad():
            permission = gate(torch.rand(1, 3, 65, 71))
        self.assertEqual(permission.shape, (1, 1, 65, 71))
        self.assertTrue(((permission >= 0) & (permission <= 1)).all())
        self.assertLess(permission.mean().item(), 0.1)

    def test_identity_at_zero_mix_and_gradient_at_nonzero_mix(self) -> None:
        module = SafeBackgroundFrontEnd(sigma=3, mix=0.5).eval()
        images = torch.rand(1, 3, 48, 48)
        with torch.no_grad():
            identity, permission = module(images, mix=0)
        self.assertTrue(torch.equal(identity, images))
        self.assertEqual(permission.shape, (1, 1, 48, 48))
        processed, _ = module(images)
        self.assertFalse(torch.equal(processed, images))
        processed.square().mean().backward()
        self.assertGreater(module.map_net.head.weight.grad.abs().sum().item(), 0)

    def test_protection_penalty_and_empty_labels(self) -> None:
        images = torch.rand(2, 3, 48, 48)
        boxes = [torch.tensor([[15.0, 15.0, 27.0, 27.0]]), torch.empty(0, 4)]
        target = make_targets(images, boxes)
        permission = torch.full((2, 1, 48, 48), 0.8, requires_grad=True)
        loss = safe_gate_loss(permission, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(permission.grad[0].abs().sum().item(), 0)
        self.assertEqual(permission.grad[1].abs().sum().item(), 0)
        protected_mean_gradient = permission.grad[0][target.protected[0]].mean().item()
        far = target.valid[0] & (target.strength[0] >= 0.8)
        far_mean_gradient = permission.grad[0][far].mean().item()
        self.assertGreater(protected_mean_gradient, 0)
        self.assertLess(far_mean_gradient, 0)

    def test_frontend_state_roundtrip(self) -> None:
        module = SafeBackgroundFrontEnd(sigma=3).eval()
        restored = SafeBackgroundFrontEnd(sigma=3).eval()
        restored.load_state_dict(copy.deepcopy(module.state_dict()))
        images = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            before = module(images)[0]
            after = restored(images)[0]
        self.assertTrue(torch.equal(before, after))

    def test_gate_loss_under_bfloat16_autocast(self) -> None:
        images = torch.rand(1, 3, 48, 48)
        targets = make_targets(images, [torch.tensor([[15.0, 15.0, 27.0, 27.0]])])
        permission = torch.full((1, 1, 48, 48), 0.2, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = safe_gate_loss(permission, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(permission.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
