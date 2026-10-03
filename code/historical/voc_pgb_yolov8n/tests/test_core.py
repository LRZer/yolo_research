from __future__ import annotations

import io
import unittest

import torch

from pgb import (
    BlurMapNet,
    GaussianBankRenderer,
    ProgressiveBlurFrontEnd,
    balanced_map_loss,
    make_targets,
    yolo_xywh_to_boxes,
)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def setUp(self) -> None:
        self.images = torch.rand(2, 3, 64, 96)
        self.boxes = [
            torch.tensor([[20.0, 15.0, 40.0, 45.0], [66.0, 22.0, 81.0, 39.0]]),
            torch.empty(0, 4),
        ]

    def test_parameter_count_and_shape(self) -> None:
        model = ProgressiveBlurFrontEnd()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 38_513)
        output, strength = model(self.images)
        self.assertEqual(output.shape, self.images.shape)
        self.assertEqual(strength.shape, (2, 1, 64, 96))

    def test_target_protection_range_empty_and_monotonicity(self) -> None:
        targets = make_targets(self.images, self.boxes, min_radius=2)
        self.assertTrue(torch.all(targets.strength[0][targets.protected[0]] == 0))
        self.assertTrue(torch.all((targets.strength >= 0) & (targets.strength <= 1)))
        self.assertFalse(targets.valid[1].any())
        # Along a ray to the left of the first protected region, blur must not
        # decrease as distance grows.
        ray = targets.strength[0, 0, 30, :18].flip(0)
        self.assertTrue(torch.all(ray[1:] >= ray[:-1]))

    def test_renderer_identity_constants_and_convex_weights(self) -> None:
        targets = make_targets(self.images, self.boxes, min_radius=2)
        renderer = GaussianBankRenderer(mix=1.0)
        zeros = torch.zeros_like(targets.strength)
        self.assertTrue(torch.equal(renderer(self.images, zeros), self.images))
        self.assertTrue(torch.equal(renderer(self.images, targets.strength, mix=0), self.images))
        ones = torch.ones_like(self.images)
        self.assertTrue(
            torch.allclose(renderer(ones, targets.strength), ones, atol=2e-6)
        )
        sigma = torch.linspace(0, 8, 1001).reshape(1, 1, 1, -1)
        weights = sum(renderer._weight(sigma, i) for i in range(5))
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights), atol=1e-6))

    def test_gradient_from_renderer_and_map_loss(self) -> None:
        targets = make_targets(self.images, self.boxes, min_radius=2)
        strength = torch.rand_like(targets.strength, requires_grad=True)
        renderer = GaussianBankRenderer(mix=1.0)
        (renderer(self.images, strength) - self.images).square().mean().backward()
        self.assertIsNotNone(strength.grad)
        self.assertGreater(strength.grad.abs().sum().item(), 0)

        model = ProgressiveBlurFrontEnd()
        output, prediction = model(self.images)
        loss = balanced_map_loss(prediction, targets) + (output - self.images).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertGreater(model.map_net.stem[0].weight.grad.abs().sum().item(), 0)

        model.zero_grad(set_to_none=True)
        output, _ = model(self.images)
        (output - self.images).square().mean().backward()
        self.assertGreater(model.map_net.head.weight.grad.abs().sum().item(), 0)

    def test_valid_pixels_ignore_boxes_and_empty_loss(self) -> None:
        valid_pixels = torch.ones(2, 1, 64, 96, dtype=torch.bool)
        valid_pixels[:, :, :, 90:] = False
        targets = make_targets(
            self.images,
            self.boxes,
            min_radius=2,
            valid_pixels=valid_pixels,
            ignore_boxes_by_image=[self.boxes[0][:1], torch.empty(0, 4)],
        )
        self.assertFalse(targets.valid[0, 0, 20:30, 25:35].any())
        self.assertFalse(targets.valid[0, 0, :, 90:].any())
        empty = make_targets(self.images, [torch.empty(0, 4), torch.empty(0, 4)])
        prediction = torch.rand_like(empty.strength, requires_grad=True)
        loss = balanced_map_loss(prediction, empty)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_yolo_coordinate_conversion(self) -> None:
        batch = {
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
            "batch_idx": torch.tensor([0.0]),
        }
        converted = yolo_xywh_to_boxes(batch, height=100, width=200, batch_size=2)
        self.assertTrue(torch.allclose(converted[0], torch.tensor([[80.0, 30.0, 120.0, 70.0]])))
        self.assertEqual(converted[1].shape, (0, 4))

    def test_odd_shapes_small_shapes_and_round_trip(self) -> None:
        model = ProgressiveBlurFrontEnd().eval()
        with torch.no_grad():
            odd_output, odd_strength = model(torch.rand(1, 3, 65, 97))
            small_output, _ = model(torch.rand(1, 3, 13, 17))
        self.assertEqual(odd_output.shape, (1, 3, 65, 97))
        self.assertEqual(odd_strength.shape, (1, 1, 65, 97))
        self.assertEqual(small_output.shape, (1, 3, 13, 17))

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = ProgressiveBlurFrontEnd().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        sample = torch.rand(1, 3, 32, 48)
        with torch.no_grad():
            expected, _ = model(sample)
            actual, _ = restored(sample)
        self.assertTrue(torch.equal(expected, actual))

    def test_full_resolution_smoke(self) -> None:
        model = BlurMapNet().eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, 640, 640))
        self.assertEqual(result.shape, (1, 1, 640, 640))


if __name__ == "__main__":
    unittest.main(verbosity=2)

