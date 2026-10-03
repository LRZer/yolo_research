from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch

from pgb.ultralytics_integration import (
    PGBDetectionModel,
    PGBDetectionTrainer,
    valid_pixels_from_batch,
)
from ultralytics.nn.tasks import DetectionModel


def synthetic_batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(2, 3, 64, 64, device=device),
        "cls": torch.tensor([[0.0], [1.0]], device=device),
        "bboxes": torch.tensor(
            [[0.35, 0.45, 0.25, 0.35], [0.65, 0.55, 0.20, 0.25]],
            device=device,
        ),
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
    }


class UltralyticsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(2027)

    def test_training_loss_reaches_frontend_and_detector(self) -> None:
        model = PGBDetectionModel("yolov8n.yaml", nc=2, verbose=False).train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        loss, items = model(synthetic_batch(torch.device("cpu")))
        self.assertEqual(set(items), {"box_loss", "cls_loss", "dfl_loss", "map_loss"})
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertGreater(model.frontend.map_net.head.weight.grad.abs().sum().item(), 0)
        detector_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("model.") and parameter.grad is not None
        ]
        self.assertTrue(detector_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in detector_gradients))

    def test_zero_mix_is_detector_equivalent_and_state_is_complete(self) -> None:
        model = PGBDetectionModel("yolov8n.yaml", nc=2, eval_mix=0.0, verbose=False).eval()
        images = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            wrapped = model(images)
            native = DetectionModel.predict(model, images)
        wrapped_tensor = wrapped[0] if isinstance(wrapped, tuple) else wrapped
        native_tensor = native[0] if isinstance(native, tuple) else native
        self.assertTrue(torch.equal(wrapped_tensor, native_tensor))
        state = copy.deepcopy(model.state_dict())
        self.assertTrue(any(key.startswith("frontend.") for key in state))
        restored = PGBDetectionModel("yolov8n.yaml", nc=2, eval_mix=0.0, verbose=False).eval()
        restored.load_state_dict(state)
        with torch.no_grad():
            recovered = restored(images)
        recovered_tensor = recovered[0] if isinstance(recovered, tuple) else recovered
        self.assertTrue(torch.equal(wrapped_tensor, recovered_tensor))

    def test_mix_schedule(self) -> None:
        trainer = object.__new__(PGBDetectionTrainer)
        trainer.pgb_zero_mix_epochs = 5
        trainer.pgb_ramp_epochs = 5
        trainer.pgb_eval_mix = 0.5
        actual = [trainer.mix_for_epoch(epoch) for epoch in range(11)]
        expected = [0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        self.assertEqual(actual, expected)

    def test_letterbox_valid_pixel_reconstruction(self) -> None:
        batch = {
            "img": torch.zeros(2, 3, 640, 640),
            "ori_shape": ((333, 500), (364, 480)),
            "ratio_pad": (
                (1.2822822822822824, 1.28),
                (1.335164835164835, 1.3333333333333333),
            ),
        }
        valid = valid_pixels_from_batch(batch, 640, 640, torch.device("cpu"))
        self.assertIsNotNone(valid)
        self.assertEqual(valid[0].sum().item(), 427 * 640)
        self.assertEqual(valid[1].sum().item(), 486 * 640)
        self.assertFalse(valid[0, 0, :106].any())
        self.assertFalse(valid[0, 0, 533:].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
