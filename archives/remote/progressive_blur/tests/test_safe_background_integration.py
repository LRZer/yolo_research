from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch

from pgb.safe_background_integration import SafeBackgroundDetectionModel, SafeBackgroundTrainer
from ultralytics.nn.tasks import DetectionModel


def batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(2, 3, 64, 64),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.35, 0.45, 0.25, 0.35], [0.65, 0.55, 0.20, 0.25]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }


class SafeBackgroundIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(2027)

    def test_map_only_updates_frontend_not_detector(self) -> None:
        model = SafeBackgroundDetectionModel(
            "yolov8n.yaml", nc=2, verbose=False, sbg_sigma=3,
            sbg_map_only=True, map_loss_weight=1,
        ).train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        sample = batch()
        loss, items = model(sample)
        self.assertEqual(set(items), {"box_loss", "cls_loss", "dfl_loss", "map_loss"})
        self.assertTrue(torch.isfinite(loss).all())
        validation_loss, _ = model.loss(sample, preds=object())
        self.assertTrue(torch.isfinite(validation_loss).all())
        loss.sum().backward()
        self.assertGreater(model.frontend.map_net.head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.model.parameters()))

    def test_joint_loss_reaches_gate_and_detector(self) -> None:
        model = SafeBackgroundDetectionModel(
            "yolov8n.yaml", nc=2, verbose=False, sbg_sigma=3,
            map_loss_weight=0.25,
        ).train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        model.set_train_mix(0.5)
        loss, items = model(batch())
        self.assertEqual(set(items), {"box_loss", "cls_loss", "dfl_loss", "map_loss"})
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertGreater(model.frontend.map_net.head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.model.parameters()))

    def test_zero_mix_matches_detector_and_state_roundtrip(self) -> None:
        model = SafeBackgroundDetectionModel(
            "yolov8n.yaml", nc=2, verbose=False, sbg_sigma=3, eval_mix=0,
        ).eval()
        images = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            wrapped = model(images)
            native = DetectionModel.predict(model, images)
        wrapped_tensor = wrapped[0] if isinstance(wrapped, tuple) else wrapped
        native_tensor = native[0] if isinstance(native, tuple) else native
        self.assertTrue(torch.equal(wrapped_tensor, native_tensor))
        state = copy.deepcopy(model.state_dict())
        restored = SafeBackgroundDetectionModel(
            "yolov8n.yaml", nc=2, verbose=False, sbg_sigma=3, eval_mix=0,
        ).eval()
        restored.load_state_dict(state)
        with torch.no_grad():
            recovered = restored(images)
        recovered_tensor = recovered[0] if isinstance(recovered, tuple) else recovered
        self.assertTrue(torch.equal(wrapped_tensor, recovered_tensor))

    def test_next_stage_reenables_blur_after_map_only_checkpoint(self) -> None:
        source = SafeBackgroundDetectionModel(
            "yolov8n.yaml", nc=2, verbose=False, sbg_sigma=3,
            eval_mix=0, sbg_map_only=True,
        )
        with torch.no_grad():
            source.frontend.map_net.head.bias.fill_(-2.0)
        trainer = object.__new__(SafeBackgroundTrainer)
        trainer.data = {"nc": 2, "channels": 3, "names": {0: "a", 1: "b"}}
        trainer.args = SimpleNamespace(cls_remap=False)
        trainer.sbg_sigma = 3.0
        trainer.sbg_far_threshold = 0.8
        trainer.sbg_protect_weight = 4.0
        trainer.sbg_far_weight = 0.25
        trainer.sbg_map_only = False
        trainer.pgb_map_loss_weight = 0.25
        trainer.pgb_eval_mix = 0.5
        trainer.pgb_target_margin = 0.1
        trainer.pgb_target_transition = 0.5
        trainer.pgb_target_min_radius = 8.0
        trainer.pgb_zero_mix_epochs = 0
        trainer.pgb_ramp_epochs = 2
        target = trainer.get_model(cfg="yolov8n.yaml", weights=source, verbose=False)
        self.assertFalse(target.sbg_map_only)
        self.assertEqual(float(target.pgb_eval_mix), 0.5)
        self.assertEqual(float(target.pgb_train_mix), 0)
        self.assertTrue(torch.equal(target.frontend.map_net.head.bias, source.frontend.map_net.head.bias))


if __name__ == "__main__":
    unittest.main(verbosity=2)
