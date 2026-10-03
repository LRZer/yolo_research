from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pgb.core import GaussianBankRenderer
from pgb.oracle_integration import (
    OracleBlurTrainer,
    OracleTrainOnlyBlurTrainer,
    apply_oracle_blur,
    apply_stochastic_oracle_blur,
    oracle_strength_from_batch,
)


def batch() -> dict:
    torch.manual_seed(2027)
    return {
        "img": torch.rand(1, 3, 64, 64),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "batch_idx": torch.tensor([0.0]),
    }


class OracleIntegrationTests(unittest.TestCase):
    def test_oracle_map_protects_box_and_blurs_background(self) -> None:
        sample = batch()
        original = sample["img"].clone()
        strength = oracle_strength_from_batch(sample)
        self.assertEqual(tuple(strength.shape), (1, 1, 64, 64))
        self.assertEqual(strength[0, 0, 32, 32].item(), 0.0)
        self.assertGreater(strength[0, 0, 0, 0].item(), 0.9)
        apply_oracle_blur(sample, GaussianBankRenderer(), mix=0.5)
        # The center lies inside the protected box and must remain exact.
        self.assertTrue(torch.equal(sample["img"][0, :, 32, 32], original[0, :, 32, 32]))
        self.assertFalse(torch.equal(sample["img"][0, :, 0, 0], original[0, :, 0, 0]))

    def test_zero_mix_is_exact_identity(self) -> None:
        sample = batch()
        original = sample["img"].clone()
        apply_oracle_blur(sample, GaussianBankRenderer(), mix=0.0)
        self.assertTrue(torch.equal(sample["img"], original))

    def test_schedule_matches_pgb(self) -> None:
        trainer = object.__new__(OracleBlurTrainer)
        trainer.oracle_zero_mix_epochs = 5
        trainer.oracle_ramp_epochs = 5
        trainer.oracle_eval_mix = 0.5
        actual = [trainer.mix_for_epoch(epoch) for epoch in range(11)]
        self.assertEqual(actual, [0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5])

    def test_stronger_blur_preserves_box_but_changes_background(self) -> None:
        sample = batch()
        original = sample["img"].clone()
        apply_oracle_blur(
            sample,
            GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16)),
            mix=0.5,
        )
        self.assertTrue(torch.equal(sample["img"][0, :, 32, 32], original[0, :, 32, 32]))
        self.assertGreater((sample["img"][0, :, 0, 0] - original[0, :, 0, 0]).abs().max().item(), 0)

    def test_train_only_uses_standard_unmasked_validator(self) -> None:
        trainer = object.__new__(OracleTrainOnlyBlurTrainer)
        trainer.args = SimpleNamespace(
            oracle_eval_mix=0.5,
            oracle_sigma_levels=(0, 2, 4, 8, 16),
            oracle_zero_mix_epochs=5,
            oracle_ramp_epochs=5,
            oracle_target_margin=0.1,
            oracle_target_transition=0.5,
            oracle_target_min_radius=8.0,
        )
        trainer.test_loader = object()
        trainer.save_dir = Path(".")
        trainer.callbacks = {}

        with patch("pgb.oracle_integration.DetectionValidator") as standard_validator:
            trainer.get_validator()
        standard_validator.assert_called_once()
        passed_args = standard_validator.call_args.kwargs["args"]
        self.assertFalse(hasattr(passed_args, "oracle_sigma_levels"))
        self.assertFalse(hasattr(passed_args, "oracle_eval_mix"))

    def test_stochastic_blur_has_exact_clean_training_images(self) -> None:
        original = batch()
        original["img"] = original["img"].repeat(16, 1, 1, 1)
        original["bboxes"] = original["bboxes"].repeat(16, 1)
        original["cls"] = original["cls"].repeat(16, 1)
        original["batch_idx"] = torch.arange(16, dtype=torch.float32)
        clean = original["img"].clone()
        torch.manual_seed(2027)
        apply_stochastic_oracle_blur(
            original,
            GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16)),
            mix=0.5,
            probability=0.5,
        )
        chosen = original["oracle_augmented"]
        self.assertGreater(chosen.sum().item(), 0)
        self.assertLess(chosen.sum().item(), 16)
        self.assertTrue(torch.equal(original["img"][~chosen], clean[~chosen]))
        self.assertFalse(torch.equal(original["img"][chosen], clean[chosen]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
