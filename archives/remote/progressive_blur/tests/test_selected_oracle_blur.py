from __future__ import annotations

import copy
import unittest

import torch

from pgb.core import GaussianBankRenderer
from pgb.oracle_integration import apply_stochastic_oracle_blur
from pgb.selected_oracle_blur import apply_stochastic_oracle_blur_selected


def make_batch(count: int = 8) -> dict:
    generator = torch.Generator().manual_seed(2005)
    return {
        "img": torch.rand((count, 3, 64, 64), generator=generator),
        "cls": torch.zeros((count, 1)),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]).repeat(count, 1),
        "batch_idx": torch.arange(count, dtype=torch.float32),
    }


class SelectedOracleBlurTests(unittest.TestCase):
    def test_image_output_matches_registered_path_at_fixed_rng(self) -> None:
        original = make_batch()
        reference = copy.deepcopy(original)
        candidate = copy.deepcopy(original)
        renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16))
        torch.manual_seed(2027)
        apply_stochastic_oracle_blur(reference, renderer, mix=0.5, probability=0.5)
        torch.manual_seed(2027)
        apply_stochastic_oracle_blur_selected(candidate, renderer, mix=0.5, probability=0.5)
        self.assertTrue(torch.equal(reference["oracle_augmented"], candidate["oracle_augmented"]))
        torch.testing.assert_close(reference["img"], candidate["img"], rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(candidate["img"][~candidate["oracle_augmented"]], original["img"][~candidate["oracle_augmented"]]))

    def test_zero_mix_and_probability_are_identity(self) -> None:
        renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16))
        for mix, probability in ((0.0, 0.5), (0.5, 0.0)):
            sample = make_batch()
            image = sample["img"].clone()
            apply_stochastic_oracle_blur_selected(sample, renderer, mix=mix, probability=probability)
            self.assertTrue(torch.equal(image, sample["img"]))

    def test_full_probability_matches_exact_path(self) -> None:
        sample = make_batch(2)
        reference = copy.deepcopy(sample)
        candidate = copy.deepcopy(sample)
        renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16))
        apply_stochastic_oracle_blur(reference, renderer, mix=0.5, probability=1.0)
        apply_stochastic_oracle_blur_selected(candidate, renderer, mix=0.5, probability=1.0)
        self.assertTrue(torch.equal(reference["img"], candidate["img"]))
        self.assertTrue(torch.equal(reference["oracle_strength"], candidate["oracle_strength"]))


if __name__ == "__main__":
    unittest.main()
