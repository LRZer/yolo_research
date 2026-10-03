from __future__ import annotations

import unittest

import torch

from pgb.blackout_integration import apply_oracle_blackout, exact_box_union_mask


class BlackoutIntegrationTests(unittest.TestCase):
    def test_exact_union_preserves_boxes_and_zeros_background(self) -> None:
        torch.manual_seed(2027)
        image = torch.rand(1, 3, 16, 16)
        batch = {
            "img": image.clone(),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]),
            "batch_idx": torch.tensor([0.0, 0.0]),
        }
        mask = exact_box_union_mask(batch)
        self.assertTrue(mask[0, 0, 4, 4])
        self.assertTrue(mask[0, 0, 12, 12])
        self.assertFalse(mask[0, 0, 8, 8])
        apply_oracle_blackout(batch)
        self.assertTrue(torch.equal(batch["img"][0, :, 4, 4], image[0, :, 4, 4]))
        self.assertTrue(torch.equal(batch["img"][0, :, 12, 12], image[0, :, 12, 12]))
        self.assertEqual(batch["img"][0, :, 8, 8].abs().sum().item(), 0.0)

    def test_empty_image_becomes_black(self) -> None:
        batch = {
            "img": torch.ones(1, 3, 8, 8),
            "cls": torch.empty((0, 1)),
            "bboxes": torch.empty((0, 4)),
            "batch_idx": torch.empty((0,)),
        }
        apply_oracle_blackout(batch)
        self.assertEqual(batch["img"].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

