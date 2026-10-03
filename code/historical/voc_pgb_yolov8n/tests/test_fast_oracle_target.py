"""Numerical-equivalence tests for the optional local Oracle target builder."""

import torch

from pgb.core import make_targets
from pgb.fast_oracle_target import make_targets_local


def test_local_target_bounded_error_and_same_protected_pixels() -> None:
    generator = torch.Generator().manual_seed(2027)
    images = torch.rand((3, 3, 72, 96), generator=generator)
    boxes = [
        torch.tensor([[1.4, 2.2, 7.8, 8.4], [42.1, 5.1, 48.5, 12.9]]),
        torch.tensor([[12.0, 10.0, 80.0, 65.0], [90.0, 60.0, 98.0, 74.0]]),
        torch.empty((0, 4)),
    ]
    valid = torch.ones((3, 1, 72, 96), dtype=torch.bool)
    valid[:, :, :3, :] = False
    for epsilon in (1e-2, 1e-4, 1e-6):
        exact = make_targets(images, boxes, valid_pixels=valid)
        local = make_targets_local(images, boxes, epsilon=epsilon, valid_pixels=valid)
        assert torch.equal(local.protected, exact.protected)
        assert torch.equal(local.valid, exact.valid)
        assert (local.strength - exact.strength).abs().max().item() <= epsilon + 1e-6


def test_local_target_random_crowding() -> None:
    generator = torch.Generator().manual_seed(2005)
    images = torch.zeros((1, 3, 96, 128))
    centers = torch.rand((32, 2), generator=generator) * torch.tensor([128, 96])
    sizes = 2 + 12 * torch.rand((32, 2), generator=generator)
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)
    # Avoid degenerate boxes at image boundaries, matching the real-label contract.
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, 128)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, 96)
    boxes = boxes[(boxes[:, 2:] > boxes[:, :2]).all(dim=1)]
    exact = make_targets(images, [boxes])
    local = make_targets_local(images, [boxes], epsilon=1e-4)
    assert torch.equal(local.protected, exact.protected)
    assert (local.strength - exact.strength).abs().max().item() <= 1.01e-4
