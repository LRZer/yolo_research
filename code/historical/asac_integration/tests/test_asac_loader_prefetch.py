"""Real four-worker test that drains stale InfiniteDataLoader prefetch with reset()."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snapshot" / "ultralytics_research"))

from ultralytics.data.build import build_dataloader  # noqa: E402


class SharedQDataset(Dataset):
    def __init__(self, shared_q: torch.Tensor, length: int = 128):
        self.shared_q = shared_q
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor([index, float(self.shared_q[0].item())], dtype=torch.float64)


class TestPrefetchReset(unittest.TestCase):
    def test_reset_makes_first_and_last_batch_observe_new_q(self):
        q = torch.tensor([0.25], dtype=torch.float64).share_memory_()
        loader = build_dataloader(
            SharedQDataset(q),
            batch=8,
            workers=4,
            shuffle=False,
            rank=-1,
            pin_memory=True,
            device=torch.device("cuda"),
        )
        self.assertEqual(loader.num_workers, 4)
        self.assertTrue(loader.pin_memory)
        first_epoch = list(loader)
        self.assertTrue(all(torch.allclose(batch[:, 1], torch.full_like(batch[:, 1], 0.25)) for batch in first_epoch))
        time.sleep(0.25)  # permit the persistent iterator to fill its next-epoch prefetch queue
        q[0] = 0.75
        loader.reset()
        second_epoch = list(loader)
        self.assertEqual(len(second_epoch), len(first_epoch))
        self.assertTrue(all(torch.allclose(batch[:, 1], torch.full_like(batch[:, 1], 0.75)) for batch in second_epoch))
        loader.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

