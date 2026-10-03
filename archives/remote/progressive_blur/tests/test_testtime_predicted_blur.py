"""Small CPU checks for the image-only inference blur renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "evaluate_testtime_predicted_blur.py"
SPEC = importlib.util.spec_from_file_location("testtime_blur", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_predicted_box_and_halo_remain_identical() -> None:
    rng = np.random.default_rng(2027)
    image = rng.random((128, 128, 3), dtype=np.float32)
    output, protected = MODULE.protect_and_blur(image, np.array([[40, 40, 80, 80]]), 8.0)
    assert np.array_equal(output[40:80, 40:80], image[40:80, 40:80])
    assert np.array_equal(output[32:88, 32:88], image[32:88, 32:88])
    assert 0.0 < protected < 1.0
    assert float(np.abs(output[:16, :16] - image[:16, :16]).mean()) > 0.05


def test_stronger_blur_changes_far_background_more() -> None:
    rng = np.random.default_rng(2005)
    image = rng.random((128, 128, 3), dtype=np.float32)
    boxes = np.array([[40, 40, 80, 80]])
    changes = []
    for sigma in (3.0, 8.0, 16.0):
        output, _ = MODULE.protect_and_blur(image, boxes, sigma)
        changes.append(float(np.abs(output[:16, :16] - image[:16, :16]).mean()))
    assert changes[0] < changes[1] < changes[2]


def test_no_predictions_blurs_whole_image() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((64, 64, 3), dtype=np.float32)
    output, protected = MODULE.protect_and_blur(image, np.empty((0, 4)), 3.0)
    assert protected == 0.0
    assert not np.array_equal(output, image)


if __name__ == "__main__":
    test_predicted_box_and_halo_remain_identical()
    test_stronger_blur_changes_far_background_more()
    test_no_predictions_blurs_whole_image()
    print("3/3 CPU renderer checks passed")
