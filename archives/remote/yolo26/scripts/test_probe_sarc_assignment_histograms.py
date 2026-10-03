"""CPU-free decision-logic tests for probe_sarc_assignment_histograms.py.

The production module is loaded with tiny import stubs so this test exercises the
histogram/JS/gate code without importing PyTorch, Ultralytics, or touching a GPU.
"""

from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("probe_sarc_assignment_histograms.py")


def load_module():
    saved_argv = sys.argv[:]
    saved_modules = {name: sys.modules.get(name) for name in (
        "torch",
        "ultralytics",
        "ultralytics.models",
        "ultralytics.models.yolo",
        "ultralytics.models.yolo.detect",
        "ultralytics.data",
        "ultralytics.data.build",
        "ultralytics.utils",
        "ultralytics.utils.tal",
    )}
    try:
        torch = types.ModuleType("torch")
        torch.nn = types.ModuleType("torch.nn")
        sys.modules["torch"] = torch
        detect = types.ModuleType("ultralytics.models.yolo.detect")
        detect.DetectionTrainer = type("DetectionTrainer", (), {})
        data_build = types.ModuleType("ultralytics.data.build")
        data_build.build_dataloader = lambda *args, **kwargs: None
        data_build.build_yolo_dataset = lambda *args, **kwargs: None
        tal = types.ModuleType("ultralytics.utils.tal")
        tal.make_anchors = lambda *args, **kwargs: None
        for name in (
            "ultralytics",
            "ultralytics.models",
            "ultralytics.models.yolo",
            "ultralytics.data",
            "ultralytics.utils",
        ):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["ultralytics.models.yolo.detect"] = detect
        sys.modules["ultralytics.data.build"] = data_build
        sys.modules["ultralytics.utils.tal"] = tal
        sys.argv = [
            str(SCRIPT),
            "--checkpoints",
            str(SCRIPT),
            "--data",
            "unused.yaml",
            "--output-dir",
            "unused",
        ]
        return runpy.run_path(str(SCRIPT), run_name="sarc_probe_test_module")
    finally:
        sys.argv = saved_argv
        for name, value in saved_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


M = load_module()
LEVELS = M["LEVEL_NAMES"]
WEIGHTINGS = M["WEIGHTINGS"]
BIN_COUNT = len(M["HIST_EDGES"]) - 1


def one_hot(index: int, count: int = 100) -> list[int]:
    values = [0] * BIN_COUNT
    values[index] = count
    return values


def result(epoch: int, per_level: dict[str, list[int]], panel: str = "fixed-panel"):
    return {
        "epoch": epoch,
        "panel_sha256": panel,
        "levels": {
            weighting: {
                level: {"histogram_counts": list(per_level[level])}
                for level in LEVELS
            }
            for weighting in WEIGHTINGS
        },
    }


class SarcDecisionTest(unittest.TestCase):
    def decision(self, per_epoch):
        results = [result(epoch, distributions) for epoch, distributions in zip((11, 21, 31, 41, 50), per_epoch)]
        rows = M["build_js_rows"](results)
        return M["make_decision"](results, rows)

    def test_js_bounds_and_known_extremes(self):
        self.assertEqual(M["js_divergence"](one_hot(2), one_hot(2)), 0.0)
        self.assertEqual(M["js_divergence"](one_hot(2), one_hot(18)), 1.0)

    def test_go_for_separated_stable_levels(self):
        stable = {"P3": one_hot(3), "P4": one_hot(11), "P5": one_hot(19)}
        decision = self.decision([stable] * 5)
        self.assertEqual(decision["verdict"], "GO")
        self.assertFalse(decision["similarity_no_go"])
        self.assertFalse(decision["instability_no_go"])

    def test_no_go_for_indistinguishable_levels(self):
        identical = {level: one_hot(8) for level in LEVELS}
        decision = self.decision([identical] * 5)
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertTrue(decision["similarity_no_go"])

    def test_no_go_for_consecutive_epoch_instability(self):
        early = {"P3": one_hot(2), "P4": one_hot(10), "P5": one_hot(18)}
        late = {"P3": one_hot(22), "P4": one_hot(10), "P5": one_hot(18)}
        decision = self.decision([early, late, late, late, late])
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertTrue(decision["instability_no_go"])
        self.assertEqual(decision["max_primary_consecutive_epoch_js"], 1.0)

    def test_panel_mismatch_is_fatal(self):
        stable = {"P3": one_hot(3), "P4": one_hot(11), "P5": one_hot(19)}
        results = [result(epoch, stable, panel=f"panel-{epoch}") for epoch in (11, 21, 31, 41, 50)]
        with self.assertRaisesRegex(RuntimeError, "panel mismatch"):
            M["make_decision"](results, M["build_js_rows"](results))


if __name__ == "__main__":
    unittest.main()
