"""Pre-GPU unit tests for the isolated ASAC implementation."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snapshot" / "ultralytics_research"))
sys.path.insert(0, str(ROOT / "scripts"))

import asac_protocol  # noqa: E402
import ultralytics.models.yolo.detect.asac_train as asac_train_module  # noqa: E402
from ultralytics.data.augment import Compose, RandomPerspective  # noqa: E402
from ultralytics.models.yolo.detect.asac_train import (  # noqa: E402
    ASACTrainer,
    AdaptiveRandomPerspective,
    install_adaptive_random_perspective,
)
from ultralytics.utils.asac import (  # noqa: E402
    ASACPanelProtocol,
    affine_parameters,
    compare_object_records,
    deterministic_unit_triplet,
    js_divergence,
    sha256_json,
    smooth_occupancy,
)


class Holder:
    def __init__(self, pre_transform):
        self.pre_transform = pre_transform


class TestAdaptiveRandomPerspective(unittest.TestCase):
    @staticmethod
    def base(scale=0.5, translate=0.1):
        return RandomPerspective(
            degrees=0.0,
            translate=translate,
            scale=scale,
            shear=0.0,
            perspective=0.0,
            size=(640, 640),
        )

    def compare_matrix_and_rng(self, q: float, expected_scale: float, expected_translate: float):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        shared_q = torch.tensor([q], dtype=torch.float64).share_memory_()
        wrapper = AdaptiveRandomPerspective(self.base(), shared_q)
        expected = self.base(scale=expected_scale, translate=expected_translate)
        random.seed(12345)
        before = random.getstate()
        expected_matrix, expected_s = expected._compute_affine_matrix(image, (640, 640))
        expected_after = random.getstate()
        random.setstate(before)
        actual_matrix, actual_s = wrapper._compute_affine_matrix(image, (640, 640))
        actual_after = random.getstate()
        np.testing.assert_array_equal(actual_matrix, expected_matrix)
        self.assertEqual(actual_s, expected_s)
        self.assertEqual(actual_after, expected_after)
        self.assertEqual(wrapper.scale, 0.5)
        self.assertEqual(wrapper.translate, 0.1)

    def test_q1_exact_official_equivalence(self):
        self.compare_matrix_and_rng(1.0, 0.5, 0.1)

    def test_q_half_exact_effective_equivalence(self):
        self.compare_matrix_and_rng(0.5, 0.25, 0.05)

    def test_finally_restores_amplitudes(self):
        wrapper = AdaptiveRandomPerspective(self.base(), torch.tensor([0.5], dtype=torch.float64).share_memory_())
        with mock.patch.object(RandomPerspective, "_compute_affine_matrix", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                wrapper._compute_affine_matrix(np.zeros((8, 8, 3), np.uint8), (8, 8))
        self.assertEqual(wrapper.scale, 0.5)
        self.assertEqual(wrapper.translate, 0.1)

    def test_aliases_are_wrapped_once(self):
        affine = self.base()
        shared = Compose([affine])
        root = Compose([shared, Holder(shared), Holder(affine)])
        q = torch.tensor([0.5], dtype=torch.float64).share_memory_()
        wrappers = install_adaptive_random_perspective(root, q)
        self.assertEqual(len(wrappers), 1)
        wrapper = wrappers[0]
        self.assertIs(shared.transforms[0], wrapper)
        self.assertIs(root.transforms[1].pre_transform.transforms[0], wrapper)
        self.assertIs(root.transforms[2].pre_transform, wrapper)
        self.assertIs(wrapper._asac_q, q)


class TestP0ParityPrimitives(unittest.TestCase):
    def test_frozen_population_identity_hash(self):
        # The full remote data test independently rebuilds these 624 IDs. This catches accidental protocol edits.
        protocol = ASACPanelProtocol()
        self.assertEqual(protocol.expected_population_count, 624)
        self.assertEqual(
            protocol.expected_population_sha256,
            "f57cffdeb8bc29ef66a21c384ac0c3cd7ceff9552c48245263cb6c7c51b14926",
        )

    def test_deterministic_affine_direction(self):
        path = "/home/ubuntu/rz/yolo_study/datasets/VOC/images/train2007/000012.jpg"
        self.assertEqual(deterministic_unit_triplet(path, 42), deterministic_unit_triplet(path, 42))
        p25 = affine_parameters([path], 0.25, 42, 0.5, 0.1)[0]
        p50 = affine_parameters([path], 0.50, 42, 0.5, 0.1)[0]
        self.assertAlmostEqual(p50["scale"] - 1.0, 2 * (p25["scale"] - 1.0))
        self.assertAlmostEqual(p50["tx"], 2 * p25["tx"])
        self.assertAlmostEqual(p50["ty"], 2 * p25["ty"])

    def test_occupancy_and_missing_assignment(self):
        protocol = ASACPanelProtocol()
        points = [(0, 0.0, 0.0), (0, 0.1, -0.1)]
        occupancy = smooth_occupancy(points, protocol.grid_bins, protocol.grid_extent, protocol.kernel_sigma)
        self.assertAlmostEqual(float(occupancy.sum()), 1.0)
        self.assertEqual(js_divergence(occupancy, occupancy), 0.0)
        self.assertEqual(compare_object_records({"points": points}, None, protocol)["stability"], 0.0)

    def test_canonical_json_hash_is_order_invariant(self):
        self.assertEqual(sha256_json({"b": 2, "a": 1}), sha256_json({"a": 1, "b": 2}))


class TestTrainerControllerParity(unittest.TestCase):
    def test_trainer_matches_standalone_protocol_all_50_epochs(self):
        standalone = asac_protocol.A2Controller()
        trainer = object.__new__(ASACTrainer)
        trainer._asac_signals = {}
        trainer._asac_s_ref = float("nan")
        trainer._asac_state_ema = float("nan")
        trainer._asac_q_target = float("nan")
        for epoch in range(1, 51):
            signal = 0.61 + 0.0003 * epoch + 0.002 * np.sin(epoch / 4)
            trainer._asac_epoch_q = standalone.q_for_epoch(epoch)
            actual_next, actual_ref, actual_m, actual_target = trainer._controller_update(epoch, signal)
            expected = standalone.observe(epoch, signal)
            if expected.next_q is None:
                self.assertIsNone(actual_next)
            else:
                self.assertAlmostEqual(actual_next, expected.next_q, places=13)
            if expected.stability_reference is None:
                self.assertTrue(np.isnan(actual_ref))
            else:
                self.assertAlmostEqual(actual_ref, expected.stability_reference, places=13)
            if expected.state_ema is None:
                self.assertTrue(np.isnan(actual_m))
            else:
                self.assertAlmostEqual(actual_m, expected.state_ema, places=13)
            if expected.q_target is None:
                self.assertIsNone(actual_target)
            else:
                self.assertAlmostEqual(actual_target, expected.q_target, places=13)


class TestCollectiveTrainingContractValidation(unittest.TestCase):
    @staticmethod
    def trainer():
        trainer = object.__new__(ASACTrainer)
        trainer.device = torch.device("cpu")
        return trainer

    @staticmethod
    def all_gather_with(statuses):
        def populate(outputs, local_status):
            for output, status in zip(outputs, statuses):
                output.fill_(status)

        return populate

    def test_single_process_preserves_validation_behavior(self):
        trainer = self.trainer()
        trainer._validate_training_contract = mock.Mock()
        with mock.patch.object(asac_train_module, "RANK", -1):
            trainer._validate_training_contract_collectively()
        trainer._validate_training_contract.assert_called_once_with()

        local_error = ValueError("local mismatch")
        trainer._validate_training_contract = mock.Mock(side_effect=local_error)
        with mock.patch.object(asac_train_module, "RANK", -1):
            with self.assertRaises(ValueError) as caught:
                trainer._validate_training_contract_collectively()
        self.assertIs(caught.exception, local_error)

    def test_peer_failure_is_observed_before_rank0_panel_setup(self):
        trainer = self.trainer()
        trainer._validate_training_contract = mock.Mock()
        collective_order = []

        def gather(outputs, local_status):
            collective_order.append("all_gather")
            self.assertEqual(int(local_status.item()), 1)
            self.all_gather_with([1, 0, 1, 1])(outputs, local_status)

        with (
            mock.patch.object(asac_train_module, "RANK", 0),
            mock.patch.object(asac_train_module.dist, "is_available", return_value=True),
            mock.patch.object(asac_train_module.dist, "is_initialized", return_value=True),
            mock.patch.object(asac_train_module.dist, "get_world_size", return_value=4),
            mock.patch.object(asac_train_module.dist, "all_gather", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, r"failed on ranks \[1\]"):
                trainer._validate_training_contract_collectively()
        self.assertEqual(collective_order, ["all_gather"])

        peer_failure = RuntimeError("peer rejected contract")
        with (
            mock.patch.object(asac_train_module.DetectionTrainer, "_setup_train"),
            mock.patch.object(
                trainer, "_validate_training_contract_collectively", side_effect=peer_failure
            ),
            mock.patch.object(asac_train_module, "ASACPanelEvaluator") as panel,
        ):
            with self.assertRaises(RuntimeError) as caught:
                trainer._setup_train()
        self.assertIs(caught.exception, peer_failure)
        panel.assert_not_called()

    def test_local_failure_participates_in_collective_and_keeps_cause(self):
        trainer = self.trainer()
        local_error = ValueError("rank-local mismatch")
        trainer._validate_training_contract = mock.Mock(side_effect=local_error)

        def gather(outputs, local_status):
            self.assertEqual(int(local_status.item()), 0)
            self.all_gather_with([1, 1, 0, 1])(outputs, local_status)

        with (
            mock.patch.object(asac_train_module, "RANK", 2),
            mock.patch.object(asac_train_module.dist, "is_available", return_value=True),
            mock.patch.object(asac_train_module.dist, "is_initialized", return_value=True),
            mock.patch.object(asac_train_module.dist, "get_world_size", return_value=4),
            mock.patch.object(asac_train_module.dist, "all_gather", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, r"failed on ranks \[2\]") as caught:
                trainer._validate_training_contract_collectively()
        self.assertIs(caught.exception.__cause__, local_error)

    def test_epoch_q_mismatch_fails_every_rank_before_batch_loading(self):
        trainer = self.trainer()
        trainer._asac_shared_q = torch.tensor([0.25], dtype=torch.float64).share_memory_()
        trainer._schedule_q = mock.Mock(return_value=0.5)

        def gather(outputs, local_packet):
            self.assertEqual(local_packet.tolist(), [1.0, 0.5])
            packets = ([1.0, 0.5], [1.0, 0.5], [1.0, 0.55], [1.0, 0.5])
            for output, packet in zip(outputs, packets):
                output.copy_(torch.tensor(packet, dtype=output.dtype))

        with (
            mock.patch.object(asac_train_module, "RANK", 0),
            mock.patch.object(asac_train_module.dist, "is_available", return_value=True),
            mock.patch.object(asac_train_module.dist, "is_initialized", return_value=True),
            mock.patch.object(asac_train_module.dist, "get_world_size", return_value=4),
            mock.patch.object(asac_train_module.dist, "all_gather", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, r"q mismatch across ranks at epoch 7"):
                trainer._set_epoch_q_collectively(7)
        self.assertEqual(trainer._asac_epoch_q, 0.5)
        self.assertEqual(trainer._asac_shared_q.item(), 0.5)

    def test_epoch_q_local_error_still_participates_in_collective(self):
        trainer = self.trainer()
        trainer._asac_shared_q = torch.tensor([0.5], dtype=torch.float64).share_memory_()
        local_error = ValueError("missing schedule entry")
        trainer._schedule_q = mock.Mock(side_effect=local_error)

        def gather(outputs, local_packet):
            self.assertEqual(int(local_packet[0].item()), 0)
            packets = ([1.0, 0.5], [0.0, float("nan")], [1.0, 0.5], [1.0, 0.5])
            for output, packet in zip(outputs, packets):
                output.copy_(torch.tensor(packet, dtype=output.dtype))

        with (
            mock.patch.object(asac_train_module, "RANK", 1),
            mock.patch.object(asac_train_module.dist, "is_available", return_value=True),
            mock.patch.object(asac_train_module.dist, "is_initialized", return_value=True),
            mock.patch.object(asac_train_module.dist, "get_world_size", return_value=4),
            mock.patch.object(asac_train_module.dist, "all_gather", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, r"q setup failed on ranks \[1\]") as caught:
                trainer._set_epoch_q_collectively(9)
        self.assertIs(caught.exception.__cause__, local_error)

    def test_epoch_record_contains_population_and_source_provenance(self):
        trainer = self.trainer()
        trainer._asac_variant = "A0"
        trainer._asac_epoch_q = 1.0
        trainer._asac_config = {"sha256": "config-hash"}
        trainer._asac_source = {
            "trainer_sha256": "trainer-hash",
            "measurement_sha256": "measurement-hash",
            "bundle_sha256": "bundle-hash",
        }
        trainer._asac_panel = SimpleNamespace(
            loader_metadata={"population_count": 624, "population_sha256": "population-hash"}
        )
        trainer.args = SimpleNamespace(seed=42)
        measurement = {
            "joint_stability": 0.8,
            "strength": 0.5,
            "elapsed_seconds": 1.25,
            "panel_content_sha256": "panel-hash",
            "autocast": {"autocast_enabled": True},
            "integrity": {"parameters_sha256": "parameters-hash"},
            "rng_sha256": "rng-hash",
            "assigner_oom_fallback": {"O2M": False, "O2O": False},
            "heads": {
                head: {
                    "mean_stability": 0.8,
                    "object_pairs": 624,
                    "assigned_both": 620,
                    "assignment_coverage": 0.99,
                }
                for head in ("O2M", "O2O")
            },
        }
        record = trainer._make_epoch_record(1, measurement)
        self.assertEqual(record["population_count"], 624)
        self.assertEqual(record["population_sha256"], "population-hash")
        self.assertEqual(record["trainer_source_sha256"], "trainer-hash")
        self.assertEqual(record["measurement_source_sha256"], "measurement-hash")
        self.assertEqual(record["source_bundle_sha256"], "bundle-hash")
        self.assertEqual(record["panel_rng_sha256"], "rng-hash")
        self.assertEqual(record["panel_assigner_oom_fallback"], {"O2M": False, "O2O": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
