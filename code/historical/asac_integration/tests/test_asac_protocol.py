"""Unit tests for the frozen standalone ASAC controller/freezer."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "asac_protocol.py"
SPEC = importlib.util.spec_from_file_location("asac_protocol", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
asac = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = asac
SPEC.loader.exec_module(asac)


def completed_a2_rows(seed: int) -> list[dict]:
    controller = asac.A2Controller()
    rows = []
    for epoch in range(1, 51):
        stability = 0.58 + 0.0015 * math.sin(epoch / 3.0 + seed) + (0.00010 + seed * 0.000001) * epoch
        observation = controller.observe(epoch, stability)
        rows.append({
            "epoch": epoch, "variant": "A2", "seed": seed, "train_q": observation.train_q,
            "effective_scale": 0.5 * observation.train_q, "effective_translate": 0.1 * observation.train_q,
            "probe_q": 0.5, "o2m_stability": stability + 0.01, "o2o_stability": stability - 0.01,
            "S_e": stability, "S_ref": observation.stability_reference, "m_e": observation.state_ema,
            "target_q": observation.q_target, "next_q": observation.next_q, "measurement_seconds": 0.01,
            "panel_hash": asac.PANEL_SHA256, "object_population_hash": asac.OBJECT_POPULATION_SHA256,
            "source_hash": "0" * 64, "config_hash": "1" * 64,
        })
    return rows


def all_seed_rows() -> dict[int, list[dict]]:
    return {seed: completed_a2_rows(seed) for seed in asac.SEEDS}


class PoisonedMapping(Mapping):
    def __init__(self, values: dict, poisoned: set[str] | None = None):
        self.values = values
        self.poisoned = poisoned or set()
        self.accessed: list[str] = []

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        self.accessed.append(key)
        if key in self.poisoned:
            raise AssertionError(f"freezer illegally read {key}")
        return self.values[key]


class ConfigTests(unittest.TestCase):
    def test_default_config_is_exact(self):
        config = asac.ASACProtocolConfig.from_mapping()
        self.assertEqual(config.seeds, (0, 1, 42))
        self.assertEqual(config.q_max, 1.0)
        self.assertEqual(config.q_probe, 0.5)
        self.assertEqual(config.feedback_update_end, 49)

    def test_serialized_lists_are_accepted(self):
        self.assertEqual(asac.ASACProtocolConfig.from_mapping(asac.ASACProtocolConfig().to_dict()), asac.ASACProtocolConfig())

    def test_unknown_and_modified_config_fail(self):
        with self.assertRaisesRegex(asac.ProtocolError, "unknown"):
            asac.ASACProtocolConfig.from_mapping({"eta": 2.5})
        with self.assertRaisesRegex(asac.ProtocolError, "q_max is frozen"):
            asac.ASACProtocolConfig.from_mapping({"q_max": 0.75})
        with self.assertRaises(asac.ProtocolError):
            asac.ASACProtocolConfig.from_mapping({"epochs": True})


class UtilityTests(unittest.TestCase):
    def test_phases_have_no_cooldown(self):
        self.assertEqual([asac.phase_for_epoch(e) for e in (1, 5, 6, 50)], ["warmup", "warmup", "feedback", "feedback"])

    def test_reference_is_median_s3_s4_s5(self):
        self.assertEqual(asac.reference_stability({3: 0.8, 4: 0.2, 5: 0.6}), 0.6)
        with self.assertRaisesRegex(asac.ProtocolError, "missing"):
            asac.reference_stability({3: 0.8, 5: 0.6})

    def test_feedback_formula(self):
        step = asac.feedback_step(q_epoch=0.5, state_ema_previous=0.6, stability=0.7, stability_reference=0.6)
        self.assertAlmostEqual(step.state_ema, 0.62)
        self.assertAlmostEqual(step.log_q_target, 0.0)
        self.assertAlmostEqual(step.q_target, 1.0)
        self.assertAlmostEqual(step.delta_log_q, 0.05)
        self.assertAlmostEqual(step.q_next, 0.5 * math.exp(0.05))

    def test_feedback_rate_limit_and_q_bounds(self):
        up = asac.feedback_step(q_epoch=1.0, state_ema_previous=1.0, stability=1.0, stability_reference=0.0)
        down = asac.feedback_step(q_epoch=0.25, state_ema_previous=0.0, stability=0.0, stability_reference=1.0)
        self.assertEqual((up.delta_log_q, up.q_next), (0.05, 1.0))
        self.assertEqual((down.delta_log_q, down.q_next), (-0.05, 0.25))

    def test_effective_affine_and_fixed_probe_are_independent(self):
        self.assertEqual(asac.effective_affine(0.5), (0.25, 0.05))
        self.assertEqual(asac.effective_affine(1.0), (0.5, 0.1))
        self.assertEqual(asac.ASACProtocolConfig().q_probe, 0.5)


class ControllerTests(unittest.TestCase):
    def test_warmup_reference_and_q6(self):
        controller = asac.A2Controller()
        observations = [controller.observe(epoch, signal) for epoch, signal in enumerate([0.1, 0.2, 0.8, 0.2, 0.6], 1)]
        self.assertEqual(observations[-1].stability_reference, 0.6)
        self.assertEqual(observations[-1].state_ema, 0.6)
        self.assertEqual(controller.q_for_epoch(6), 0.5)

    def test_update_is_strictly_causal(self):
        controller = asac.A2Controller()
        for epoch in range(1, 6):
            controller.observe(epoch, 0.5)
        q6 = controller.q_for_epoch(6)
        observation6 = controller.observe(6, 0.7)
        self.assertEqual(observation6.train_q, q6)
        self.assertGreater(controller.q_for_epoch(7), q6)

    def test_out_of_order_and_invalid_signal_fail(self):
        controller = asac.A2Controller()
        with self.assertRaisesRegex(asac.ProtocolError, "expected epoch 1"):
            controller.observe(2, 0.5)
        with self.assertRaises(asac.ProtocolError):
            controller.observe(1, float("nan"))

    def test_epoch49_controls_q50_but_epoch50_produces_no_q51(self):
        controller = asac.A2Controller()
        for epoch in range(1, 50):
            observation = controller.observe(epoch, 0.61)
        self.assertIsNotNone(observation.next_q)
        q50 = controller.q_for_epoch(50)
        final = controller.observe(50, 0.99)
        self.assertEqual(final.train_q, q50)
        self.assertIsNone(final.next_q)
        self.assertIsNone(final.q_target)
        self.assertEqual(len(controller.schedule()), 50)

    def test_every_action_step_respects_log_rate_limit(self):
        qs = [row["train_q"] for row in completed_a2_rows(42)]
        self.assertTrue(all(abs(math.log(b) - math.log(a)) <= 0.05 + 1e-12 for a, b in zip(qs[5:], qs[6:])))


class FreezerTests(unittest.TestCase):
    def setUp(self):
        self.rows = all_seed_rows()

    def test_requires_all_three_live_a2_runs(self):
        with self.assertRaisesRegex(asac.ProtocolError, "exactly"):
            asac.freeze_control_schedules({42: self.rows[42]})

    def test_a1_is_frozen_cross_seed_derangement(self):
        bundle = asac.freeze_control_schedules(self.rows)
        self.assertEqual(asac.A1_SOURCE_SEED, {0: 42, 1: 0, 42: 1})
        for target, source in asac.A1_SOURCE_SEED.items():
            target_q = [row["train_q"] for row in bundle.schedules["A1"][target]]
            source_q = [row["train_q"] for row in self.rows[source]]
            self.assertEqual([q.hex() for q in target_q], [q.hex() for q in source_q])
            self.assertNotEqual(target, source)

    def test_a3_left_shift_mapping_and_multiset(self):
        mapping = asac.a3_source_epoch_map()
        self.assertEqual((mapping[1], mapping[5], mapping[6], mapping[40], mapping[41], mapping[50]), (1, 5, 16, 50, 6, 15))
        bundle = asac.freeze_control_schedules(self.rows)
        for seed in asac.SEEDS:
            rows = bundle.schedules["A3"][seed]
            self.assertEqual([row["train_q"] for row in rows[:5]], [0.5] * 5)
            self.assertEqual(sorted(row["train_q"].hex() for row in rows[5:]), sorted(row["train_q"].hex() for row in self.rows[seed][5:]))

    def test_a4_is_exact_same_seed_replay(self):
        bundle = asac.freeze_control_schedules(self.rows)
        for seed in asac.SEEDS:
            replay = bundle.schedules["A4"][seed]
            self.assertEqual([row["train_q"].hex() for row in replay], [row["train_q"].hex() for row in self.rows[seed]])
            self.assertEqual([row["source_epoch"] for row in replay], list(range(1, 51)))

    def test_manifest_pins_source_mappings_and_hashes(self):
        bundle = asac.freeze_control_schedules(
            self.rows, input_state_sha256_by_seed={seed: str(seed).zfill(64) for seed in asac.SEEDS}, source_sha256="f" * 64,
        )
        manifest = bundle.manifest
        self.assertEqual(manifest["a3_shift"], {"direction": "left", "epochs": 10, "phase": [6, 50]})
        self.assertEqual(manifest["a3_source_epoch_by_target_epoch"]["6"], 16)
        self.assertEqual(len(manifest["a3_source_epoch_mapping_sha256"]), 64)
        self.assertEqual(manifest["a1_source_seed_by_target_seed"], {"0": 42, "1": 0, "42": 1})
        self.assertEqual(manifest["performance_fields_consumed"], [])
        self.assertEqual(manifest["optional_integrity_fields_consumed"], ["seed"])

    def test_incomplete_or_invalid_hash_sets_fail_closed(self):
        with self.assertRaisesRegex(asac.ProtocolError, "hashes must cover exactly"):
            asac.freeze_control_schedules(self.rows, input_state_sha256_by_seed={42: "0" * 64})
        with self.assertRaisesRegex(asac.ProtocolError, "lowercase SHA256"):
            asac.freeze_control_schedules(
                self.rows,
                input_state_sha256_by_seed={seed: "0" * 64 for seed in asac.SEEDS},
                source_sha256="not-a-sha",
            )

    def test_freezer_is_deterministic_under_row_reordering(self):
        first = asac.freeze_control_schedules(self.rows)
        second = asac.freeze_control_schedules({seed: list(reversed(rows)) for seed, rows in self.rows.items()})
        self.assertEqual(first.schedules, second.schedules)
        self.assertEqual(first.manifest["output_schedule_sha256"], second.manifest["output_schedule_sha256"])

    def test_freezer_reads_only_schedule_values(self):
        wrapped = {}
        for seed, rows in self.rows.items():
            wrapped[seed] = []
            for row in rows:
                values = dict(row)
                values["opaque_provenance"] = object()
                wrapped[seed].append(PoisonedMapping(values, {"opaque_provenance"}))
        asac.freeze_control_schedules(wrapped)
        self.assertTrue(all("opaque_provenance" not in row.accessed for rows in wrapped.values() for row in rows))

    def test_performance_fields_are_rejected_without_value_access(self):
        wrapped = {seed: [PoisonedMapping(dict(row)) for row in rows] for seed, rows in self.rows.items()}
        wrapped[1][10].values["val/mAP50-95"] = object()
        wrapped[1][10].poisoned.add("val/mAP50-95")
        with self.assertRaisesRegex(asac.ProtocolError, "forbidden performance"):
            asac.freeze_control_schedules(wrapped)
        self.assertNotIn("val/mAP50-95", wrapped[1][10].accessed)

    def test_common_performance_aliases_are_rejected(self):
        for key in ("train_loss", "precision", "recall", "fitness", "results.csv", "test_ap", "val_accuracy", "auc"):
            rows = all_seed_rows()
            rows[0][0][key] = 0.0
            with self.subTest(key=key), self.assertRaises(asac.ProtocolError):
                asac.freeze_control_schedules(rows)

    def test_bad_epoch_seed_warmup_bound_and_rate_limit_fail_closed(self):
        cases = []
        duplicate = all_seed_rows(); duplicate[0][-1]["epoch"] = 49; cases.append(duplicate)
        wrong_seed = all_seed_rows(); wrong_seed[1][0]["seed"] = 42; cases.append(wrong_seed)
        warmup = all_seed_rows(); warmup[42][5]["train_q"] = 0.51; cases.append(warmup)
        bound = all_seed_rows(); bound[42][20]["train_q"] = 1.01; cases.append(bound)
        rate = all_seed_rows(); rate[42][20]["train_q"] = rate[42][19]["train_q"] * math.exp(0.051); cases.append(rate)
        for rows in cases:
            with self.subTest(), self.assertRaises(asac.ProtocolError):
                asac.freeze_control_schedules(rows)

    def test_raw_input_and_written_schedule_hashes_match(self):
        raw = {seed: b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows) for seed, rows in self.rows.items()}
        bundle = asac.freeze_jsonl_bytes_by_seed(raw)
        self.assertEqual(bundle.manifest["input_state_sha256_by_seed"], {str(seed): asac.sha256_bytes(raw[seed]) for seed in asac.SEEDS})
        with tempfile.TemporaryDirectory() as directory:
            paths = asac.write_frozen_schedules(bundle, directory)
            for variant in ("A1", "A3", "A4"):
                for seed in asac.SEEDS:
                    self.assertEqual(asac.sha256_file(paths[f"{variant.lower()}_seed{seed}"]), bundle.manifest["output_schedule_sha256"][variant][str(seed)])

    def test_unpinned_bundle_cannot_be_written(self):
        bundle = asac.freeze_control_schedules(self.rows)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(asac.ProtocolError, "unpinned"):
            asac.write_frozen_schedules(bundle, directory)

    def test_jsonl_rejects_duplicate_nonfinite_and_blank(self):
        with self.assertRaisesRegex(asac.ProtocolError, "duplicate JSON key"):
            asac.parse_a2_state_jsonl(b'{"epoch":1,"epoch":2}\n')
        with self.assertRaisesRegex(asac.ProtocolError, "non-finite"):
            asac.parse_a2_state_jsonl(b'{"epoch":NaN}\n')
        with self.assertRaisesRegex(asac.ProtocolError, "blank"):
            asac.parse_a2_state_jsonl(b'{"epoch":1}\n\n')


if __name__ == "__main__":
    unittest.main(verbosity=2)
