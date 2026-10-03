"""CPU-only tests for the ASAC probe's geometry, metric, and decision logic."""

from __future__ import annotations

import importlib.util
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


SCRIPT = Path(__file__).with_name("probe_asac_assignment_stability.py")
SPEC = importlib.util.spec_from_file_location("asac_probe_test_module", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def protocol():
    return {
        "expected_epochs": [11, 21, 31, 41, 50],
        "strengths": [0.25, 0.5, 0.75],
        "min_object_pairs": 100,
        "min_assignment_coverage": 0.5,
        "monotonic_tolerance": 0.01,
        "min_weak_strong_drop": 0.02,
        "min_monotonic_checkpoints": 3,
        "min_spearman": 0.2,
        "min_correlated_checkpoints": 2,
    }


def checkpoints(panel="fixed-panel"):
    return [{"epoch": epoch, "panel_sha256": panel} for epoch in (11, 21, 31, 41, 50)]


def summaries(curves=None, rho=0.3, objects=150, matched=150):
    strengths = (0.25, 0.5, 0.75)
    curves = curves or {"O2M": (0.90, 0.82, 0.70), "O2O": (0.88, 0.80, 0.68)}
    rows = []
    for epoch in (11, 21, 31, 41, 50):
        for head in MODULE.HEADS:
            for strength, stability in zip(strengths, curves[head]):
                rows.append(
                    {
                        "epoch": epoch,
                        "head": head,
                        "strength": strength,
                        "object_pairs": objects,
                        "assigned_both": matched,
                        "assignment_coverage": matched / objects,
                        "mean_stability": stability,
                        "spearman_stability_quality_assigned": rho,
                    }
                )
    return rows


def frozen_args(**updates):
    values = {key: value for key, value in MODULE.FROZEN_PROTOCOL.items() if key != "panel_images"}
    values.update(updates)
    return SimpleNamespace(**values)


TEST_ROOT = Path(tempfile.gettempdir()).resolve() / "asac_b50_unit"


def b50_manifest(data=None, model=None, project=None):
    project = Path(project or (TEST_ROOT / "runs")).resolve()
    model = Path(model or (TEST_ROOT / "configs" / "models" / "yolo26n_baseline.yaml")).resolve()
    data = Path(data or (TEST_ROOT / "configs" / "datasets" / "voc_server.yaml")).resolve()
    values = dict(MODULE.FROZEN_B50_ARGS)
    values.update(
        {
            "model": str(model),
            "data": str(data),
            "project": str(project),
            "save_dir": str((project / values["name"]).resolve()),
        }
    )
    return values


def b50_checkpoint_manifest(data=None, model=None, project=None):
    return MODULE.checkpoint_runtime_manifest(
        b50_manifest(data=data, model=model, project=project)
    )


def b50_train_args(data=None, model=None, project=None, include_tqgb=False):
    values = b50_checkpoint_manifest(data=data, model=model, project=project)
    if not include_tqgb:
        for key in MODULE.SEMANTIC_DEFAULTS:
            values.pop(key)
    return values


def checkpoint_payload(epoch, data=None, stripped=False):
    return {
        "epoch": -1 if stripped else epoch - 1,
        "train_args": b50_train_args(data),
        "train_results": {"epoch": list(range(1, epoch + 1)), "metric": [0.0] * epoch},
        "ema": object(),
        "git": {"commit": "fixed"},
        "version": "test",
    }


def expected_structure(**updates):
    payload = {
        "model_class": MODULE.EXPECTED_B50_MODEL_STRUCTURE["model_class"],
        "yaml": {"nc": 20, "scale": "n"},
        "names": {index: name for index, name in enumerate(MODULE.EXPECTED_B50_MODEL_STRUCTURE["names"])},
        "head_class": MODULE.EXPECTED_B50_MODEL_STRUCTURE["head_class"],
        "nc": 20,
        "stride": [8.0, 16.0, 32.0],
        "parameter_count": MODULE.EXPECTED_B50_MODEL_STRUCTURE["parameter_count"],
    }
    payload.update(updates)
    payload["sha256"] = MODULE.sha256_json(payload)
    return payload


def create_provenance_fixture(directory):
    base = Path(directory).resolve()
    project = base / "runs"
    run = project / MODULE.FROZEN_B50_ARGS["name"]
    weights = run / "weights"
    model = base / "configs" / "models" / "yolo26n_baseline.yaml"
    data = base / "configs" / "datasets" / "voc_server.yaml"
    weights.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    model.write_text("nc: 20\nscale: n\n", encoding="utf-8")
    data.write_text("path: .\ntrain: images/train\nnames: []\n", encoding="utf-8")
    frozen = b50_manifest(data=data, model=model, project=project)
    run_args = dict(frozen)
    for key in MODULE.SEMANTIC_DEFAULTS:
        run_args.pop(key)
    (run / "args.yaml").write_text(yaml.safe_dump(run_args, sort_keys=False), encoding="utf-8")
    (run / "results.csv").write_text(
        "epoch,metric\n" + "".join(f"{epoch},0\n" for epoch in range(1, 51)), encoding="utf-8"
    )
    records = []
    structure = expected_structure()
    frozen_checkpoint = MODULE.checkpoint_runtime_manifest(frozen)
    for filename, epoch in MODULE.EXPECTED_CHECKPOINT_FILES.items():
        trajectory_args = dict(frozen_checkpoint)
        records.append(
            {
                "path": str(weights / filename),
                "epoch": epoch,
                "trajectory_args": trajectory_args,
                "trajectory_args_sha256": MODULE.sha256_json(trajectory_args),
                "checkpoint_sha256": MODULE.EXPECTED_CHECKPOINT_SHA256[filename],
                "git": {"commit": "fixed"},
                "model_structure": copy.deepcopy(structure),
            }
        )
    return run, data, frozen, records


def mutate_all_records(records, **updates):
    changed = copy.deepcopy(records)
    for row in changed:
        row["trajectory_args"].update(updates)
        row["trajectory_args_sha256"] = MODULE.sha256_json(row["trajectory_args"])
    return changed


class GeometryAndMetricTest(unittest.TestCase):
    def test_affine_parameters_are_deterministic_and_nested(self):
        weak = MODULE.affine_parameters(["/train/a.jpg"], 0.25, 42, 0.5, 0.1)[0]
        strong = MODULE.affine_parameters(["/train/a.jpg"], 0.75, 42, 0.5, 0.1)[0]
        repeat = MODULE.affine_parameters(["/train/a.jpg"], 0.25, 42, 0.5, 0.1)[0]
        self.assertEqual(weak, repeat)
        self.assertAlmostEqual((strong["scale"] - 1.0) / (weak["scale"] - 1.0), 3.0)
        self.assertAlmostEqual(strong["tx"] / weak["tx"], 3.0)
        self.assertAlmostEqual(strong["ty"] / weak["ty"], 3.0)

    def test_smooth_occupancy_js_extremes(self):
        point = [(0, 0.0, 0.0)]
        same_left = MODULE.smooth_occupancy(point, 13, 1.5, 0.25)
        same_right = MODULE.smooth_occupancy(point, 13, 1.5, 0.25)
        other_level = MODULE.smooth_occupancy([(2, 0.0, 0.0)], 13, 1.5, 0.25)
        self.assertAlmostEqual(MODULE.js_divergence(same_left, same_right), 0.0)
        self.assertAlmostEqual(MODULE.js_divergence(same_left, other_level), 1.0)

    def test_quality_is_not_used_as_occupancy_weight(self):
        left = {"points": [(1, 0.0, 0.0)], "qualities": [0.1], "target_scores": [0.2]}
        right = {"points": [(1, 0.0, 0.0)], "qualities": [0.9], "target_scores": [0.8]}
        result = MODULE.compare_object_records(left, right, 13, 1.5, 0.25)
        self.assertAlmostEqual(result["stability"], 1.0)
        self.assertAlmostEqual(result["quality_agreement"], 1.0 / 9.0)

    def test_missing_assignment_scores_zero_stability(self):
        present = {"points": [(0, 0.0, 0.0)], "qualities": [0.5], "target_scores": [0.5]}
        result = MODULE.compare_object_records(present, None, 13, 1.5, 0.25)
        self.assertFalse(result["assigned_both"])
        self.assertEqual(result["stability"], 0.0)
        self.assertEqual(result["js_base2"], 1.0)

    def test_spearman_known_order_and_ties(self):
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertIsNone(MODULE.spearman([1, 1, 1], [1, 2, 3]))

    def test_geometry_survival_reports_retained_and_dropped_distributions(self):
        rows = [
            {
                "area_fraction": 0.01,
                "sqrt_area_fraction": 0.1,
                "box_boundary_margin": 0.2,
                "center_edge_distance": 0.3,
                "retained_q0.25": True,
                "retained_q0.50": True,
                "retained_q0.75": True,
                "retained_all": True,
            },
            {
                "area_fraction": 0.25,
                "sqrt_area_fraction": 0.5,
                "box_boundary_margin": 0.0,
                "center_edge_distance": 0.1,
                "retained_q0.25": True,
                "retained_q0.50": False,
                "retained_q0.75": False,
                "retained_all": False,
            },
        ]
        summary = MODULE.summarize_geometry(rows, [0.25, 0.5, 0.75])
        self.assertEqual(summary["clean_gt_total"], 2)
        self.assertEqual(summary["intersection_count"], 1)
        self.assertEqual(summary["dropped_distributions"]["area_fraction"]["p50"], 0.25)


class RuntimeContractTest(unittest.TestCase):
    def make_fake_trainer(self, o2o_topk=7, o2o_topk2=1):
        Assigner = type("TaskAlignedAssigner", (), {})
        Assigner.__module__ = "ultralytics.utils.tal"
        Loss = type("v8DetectionLoss", (), {"get_assigned_targets_and_loss": lambda self: None})
        Loss.__module__ = "ultralytics.utils.loss"
        Criterion = type("E2ELoss", (), {})
        Criterion.__module__ = "ultralytics.utils.loss"
        Head = type("Detect", (), {})
        Head.__module__ = "ultralytics.nn.modules.head"
        o2m, o2o = Loss(), Loss()
        o2m.assigner, o2o.assigner = Assigner(), Assigner()
        o2m.assigner.topk, o2m.assigner.topk2, o2m.stride = 10, 10, [8, 16, 32]
        o2o.assigner.topk, o2o.assigner.topk2, o2o.stride = o2o_topk, o2o_topk2, [8, 16, 32]
        criterion = Criterion()
        criterion.one2many, criterion.one2one = o2m, o2o
        head = Head()
        head.one2one_cv2, head.end2end = object(), True
        model = SimpleNamespace(criterion=criterion, model=[head])
        return SimpleNamespace(model=model)

    def test_exact_native_assigner_contract_passes(self):
        record = MODULE.validate_criterion_contract(self.make_fake_trainer(), [8.0, 16.0, 32.0])
        self.assertEqual(record["branches"]["one2one"]["topk2"], 1)

    def test_legacy_direct_top1_o2o_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "topk/topk2"):
            MODULE.validate_criterion_contract(self.make_fake_trainer(o2o_topk=1, o2o_topk2=1), [8, 16, 32])

    def test_integrity_hash_change_is_fatal(self):
        before = {
            "parameter_sha256": "p",
            "bn_buffer_sha256": "b",
            "parameter_tensors": 10,
            "bn_buffer_tensors": 3,
            "bn_total": 2,
            "bn_all_eval": True,
            "optimizer_created": False,
        }
        MODULE.assert_integrity_unchanged(before, dict(before), "checkpoint.pt")
        changed = dict(before, bn_buffer_sha256="changed")
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            MODULE.assert_integrity_unchanged(before, changed, "checkpoint.pt")


class ProvenanceTest(unittest.TestCase):
    def test_numbered_checkpoint_requires_unstripped_exact_epoch(self):
        record = MODULE.checkpoint_record_from_payload("/tmp/weights/epoch10.pt", checkpoint_payload(11))
        self.assertEqual(record["epoch"], 11)
        self.assertEqual(record["trajectory_args"]["tqgb_cls_m2o_gain"], 0.0)
        self.assertEqual(record["trajectory_args"]["tqgb_cls_o2m_gain"], 0.0)
        self.assertEqual(record["trajectory_args"]["tqgb_direction"], "none")
        with self.assertRaisesRegex(RuntimeError, "stripped numbered"):
            MODULE.checkpoint_record_from_payload(
                "/tmp/weights/epoch10.pt", checkpoint_payload(11, stripped=True)
            )

    def test_best_checkpoint_is_always_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "best.pt"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/best.pt", checkpoint_payload(50))

    def test_last_requires_full_50_row_evidence(self):
        record = MODULE.checkpoint_record_from_payload(
            "/tmp/weights/last.pt", checkpoint_payload(50, stripped=True)
        )
        self.assertTrue(record["stripped"])
        bad = checkpoint_payload(50, stripped=True)
        bad["train_results"] = {"epoch": list(range(1, 50)), "metric": [0.0] * 49}
        with self.assertRaisesRegex(RuntimeError, "do not prove completion"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/last.pt", bad)

    def test_stripped_last_recovers_only_save_dir_from_exact_run_path(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve() / "runs" / MODULE.FROZEN_B50_ARGS["name"]
            path = run / "weights" / "last.pt"
            payload = checkpoint_payload(50, stripped=True)
            payload["train_args"].pop("save_dir")
            record = MODULE.checkpoint_record_from_payload(path, payload)
            self.assertEqual(record["trajectory_args"]["save_dir"], str(run.resolve()))
            self.assertEqual(
                record["terminal_stripped_recovery"]["recovered_fields"], ["save_dir"]
            )
            self.assertEqual(
                record["terminal_stripped_recovery"]["save_dir_source"],
                "checkpoint_path.parent.parent",
            )

    def test_terminal_recovery_rejects_any_second_missing_field(self):
        payload = checkpoint_payload(50, stripped=True)
        payload["train_args"].pop("save_dir")
        payload["train_args"].pop("lr0")
        with self.assertRaisesRegex(RuntimeError, "lacks frozen B50 manifest fields.*lr0.*save_dir"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/last.pt", payload)

    def test_unstripped_last_cannot_recover_save_dir(self):
        payload = checkpoint_payload(50, stripped=False)
        payload["train_args"].pop("save_dir")
        with self.assertRaisesRegex(RuntimeError, "lacks frozen B50 manifest fields.*save_dir"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/last.pt", payload)

    def test_realistic_five_checkpoint_panel_accepts_only_terminal_save_dir_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, _ = create_provenance_fixture(directory)
            records = []
            for filename, epoch in MODULE.EXPECTED_CHECKPOINT_FILES.items():
                train_args = b50_train_args(
                    data=data, model=frozen["model"], project=frozen["project"]
                )
                stripped = filename == "last.pt"
                if stripped:
                    train_args.pop("save_dir")
                payload = {
                    "epoch": -1 if stripped else epoch - 1,
                    "train_args": train_args,
                    "train_results": {
                        "epoch": list(range(1, epoch + 1)),
                        "metric": [0.0] * epoch,
                    },
                    "ema": object(),
                    "git": {"commit": "fixed"},
                    "version": "test",
                }
                record = MODULE.checkpoint_record_from_payload(
                    run / "weights" / filename, payload
                )
                record["checkpoint_sha256"] = MODULE.EXPECTED_CHECKPOINT_SHA256[filename]
                record["model_structure"] = expected_structure()
                records.append(record)
            evidence = MODULE.validate_b50_trajectory(
                records, str(data), frozen_manifest=frozen
            )
            self.assertEqual(len({row["trajectory_args_sha256"] for row in records}), 1)
            self.assertEqual(evidence["trajectory_args"]["save_dir"], str(run.resolve()))
            recovered = [row for row in records if row["terminal_stripped_recovery"]]
            self.assertEqual([row["filename"] for row in recovered], ["last.pt"])

    def test_launch_and_checkpoint_warmup_bias_lr_runtime_transform_is_exact(self):
        self.assertEqual(MODULE.FROZEN_B50_ARGS["optimizer"], "auto")
        self.assertEqual(MODULE.FROZEN_B50_ARGS["warmup_bias_lr"], 0.1)
        self.assertEqual(MODULE.FROZEN_B50_CHECKPOINT_ARGS["warmup_bias_lr"], 0.0)
        self.assertNotEqual(
            MODULE.FROZEN_B50_MANIFEST_SHA256,
            MODULE.FROZEN_B50_CHECKPOINT_MANIFEST_SHA256,
        )
        transformed = MODULE.checkpoint_runtime_manifest(MODULE.FROZEN_B50_ARGS)
        differing = {
            key
            for key in MODULE.FROZEN_B50_ARGS
            if transformed[key] != MODULE.FROZEN_B50_ARGS[key]
        }
        self.assertEqual(differing, {"warmup_bias_lr"})

    def test_checkpoint_launch_value_and_args_runtime_value_are_both_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, records = create_provenance_fixture(directory)
            wrong_checkpoint_records = mutate_all_records(records, warmup_bias_lr=0.1)
            with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                MODULE.validate_b50_trajectory(
                    wrong_checkpoint_records, str(data), frozen_manifest=frozen
                )

            wrong_run_args = dict(frozen, warmup_bias_lr=0.0)
            (run / "args.yaml").write_text(
                yaml.safe_dump(wrong_run_args, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "args.yaml is not the frozen B50 manifest"):
                MODULE.validate_b50_trajectory(records, str(data), frozen_manifest=frozen)

    def test_checkpoint_rejects_missing_nonsemantic_manifest_field(self):
        payload = checkpoint_payload(11)
        payload["train_args"].pop("lr0")
        with self.assertRaisesRegex(RuntimeError, "lacks frozen B50 manifest fields.*lr0"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/epoch10.pt", payload)

    def test_checkpoint_rejects_fields_outside_frozen_manifest(self):
        payload = checkpoint_payload(11)
        payload["train_args"]["future_training_gain"] = 0.25
        with self.assertRaisesRegex(RuntimeError, "fields outside the frozen B50 manifest"):
            MODULE.checkpoint_record_from_payload("/tmp/weights/epoch10.pt", payload)

    def test_exact_frozen_run_passes_and_mixed_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            evidence = MODULE.validate_b50_trajectory(records, str(data), frozen_manifest=frozen)
            self.assertEqual(evidence["run_evidence"]["results_csv"]["rows"], 50)
            self.assertEqual(evidence["frozen_manifest_sha256"], MODULE.sha256_json(frozen))
            self.assertTrue(evidence["run_evidence"]["project_name_relation_verified"])
            mixed = copy.deepcopy(records)
            mixed[1]["trajectory_args"] = dict(mixed[1]["trajectory_args"], seed=0)
            mixed[1]["trajectory_args_sha256"] = MODULE.sha256_json(mixed[1]["trajectory_args"])
            with self.assertRaisesRegex(RuntimeError, "mixed trajectories"):
                MODULE.validate_b50_trajectory(mixed, str(data), frozen_manifest=frozen)

    def test_consistent_wrong_model_data_and_name_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            attacks = {
                "model": str((Path(directory) / "not-yolo26n.yaml").resolve()),
                "data": str((Path(directory) / "not-voc.yaml").resolve()),
                "name": "not-the-b50-run",
            }
            for field, value in attacks.items():
                with self.subTest(field=field):
                    malicious = mutate_all_records(records, **{field: value})
                    with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                        MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_every_augmentation_field_is_individually_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            attacks = {
                "hsv_h": 0.02,
                "hsv_s": 0.6,
                "hsv_v": 0.3,
                "degrees": 25.0,
                "translate": 0.2,
                "scale": 0.4,
                "shear": 2.0,
                "perspective": 0.001,
                "flipud": 0.1,
                "fliplr": 0.4,
                "bgr": 0.1,
                "mosaic": 0.8,
                "close_mosaic": 0,
                "mixup": 0.1,
                "cutmix": 0.1,
                "copy_paste": 0.1,
                "copy_paste_mode": "mixup",
                "auto_augment": "autoaugment",
                "erasing": 0.2,
            }
            for field, value in attacks.items():
                with self.subTest(field=field):
                    malicious = mutate_all_records(records, **{field: value})
                    with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                        MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_nonzero_tqgb_gains_and_wrong_direction_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            attacks = {
                "tqgb_cls_m2o_gain": 0.08,
                "tqgb_cls_o2m_gain": 0.15,
                "tqgb_direction": "bidirectional",
            }
            for field, value in attacks.items():
                with self.subTest(field=field):
                    malicious = mutate_all_records(records, **{field: value})
                    with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                        MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_full_optimizer_loss_and_amp_manifest_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            attacks = {
                "optimizer": "SGD",
                "amp": "fp16",
                "cls_remap": False,
                "lr0": 0.02,
                "lrf": 0.1,
                "weight_decay": 0.001,
                "warmup_bias_lr": 0.1,
                "box": 8.0,
                "cls": 0.75,
                "dfl": 2.0,
                "o2o_kd_gain": 0.1,
            }
            for field, value in attacks.items():
                with self.subTest(field=field):
                    malicious = mutate_all_records(records, **{field: value})
                    with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                        MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_consistent_reaudit_malicious_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, records = create_provenance_fixture(directory)
            malicious = mutate_all_records(
                records,
                model=str((Path(directory) / "not-yolo26n.yaml").resolve()),
                degrees=25.0,
                tqgb_cls_m2o_gain=0.08,
                tqgb_cls_o2m_gain=0.15,
                tqgb_direction="bidirectional",
                project=str((Path(directory) / "other-project").resolve()),
            )
            # Reproduce the audit's strongest counterexample: every checkpoint
            # and args.yaml agree with one another, but all describe the wrong run.
            malicious_run_args = dict(malicious[0]["trajectory_args"])
            (run / "args.yaml").write_text(
                yaml.safe_dump(malicious_run_args, sort_keys=False), encoding="utf-8"
            )
            self.assertEqual(len({row["trajectory_args_sha256"] for row in malicious}), 1)
            with self.assertRaisesRegex(RuntimeError, "not the frozen B50 manifest"):
                MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_run_directory_must_equal_project_and_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, records = create_provenance_fixture(directory)
            wrong_project = (Path(directory) / "other-project").resolve()
            wrong_frozen = dict(
                frozen,
                project=str(wrong_project),
                save_dir=str((wrong_project / frozen["name"]).resolve()),
            )
            records = mutate_all_records(
                records, project=wrong_frozen["project"], save_dir=wrong_frozen["save_dir"]
            )
            run_args = dict(wrong_frozen)
            for key in MODULE.SEMANTIC_DEFAULTS:
                run_args.pop(key)
            (run / "args.yaml").write_text(yaml.safe_dump(run_args, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "run directory must equal project/name"):
                MODULE.validate_b50_trajectory(records, str(data), frozen_manifest=wrong_frozen)

    def test_wrong_yolo26n_voc20_structure_is_rejected_even_when_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            wrong = expected_structure(nc=80, yaml={"nc": 80, "scale": "n"})
            malicious = copy.deepcopy(records)
            for row in malicious:
                row["model_structure"] = copy.deepcopy(wrong)
            with self.assertRaisesRegex(RuntimeError, "not the frozen YOLO26n/VOC20 model structure"):
                MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_args_yaml_must_independently_match_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, records = create_provenance_fixture(directory)
            wrong = dict(frozen, amp="fp16")
            (run / "args.yaml").write_text(yaml.safe_dump(wrong, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "args.yaml is not the frozen B50 manifest"):
                MODULE.validate_b50_trajectory(records, str(data), frozen_manifest=frozen)

    def test_missing_checkpoint_or_args_yaml_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frozen, records = create_provenance_fixture(directory)
            missing_checkpoint = copy.deepcopy(records)
            for row in missing_checkpoint:
                row["trajectory_args"].pop("lr0")
                row["trajectory_args_sha256"] = MODULE.sha256_json(row["trajectory_args"])
            with self.assertRaisesRegex(RuntimeError, "lacks frozen B50 manifest fields.*lr0"):
                MODULE.validate_b50_trajectory(missing_checkpoint, str(data), frozen_manifest=frozen)

            missing_yaml = dict(frozen)
            missing_yaml.pop("box")
            (run / "args.yaml").write_text(yaml.safe_dump(missing_yaml, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lacks frozen B50 manifest fields.*box"):
                MODULE.validate_b50_trajectory(records, str(data), frozen_manifest=frozen)

    def test_manifest_hash_cannot_be_spoofed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            malicious = copy.deepcopy(records)
            malicious[0]["trajectory_args_sha256"] = "same-signature-is-not-proof"
            with self.assertRaisesRegex(RuntimeError, "manifest hash is invalid"):
                MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)

    def test_checkpoint_content_hash_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, frozen, records = create_provenance_fixture(directory)
            malicious = copy.deepcopy(records)
            malicious[0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "does not match the pinned B50 artifacts"):
                MODULE.validate_b50_trajectory(malicious, str(data), frozen_manifest=frozen)


class DecisionTest(unittest.TestCase):
    def test_go_for_monotonic_quality_relevant_signal(self):
        decision = MODULE.make_decision(checkpoints(), summaries(), protocol())
        self.assertEqual(decision["verdict"], "GO")
        self.assertEqual(decision["head_evidence"]["O2M"]["correlated_checkpoint_count"], 5)

    def test_no_go_for_nonmonotonic_o2o_signal(self):
        curves = {"O2M": (0.90, 0.82, 0.70), "O2O": (0.70, 0.90, 0.80)}
        decision = MODULE.make_decision(checkpoints(), summaries(curves=curves), protocol())
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertIn("intensity-sensitive", decision["reason"])

    def test_no_go_for_quality_irrelevant_signal(self):
        decision = MODULE.make_decision(checkpoints(), summaries(rho=0.1), protocol())
        self.assertEqual(decision["verdict"], "NO-GO")
        self.assertIn("quality", decision["reason"])

    def test_inconclusive_for_too_few_assigned_pairs(self):
        decision = MODULE.make_decision(checkpoints(), summaries(objects=150, matched=80), protocol())
        self.assertEqual(decision["verdict"], "INCONCLUSIVE")
        self.assertTrue(decision["sample_insufficiency"])

    def test_panel_mismatch_is_fatal(self):
        rows = checkpoints()
        rows[-1]["panel_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "panel mismatch"):
            MODULE.make_decision(rows, summaries(), protocol())

    def test_any_frozen_cli_deviation_forces_non_preregistered(self):
        args = frozen_args(min_spearman=0.0)
        status = MODULE.protocol_status(args)
        self.assertFalse(status["preregistered"])
        diagnostic_protocol = protocol()
        diagnostic_protocol.update(status)
        decision = MODULE.make_decision(checkpoints(), summaries(), diagnostic_protocol)
        self.assertEqual(decision["verdict"], "NON_PREREGISTERED")
        self.assertIn("min_spearman", decision["protocol_deviations"])

    def test_exact_frozen_parameters_are_preregistered(self):
        self.assertTrue(MODULE.protocol_status(frozen_args())["preregistered"])


if __name__ == "__main__":
    unittest.main()
