"""Focused fail-closed tests for the ASAC launcher."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = load_module("train_asac_launcher_test", ROOT / "scripts" / "train_asac.py")
protocol = load_module("asac_protocol_launcher_test", ROOT / "scripts" / "asac_protocol.py")


def frozen_fixture(directory: Path):
    raw_by_seed = {}
    state_paths = {}
    for seed in protocol.SEEDS:
        rows = [{"epoch": epoch, "variant": "A2", "seed": seed, "train_q": 0.5} for epoch in range(1, 51)]
        raw = b"".join(protocol.canonical_json_bytes(row) for row in rows)
        path = directory / f"a2_seed{seed}.jsonl"
        path.write_bytes(raw)
        raw_by_seed[seed] = raw
        state_paths[str(seed)] = str(path)
    bundle = protocol.freeze_jsonl_bytes_by_seed(raw_by_seed)
    paths = protocol.write_frozen_schedules(bundle, directory / "frozen")
    return bundle, paths, state_paths


def trusted_frozen_hashes(paths: dict, state_paths: dict[str, str]) -> dict:
    return {
        "trusted_freeze_manifest_sha256": launcher.file_sha256(paths["manifest"]),
        "trusted_a2_state_sha256_by_seed": {
            seed: launcher.file_sha256(Path(path)) for seed, path in state_paths.items()
        },
    }


def valid_golden_payload(tree_sha256: str) -> dict:
    autocast = {
        "autocast_dtype": "torch.bfloat16",
        "autocast_enabled": True,
        "inference_mode": True,
        "input_dtype": "torch.float32",
        "o2m_boxes_dtype": "torch.bfloat16",
        "o2m_scores_dtype": "torch.bfloat16",
        "o2o_boxes_dtype": "torch.bfloat16",
        "o2o_scores_dtype": "torch.bfloat16",
    }
    branches = {}
    for name, (topk, topk2) in {"one2many": (10, 10), "one2one": (7, 1)}.items():
        branches[name] = {
            "loss_class": "ultralytics.utils.loss.v8DetectionLoss",
            "assigner_class": "ultralytics.utils.tal.TaskAlignedAssigner",
            "topk": topk,
            "topk2": topk2,
            "stride": [8.0, 16.0, 32.0],
        }
    return {
        "schema": "asac-panel-golden-smoke-v1",
        "status": "PASS",
        "checkpoint_sha256": launcher.PINNED_P0_CHECKPOINT_SHA256,
        "data_sha256": launcher.PINNED_DATA_YAML_SHA256,
        "absolute_tolerance": 1e-12,
        "physical_gpu": 4,
        "visible_gpu_count": 1,
        "expected": dict(launcher.PINNED_GOLDEN_EXPECTED),
        "differences": {"O2M": -1.1102230246251565e-16, "O2O": 0.0},
        "measurement": {
            "strength": 0.5,
            "panel_content_sha256": launcher.PINNED_PANEL_CONTENT_SHA256,
            "assigner_oom_fallback": {"O2M": False, "O2O": False},
            "rng_sha256": "a" * 64,
            "integrity": {
                "parameters_sha256": "b" * 64,
                "registered_buffers_sha256": "c" * 64,
                "bn_buffers_sha256": "d" * 64,
                "extra_tensor_attributes_sha256": "e" * 64,
            },
            "autocast": autocast,
            "heads": {
                "O2M": {
                    "assigned_both": 624,
                    "assignment_coverage": 1.0,
                    "mean_stability": 0.9545479413122413,
                    "object_pairs": 624,
                },
                "O2O": {
                    "assigned_both": 624,
                    "assignment_coverage": 1.0,
                    "mean_stability": 0.9192160195274237,
                    "object_pairs": 624,
                },
            },
        },
        "panel": {
            "dataset_length": 16_551,
            "loader_workers": 4,
            "loader_pin_memory": True,
            "panel_content_sha256": launcher.PINNED_PANEL_CONTENT_SHA256,
            "population_count": 624,
            "population_sha256": launcher.PINNED_POPULATION_SHA256,
            "protocol": {
                "expected_panel_content_sha256": launcher.PINNED_PANEL_CONTENT_SHA256,
                "expected_population_count": 624,
                "expected_population_sha256": launcher.PINNED_POPULATION_SHA256,
            },
        },
        "criterion_contract": {
            "criterion_class": "ultralytics.utils.loss.E2ELoss",
            "head_class": "ultralytics.nn.modules.head.Detect",
            "head_runtime_end2end": False,
            "dual_head_modules_present": True,
            "branches": branches,
        },
        "golden_script": {"sha256": launcher.PINNED_CUSTOM_ASAC_SHA256["golden_script"]},
        "source_records": {
            "ultralytics/utils/asac.py": {"sha256": launcher.PINNED_CUSTOM_ASAC_SHA256["runtime"]}
        },
        "isolated_source_manifest": {"tree_sha256": tree_sha256},
    }


class OutputDirectoryTests(unittest.TestCase):
    def test_formal_output_must_be_new_and_exist_ok_false(self):
        with tempfile.TemporaryDirectory() as directory:
            train = {"project": directory, "name": "formal_run", "exist_ok": False}
            target = launcher.exact_output_directory(train, True)
            self.assertEqual(target, (Path(directory) / "formal_run").resolve())
            target.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing implicit name increment"):
                launcher.exact_output_directory(train, True)
            with self.assertRaisesRegex(ValueError, "exist_ok=false"):
                launcher.exact_output_directory({**train, "name": "other", "exist_ok": True}, True)

    def test_smoke_cannot_overwrite_or_auto_increment_either(self):
        with tempfile.TemporaryDirectory() as directory:
            train = {"project": directory, "name": "smoke", "exist_ok": False}
            target = launcher.exact_output_directory(train, False)
            target.mkdir()
            with self.assertRaises(FileExistsError):
                launcher.exact_output_directory(train, False)
            with self.assertRaisesRegex(ValueError, "every ASAC launch"):
                launcher.exact_output_directory({**train, "name": "other", "exist_ok": True}, False)

    def test_dangling_target_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "dangling"
            try:
                link.symlink_to(root / "missing", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")
            with self.assertRaises(FileExistsError):
                launcher.exact_output_directory(
                    {"project": str(root), "name": link.name, "exist_ok": False}, False
                )

    def test_external_distributed_environment_is_rejected(self):
        launcher.reject_external_distributed_environment({})
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "existing distributed job"):
                launcher.reject_external_distributed_environment({key: "0"})

    def test_name_must_be_one_component(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one non-empty path component"):
                launcher.exact_output_directory(
                    {"project": directory, "name": "nested/run", "exist_ok": False}, True
                )


class EvidenceTests(unittest.TestCase):
    def test_source_manifest_is_deterministic_and_ignores_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "pkg" / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / "pkg" / "__pycache__").mkdir()
            (root / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"runtime")
            first = launcher.source_tree_manifest(root)
            second = launcher.source_tree_manifest(root)
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            self.assertEqual([item["path"] for item in first["files"]], ["pkg/module.py"])
            (root / "pkg" / "module.py").write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(first["tree_sha256"], launcher.source_tree_manifest(root)["tree_sha256"])

    def test_voc_manifest_uses_relative_image_sizes_and_label_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "train2012" / "a.jpg"
            label = root / "labels" / "train2012" / "a.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"abc")
            label.write_bytes(b"0 0.5 0.5 0.1 0.1\n\n")
            manifest = launcher.voc_train_manifest(root)
            image_line = b"images/train2012/a.jpg 3\n"
            label_line = f"{launcher.file_sha256(label)}  labels/train2012/a.txt\n".encode()
            self.assertEqual((manifest["image_count"], manifest["label_count"], manifest["gt_lines"]), (1, 1, 1))
            self.assertEqual(manifest["image_relative_path_space_size_sha256"], hashlib.sha256(image_line).hexdigest())
            self.assertEqual(manifest["ordered_label_sha256sum_bundle"], hashlib.sha256(label_line).hexdigest())

    def test_launch_record_hashes_every_required_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in ("config", "base", "model", "data"):
                path = root / f"{name}.yaml"
                path.write_text(f"kind: {name}\n", encoding="utf-8")
                artifacts[name] = path
            train = {"model": str(artifacts["model"]), "data": str(artifacts["data"])}
            base_sha = launcher.file_sha256(artifacts["base"])
            envelope = launcher.build_launch_record(
                spec={"id": "test"},
                spec_path=artifacts["config"],
                train=train,
                asac={"schema": launcher.SCHEMA, "variant": "A2"},
                base_path=artifacts["base"],
                base_sha256=base_sha,
                output_directory=root / "run",
                schedule_evidence={},
                golden_evidence={},
                formal=False,
            )
            record = envelope["record"]
            self.assertEqual(envelope["record_sha256"], launcher.canonical_sha256(record))
            self.assertEqual(record["artifacts"]["base_args"]["sha256"], base_sha)
            self.assertEqual(set(record["artifacts"]["custom_asac_files"]), set(launcher.CUSTOM_ASAC_FILES))
            self.assertGreater(record["isolated_source_manifest"]["file_count"], 300)

    def test_pinned_custom_and_full_source_hashes_match_current_snapshot(self):
        actual_custom = {
            name: launcher.file_sha256(launcher.CUSTOM_ASAC_FILES[name])
            for name in launcher.PINNED_CUSTOM_ASAC_SHA256
        }
        self.assertEqual(actual_custom, launcher.PINNED_CUSTOM_ASAC_SHA256)
        self.assertEqual(
            launcher.source_tree_manifest(launcher.EXPECTED_SOURCE)["tree_sha256"],
            launcher.PINNED_SOURCE_TREE_SHA256,
        )

    def test_formal_launch_rejects_unpinned_model_and_data_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in ("config", "base", "model", "data"):
                path = root / f"{name}.yaml"
                path.write_text(f"kind: {name}\n", encoding="utf-8")
                artifacts[name] = path
            with self.assertRaisesRegex(RuntimeError, "model/data YAML hash mismatch"):
                launcher.build_launch_record(
                    spec={"id": "test"},
                    spec_path=artifacts["config"],
                    train={"model": str(artifacts["model"]), "data": str(artifacts["data"])},
                    asac={"schema": launcher.SCHEMA, "variant": "A2"},
                    base_path=artifacts["base"],
                    base_sha256=launcher.file_sha256(artifacts["base"]),
                    output_directory=root / "run",
                    schedule_evidence={},
                    golden_evidence={},
                    formal=True,
                )

    def test_launch_record_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch_record.json"
            launcher.persist_json_exclusive(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            with self.assertRaises(FileExistsError):
                launcher.persist_json_exclusive(path, {"value": 2})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


class GoldenGateTests(unittest.TestCase):
    def test_formal_golden_artifact_passes_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            payload = valid_golden_payload(launcher.PINNED_SOURCE_TREE_SHA256)
            path.write_text(json.dumps(payload), encoding="utf-8")
            resolved, evidence = launcher.resolve_golden_artifact(
                {"schema": launcher.SCHEMA, "variant": "A2", "golden_artifact": str(path)},
                True,
                launcher.PINNED_SOURCE_TREE_SHA256,
            )
            self.assertNotIn("golden_artifact", resolved)
            self.assertTrue(evidence["validated"])
            self.assertEqual(evidence["artifact"]["sha256"], launcher.file_sha256(path))

    def test_formal_golden_artifact_tampering_fails_closed(self):
        base = valid_golden_payload(launcher.PINNED_SOURCE_TREE_SHA256)
        cases = {
            "status": lambda value: value.update(status="FAIL"),
            "difference": lambda value: value["differences"].update(O2O=2e-12),
            "criterion": lambda value: value["criterion_contract"]["branches"]["one2one"].update(topk=1),
            "runtime": lambda value: value["source_records"]["ultralytics/utils/asac.py"].update(sha256="0" * 64),
            "tree": lambda value: value["isolated_source_manifest"].update(tree_sha256="0" * 64),
            "oom": lambda value: value["measurement"]["assigner_oom_fallback"].update(O2O=True),
            "rng": lambda value: value["measurement"].update(rng_sha256="not-a-sha"),
            "integrity": lambda value: value["measurement"]["integrity"].update(extra="f" * 64),
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, (name, mutate) in enumerate(cases.items()):
                payload = copy.deepcopy(base)
                mutate(payload)
                path = Path(directory) / f"golden_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    launcher.resolve_golden_artifact(
                        {"schema": launcher.SCHEMA, "variant": "A2", "golden_artifact": str(path)},
                        True,
                        launcher.PINNED_SOURCE_TREE_SHA256,
                    )

    def test_formal_requires_golden_and_nonformal_cannot_claim_one(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit golden_artifact"):
            launcher.resolve_golden_artifact({"variant": "A2"}, True, launcher.PINNED_SOURCE_TREE_SHA256)
        with self.assertRaisesRegex(ValueError, "nonformal"):
            launcher.resolve_golden_artifact(
                {"variant": "A2", "golden_artifact": "not-used.json"},
                False,
                launcher.PINNED_SOURCE_TREE_SHA256,
            )


class FrozenScheduleTests(unittest.TestCase):
    def test_formal_control_requires_independent_a2_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths, state_paths = frozen_fixture(root)
            evidence = {}
            resolved = launcher.resolve_control_schedule(
                {
                    "schema": launcher.SCHEMA,
                    "variant": "A1",
                    "formal": True,
                    "schedule_file": str(paths["a1_seed42"]),
                    "freeze_manifest_file": str(paths["manifest"]),
                    "a2_state_files": state_paths,
                    **trusted_frozen_hashes(paths, state_paths),
                },
                {"seed": 42},
                evidence,
            )
            self.assertEqual(set(resolved["schedule"]), set(range(1, 51)))
            self.assertTrue(evidence["independent_reconstruction"])
            self.assertEqual(set(evidence["a2_states"]), {"0", "1", "42"})

    def test_paired_schedule_and_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths, state_paths = frozen_fixture(root)
            schedule_path, manifest_path = paths["a3_seed42"], paths["manifest"]
            rows = json.loads(schedule_path.read_text(encoding="utf-8"))
            rows[10]["train_q"] = 0.51
            schedule_path.write_bytes(protocol.canonical_json_bytes(rows))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_schedule_sha256"]["A3"]["42"] = launcher.file_sha256(schedule_path)
            manifest_path.write_bytes(protocol.canonical_json_bytes(manifest))
            with self.assertRaisesRegex(RuntimeError, "independent reconstruction"):
                launcher.resolve_control_schedule(
                    {
                        "schema": launcher.SCHEMA,
                        "variant": "A3",
                        "formal": True,
                        "schedule_file": str(schedule_path),
                        "freeze_manifest_file": str(manifest_path),
                        "a2_state_files": state_paths,
                        **trusted_frozen_hashes(
                            {**paths, "manifest": paths["manifest"]}, state_paths
                        ),
                    },
                    {"seed": 42},
                )

    def test_formal_control_without_states_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "explicit trusted"):
            launcher.resolve_control_schedule(
                {
                    "schema": launcher.SCHEMA,
                    "variant": "A4",
                    "formal": True,
                    "schedule_file": "schedule.json",
                    "freeze_manifest_file": "manifest.json",
                },
                {"seed": 42},
            )


if __name__ == "__main__":
    unittest.main()
