"""One-checkpoint, one-batch, zero-update ASAC v5 runtime smoke gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import pynvml


FAULT_PATTERN = re.compile(
    r"Uncorrected hardware memory error|Sending SIGBUS|Memory failure|"
    r"\[Hardware Error\]|EDAC.*(?:UE|CE|error)|Bus error",
    re.IGNORECASE,
)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def nvml_record(index: int) -> dict:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return {
            "physical_device": index,
            "name": pynvml.nvmlDeviceGetName(handle),
            "gpu_util_pct": int(util.gpu),
            "memory_used_mib": memory.used / 1048576.0,
            "compute_pids": [int(process.pid) for process in processes],
        }
    finally:
        pynvml.nvmlShutdown()


def kernel_health(since: str) -> dict:
    completed = subprocess.run(
        ["journalctl", "-k", "--since", since, "--no-pager"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read kernel log: {completed.stderr.strip()}")
    matching = [line for line in completed.stdout.splitlines() if FAULT_PATTERN.search(line)]
    return {"since": since, "checked_at": now(), "qualifying_event_count": len(matching), "events": matching}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physical-device", type=int, default=4, choices=(4, 5, 6, 7))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    pre_nvml = nvml_record(args.physical_device)
    if pre_nvml["compute_pids"]:
        raise RuntimeError(f"physical GPU {args.physical_device} has compute processes: {pre_nvml}")

    import probe_asac_assignment_stability as p

    checkpoint = (
        "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/"
        "b50_no_kd_e50_g4567_s42/weights/epoch10.pt"
    )
    smoke_args = SimpleNamespace(
        data="/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/datasets/voc_server.yaml",
        batch=64,
        imgsz=640,
        workers=4,
        seed=42,
        max_scale_delta=0.50,
        max_translate=0.10,
        min_visibility=0.70,
        min_box_pixels=2.0,
        expected_strides=[8.0, 16.0, 32.0],
        grid_bins=13,
        grid_extent=1.5,
        kernel_sigma=0.25,
    )

    loader = None
    try:
        p.initialize_runtime(args.physical_device)
        trainer = p.make_trainer(checkpoint, smoke_args)
        contract = p.validate_criterion_contract(trainer, smoke_args.expected_strides)
        loader, loader_meta = p.make_fixed_loader(trainer, smoke_args)
        before = p.integrity_snapshot(trainer)
        if not before["model_training"] or not before["bn_all_eval"] or before["optimizer_created"]:
            raise RuntimeError(f"invalid zero-update precondition: {before}")

        with p.torch.inference_mode():
            raw = next(iter(loader))
            paths = [str(value) for value in raw["im_file"]]
            batch = trainer.preprocess_batch(raw)
            batch_size, _, height, width = batch["img"].shape
            if batch_size != 64 or len(paths) != 64 or len(set(paths)) != 64:
                raise RuntimeError(
                    f"smoke batch is not 64 unique images: bs={batch_size}, "
                    f"paths={len(paths)}, unique={len(set(paths))}"
                )
            canonical = p.canonical_boxes(batch, batch_size)
            base_maps = p.base_local_maps(batch, batch_size)
            identity = [{"scale": 1.0, "tx": 0.0, "ty": 0.0} for _ in range(batch_size)]
            clean, clean_dtype = p.forward_assignments(trainer, batch)
            affine = p.affine_parameters(
                paths, 0.25, smoke_args.seed, smoke_args.max_scale_delta, smoke_args.max_translate
            )
            view_batch, view_maps = p.transform_batch(batch, affine, smoke_args)
            view, view_dtype = p.forward_assignments(trainer, view_batch)
            if view_dtype != clean_dtype:
                raise RuntimeError(f"dtype changed: clean={clean_dtype}, view={view_dtype}")
            counts = {"clean": {}, "view_q0.25": {}}
            for head, (assignment, prediction, strides) in clean.items():
                counts["clean"][head] = len(
                    p.extract_object_records(
                        assignment,
                        prediction,
                        base_maps,
                        canonical,
                        identity,
                        (height, width),
                        strides,
                        smoke_args,
                    )
                )
            for head, (assignment, prediction, strides) in view.items():
                counts["view_q0.25"][head] = len(
                    p.extract_object_records(
                        assignment,
                        prediction,
                        view_maps,
                        canonical,
                        affine,
                        (height, width),
                        strides,
                        smoke_args,
                    )
                )
            if any(value <= 0 for group in counts.values() for value in group.values()):
                raise RuntimeError(f"empty assignment object set: {counts}")

        after = p.integrity_snapshot(trainer)
        p.assert_integrity_unchanged(before, after, checkpoint)
        health = kernel_health(started)
        if health["qualifying_event_count"]:
            raise RuntimeError(f"hardware/runtime fault during smoke: {health['events']}")
        result = {
            "status": "PASS",
            "started_at": started,
            "finished_at": now(),
            "probe_sha256": p.file_sha256(p.__file__),
            "provenance_revision": p.PROVENANCE_REVISION,
            "pre_nvml": pre_nvml,
            "post_nvml": nvml_record(args.physical_device),
            "criterion": contract,
            "loader": loader_meta,
            "clean_dtype": clean_dtype,
            "view_dtype": view_dtype,
            "object_counts": counts,
            "integrity_before": before,
            "integrity_after": after,
            "kernel_health": health,
        }
        (output / "smoke_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        failure = {
            "status": "FAIL",
            "started_at": started,
            "failed_at": now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        try:
            failure["kernel_health"] = kernel_health(started)
        except Exception as health_error:
            failure["kernel_health_error"] = repr(health_error)
        (output / "smoke_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    finally:
        if loader is not None and hasattr(loader, "close"):
            loader.close()


if __name__ == "__main__":
    main()
