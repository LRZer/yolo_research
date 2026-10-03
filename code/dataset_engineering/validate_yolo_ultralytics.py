#!/usr/bin/env python3
"""Full Ultralytics parser/cache validation for prepared detection datasets."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test/processed_yolo")
OUT = ROOT / "ULTRALYTICS_VALIDATION.json"


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    report = {"status": "running", "started": time.time(), "datasets": [], "errors": []}
    atomic_json(OUT, report)
    for dataset_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.is_file():
            continue
        item = {"dataset": dataset_dir.name, "yaml": str(yaml_path), "splits": {}, "errors": []}
        try:
            data = check_det_dataset(str(yaml_path), autodownload=False)
            if Path(data["val"]).resolve() != Path(data["test"]).resolve():
                raise AssertionError("val and test are not the same test directory")
            for split in ("train", "test"):
                print(f"ULTRALYTICS START {dataset_dir.name} {split}", flush=True)
                ds = YOLODataset(
                    img_path=data[split], data=data, task="detect", imgsz=640,
                    augment=False, cache=False, prefix=f"{dataset_dir.name}/{split}: ",
                    rect=False, batch_size=1, stride=32,
                )
                boxes = sum(len(label["cls"]) for label in ds.labels)
                item["splits"][split] = {
                    "images_loaded": len(ds.im_files),
                    "labels_loaded": len(ds.labels),
                    "boxes_loaded": boxes,
                }
                print(f"ULTRALYTICS PASS {dataset_dir.name} {split} images={len(ds.im_files)} boxes={boxes}", flush=True)
        except BaseException as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            report["errors"].append(f"{dataset_dir.name}: {type(exc).__name__}: {exc}")
            print(f"ULTRALYTICS FAIL {dataset_dir.name}: {type(exc).__name__}: {exc}", flush=True)
        report["datasets"].append(item)
        atomic_json(OUT, report)
    report["finished"] = time.time()
    report["status"] = "all_datasets_loaded_by_ultralytics" if not report["errors"] else "failed"
    atomic_json(OUT, report)
    print(report["status"], flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
