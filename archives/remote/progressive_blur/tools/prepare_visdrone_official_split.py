"""Create non-destructive train/val/test lists from VisDrone source provenance.

The existing processed YOLO tree is read-only. The original official train,
val, and test-dev memberships are recovered from source_map.tsv, instead of
using the earlier merged-train / test-as-val exploratory protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/ubuntu/rz/object_detection_dataset_test/processed_yolo/VisDrone2019-DET")
DEST = ROOT / "data_splits" / "visdrone_official"
EXPECTED = {"train": 6471, "val": 548, "test": 1610}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(source_image: str) -> str:
    if "/VisDrone2019-DET-train.zip/" in source_image:
        return "train"
    if "/VisDrone2019-DET-val.zip/" in source_image:
        return "val"
    if "/VisDrone2019-DET-test-dev.zip/" in source_image:
        return "test"
    raise ValueError(f"unrecognized source archive: {source_image}")


def main() -> None:
    source_map = SOURCE / "source_map.tsv"
    with source_map.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    groups: dict[str, list[Path]] = {name: [] for name in EXPECTED}
    seen: set[Path] = set()
    for row in records:
        group = classify(row["source_image"])
        processed_split = "test" if group == "test" else "train"
        if row["split"] != processed_split:
            raise ValueError(f"split conflict: {row}")
        image = SOURCE / "images" / processed_split / row["derived_image"]
        label = SOURCE / "labels" / processed_split / (image.stem + ".txt")
        if image in seen or not image.is_file() or not label.is_file():
            raise ValueError(f"missing or duplicate image/label: {image}")
        seen.add(image)
        groups[group].append(image)
    counts = Counter({group: len(images) for group, images in groups.items()})
    if dict(counts) != EXPECTED:
        raise ValueError(f"unexpected official-split counts: {dict(counts)}")

    DEST.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for group, images in groups.items():
        output = DEST / f"{group}.txt"
        output.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
        manifests[group] = {"count": len(images), "sha256": sha256(output)}
    source_yaml = yaml.safe_load((SOURCE / "dataset.yaml").read_text(encoding="utf-8"))
    data = {
        "train": str(DEST / "train.txt"),
        "val": str(DEST / "val.txt"),
        "test": str(DEST / "test.txt"),
        "names": source_yaml["names"],
    }
    yaml_path = DEST / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit = {
        "status": "official_provenance_split_ready",
        "source_map": str(source_map),
        "source_map_sha256": sha256(source_map),
        "processed_data_read_only": str(SOURCE),
        "yaml": str(yaml_path),
        "manifests": manifests,
        "all_images_and_labels_exist": True,
        "test_dev_not_used_for_training_or_validation": True,
    }
    (DEST / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
