"""Create a deterministic image-level VOC train/development split.

The script never modifies the shared VOC tree. It writes absolute-path manifests,
a dataset YAML and an auditable class/box distribution report under this project.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
VOC = Path("/home/ubuntu/rz/yolo_study/datasets/VOC")
OUTPUT = PROJECT / "data_splits"
VIEW = PROJECT / "voc_view"
SEED = 2027
DEV_FRACTION = 0.10
TRAIN_SPLITS = ("train2007", "val2007", "train2012", "val2012")
CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def stable_key(path: Path) -> str:
    return hashlib.sha256(f"{SEED}|{path.as_posix()}".encode()).hexdigest()


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_classes(image: Path) -> tuple[list[int], set[int]]:
    label = label_path(image)
    if not label.is_file():
        raise FileNotFoundError(f"missing label for {image}: {label}")
    classes: list[int] = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid label row {label}:{line_number}")
        class_id = int(float(fields[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"invalid class id {class_id} in {label}")
        classes.append(class_id)
    if not classes:
        raise ValueError(f"unexpected empty VOC label: {label}")
    return classes, set(classes)


def digest(lines: list[Path]) -> str:
    content = "".join(f"{path.as_posix()}\n" for path in lines).encode()
    return hashlib.sha256(content).hexdigest()


def ensure_task_local_view() -> None:
    """Create symlinked image/label directories so caches stay task-local."""

    for role in ("train", "dev", "test"):
        for kind in ("images", "labels"):
            for split in (*TRAIN_SPLITS, "test2007"):
                target = VOC / kind / split
                link = VIEW / role / kind / split
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != target.resolve():
                        raise AssertionError(f"unexpected symlink target: {link}")
                elif link.exists():
                    raise FileExistsError(f"task-local view path is not a symlink: {link}")
                else:
                    link.symlink_to(target, target_is_directory=True)


def view_path(image: Path, role: str) -> Path:
    return VIEW / role / "images" / image.parent.name / image.name


def stats(records: list[dict]) -> dict:
    image_counts = [0] * len(CLASS_NAMES)
    box_counts = [0] * len(CLASS_NAMES)
    for record in records:
        for class_id in record["present"]:
            image_counts[class_id] += 1
        for class_id in record["classes"]:
            box_counts[class_id] += 1
    return {
        "images": len(records),
        "boxes": sum(box_counts),
        "class_image_counts": dict(zip(CLASS_NAMES, image_counts)),
        "class_box_counts": dict(zip(CLASS_NAMES, box_counts)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ensure_task_local_view()
    images = sorted(
        image
        for split in TRAIN_SPLITS
        for image in (VOC / "images" / split).glob("*.jpg")
    )
    if len(images) != 16_551:
        raise AssertionError(f"expected 16551 trainval images, got {len(images)}")
    records = []
    presence = [0] * len(CLASS_NAMES)
    for image in images:
        classes, present = read_classes(image)
        for class_id in present:
            presence[class_id] += 1
        records.append({"image": image, "classes": classes, "present": present})

    # Assign each multi-label image to its rarest present class, then take a
    # deterministic proportional sample inside each stratum.
    strata: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        primary = min(record["present"], key=lambda class_id: (presence[class_id], class_id))
        strata[primary].append(record)
    target_dev = round(len(records) * DEV_FRACTION)
    quotas = {class_id: math.floor(len(group) * DEV_FRACTION) for class_id, group in strata.items()}
    remaining = target_dev - sum(quotas.values())
    fractional_order = sorted(
        strata,
        key=lambda class_id: (
            -(len(strata[class_id]) * DEV_FRACTION - quotas[class_id]),
            class_id,
        ),
    )
    for class_id in fractional_order[:remaining]:
        quotas[class_id] += 1

    dev_paths: set[Path] = set()
    for class_id, group in strata.items():
        ordered = sorted(group, key=lambda record: stable_key(record["image"]))
        dev_paths.update(record["image"] for record in ordered[: quotas[class_id]])
    train_records = [record for record in records if record["image"] not in dev_paths]
    dev_records = [record for record in records if record["image"] in dev_paths]
    if len(train_records) != 14_896 or len(dev_records) != 1_655:
        raise AssertionError("unexpected 90/10 split sizes")

    test_images = sorted((VOC / "images" / "test2007").glob("*.jpg"))
    if len(test_images) != 4_952:
        raise AssertionError(f"expected 4952 test images, got {len(test_images)}")
    test_records = []
    for image in test_images:
        classes, present = read_classes(image)
        test_records.append({"image": image, "classes": classes, "present": present})

    train_source_paths = sorted(record["image"] for record in train_records)
    dev_source_paths = sorted(record["image"] for record in dev_records)
    if set(train_source_paths) & set(dev_source_paths):
        raise AssertionError("train/development overlap")
    train_paths = [view_path(path, "train") for path in train_source_paths]
    dev_paths_sorted = [view_path(path, "dev") for path in dev_source_paths]
    test_view_paths = [view_path(path, "test") for path in test_images]
    manifests = {
        "train": OUTPUT / f"voc_train90_seed{SEED}.txt",
        "val": OUTPUT / f"voc_dev10_seed{SEED}.txt",
        "test": OUTPUT / "voc2007_test.txt",
    }
    for name, paths in (
        ("train", train_paths),
        ("val", dev_paths_sorted),
        ("test", test_view_paths),
    ):
        manifests[name].write_text(
            "".join(f"{path.as_posix()}\n" for path in paths), encoding="utf-8"
        )

    yaml_path = OUTPUT / f"voc_strict_seed{SEED}.yaml"
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {PROJECT.as_posix()}",
                f"train: {manifests['train'].as_posix()}",
                f"val: {manifests['val'].as_posix()}",
                f"test: {manifests['test'].as_posix()}",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )

    train_stats, dev_stats, test_stats = (
        stats(train_records),
        stats(dev_records),
        stats(test_records),
    )
    full_stats = stats(records)
    class_audit = {}
    for name in CLASS_NAMES:
        full_count = full_stats["class_image_counts"][name]
        class_audit[name] = {
            "all_trainval_images": full_count,
            "train_images": train_stats["class_image_counts"][name],
            "dev_images": dev_stats["class_image_counts"][name],
            "dev_fraction": dev_stats["class_image_counts"][name] / full_count,
            "all_trainval_boxes": full_stats["class_box_counts"][name],
            "train_boxes": train_stats["class_box_counts"][name],
            "dev_boxes": dev_stats["class_box_counts"][name],
        }
    max_deviation = max(
        abs(values["dev_fraction"] - DEV_FRACTION) for values in class_audit.values()
    )
    if max_deviation > 0.025:
        raise AssertionError(f"class image-ratio deviation too large: {max_deviation}")

    report = {
        "status": "verified",
        "algorithm": "rarest-present-class stratification with SHA256 seeded ordering",
        "seed": SEED,
        "development_fraction_target": DEV_FRACTION,
        "max_class_image_fraction_deviation": max_deviation,
        "shared_voc_images_and_labels_unchanged": True,
        "task_local_symlink_view": str(VIEW),
        "yaml": str(yaml_path),
        "manifests": {
            name: {"path": str(path), "sha256": digest(paths)}
            for (name, path), paths in zip(
                manifests.items(), (train_paths, dev_paths_sorted, test_view_paths)
            )
        },
        "train": train_stats,
        "development": dev_stats,
        "test": test_stats,
        "class_audit": class_audit,
    }
    report_path = OUTPUT / "voc_split_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "train_images": train_stats["images"],
        "development_images": dev_stats["images"],
        "test_images": test_stats["images"],
        "max_class_image_fraction_deviation": max_deviation,
        "audit": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
