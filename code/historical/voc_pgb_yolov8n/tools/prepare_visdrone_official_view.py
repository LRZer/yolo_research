"""Create a task-owned symlink view so Ultralytics label caches stay local."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data_splits" / "visdrone_official"
VIEW = SPLIT / "view"
EXPECTED = {"train": 6471, "val": 548, "test": 1610}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if VIEW.exists():
        raise FileExistsError(VIEW)
    VIEW.mkdir()
    audit = {"status": "task_owned_symlink_view_ready", "source_image_and_label_files_modified": False,
             "cache_directory": str(VIEW / "labels"), "splits": {}}
    for group, expected in EXPECTED.items():
        images = [Path(line) for line in (SPLIT / f"{group}.txt").read_text(encoding="utf-8").splitlines()]
        if len(images) != expected:
            raise ValueError(f"wrong {group} manifest count: {len(images)}")
        image_dir = VIEW / "images" / group
        label_dir = VIEW / "labels" / group
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in images:
            original_label = image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"
            if not image.is_file() or not original_label.is_file():
                raise FileNotFoundError(image)
            image_link = image_dir / image.name
            label_link = label_dir / original_label.name
            if image_link.exists() or label_link.exists():
                raise ValueError(f"duplicate basename in {group}: {image.name}")
            image_link.symlink_to(image)
            label_link.symlink_to(original_label)
        image_count = sum(1 for path in image_dir.iterdir() if path.is_file())
        label_count = sum(1 for path in label_dir.iterdir() if path.is_file())
        if image_count != expected or label_count != expected:
            raise RuntimeError(f"incomplete {group} link view")
        audit["splits"][group] = {"image_links": image_count, "label_links": label_count}
    old_yaml = yaml.safe_load((SPLIT / "dataset.yaml").read_text(encoding="utf-8"))
    data = {
        "train": str(VIEW / "images" / "train"),
        "val": str(VIEW / "images" / "val"),
        "test": str(VIEW / "images" / "test"),
        "names": old_yaml["names"],
    }
    output = SPLIT / "view_dataset.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit["view_yaml"] = str(output)
    audit["view_yaml_sha256"] = sha256(output)
    (SPLIT / "view_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
