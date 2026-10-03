#!/usr/bin/env python3
"""Create YOLO-detection derivatives without modifying any original dataset file.

The generated layout uses absolute symbolic links for images and new YOLO label
files.  Every training YAML deliberately points ``val`` and ``test`` to the
same test split, per the experiment policy requested on 2026-09-21.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path("/home/ubuntu/rz/object_detection_dataset_test").resolve()
FINAL = ROOT / "processed_yolo"
STAGE = ROOT / f"processed_yolo.__staging__.{os.getpid()}"
POLICY = "train_plus_val_when_available__test_used_as_val_each_epoch"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize(value: str) -> str:
    value = value.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


class Writer:
    def __init__(self, slug: str, names: list[str], policy: str, notes: list[str] | None = None):
        self.slug = slug
        self.root = STAGE / slug
        self.names = list(names)
        self.policy = policy
        self.notes = list(notes or [])
        self.stats = {
            "train": {"images": 0, "labels": 0, "boxes": 0, "empty_labels": 0},
            "test": {"images": 0, "labels": 0, "boxes": 0, "empty_labels": 0},
        }
        self.corrections = Counter()
        self.source_annotations: list[str] = []
        self._used: set[tuple[str, str]] = set()
        for split in ("train", "test"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=False)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=False)
        self._map = (self.root / "source_map.tsv").open("w", encoding="utf-8", newline="\n")
        self._map.write("split\tderived_image\tsource_image\n")

    def source(self, path: Path):
        text = str(path.resolve())
        if text not in self.source_annotations:
            self.source_annotations.append(text)

    def add(
        self,
        split: str,
        src: Path,
        width: int,
        height: int,
        boxes: list[tuple[int, float, float, float, float]],
        tag: str,
        identity: str | int,
    ):
        if split not in self.stats:
            raise ValueError(split)
        src = src.resolve(strict=True)
        if src.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"not an image: {src}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions {width}x{height}: {src}")
        base = sanitize(f"{tag}_{identity}_{src.name}")
        key = (split, base.lower())
        if key in self._used:
            digest = hashlib.sha256(str(src).encode()).hexdigest()[:12]
            base = sanitize(f"{tag}_{identity}_{digest}_{src.name}")
            key = (split, base.lower())
        if key in self._used:
            raise RuntimeError(f"derived filename collision: {split}/{base}")
        self._used.add(key)
        image_out = self.root / "images" / split / base
        label_out = self.root / "labels" / split / (Path(base).stem + ".txt")
        os.symlink(str(src), str(image_out))
        rows: list[str] = []
        for cls, x, y, w, h in boxes:
            if not 0 <= int(cls) < len(self.names):
                raise ValueError(f"class {cls} outside 0..{len(self.names)-1}: {src}")
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            clipped = (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
                       max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))
            if any(abs(a - b) > 1e-8 for a, b in zip((x1, y1, x2, y2), clipped)):
                self.corrections["boxes_clipped_to_image"] += 1
            x1, y1, x2, y2 = clipped
            if x2 <= x1 or y2 <= y1:
                self.corrections["nonpositive_boxes_skipped"] += 1
                continue
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            nw = (x2 - x1) / width
            nh = (y2 - y1) / height
            rows.append(f"{int(cls)} {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
        unique_rows = list(dict.fromkeys(rows))
        if len(unique_rows) != len(rows):
            self.corrections["duplicate_yolo_rows_removed"] += len(rows) - len(unique_rows)
        rows = unique_rows
        label_out.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
        self._map.write(f"{split}\t{base}\t{src}\n")
        stat = self.stats[split]
        stat["images"] += 1
        stat["labels"] += 1
        stat["boxes"] += len(rows)
        if not rows:
            stat["empty_labels"] += 1

    def finish(self):
        self._map.close()
        dataset_yaml = {
            "path": str((FINAL / self.slug).resolve()),
            "train": "images/train",
            "val": "images/test",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.names)},
        }
        with (self.root / "dataset.yaml").open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)
        report = {
            "dataset": self.slug,
            "status": "converted_and_structurally_verified_pending_global_audit",
            "policy": self.policy,
            "raw_files_modified": False,
            "image_storage": "absolute_symlinks_to_verified_raw_images",
            "yaml_val_points_to": "images/test",
            "yaml_test_points_to": "images/test",
            "class_count": len(self.names),
            "names": self.names,
            "stats": self.stats,
            "corrections": dict(self.corrections),
            "source_annotations": self.source_annotations,
            "notes": self.notes,
        }
        write_json(self.root / "conversion_report.json", report)
        lines = [
            f"# {self.slug} YOLO派生数据",
            "",
            f"- 策略：{self.policy}",
            "- `val`与`test`均指向`images/test`，训练时每轮评测test。",
            "- 图片为指向已验收原图的绝对软链接；标签为新生成YOLO检测标签；原始文件未修改。",
            f"- train：{self.stats['train']['images']}图，{self.stats['train']['boxes']}框。",
            f"- test：{self.stats['test']['images']}图，{self.stats['test']['boxes']}框。",
            f"- 类别数：{len(self.names)}。",
            f"- 派生修正：`{json.dumps(dict(self.corrections), ensure_ascii=False)}`。",
        ]
        if self.notes:
            lines += ["", "## 注意事项", ""] + [f"- {n}" for n in self.notes]
        (self.root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report


def anns_by_image(data):
    out = defaultdict(list)
    for a in data.get("annotations", data.get("annotation", [])):
        out[a["image_id"]].append(a)
    return out


def category_union(paths: list[Path], allowed=None):
    found = {}
    for p in paths:
        for c in read_json(p).get("categories", []):
            if allowed is None or c["id"] in allowed:
                found[c["id"]] = c["name"]
    ids = sorted(found)
    return ids, [found[i] for i in ids], {v: i for i, v in enumerate(ids)}


def add_coco(writer: Writer, json_path: Path, out_split: str, tag: str, image_resolver, category_map, skip_categories=None):
    writer.source(json_path)
    data = read_json(json_path)
    grouped = anns_by_image(data)
    skip_categories = set(skip_categories or [])
    for im in data["images"]:
        file_name = im.get("file_name", im.get("filename"))
        src = image_resolver(file_name, im)
        boxes = []
        for a in grouped.get(im["id"], []):
            cid = a["category_id"]
            if cid in skip_categories:
                writer.corrections["ignored_category_boxes_skipped"] += 1
                continue
            if cid not in category_map:
                raise ValueError(f"unmapped category {cid} in {json_path}")
            x, y, w, h = a["bbox"]
            boxes.append((category_map[cid], x, y, w, h))
        writer.add(out_split, src, int(im["width"]), int(im["height"]), boxes, tag, im["id"])


def convert_hit():
    base = next((ROOT / "HIT-UAV" / "raw").glob("**/normal_json"))
    paths = [base / "annotations" / f"{s}.json" for s in ("train", "val", "test")]
    allowed = {0, 1, 2, 3}
    _, names, cmap = category_union(paths, allowed)
    w = Writer("HIT-UAV", names, "official train+val -> train; official test -> test; horizontal thermal detection",
               ["DontCare(category_id=4) is excluded, not trained as an ordinary class.",
                "Only normal_json thermal images are linked; rotated/XML/image duplicates are not duplicated."])
    for source, out in (("train", "train"), ("val", "train"), ("test", "test")):
        add_coco(w, base / "annotations" / f"{source}.json", out, source,
                 lambda fn, im, s=source: base / s / fn, cmap, {4})
    return w.finish()


def convert_duo():
    base = ROOT / "DUO" / "raw" / "DUO.zip" / "DUO"
    paths = [base / "annotations" / "instances_train.json", base / "annotations" / "instances_test.json"]
    _, names, cmap = category_union(paths)
    w = Writer("DUO", names, "official train -> train; official test -> test",
               ["Derived names include split/id prefixes, so identical basenames across splits cannot collide."])
    for source in ("train", "test"):
        add_coco(w, base / "annotations" / f"instances_{source}.json", source, source,
                 lambda fn, im, s=source: base / "images" / s / fn, cmap)
    return w.finish()


def convert_ap10k():
    base = ROOT / "AP-10K" / "raw" / "ap-10k.zip" / "ap-10k"
    paths = [base / "annotations" / f"ap10k-{s}-split1.json" for s in ("train", "val", "test")]
    _, names, cmap = category_union(paths)
    w = Writer("AP-10K", names, "official split1 train+val -> train; official split1 test -> test",
               ["These boxes derive from an animal pose dataset and are not asserted to exhaustively label every object in each image."])
    for source, out in (("train", "train"), ("val", "train"), ("test", "test")):
        add_coco(w, base / "annotations" / f"ap10k-{source}-split1.json", out, source,
                 lambda fn, im: base / "data" / fn, cmap)
    return w.finish()


def convert_seaclear():
    base = ROOT / "SeaClear" / "raw" / "Seaclear Marine Debris Dataset.rar"
    jp = base / "dataset.json"
    data = read_json(jp)
    _, names, cmap = category_union([jp])
    actual = {p.name: p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    ordered = sorted(data["images"], key=lambda im: hashlib.sha256(im["file_name"].encode()).hexdigest())
    test_ids = {im["id"] for im in ordered[: round(len(ordered) * 0.20)]}
    grouped = anns_by_image(data)
    w = Writer("SeaClear", names, "fixed SHA256-sorted 80/20 custom train/test split; test used each epoch",
               ["This is a custom image-level split, not claimed as an official SeaClear protocol."])
    w.source(jp)
    for im in data["images"]:
        split = "test" if im["id"] in test_ids else "train"
        boxes = [(cmap[a["category_id"]], *a["bbox"]) for a in grouped.get(im["id"], [])]
        w.add(split, actual[im["file_name"]], int(im["width"]), int(im["height"]), boxes, split, im["id"])
    return w.finish()


def convert_exdark():
    raw = ROOT / "ExDark" / "raw"
    ann_root = raw / "groundtruth.tar.gz" / "ExDark_Annno"
    protocol = ROOT / "ExDark" / "archives" / "protocol" / "imageclasslist.txt"
    class_names = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
    cmap = {n.lower(): i for i, n in enumerate(class_names)}
    images = {}
    for p in raw.glob("*.tar.gz/*/*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._"):
            images[p.name.lower()] = p
    labels = {}
    for p in ann_root.glob("*/*.txt"):
        if p.is_file() and not p.name.startswith("._"):
            labels[p.name[:-4].lower()] = p
    w = Writer("ExDark", class_names, "official train+val -> train; official test -> test",
               ["One JPG/jpg case-only mismatch is resolved by a recorded case-insensitive lookup; raw names remain unchanged."])
    w.source(protocol)
    rows = protocol.read_text(encoding="utf-8", errors="strict").splitlines()[1:]
    for idx, row in enumerate(rows):
        if not row.strip():
            continue
        parts = row.split()
        name, split_code = parts[0], int(parts[-1])
        split = "test" if split_code == 3 else "train"
        src = images[name.lower()]
        lp = labels[name.lower()]
        boxes = []
        for line in lp.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line or line.startswith("%"):
                continue
            v = line.split()
            boxes.append((cmap[v[0].lower()], float(v[1]), float(v[2]), float(v[3]), float(v[4])))
        with Image.open(src) as im:
            width, height = im.size
        w.add(split, src, width, height, boxes, f"p{split_code}", idx)
    return w.finish()


def convert_orca():
    ann_root = ROOT / "ORCA" / "archives" / "split_annotations"
    paths = [ann_root / "class_level_train.json", ann_root / "class_level_val_seen.json"]
    _, names, cmap = category_union(paths)
    image_root = ROOT / "ORCA" / "raw" / "images.zip"
    w = Writer("ORCA-class-level-seen", names,
               "official class_level_train -> train; class_level_val_seen -> test; unseen categories excluded",
               ["This is the closed-set 24-class class-level seen protocol, not the canonical 670-common-name dataset.",
                "Caption label fields are ignored; only COCO category_id and bbox are converted."])
    add_coco(w, paths[0], "train", "train", lambda fn, im: image_root / fn, cmap)
    add_coco(w, paths[1], "test", "test", lambda fn, im: image_root / fn, cmap)
    return w.finish()


def convert_hrsid():
    base = ROOT / "HRSID" / "raw" / "HRSID_JPG.rar" / "HRSID_JPG"
    paths = [base / "annotations" / "train2017.json", base / "annotations" / "test2017.json"]
    _, names, cmap = category_union(paths)
    w = Writer("HRSID-JPG", names, "JPG variant train2017 -> train; test2017 -> test",
               ["The inconsistent train_test2017 combined JSON is not used.",
                "PNG-version GitHub labels are not mixed with this JPG archive."])
    for source in ("train", "test"):
        add_coco(w, base / "annotations" / f"{source}2017.json", source, source,
                 lambda fn, im: base / "JPEGImages" / fn, cmap)
    return w.finish()


def convert_llvip(modality: str):
    base = ROOT / "LLVIP" / "raw" / "LLVIP.zip" / "LLVIP"
    w = Writer(f"LLVIP-{modality}", ["person"], f"official {modality} train -> train; test -> test",
               ["VOC coordinates are interpreted as 1-based inclusive pixels: [xmin-1,ymin-1,xmax,ymax).",
                "This preserves the five xmin=xmax boxes as one-pixel-wide derived boxes.",
                "Visible and infrared variants share annotations but are prepared as separate single-modality YOLO datasets."])
    w.source(base / "Annotations")
    for split in ("train", "test"):
        for idx, src in enumerate(sorted((base / modality / split).glob("*.jpg"))):
            xp = base / "Annotations" / f"{src.stem}.xml"
            root = ET.parse(xp).getroot()
            width = int(root.findtext("size/width"))
            height = int(root.findtext("size/height"))
            boxes = []
            for obj in root.findall("object"):
                if obj.findtext("name").strip().lower() != "person":
                    raise ValueError(f"unexpected LLVIP class in {xp}")
                b = obj.find("bndbox")
                xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
                xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
                x1, y1, x2, y2 = xmin - 1.0, ymin - 1.0, xmax, ymax
                if xmax == xmin:
                    w.corrections["inclusive_one_pixel_width_boxes"] += 1
                boxes.append((0, x1, y1, x2 - x1, y2 - y1))
            w.add(split, src, width, height, boxes, split, idx)
    return w.finish()


def convert_visdrone():
    classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    locations = {
        "train": ROOT / "VisDrone2019-DET" / "raw" / "VisDrone2019-DET-train.zip" / "VisDrone2019-DET-train",
        "val": ROOT / "VisDrone2019-DET" / "raw" / "VisDrone2019-DET-val.zip" / "VisDrone2019-DET-val",
        "test-dev": ROOT / "VisDrone2019-DET" / "raw" / "VisDrone2019-DET-test-dev.zip",
    }
    w = Writer("VisDrone2019-DET", classes, "official train+val -> train; labeled test-dev -> test",
               ["Categories 0 (ignored regions) and 11 (others) are excluded.",
                "Nonpositive original boxes are excluded from the derived labels and counted."])
    for source, base in locations.items():
        out = "test" if source == "test-dev" else "train"
        w.source(base / "annotations")
        for idx, src in enumerate(sorted((base / "images").glob("*"))):
            if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(src) as im:
                width, height = im.size
            boxes = []
            ap = base / "annotations" / f"{src.stem}.txt"
            for line in ap.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                x, y, bw, bh, score, cat, trunc, occ = map(float, line.split(",")[:8])
                cat = int(cat)
                if cat not in range(1, 11) or score <= 0:
                    w.corrections["ignored_or_other_rows_skipped"] += 1
                    continue
                if bw <= 0 or bh <= 0:
                    w.corrections["nonpositive_boxes_skipped"] += 1
                    continue
                boxes.append((cat - 1, x, y, bw, bh))
            w.add(out, src, width, height, boxes, source, idx)
    return w.finish()


def convert_zerowaste():
    base = ROOT / "ZeroWaste-f" / "raw" / "zerowaste-f-final.zip" / "splits_final_deblurred"
    paths = [base / s / "labels.json" for s in ("train", "val", "test")]
    _, names, cmap = category_union(paths)
    w = Writer("ZeroWaste-f", names, "official train+val -> train; official test -> test",
               ["Out-of-bounds boxes are clipped only in derived YOLO labels; raw COCO labels remain unchanged.",
                "Derived names include the source split, so cross-split basename collisions cannot collide.",
                "No attempt is made to mitigate adjacent-frame leakage, per the requested experiment policy."])
    for source, out in (("train", "train"), ("val", "train"), ("test", "test")):
        add_coco(w, base / source / "labels.json", out, source,
                 lambda fn, im, s=source: base / s / "data" / fn, cmap)
    return w.finish()


def validate_dataset(root: Path):
    data = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    if data["val"] != "images/test" or data["test"] != "images/test":
        raise AssertionError(f"test-each-epoch policy missing in {root}")
    nc = len(data["names"])
    result = {"dataset": root.name, "splits": {}, "errors": []}
    for split in ("train", "test"):
        images = sorted((root / "images" / split).iterdir())
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if len(images) != len(labels):
            result["errors"].append(f"image/label count mismatch {split}: {len(images)} vs {len(labels)}")
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            result["errors"].append(f"image/label stem mismatch {split}")
        boxes = 0
        broken_links = 0
        for p in images:
            if not p.is_symlink() or not p.exists():
                broken_links += 1
        for p in labels:
            for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                vals = line.split()
                if len(vals) != 5:
                    result["errors"].append(f"{p}:{line_no}: expected 5 fields")
                    continue
                cls = int(vals[0]); x, y, w, h = map(float, vals[1:])
                if not (0 <= cls < nc and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                        and x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6
                        and x + w / 2 <= 1 + 1e-6 and y + h / 2 <= 1 + 1e-6):
                    result["errors"].append(f"{p}:{line_no}: invalid normalized row {line}")
                boxes += 1
        if broken_links:
            result["errors"].append(f"broken/non-symlink images {split}: {broken_links}")
        result["splits"][split] = {"images": len(images), "labels": len(labels), "boxes": boxes, "broken_links": broken_links}
    if result["errors"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def main():
    if FINAL.exists():
        raise SystemExit(f"refusing to overwrite existing output: {FINAL}")
    if STAGE.exists():
        raise SystemExit(f"unexpected staging path exists: {STAGE}")
    STAGE.mkdir(parents=False)
    started = time.time()
    converters = [
        convert_hit, convert_duo, convert_ap10k, convert_seaclear, convert_exdark,
        convert_orca, convert_hrsid, lambda: convert_llvip("infrared"),
        lambda: convert_llvip("visible"), convert_visdrone, convert_zerowaste,
    ]
    reports = []
    try:
        for fn in converters:
            print(f"START {fn.__name__}", flush=True)
            report = fn()
            reports.append(report)
            print(f"DONE {report['dataset']} {report['stats']}", flush=True)
        audits = [validate_dataset(p) for p in sorted(STAGE.iterdir()) if p.is_dir()]
        global_report = {
            "status": "all_yolo_derivatives_structurally_verified",
            "created": time.time(),
            "elapsed_seconds": time.time() - started,
            "root_after_publish": str(FINAL),
            "policy": POLICY,
            "datasets": reports,
            "structural_audits": audits,
            "errors": [],
            "raw_files_modified": False,
        }
        write_json(STAGE / "PREPARATION_REPORT.json", global_report)
        lines = [
            "# YOLO训练前准备报告", "",
            f"- 状态：`{global_report['status']}`",
            "- 所有YAML的`val`和`test`均指向各自`images/test`；训练每轮评测test。",
            "- 有原val的数据已合并到train；没有创建val划分。",
            "- 原图使用绝对软链接，原标签不改；YOLO标签及修正规则位于各派生目录。",
            "", "| 派生数据集 | train图/框 | test图/框 | 类别 |", "|---|---:|---:|---:|",
        ]
        for r in reports:
            a, b = r["stats"]["train"], r["stats"]["test"]
            lines.append(f"| {r['dataset']} | {a['images']}/{a['boxes']} | {b['images']}/{b['boxes']} | {r['class_count']} |")
        lines += ["", "详细的来源映射、派生修正和注意事项见各目录的`conversion_report.json`与`README.md`。"]
        (STAGE / "PREPARATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        commands = ["# 训练命令模板", "", "以下命令只作为模板，本次准备过程没有启动训练。", ""]
        for r in reports:
            y = FINAL / r["dataset"] / "dataset.yaml"
            commands += [f"## {r['dataset']}", "", "```bash", f"CUDA_VISIBLE_DEVICES=4,5,6,7 yolo detect train model=yolo26n.pt data={y} epochs=150 device=0,1,2,3", "```", ""]
        (STAGE / "TRAINING_COMMANDS.md").write_text("\n".join(commands), encoding="utf-8")
        os.rename(STAGE, FINAL)
        print(f"PUBLISHED {FINAL}", flush=True)
    except BaseException:
        print(f"FAILED staging preserved at {STAGE}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
