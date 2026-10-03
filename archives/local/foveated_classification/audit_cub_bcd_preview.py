"""Independent saved-PNG and segmentation audit for the CUB B/C/D preview set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_PREVIEW = Path(__file__).resolve().parent / "previews" / "cub_bcd_native500_v1_2026-09-25"
DEFAULT_SEGMENTATIONS = Path(r"E:\datasets\CUB-200-2011\extracted\segmentations")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--segmentations", type=Path, default=DEFAULT_SEGMENTATIONS)
    args = parser.parse_args()
    preview = args.preview.resolve()
    segroot = args.segmentations.resolve()
    manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    mean = np.rint(manifest["train_rgb_mean_0_255"]).astype(np.uint8)
    results = []
    for sample in manifest["samples"]:
        assert sample["official_train"] is True
        image_id, stage = sample["image_id"], sample["stage"]
        subdir = preview / f"{image_id:05d}_{stage}"
        groups = {}
        for name, expected_sha in sample["output_sha256"].items():
            image_path = subdir / f"{name}.png"
            assert digest(image_path) == expected_sha, image_path
            with Image.open(image_path) as image:
                assert image.mode == "RGB" and image.size == (500, 500), image_path
                groups[name] = np.asarray(image, dtype=np.uint8)
        xoff, yoff = sample["padding_offset_xy"]
        width, height = sample["size_wh"]
        source_path = Path(manifest["source_data"]) / "images" / sample["relative_image_path"]
        assert digest(source_path) == sample["source_sha256"]
        with Image.open(source_path) as source:
            original = np.asarray(source.convert("RGB"), dtype=np.uint8)
        assert np.array_equal(groups["A_original"][yoff : yoff + height, xoff : xoff + width], original)

        valid = np.zeros((500, 500), dtype=bool)
        valid[yoff : yoff + height, xoff : xoff + width] = True
        for name, pixels in groups.items():
            assert np.all(pixels[~valid] == mean), (stage, name, "padding differs")

        px1, py1, px2, py2 = sample["metrics"]["protected_box_xyxy"]
        xx = np.arange(width, dtype=np.float32)[None, :] + 0.5
        yy = np.arange(height, dtype=np.float32)[:, None] + 0.5
        protected = (xx >= px1) & (xx <= px2) & (yy >= py1) & (yy <= py2)
        c = groups["C_context_removed"][yoff : yoff + height, xoff : xoff + width]
        d = groups["D_gradual_background"][yoff : yoff + height, xoff : xoff + width]
        assert np.array_equal(c[protected], original[protected]), (stage, "C damaged protected pixels")
        assert np.array_equal(d[protected], original[protected]), (stage, "D damaged protected pixels")

        segmentation_path = (segroot / sample["relative_image_path"]).with_suffix(".png")
        with Image.open(segmentation_path) as segmentation:
            layer = segmentation.getchannel(0) if segmentation.mode == "LA" else segmentation.convert("L")
            mask = np.asarray(layer, dtype=np.uint8)
        assert mask.shape == (height, width)
        changed_d = np.any(d != original, axis=2)
        thresholds = {}
        for label, bird in {
            "mask_eq_255": mask == 255,
            "mask_ge_153": mask >= 153,
            "mask_gt_0": mask > 0,
        }.items():
            count = int(bird.sum())
            assert count > 0
            thresholds[label] = {
                "bird_pixel_count": count,
                "protected_fraction": float(np.mean(protected[bird])),
                "D_changed_fraction": float(np.mean(changed_d[bird])),
                "D_mean_abs_rgb_difference": float(np.abs(d.astype(np.int16) - original.astype(np.int16))[bird].mean()),
            }
        results.append({"stage": stage, "image_id": image_id, "checks_passed": True, "mask_metrics": thresholds})
        print(stage, image_id, "strict_bird_protected", f"{thresholds['mask_eq_255']['protected_fraction']:.3%}", flush=True)

    report = {"preview_directory": str(preview), "all_saved_png_and_source_checks_passed": True, "samples": results}
    (preview / "SAVED_IMAGE_AND_MASK_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B/C/D 保存图与鸟体保护审计",
        "",
        "8张图片全部来自官方train；逐张验证源JPG和输出PNG哈希、500×500 RGB、A有效像素与原图逐像素相同、各组填充相同、C/D保护区逐像素等于原图；均通过。掩膜只用于此后验审计，不参与变换。",
        "",
        "`mask_eq_255`是严格鸟体像素；`mask_ge_153`和`mask_gt_0`作为阈值敏感性。保护率指鸟体像素落在P内的比例；D改变率指鸟体像素的RGB与A不同的比例（不等同于语义受损）。",
        "",
        "| 选图类型 | ID | 严格鸟体在保护区内 | 严格鸟体被D改变 | ≥153鸟体在保护区内 | >0鸟体在保护区内 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        mm = item["mask_metrics"]
        lines.append(
            f"| {item['stage']} | {item['image_id']} | {mm['mask_eq_255']['protected_fraction']:.2%} | "
            f"{mm['mask_eq_255']['D_changed_fraction']:.2%} | {mm['mask_ge_153']['protected_fraction']:.2%} | "
            f"{mm['mask_gt_0']['protected_fraction']:.2%} |"
        )
    (preview / "SAVED_IMAGE_AND_MASK_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("All 8 preview files and bird masks audited successfully", flush=True)


if __name__ == "__main__":
    main()
