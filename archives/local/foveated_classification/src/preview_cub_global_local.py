"""Exact GT-box crop previews only: no training, GPU, or source-data writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont


SAMPLE_IDS = (10140, 2568, 6223, 4786, 1767, 1592)
SIDE = 500
FILL = (124, 128, 111)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("A Chinese-capable font is required for size labels")


def pad(image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    assert 0 < width <= SIDE and 0 < height <= SIDE
    offset = ((SIDE - width) // 2, (SIDE - height) // 2)
    canvas = Image.new("RGB", (SIDE, SIDE), FILL)
    canvas.paste(image, offset)
    assert canvas.crop((*offset, offset[0] + width, offset[1] + height)).tobytes() == image.tobytes()
    return canvas, offset


def comparison(panels: list[Image.Image], headings: list[list[str]], title: str) -> Image.Image:
    margin, gap, top = 20, 18, 176
    canvas = Image.new("RGB", (margin * 2 + SIDE * 3 + gap * 2, 734), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, font=font(27), fill=(20, 25, 30))
    for index, (panel, lines) in enumerate(zip(panels, headings)):
        x = margin + index * (SIDE + gap)
        for line_index, line in enumerate(lines):
            draw.text((x, 57 + line_index * 33), line, font=font(22), fill=(25, 30, 35))
        canvas.paste(panel, (x, top))
    draw.text((margin, 692), "尺寸均为宽×高；按真实框裁剪，无外扩、无模糊。补边色为训练集均值色。", font=font(22), fill=(50, 55, 60))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.out}")

    prior = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    samples = {int(sample["image_id"]): sample for sample in prior["samples"]}
    image_index = {}
    for line in (args.data / "images.txt").read_text().splitlines():
        image_id, name = line.split(maxsplit=1)
        image_index[int(image_id)] = name
    boxes = {}
    for line in (args.data / "bounding_boxes.txt").read_text().splitlines():
        fields = line.split()
        boxes[int(fields[0])] = tuple(map(float, fields[1:5]))
    train_ids = {int(line.split()[0]) for line in (args.data / "train_test_split.txt").read_text().splitlines()
                 if line.split()[1] == "1"}
    assert len(train_ids) == 5994

    args.out.mkdir(parents=True)
    rows, overview_rows, records = [], [], []
    for image_id in SAMPLE_IDS:
        expected = samples[image_id]
        relative = image_index[image_id]
        assert image_id in train_ids and relative == expected["relative_image_path"]
        source = args.data / "images" / relative
        assert sha(source) == expected["source_sha256"], "Original source differs from approved prior preview"
        raw = boxes[image_id]
        assert list(raw) == expected["raw_box_xywh"]
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        assert [width, height] == expected["size_wh"]
        x, y, box_width, box_height = raw
        assert all(math.isfinite(v) for v in raw) and box_width > 0 and box_height > 0
        # Coordinates follow the existing project's xywh convention; upper endpoints are exclusive.
        left = max(0, min(width, math.floor(x)))
        top = max(0, min(height, math.floor(y)))
        right = max(0, min(width, math.ceil(x + box_width)))
        bottom = max(0, min(height, math.ceil(y + box_height)))
        assert right > left and bottom > top
        was_clipped = x < 0 or y < 0 or x + box_width > width or y + box_height > height
        cropped = original.crop((left, top, right, bottom))
        crop_width, crop_height = cropped.size
        scale = SIDE / max(crop_width, crop_height)
        resized_width = max(1, min(SIDE, round(crop_width * scale)))
        resized_height = max(1, min(SIDE, round(crop_height * scale)))
        resized = cropped.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        assert max(resized.size) == SIDE
        assert abs(resized_width - crop_width * scale) <= 0.5 + 1e-9
        assert abs(resized_height - crop_height * scale) <= 0.5 + 1e-9
        global_input, global_offset = pad(original)
        crop_native_panel, _ = pad(cropped)
        local_input, local_offset = pad(resized)
        annotated_global = global_input.copy()
        ImageDraw.Draw(annotated_global).rectangle(
            (left + global_offset[0], top + global_offset[1], right - 1 + global_offset[0], bottom - 1 + global_offset[1]),
            outline=(20, 220, 60), width=3,
        )
        folder = args.out / f"{image_id:05d}_{expected['stage']}"
        folder.mkdir()
        images = {
            "original_native.png": original,
            "global_input_500.png": global_input,
            "crop_native.png": cropped,
            "crop_resized.png": resized,
            "local_input_500.png": local_input,
        }
        for name, image in images.items():
            image.save(folder / name)
        headings = [
            [f"原图 {width}×{height} px", "绿框仅用于展示真实框", "全局模型输入不含绿框"],
            [f"目标原始裁剪 {crop_width}×{crop_height} px", "此列保持裁剪原生像素", "越界框已裁到图内" if was_clipped else "严格按框裁剪，未外扩"],
            [f"放大后 {resized_width}×{resized_height} px", f"等比例放大约 {scale:.3f} 倍", "补边后输入 500×500 px"],
        ]
        comparison([annotated_global, crop_native_panel, local_input], headings,
                   f"CUB 训练样例 {image_id}  原图 → 目标裁剪 → 等比例放大").save(folder / "comparison_labeled.png")
        row = {
            "image_id": image_id, "original_width": width, "original_height": height,
            "crop_width": crop_width, "crop_height": crop_height,
            "resized_width": resized_width, "resized_height": resized_height,
            "nominal_scale": scale, "input_width": SIDE, "input_height": SIDE,
            "bbox_clipped_to_image": was_clipped,
        }
        rows.append(row)
        records.append({**row, "relative_path": relative, "official_train": True,
                        "source_sha256": sha(source), "raw_box_xywh": list(raw),
                        "integer_crop_xyxy_exclusive": [left, top, right, bottom],
                        "global_padding_offset_xy": list(global_offset),
                        "local_padding_offset_xy": list(local_offset),
                        "output_sha256": {p.name: sha(p) for p in sorted(folder.glob("*.png"))}})
        overview_rows.append((image_id, annotated_global, local_input, row))

    with (args.out / "dimensions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tile_width, tile_height = 626, 438
    overview = Image.new("RGB", (20 + tile_width * 2, 94 + tile_height * 3), (247, 248, 250))
    draw = ImageDraw.Draw(overview)
    draw.text((20, 10), "目标局部放大预览：原图真实框 → 局部模型输入", font=font(29), fill=(20, 25, 30))
    draw.text((20, 53), "尺寸均为宽×高；右图长边500，补边到500×500；此总览为缩略展示。", font=font(22), fill=(45, 50, 55))
    for index, (image_id, global_panel, local_panel, row) in enumerate(overview_rows):
        origin_x = 20 + (index % 2) * tile_width
        origin_y = 94 + (index // 2) * tile_height
        draw.text((origin_x, origin_y),
                  f"{image_id}  原图 {row['original_width']}×{row['original_height']}", font=font(22), fill=(20, 25, 30))
        draw.text((origin_x, origin_y + 32),
                  f"裁剪 {row['crop_width']}×{row['crop_height']} → 放大 {row['resized_width']}×{row['resized_height']}", font=font(22), fill=(20, 25, 30))
        overview.paste(global_panel.resize((294, 294), Image.Resampling.BICUBIC), (origin_x, origin_y + 72))
        overview.paste(local_panel.resize((294, 294), Image.Resampling.BICUBIC), (origin_x + 306, origin_y + 72))
        draw.text((origin_x, origin_y + 376), f"约放大 {row['nominal_scale']:.3f} 倍；输入 500×500", font=font(21), fill=(45, 50, 55))
    overview.save(args.out / "overview_6_samples.png")
    manifest = {
        "purpose": "Preview only; no training started", "data_root": str(args.data),
        "source_split": "official_train_only", "gt_bbox_required": True,
        "bbox_margin": 0, "blur": False, "augmentation": False,
        "coordinates": "same xywh convention as prior project; floor lower, ceil upper, clip to image; upper endpoints exclusive",
        "resize": "Pillow RGB BICUBIC, isotropic nominal scale 500/max(crop_width,crop_height); round dimensions",
        "local_canvas": [SIDE, SIDE], "padding_rgb": list(FILL),
        "green_box_display_only": True, "model_input_annotations": False,
        "normalization_applied_to_saved_png": False, "pillow_version": PIL.__version__,
        "script_sha256": sha(Path(__file__)), "bbox_file_sha256": sha(args.data / "bounding_boxes.txt"),
        "samples": records, "overview_sha256": sha(args.out / "overview_6_samples.png"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "samples": rows, "training_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
