"""Generate deterministic, GT-box Oracle B/C/D preview images from CUB train images.

This script does not train a classifier or modify the source dataset. The methods
match CUB_STAGE1_PROTOCOL_V3_NATIVE500_DISCUSSION_2026-09-25.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter


DEFAULT_DATA = Path(r"E:\datasets\CUB-200-2011\extracted\CUB_200_2011")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "previews" / "cub_bcd_native500_v1_2026-09-25"
CANVAS = 500


@dataclass(frozen=True)
class Record:
    image_id: int
    relpath: str
    class_id: int
    train: bool
    raw_box: tuple[float, float, float, float]
    width: int
    height: int
    image_mode: str

    @property
    def clipped_box(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.raw_box
        return (
            max(0.0, min(float(self.width), x)),
            max(0.0, min(float(self.height), y)),
            max(0.0, min(float(self.width), x + w)),
            max(0.0, min(float(self.height), y + h)),
        )

    @property
    def clipped_area_fraction(self) -> float:
        x1, y1, x2, y2 = self.clipped_box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (self.width * self.height)

    @property
    def raw_box_outside(self) -> bool:
        x, y, w, h = self.raw_box
        return x < 0 or y < 0 or x + w > self.width or y + h > self.height


def read_lines(root: Path, filename: str) -> list[list[str]]:
    return [line.split() for line in (root / filename).read_text(encoding="utf-8").splitlines()]


def metadata(root: Path) -> list[Record]:
    images = {int(row[0]): row[1] for row in read_lines(root, "images.txt")}
    labels = {int(row[0]): int(row[1]) for row in read_lines(root, "image_class_labels.txt")}
    splits = {int(row[0]): row[1] == "1" for row in read_lines(root, "train_test_split.txt")}
    boxes = {int(row[0]): tuple(float(v) for v in row[1:5]) for row in read_lines(root, "bounding_boxes.txt")}
    ids = set(images)
    if ids != set(labels) or ids != set(splits) or ids != set(boxes) or len(ids) != 11788:
        raise ValueError("CUB metadata IDs do not match the audited 11,788 images")
    result = []
    for image_id in sorted(ids):
        with Image.open(root / "images" / images[image_id]) as image:
            width, height = image.size
            mode = image.mode
        result.append(Record(image_id, images[image_id], labels[image_id], splits[image_id], boxes[image_id], width, height, mode))
    return result


def training_rgb_mean(root: Path, rows: list[Record]) -> tuple[np.ndarray, int]:
    rgb_sum = np.zeros(3, dtype=np.uint64)
    pixel_count = 0
    for row in rows:
        if not row.train:
            continue
        with Image.open(root / "images" / row.relpath) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        rgb_sum += rgb.sum(axis=(0, 1), dtype=np.uint64)
        pixel_count += row.width * row.height
    if sum(row.train for row in rows) != 5994 or pixel_count <= 0:
        raise ValueError("Official training split was not the expected 5,994 images")
    return rgb_sum.astype(np.float64) / pixel_count, pixel_count


def choose_representatives(rows: list[Record]) -> list[tuple[str, Record]]:
    train = [r for r in rows if r.train]
    chosen: list[tuple[str, Record]] = []
    used_ids: set[int] = set()
    used_classes: set[int] = set()

    def add(name: str, pool: list[Record], target_fraction: float) -> None:
        options = [r for r in pool if r.image_id not in used_ids and r.clipped_area_fraction > 0]
        distinct = [r for r in options if r.class_id not in used_classes]
        if distinct:
            options = distinct
        if not options:
            raise ValueError(f"No training candidate for {name}")
        row = min(options, key=lambda r: (abs(r.clipped_area_fraction - target_fraction), r.image_id))
        chosen.append((name, row))
        used_ids.add(row.image_id)
        used_classes.add(row.class_id)

    wide = [r for r in train if r.width == 500 and 330 <= r.height <= 375]
    add("wide_small_bird", wide, 0.14)
    add("wide_medium_bird", wide, 0.32)
    add("wide_large_bird", wide, 0.58)
    add("portrait", [r for r in train if r.height == 500 and 300 <= r.width <= 375], 0.32)
    add("square", [r for r in train if r.width == r.height == 500], 0.32)
    add("small_native_image", [r for r in train if max(r.width, r.height) <= 350], 0.32)
    add("out_of_bounds_box", [r for r in train if r.raw_box_outside], 0.30)
    add("grayscale_jpg", [r for r in train if r.image_mode == "L"], 0.30)
    return chosen


def protected_geometry(row: Record) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float, float]:
    x1, y1, x2, y2 = row.clipped_box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty clipped box: image {row.image_id}")
    dx_expand = 0.1 * (x2 - x1)
    dy_expand = 0.1 * (y2 - y1)
    px1 = max(0.0, x1 - dx_expand)
    py1 = max(0.0, y1 - dy_expand)
    px2 = min(float(row.width), x2 + dx_expand)
    py2 = min(float(row.height), y2 + dy_expand)
    xx = np.arange(row.width, dtype=np.float32)[None, :] + 0.5
    yy = np.arange(row.height, dtype=np.float32)[:, None] + 0.5
    dist_x = np.maximum(np.maximum(px1 - xx, xx - px2), 0.0)
    dist_y = np.maximum(np.maximum(py1 - yy, yy - py2), 0.0)
    distance = np.sqrt(dist_x * dist_x + dist_y * dist_y)
    scale = max(row.width, row.height) / 500.0
    transition = 70.0 * scale
    q = np.clip(distance / transition, 0.0, 1.0)
    smooth = q * q * (3.0 - 2.0 * q)
    sigma = 8.0 * scale * smooth
    return smooth, sigma, (px1, py1, px2, py2), scale, transition


def blur_valid_image(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return rgb.astype(np.float32)
    return gaussian_filter(rgb.astype(np.float32), sigma=(sigma, sigma, 0.0), mode="reflect", truncate=3.0)


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def group_images(rgb: np.ndarray, row: Record, mean_rgb: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    smooth, sigma, pbox, scale, transition = protected_geometry(row)
    levels = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32) * scale
    filtered = [blur_valid_image(rgb, float(level)) for level in levels]
    a = rgb.copy()
    b = to_uint8(filtered[-1])
    c = to_uint8(rgb.astype(np.float32) * (1.0 - smooth[:, :, None]) + mean_rgb[None, None, :] * smooth[:, :, None])
    sigma = np.clip(sigma, 0.0, float(levels[-1]))
    d_float = np.empty_like(filtered[0])
    covered = np.zeros(sigma.shape, dtype=bool)
    for index in range(3):
        lo, hi = float(levels[index]), float(levels[index + 1])
        selected = (sigma >= lo) & (sigma <= hi) if index == 2 else (sigma >= lo) & (sigma < hi)
        covered |= selected
        blend = ((sigma[selected] - lo) / (hi - lo))[:, None]
        d_float[selected] = filtered[index][selected] * (1.0 - blend) + filtered[index + 1][selected] * blend
    if not np.all(covered):
        raise AssertionError("Some pixels were not assigned a blur interpolation interval")
    d = to_uint8(d_float)
    protected_pixels = smooth == 0
    c[protected_pixels] = rgb[protected_pixels]
    d[protected_pixels] = rgb[protected_pixels]
    if not np.array_equal(c[protected_pixels], rgb[protected_pixels]) or not np.array_equal(d[protected_pixels], rgb[protected_pixels]):
        raise AssertionError("Protected pixels changed")
    info = {
        "clipped_box_xyxy": list(row.clipped_box),
        "protected_box_xyxy": list(pbox),
        "native_scale": scale,
        "transition_px": transition,
        "max_sigma_px": float(levels[-1]),
        "protected_valid_pixel_fraction": float(np.mean(protected_pixels)),
        "half_or_more_blur_valid_pixel_fraction": float(np.mean(sigma >= 4.0 * scale)),
        "fully_blurred_valid_pixel_fraction": float(np.mean(sigma >= 8.0 * scale - 1e-5)),
        "mean_sigma_px": float(sigma.mean()),
        "mean_abs_change_B_valid": float(np.abs(b.astype(np.int16) - rgb.astype(np.int16)).mean()),
        "mean_abs_change_C_valid": float(np.abs(c.astype(np.int16) - rgb.astype(np.int16)).mean()),
        "mean_abs_change_D_valid": float(np.abs(d.astype(np.int16) - rgb.astype(np.int16)).mean()),
    }
    return {"A_original": a, "B_global_blur": b, "C_context_removed": c, "D_gradual_background": d}, info


def canvas_image(rgb: np.ndarray, mean_rgb: np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
    height, width = rgb.shape[:2]
    if width > CANVAS or height > CANVAS:
        raise ValueError("Source image exceeds 500x500")
    left = (CANVAS - width) // 2
    top = (CANVAS - height) // 2
    fill = tuple(int(v) for v in to_uint8(mean_rgb))
    canvas = Image.new("RGB", (CANVAS, CANVAS), fill)
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (left, top))
    return canvas, (left, top)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contact_sheet(row: Record, stage: str, images: dict[str, Image.Image], destination: Path) -> None:
    gap, title_height, label_height = 16, 42, 30
    sheet = Image.new("RGB", (2 * CANVAS + 3 * gap, 2 * (CANVAS + label_height) + title_height + 3 * gap), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"CUB train ID {row.image_id} | {stage} | {row.width}x{row.height} | box area {row.clipped_area_fraction:.1%}"
    draw.text((gap, 8), title, fill="black", font=font(18))
    for index, (group, image) in enumerate(images.items()):
        col, r = index % 2, index // 2
        x = gap + col * (CANVAS + gap)
        y = title_height + gap + r * (CANVAS + label_height + gap)
        draw.text((x, y), group, fill="black", font=font(19))
        sheet.paste(image, (x, y + label_height))
    sheet.save(destination, format="PNG")


def annotation_preview(row: Record, original: Image.Image, pbox: list[float], offset: tuple[int, int], destination: Path) -> None:
    marked = original.copy()
    draw = ImageDraw.Draw(marked)
    left, top = offset
    x1, y1, x2, y2 = row.clipped_box
    draw.rectangle((left + x1, top + y1, left + x2, top + y2), outline=(255, 40, 20), width=3)
    px1, py1, px2, py2 = pbox
    draw.rectangle((left + px1, top + py1, left + px2, top + py2), outline=(20, 235, 255), width=3)
    marked.save(destination, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.data.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preview directory: {output}")

    rows = metadata(root)
    selected = choose_representatives(rows)
    mean_rgb, train_pixel_count = training_rgb_mean(root, rows)
    output.mkdir(parents=True)
    manifest: dict = {
        "source_data": str(root),
        "source_split_used_for_samples": "official_train_only",
        "source_train_images_used_for_rgb_mean": 5994,
        "source_train_valid_pixel_count": train_pixel_count,
        "train_rgb_mean_0_255": mean_rgb.tolist(),
        "protocol": "CUB_STAGE1_PROTOCOL_V3_NATIVE500_DISCUSSION_2026-09-25.md",
        "canvas": [500, 500],
        "implementation": "numpy + scipy.ndimage.gaussian_filter(mode=reflect, truncate=3.0); no geometric resizing",
        "samples": [],
    }
    readme = [
        "# CUB B/C/D 实际图片预览（仅官方训练图片）",
        "",
        "本目录仅用于用户检查图像处理效果；没有训练分类模型、没有改动原始图片或标注，也没有使用官方测试图片。所有 A/B/C/D 单张 PNG 都是真实 500×500 模型候选输入，联系表只是并排展示。",
        "",
        f"训练集有效像素 RGB 均值（0–255）：{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}。画布填充使用四舍五入的该颜色。",
        "",
        "B：整幅有效图 σ=8×(原图长边/500) 模糊；C：真值框外扩10%内清晰，框外按70×(长边/500)过渡至均值色；D：同一保护区外按0/2/4/8×(长边/500)多尺度高斯连续混合。红框=原始框裁边，青框=保护框；框线只在 *_boxes.png，绝不叠在模型候选输入图上。",
        "",
        "| 用途 | 类别/ID | 尺寸 | 框面积 | 对比图 | 标注示意 |",
        "|---|---|---:|---:|---|---|",
    ]
    for stage, row in selected:
        if not row.train:
            raise AssertionError("A test image was selected")
        source_path = root / "images" / row.relpath
        with Image.open(source_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        groups, info = group_images(rgb, row, mean_rgb)
        if rgb.shape[:2] != (row.height, row.width):
            raise AssertionError("Unexpected decoded image dimensions")
        sample_dir = output / f"{row.image_id:05d}_{stage}"
        sample_dir.mkdir()
        canvases: dict[str, Image.Image] = {}
        offsets = set()
        file_hashes = {}
        for group, pixels in groups.items():
            canvas, offset = canvas_image(pixels, mean_rgb)
            offsets.add(offset)
            path = sample_dir / f"{group}.png"
            canvas.save(path, format="PNG")
            if Image.open(path).size != (500, 500):
                raise AssertionError("Saved image is not 500x500")
            canvases[group] = canvas
            file_hashes[group] = sha256(path)
        if len(offsets) != 1:
            raise AssertionError("Groups do not have matching padding")
        offset = next(iter(offsets))
        sheet = sample_dir / "A_B_C_D_contact_sheet.png"
        boxes = sample_dir / "A_original_boxes.png"
        contact_sheet(row, stage, canvases, sheet)
        annotation_preview(row, canvases["A_original"], info["protected_box_xyxy"], offset, boxes)
        entry = {
            "stage": stage,
            "image_id": row.image_id,
            "class_id": row.class_id,
            "relative_image_path": row.relpath,
            "official_train": row.train,
            "source_sha256": sha256(source_path),
            "source_mode": row.image_mode,
            "size_wh": [row.width, row.height],
            "raw_box_xywh": list(row.raw_box),
            "raw_box_outside": row.raw_box_outside,
            "box_area_fraction": row.clipped_area_fraction,
            "padding_offset_xy": list(offset),
            "metrics": info,
            "output_sha256": file_hashes,
        }
        manifest["samples"].append(entry)
        readme.append(f"| {stage} | {row.relpath.split('/')[0]} / {row.image_id} | {row.width}×{row.height} | {row.clipped_area_fraction:.1%} | [A/B/C/D]({sample_dir.name}/A_B_C_D_contact_sheet.png) | [框]({sample_dir.name}/A_original_boxes.png) |")
        print(f"{stage}: ID={row.image_id} {row.relpath} {row.width}x{row.height} protected={info['protected_valid_pixel_fraction']:.1%} D_changed={info['mean_abs_change_D_valid']:.2f}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Saved {len(selected)} audited train-image previews to {output}", flush=True)


if __name__ == "__main__":
    main()
