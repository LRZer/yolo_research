"""Generate a read-only CUB-200-2011 dataset profile from the official files."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(r"E:\datasets\CUB-200-2011")
BASE = ROOT / "extracted" / "CUB_200_2011"
MASK_ROOT = ROOT / "extracted" / "segmentations"
OUT = Path(__file__).with_name("CUB_DATASET_FULL_PROFILE_2026-09-25.md")
JSON_OUT = Path(__file__).with_name("CUB_DATASET_FULL_PROFILE_2026-09-25.json")


def rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def fmt_bytes(size: int) -> str:
    return f"{size:,} B ({size / 1_000_000_000:.3f} GB; {size / 2**30:.3f} GiB)"


def main() -> None:
    classes = {int(key): name for key, name in rows(BASE / "classes.txt")}
    images = {int(key): name for key, name in rows(BASE / "images.txt")}
    labels = {int(key): int(label) for key, label in rows(BASE / "image_class_labels.txt")}
    split = {int(key): int(value) for key, value in rows(BASE / "train_test_split.txt")}
    boxes = {int(key): tuple(float(v) for v in values) for key, *values in rows(BASE / "bounding_boxes.txt")}
    part_names = {int(key): name for key, *words in rows(BASE / "parts" / "parts.txt") for name in [" ".join(words)]}
    attribute_names = {int(key): name for key, name in rows(ROOT / "extracted" / "attributes.txt")}
    certainty_names = {int(key): name for key, *words in rows(BASE / "attributes" / "certainties.txt") for name in [" ".join(words)]}

    if not (set(images) == set(labels) == set(split) == set(boxes)):
        raise RuntimeError("Official image/label/split/box ID sets differ")
    if len(classes) != 200 or len(attribute_names) != 312 or len(part_names) != 15:
        raise RuntimeError("Unexpected class, attribute, or part count")

    class_total: Counter[int] = Counter()
    class_train: Counter[int] = Counter()
    class_test: Counter[int] = Counter()
    image_modes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    mask_value_pixels: Counter[int] = Counter()
    image_widths: list[float] = []
    image_heights: list[float] = []
    box_area_ratios: list[float] = []
    oob: list[tuple[int, float]] = []
    image_bytes = 0
    mask_bytes = 0
    missing: list[str] = []
    non_rgb: list[int] = []
    non_l_mask: list[int] = []
    for image_id, relative in images.items():
        class_id = labels[image_id]
        class_total[class_id] += 1
        (class_train if split[image_id] == 1 else class_test)[class_id] += 1
        image_path = BASE / "images" / relative
        mask_path = MASK_ROOT / Path(relative).with_suffix(".png")
        if not image_path.is_file() or not mask_path.is_file():
            missing.append(relative)
            continue
        image_bytes += image_path.stat().st_size
        mask_bytes += mask_path.stat().st_size
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            image_modes[image.mode] += 1
            if image.mode != "RGB":
                non_rgb.append(image_id)
        with Image.open(mask_path) as mask:
            mask.load()
            mask_modes[mask.mode] += 1
            if mask.mode != "L":
                non_l_mask.append(image_id)
            if mask.size != (width, height):
                raise RuntimeError(f"Mask size mismatch for image {image_id}")
            channel = mask.getchannel("L") if mask.mode == "LA" else mask
            mask_value_pixels.update({i: count for i, count in enumerate(channel.histogram()) if count})
        image_widths.append(width)
        image_heights.append(height)
        dimension_counts[f"{width}×{height}"] += 1
        x, y, box_width, box_height = boxes[image_id]
        box_area_ratios.append((box_width * box_height) / (width * height))
        excess = max(0.0, -x, -y, x + box_width - width, y + box_height - height)
        if excess > 0 or box_width <= 0 or box_height <= 0:
            oob.append((image_id, excess))
    if missing:
        raise RuntimeError(f"Missing images/masks: {missing[:5]}")
    if len(set((BASE / "images").rglob("*.jpg"))) != len(images):
        raise RuntimeError("JPG disk count differs from manifest")
    if len(set(MASK_ROOT.rglob("*.png"))) != len(images):
        raise RuntimeError("PNG disk count differs from manifest")

    part_rows = 0
    visible_part_rows = 0
    part_visibility: Counter[int] = Counter()
    part_pair_counts: Counter[tuple[int, int]] = Counter()
    for image_id, part_id, _x, _y, visible in rows(BASE / "parts" / "part_locs.txt"):
        image_id, part_id = int(image_id), int(part_id)
        part_rows += 1
        part_pair_counts[(image_id, part_id)] += 1
        if int(visible) == 1:
            visible_part_rows += 1
            part_visibility[part_id] += 1
    if part_rows != len(images) * len(part_names) or len(part_pair_counts) != part_rows:
        raise RuntimeError("Part annotation coverage is not exactly one per image/part")
    click_rows = sum(1 for _ in (BASE / "parts" / "part_click_locs.txt").open(encoding="utf-8"))

    attribute_rows = 0
    positive_attribute_rows = 0
    extra_attribute_field_rows = 0
    extra_attribute_field_images: Counter[int] = Counter()
    certainty_counts: Counter[int] = Counter()
    attr_image_counts: Counter[int] = Counter()
    with (BASE / "attributes" / "image_attribute_labels.txt").open(encoding="utf-8") as stream:
        for line in stream:
            columns = line.split()
            if len(columns) == 6 and columns[4] == "0":
                extra_attribute_field_rows += 1
                extra_attribute_field_images[int(columns[0])] += 1
            elif len(columns) != 5:
                raise RuntimeError(f"Unexpected attribute record format: {line[:100]!r}")
            image_id, attribute_id, present, certainty_id = columns[:4]
            attribute_rows += 1
            positive_attribute_rows += int(present)
            certainty_counts[int(certainty_id)] += 1
            attr_image_counts[int(image_id)] += 1
    matrix_rows = rows(BASE / "attributes" / "class_attribute_labels_continuous.txt")
    if len(matrix_rows) != 200 or any(len(line) != 312 for line in matrix_rows):
        raise RuntimeError("Class attribute matrix is not 200×312")

    archive_main = (ROOT / "original_archives" / "CUB_200_2011.tgz").stat().st_size
    archive_masks = (ROOT / "original_archives" / "segmentations.tgz").stat().st_size
    total_pixels = sum(mask_value_pixels.values())
    summary = {
        "classes": len(classes),
        "images": len(images),
        "train_images": sum(class_train.values()),
        "test_images": sum(class_test.values()),
        "bounding_boxes": len(boxes),
        "masks": len(images),
        "parts": len(part_names),
        "part_rows": part_rows,
        "visible_part_rows": visible_part_rows,
        "part_click_rows": click_rows,
        "attributes": len(attribute_names),
        "image_attribute_rows": attribute_rows,
        "positive_attribute_rows": positive_attribute_rows,
        "extra_attribute_field_rows": extra_attribute_field_rows,
        "extra_attribute_field_images": dict(extra_attribute_field_images),
        "attribute_rows_per_image_min": min(attr_image_counts.values()),
        "attribute_rows_per_image_max": max(attr_image_counts.values()),
        "class_attribute_matrix": [len(matrix_rows), len(matrix_rows[0])],
        "archive_main_bytes": archive_main,
        "archive_mask_bytes": archive_masks,
        "image_bytes": image_bytes,
        "mask_bytes": mask_bytes,
        "image_modes": dict(image_modes),
        "mask_modes": dict(mask_modes),
        "unique_dimensions": len(dimension_counts),
        "width_min_median_max": [min(image_widths), percentile(image_widths, 0.5), max(image_widths)],
        "height_min_median_max": [min(image_heights), percentile(image_heights, 0.5), max(image_heights)],
        "box_area_ratio_p10_p50_p90": [percentile(box_area_ratios, x) for x in (0.1, 0.5, 0.9)],
        "class_size_min_max": [min(class_total.values()), max(class_total.values())],
        "oob_box_count": len(oob),
        "oob_box_max_pixels": max((amount for _, amount in oob), default=0),
        "mask_value_pixels": dict(sorted(mask_value_pixels.items())),
        "certainty_counts": dict(sorted(certainty_counts.items())),
        "per_class": [
            {"id": class_id, "name": classes[class_id], "all": class_total[class_id], "train": class_train[class_id], "test": class_test[class_id]}
            for class_id in sorted(classes)
        ],
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    text: list[str] = [
        "# CUB-200-2011 数据集全量信息（本机原始文件复算）",
        "",
        "日期：2026-09-25。统计来源：已校验的 Caltech 官方原始包及解压文件；只读扫描，未修改图片或标注，未运行模型。",
        "",
        "来源：[Caltech 官方页面](https://www.vision.caltech.edu/datasets/cub_200_2011/)、[主包记录](https://data.caltech.edu/records/65de6-vp158)、[分割包记录](https://data.caltech.edu/records/w9d68-gec53)。图片限非商业研究和教学使用。官方提醒它与 ImageNet 图片有重叠；若用 ImageNet 预训练权重，测试可能存在预训练污染。",
        "",
        "## 1. 它是什么、掩膜是什么",
        "",
        "这是 200 种鸟的细粒度**单标签图像分类**数据集。每张图有一个物种类别、一个主要鸟的矩形框、15 类部位点和属性标注。官方另行发布的 `segmentations.tgz` 为逐像素鸟体掩膜：图片与掩膜同名、同尺寸；黑色(0)表示背景、白色(255)表示鸟体，轮廓处可见中间灰度。实际文件不是严格二值，出现 0/51/102/153/204/255 六档，因此阈值规则须在实验前说明；2 张为 `LA` 模式但 alpha 恒为 255。它是**发布的参考前景标注**，不同于我们未来由模型预测的模糊强度图。第一阶段按 GT 框中心做实验不依赖分割包；掩膜可用于诊断被模糊的鸟体像素比例，正式未知图像推理不能读取它。",
        "",
        "## 2. 总量与划分",
        "",
        "| 指标 | 实测值 |",
        "|---|---:|",
        f"| 鸟类物种 | {len(classes)} |",
        f"| 图像 | {len(images):,} |",
        f"| 官方训练集 | {sum(class_train.values()):,} |",
        f"| 官方测试集 | {sum(class_test.values()):,} |",
        f"| 矩形框 | {len(boxes):,}（每图1个） |",
        f"| 独立分割掩膜 | {len(images):,}（与图像逐一匹配） |",
        f"| 每类图像数 | {min(class_total.values())}–{max(class_total.values())} |",
        f"| 每类训练图 | {min(class_train.values())}–{max(class_train.values())} |",
        f"| 每类测试图 | {min(class_test.values())}–{max(class_test.values())} |",
        f"| 图像文件合计 | {fmt_bytes(image_bytes)} |",
        f"| 分割PNG合计 | {fmt_bytes(mask_bytes)} |",
        f"| 主压缩包 | {fmt_bytes(archive_main)} |",
        f"| 分割压缩包 | {fmt_bytes(archive_masks)} |",
        "",
        "官方 `train_test_split.txt` 中 1 表示训练、0 表示测试。若要有开发验证集，只从 5,994 张官方训练图内再划；不要把官方测试集拿来反复调半径或模糊强度。",
        "",
        "## 3. 图像、框与掩膜的技术细节",
        "",
        f"- 图像共有 {len(dimension_counts):,} 种原始尺寸；宽度最小/中位/最大为 {min(image_widths):.0f}/{percentile(image_widths, 0.5):.0f}/{max(image_widths):.0f} 像素，高度为 {min(image_heights):.0f}/{percentile(image_heights, 0.5):.0f}/{max(image_heights):.0f} 像素。最常见尺寸：" + "、".join(f"{size}（{count:,}张）" for size, count in dimension_counts.most_common(10)) + "。",
        f"- JPG 模式：" + "、".join(f"{mode} {count:,}张" for mode, count in image_modes.items()) + "；PNG 模式：" + "、".join(f"{mode} {count:,}张" for mode, count in mask_modes.items()) + f"。全部 {len(images):,} 对都能完整像素解码且尺寸相同。",
        f"- 类别规模相对均衡：144类恰好60张；194类有30张官方训练图，另6类有29张；每类官方测试图为 {min(class_test.values())}–{max(class_test.values())} 张。",
        f"- 原始框采用像素坐标 `(x, y, width, height)`，框面积占图像面积的第10/50/90百分位分别约为 {percentile(box_area_ratios, 0.1):.1%}/{percentile(box_area_ratios, 0.5):.1%}/{percentile(box_area_ratios, 0.9):.1%}。有 {len(oob)} 张原始框越过图片边缘，最大越界 {max(amount for _, amount in oob):.0f} 像素；不能直接假定框始终在图内，原始文件未改动。",
        "- 分割图的六档像素数量：" + "、".join(f"{value}→{count:,}" for value, count in sorted(mask_value_pixels.items())) + f"；共 {total_pixels:,} 个掩膜像素。将哪一级当作前景，必须在具体实验中预先定义。",
        "",
        "## 4. 额外标注",
        "",
        f"- 部位：`parts/parts.txt` 定义 {len(part_names)} 种，`parts/part_locs.txt` 有 {part_rows:,} 条聚合记录，正好每图每部位一条；其中可见标记为1的有 {visible_part_rows:,} 条（{visible_part_rows / part_rows:.1%}）。不可见点通常坐标为0，不应当作真实位置。`parts/part_click_locs.txt` 有 {click_rows:,} 条众包原始点击记录。",
        f"- 属性：实际 `attributes.txt` 有 {len(attribute_names)} 个属性；`attributes/image_attribute_labels.txt` 有 {attribute_rows:,} 条记录，每图最少/最多 {min(attr_image_counts.values())}/{max(attr_image_counts.values())} 条，标为存在的记录 {positive_attribute_rows:,} 条。每条含属性是否存在、确定度和标注耗时。`class_attribute_labels_continuous.txt` 是 {len(matrix_rows)}×{len(matrix_rows[0])} 的类级属性比例矩阵。包内 README 概览写作“322”，但实际名单与矩阵及官方网页均为 312，使用时以文件内容为准。",
        f"- 属性源文件另有 {extra_attribute_field_rows} 条六列记录（图像ID 2275、9364各303条，均在第5列多一个`0`）；其它记录为 README 描述的五列。本报告只把前四列用于标签统计、最后一列视为耗时，不改动原文件。未来若使用逐图属性，必须显式处理并审计这一格式异常。",
        "- 15个部位的可见次数：" + "、".join(f"{part_names[i]} {part_visibility[i]:,}" for i in sorted(part_names)) + "。",
        "- 属性确定度编码：" + "、".join(f"{i}={certainty_names[i]}（{certainty_counts[i]:,}条）" for i in sorted(certainty_names)) + "。",
        "",
        "## 5. 文件结构与字段",
        "",
        "- `images/<species>/<file>.jpg`：原始照片；`images.txt`：图像 ID 到相对路径；`classes.txt`：200 类 ID 与名称；`image_class_labels.txt`：图像 ID 到类别。",
        "- `train_test_split.txt`：官方训练/测试标记；`bounding_boxes.txt`：每图一个 `(x,y,w,h)` 框；`parts/part_locs.txt`：每图每部位 `(x,y,visible)`。",
        "- `attributes/`：逐图属性判定、确定度及类级属性比例；压缩包顶层另有 `attributes.txt` 属性名称表。",
        "- 独立分割包解压到 `segmentations/<species>/<file>.png`，与 JPG 相对路径对应。完整官方包与解压件在服务器 `/home/ubuntu/rz/foveated_classification_research/data/CUB-200-2011/`，本机在 `E:\\datasets\\CUB-200-2011\\`。",
        "",
        "## 6. 200 类逐类图片、官方划分、框与掩膜数量",
        "",
        "每图恰好一个类别、一个框、一张掩膜，所以每类框数和掩膜数均等于该类图片总数。以下数字直接由 `image_class_labels.txt` 与 `train_test_split.txt` 复算。",
        "",
        "| ID | 官方类名 | 全部图片 | 训练 | 测试 | 框 | 掩膜 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for class_id in sorted(classes):
        count = class_total[class_id]
        text.append(f"| {class_id} | {classes[class_id]} | {count} | {class_train[class_id]} | {class_test[class_id]} | {count} | {count} |")
    text.extend(["", "注：这些是数据集统计，不是模型准确率或检测 AP；本研究尚未跑 CUB 分类模型。", ""])
    OUT.write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2, ensure_ascii=False))
    print(f"REPORT {OUT}")
    print(f"JSON {JSON_OUT}")


if __name__ == "__main__":
    main()
