"""Export real VOC examples processed by the registered Oracle-PGB rule."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.core import GaussianBankRenderer, make_targets


CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
COLORS = [
    tuple(int(value) for value in color)
    for color in np.random.default_rng(2027).integers(48, 256, size=(20, 3))
]


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"image path has no images component: {image_path}") from error
    return Path(*parts).with_suffix(".txt")


def read_labels(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                rows.append([float(value) for value in fields[:5]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def choose_examples(paths: list[Path]) -> list[tuple[Path, np.ndarray, str]]:
    """Greedily cover every class with distinct deterministic images."""

    records = [(path, read_labels(label_path(path))) for path in paths]
    selected: list[tuple[Path, np.ndarray, str]] = []
    used: set[Path] = set()
    for class_id, class_name in enumerate(CLASS_NAMES):
        candidates = [
            (path, labels)
            for path, labels in records
            if path not in used and labels.size and class_id in labels[:, 0].astype(int)
        ]
        if not candidates:
            raise RuntimeError(f"no unused development image contains {class_name}")
        # Prefer an image with several objects so the multi-target behavior is visible.
        path, labels = max(candidates, key=lambda item: (len(item[1]), -len(str(item[0]))))
        used.add(path)
        selected.append((path, labels, class_name))
    return selected


def letterbox(image: np.ndarray, labels: np.ndarray, size: int = 640):
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    boxes = np.empty((len(labels), 4), dtype=np.float32)
    if len(labels):
        cx = labels[:, 1] * width
        cy = labels[:, 2] * height
        bw = labels[:, 3] * width
        bh = labels[:, 4] * height
        boxes[:, 0] = (cx - bw / 2) * gain + left
        boxes[:, 1] = (cy - bh / 2) * gain + top
        boxes[:, 2] = (cx + bw / 2) * gain + left
        boxes[:, 3] = (cy + bh / 2) * gain + top
    valid = torch.zeros((1, 1, size, size), dtype=torch.bool)
    valid[:, :, top : top + resized_height, left : left + resized_width] = True
    return canvas, boxes, valid, (height, width), (top, left, resized_height, resized_width)


def draw_boxes(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    output = image.copy()
    for box, row in zip(boxes, labels):
        class_id = int(row[0])
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        color = COLORS[class_id]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = CLASS_NAMES[class_id]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        text_top = max(0, y1 - th - 7)
        cv2.rectangle(output, (x1, text_top), (x1 + tw + 5, y1), color, -1)
        cv2.putText(
            output, text, (x1 + 2, max(th + 1, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return output


def title(image: np.ndarray, value: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(output, value, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data_splits" / "voc_dev10_seed2027.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "oracle_visualizations",
    )
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument(
        "--sigma-levels",
        default="0,1,2,4,8",
        help="comma-separated strictly increasing Gaussian sigma levels",
    )
    args = parser.parse_args()
    if not 0 <= args.mix <= 1:
        raise ValueError("mix must be in [0,1]")
    sigma_levels = tuple(float(value) for value in args.sigma_levels.split(","))

    output = args.output.resolve()
    folders = {
        name: output / name
        for name in ("original_640", "strength_map", "oracle_processed", "absolute_difference", "comparison_panels")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    paths = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = choose_examples(paths)
    renderer = GaussianBankRenderer(sigma_levels=sigma_levels, mix=args.mix).eval()
    records: list[dict[str, object]] = []

    for index, (path, labels, covered_class) in enumerate(selected, start=1):
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"cannot decode {path}")
        original, boxes, valid, source_shape, letterbox_meta = letterbox(source, labels)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        box_tensor = torch.from_numpy(boxes)
        targets = make_targets(
            tensor,
            [box_tensor],
            margin=0.10,
            transition=0.50,
            min_radius=8.0,
            valid_pixels=valid,
        )
        with torch.no_grad():
            processed_tensor = renderer(tensor, targets.strength, mix=args.mix)
        processed_rgb = (
            processed_tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255
        ).round().astype(np.uint8)
        processed = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        strength = targets.strength[0, 0].numpy()
        strength_u8 = np.clip(strength * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(strength_u8, cv2.COLORMAP_TURBO)
        difference = cv2.absdiff(original, processed)
        difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        difference_heat = cv2.applyColorMap(
            cv2.normalize(difference_gray, None, 0, 255, cv2.NORM_MINMAX),
            cv2.COLORMAP_INFERNO,
        )

        filename = f"{index:02d}_{covered_class}_{path.stem}.jpg"
        cv2.imwrite(str(folders["original_640"] / filename), original, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(folders["strength_map"] / filename), heatmap, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(folders["oracle_processed"] / filename), processed, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(folders["absolute_difference"] / filename), difference_heat, [cv2.IMWRITE_JPEG_QUALITY, 96])

        original_boxed = draw_boxes(original, boxes, labels)
        processed_boxed = draw_boxes(processed, boxes, labels)
        heatmap_boxed = draw_boxes(heatmap, boxes, labels)
        panel = np.vstack(
            [
                np.hstack([title(original_boxed, "Original + ground-truth boxes"), title(heatmap_boxed, "Oracle strength: blue=protect, red=blur")]),
                np.hstack([title(processed_boxed, f"Oracle processed (mix={args.mix:.2f}, max sigma={sigma_levels[-1]:g})"), title(difference_heat, "Absolute pixel difference")]),
            ]
        )
        cv2.imwrite(str(folders["comparison_panels"] / filename), panel, [cv2.IMWRITE_JPEG_QUALITY, 96])

        present = sorted({CLASS_NAMES[int(class_id)] for class_id in labels[:, 0]})
        valid_array = valid[0, 0].numpy()
        records.append(
            {
                "file": filename,
                "source": str(path),
                "covered_class": covered_class,
                "source_height": source_shape[0],
                "source_width": source_shape[1],
                "objects": len(labels),
                "classes": ";".join(present),
                "protected_fraction_of_valid": float(((strength == 0) & valid_array).sum() / valid_array.sum()),
                "mean_strength_on_valid": float(strength[valid_array].mean()),
                "mean_absolute_pixel_change": float(difference[valid_array].mean()),
                "mix": args.mix,
                "sigma_levels": ";".join(f"{value:g}" for value in sigma_levels),
                "letterbox_top_left_height_width": json.dumps(letterbox_meta),
            }
        )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (output / "README.md").write_text(
        f"""# Oracle-PGB real VOC visualizations

These are real images from the fixed VOC development split, processed with the
registered Oracle geometry: 10% box margin, 0.5 box-scale transition and an
8-pixel minimum radius. This export uses Gaussian levels
{','.join(f'{value:g}' for value in sigma_levels)} and residual mix
{args.mix:.2f}.

- `original_640/`: clean letterboxed detector inputs.
- `strength_map/`: blue protects detail (strength near 0); red applies stronger
  blur (strength near 1).
- `oracle_processed/`: clean processed detector inputs, without drawn boxes.
- `absolute_difference/`: heatmap of the pixel change.
- `comparison_panels/`: 2x2 explanation panels with ground-truth boxes.
- `manifest.csv`: source paths, classes and quantitative processing statistics.

One distinct image was selected for each of the 20 VOC classes, preferring
multi-object examples. Ground-truth boxes are used only because this is an
Oracle diagnostic; this cannot be reproduced at ordinary inference time.
""",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "examples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
