"""Create a deterministic tiny detection dataset for trainer-path auditing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "audit_data"


def make_split(split: str, count: int, seed: int) -> dict[str, int]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    box_count = 0
    for index in range(count):
        array = rng.integers(0, 35, size=(128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
        draw = ImageDraw.Draw(image)
        labels: list[str] = []
        instances = 1 + int(index % 3 == 0)
        for instance in range(instances):
            cls = (index + instance) % 2
            width = 24 + (index * 3 + instance * 5) % 24
            height = 20 + (index * 5 + instance * 7) % 28
            x1 = 6 + (index * 11 + instance * 41) % (116 - width)
            y1 = 7 + (index * 13 + instance * 37) % (115 - height)
            x2, y2 = x1 + width, y1 + height
            color = (225, 45, 45) if cls == 0 else (45, 225, 75)
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=color)
            x_center = (x1 + x2) / 2 / 128
            y_center = (y1 + y2) / 2 / 128
            labels.append(
                f"{cls} {x_center:.8f} {y_center:.8f} {width / 128:.8f} {height / 128:.8f}"
            )
            box_count += 1
        image.save(image_dir / f"{split}_{index:03d}.png")
        (label_dir / f"{split}_{index:03d}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )
    return {"images": count, "boxes": box_count}


def main() -> None:
    train = make_split("train", 12, 2027)
    val = make_split("val", 6, 2005)
    yaml = DATASET / "tiny.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {DATASET.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: red_object",
                "  1: green_object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {"status": "created", "train": train, "val": val, "yaml": str(yaml)}
    (DATASET / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

