"""CPU-only selected-only Oracle equivalence on four real VisDrone frames.

This is not an Ultralytics dataloader/DDP test and does not modify data.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgb.core import GaussianBankRenderer
from pgb.oracle_integration import apply_stochastic_oracle_blur
from pgb.selected_oracle_blur import apply_stochastic_oracle_blur_selected
from tools.export_oracle_visualizations import label_path, letterbox, read_labels


STEMS = (
    "val_519_0000359_03529_d_0000712",
    "val_328_0000289_05001_d_0000836",
    "val_88_0000155_00401_d_0000001",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("view", type=Path)
    args = parser.parse_args()
    image_dir = args.view / "images" / "val"
    first = sorted(image_dir.glob("*.jpg"))[0]
    paths = [first, *(image_dir / f"{stem}.jpg" for stem in STEMS)]
    images = []
    all_boxes = []
    all_indices = []
    counts = []
    for index, path in enumerate(paths):
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"cannot decode {path}")
        labels = read_labels(label_path(path))
        letterboxed, boxes, _, _, _ = letterbox(source, labels, size=960)
        images.append(torch.from_numpy(cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255)
        xyxy = torch.from_numpy(boxes)
        xywh = torch.stack(((xyxy[:, 0] + xyxy[:, 2]) / 2,
                            (xyxy[:, 1] + xyxy[:, 3]) / 2,
                            xyxy[:, 2] - xyxy[:, 0],
                            xyxy[:, 3] - xyxy[:, 1]), dim=1) / 960
        all_boxes.append(xywh)
        all_indices.append(torch.full((len(xywh),), float(index)))
        counts.append(len(xywh))
    batch = {
        "img": torch.stack(images),
        "bboxes": torch.cat(all_boxes),
        "batch_idx": torch.cat(all_indices),
        "cls": torch.zeros((sum(counts), 1)),
    }
    renderer = GaussianBankRenderer(sigma_levels=(0, 2, 4, 8, 16))
    output = {}
    for name, function in (("registered_full_batch", apply_stochastic_oracle_blur),
                           ("candidate_selected_only", apply_stochastic_oracle_blur_selected)):
        sample = copy.deepcopy(batch)
        torch.manual_seed(2027)
        started = time.perf_counter()
        function(sample, renderer, mix=0.5, probability=0.5)
        output[name] = {
            "seconds_single_cpu_call": time.perf_counter() - started,
            "chosen": sample["oracle_augmented"].tolist(),
            "images": sample["img"],
        }
    a = output["registered_full_batch"]
    b = output["candidate_selected_only"]
    max_difference = float((a["images"] - b["images"]).abs().max())
    result = {
        "device": "CPU",
        "image_size": 960,
        "image_stems": [p.stem for p in paths],
        "box_counts": counts,
        "registered_chosen": a["chosen"],
        "candidate_chosen": b["chosen"],
        "chosen_equal": a["chosen"] == b["chosen"],
        "max_output_absolute_difference": max_difference,
        "registered_seconds_single_cpu_call": a["seconds_single_cpu_call"],
        "candidate_seconds_single_cpu_call": b["seconds_single_cpu_call"],
        "caveat": "one unreplicated CPU batch with source-image LetterBox, no Mosaic/DDP; do not infer whole-epoch GPU speedup",
    }
    if not result["chosen_equal"] or max_difference > 1e-6:
        raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
