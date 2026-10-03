"""CPU-only, annotation-free edge/texture previews; no training or dataset writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


SAMPLES = [
    (10140, "wide_small_bird", "173.Orange_crowned_Warbler/Orange_Crowned_Warbler_0051_168002.jpg", "eac9794ba5e6ebb2c102b4ef0906f38c44904ae4f44cf94482802178ee9a8224"),
    (2568, "wide_medium_bird", "045.Northern_Fulmar/Northern_Fulmar_0047_43698.jpg", "34050877c7cbee369c5588a06b4a8071184702ddcb7b93bd620876bceb6f675b"),
    (6223, "wide_large_bird", "107.Common_Raven/Common_Raven_0007_102020.jpg", "0d0e55a072578cbc86cd3b130a9671b6c3ff6766ee1b1310943e5cf58ca1e0b6"),
    (4786, "portrait", "082.Ringed_Kingfisher/Ringed_Kingfisher_0008_72943.jpg", "d597072353353a29bf04f00c8628a3d86ab9967739f6646255d34ebdb02af07a"),
    (9234, "square", "158.Bay_breasted_Warbler/Bay_Breasted_Warbler_0099_159753.jpg", "9cfe6a6c298e7a82d89df1bc749979c03f48798821374f9b7ab4d9d188ffb862"),
    (1767, "small_native_image", "032.Mangrove_Cuckoo/Mangrove_Cuckoo_0020_794629.jpg", "2cb2069b307dcedad930a6124caa37dbced8558c5eae4164faa351bf7d1ae438"),
    (1592, "american_crow", "029.American_Crow/American_Crow_0003_25130.jpg", "611399b27f6040cd9725d9b683b37c344006cbb67891800803fae23478deb200"),
    (3617, "grayscale_source", "063.Ivory_Gull/Ivory_Gull_0040_49180.jpg", "b68ead27a51c5f995ffb4b61b5c07d61ef6605cd6113abd5ebe193c21fa6059b"),
]
METHODS = ["original", "scharr", "canny", "dog", "gabor"]
LABELS = {
    "original": "Original RGB",
    "scharr": "Scharr: continuous edges",
    "canny": "Canny: binary edges",
    "dog": "DoG: multiscale detail",
    "gabor": "Gabor: oriented texture",
}
TRAIN_PADDING = (124, 128, 111)
BORDER = cv2.BORDER_REFLECT_101


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(response: np.ndarray, gain: float) -> np.ndarray:
    assert np.isfinite(response).all()
    return np.rint(np.clip(response * gain, 0.0, 1.0) * 255).astype(np.uint8)


def gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    radius = math.ceil(3 * sigma)
    return cv2.GaussianBlur(gray, (2 * radius + 1, 2 * radius + 1), sigmaX=sigma,
                            sigmaY=sigma, borderType=BORDER)


def gabor_kernels() -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for wavelength in (4.0, 8.0):
        sigma = 0.56 * wavelength
        radius = math.ceil(3 * sigma)
        for angle in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
            phases = []
            for phase in (0.0, math.pi / 2):
                kernel = cv2.getGaborKernel((2 * radius + 1, 2 * radius + 1),
                                            sigma, angle, wavelength, 0.5, phase,
                                            ktype=cv2.CV_32F)
                kernel -= kernel.mean()
                kernel /= np.abs(kernel).sum()
                phases.append(kernel)
            result.append((phases[0], phases[1]))
    return result


KERNELS = gabor_kernels()


def extract(gray_u8: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    gray = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1 / 32, borderType=BORDER)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=1 / 32, borderType=BORDER)
    scharr = np.hypot(gx, gy)
    canny_input = np.rint(gaussian(gray, 1.0) * 255).clip(0, 255).astype(np.uint8)
    canny = cv2.Canny(canny_input, threshold1=50, threshold2=100, apertureSize=3,
                       L2gradient=True)
    blur1, blur2, blur4 = [gaussian(gray, sigma) for sigma in (1.0, 2.0, 4.0)]
    dog = np.maximum(np.abs(blur1 - blur2), np.abs(blur2 - blur4))
    gabor = np.zeros_like(gray)
    for even_kernel, odd_kernel in KERNELS:
        even = cv2.filter2D(gray, cv2.CV_32F, even_kernel, borderType=BORDER)
        odd = cv2.filter2D(gray, cv2.CV_32F, odd_kernel, borderType=BORDER)
        np.maximum(gabor, np.hypot(even, odd), out=gabor)
    maps = {"scharr": quantize(scharr, 4.0), "canny": canny,
            "dog": quantize(dog, 8.0), "gabor": quantize(gabor, 8.0)}
    stats = {}
    for name, value in maps.items():
        assert value.shape == gray_u8.shape and value.dtype == np.uint8
        stats[name] = {"mean_0_255": float(value.mean()),
                       "nonzero_fraction": float(np.mean(value > 0)),
                       "saturated_fraction": float(np.mean(value == 255))}
    assert set(np.unique(canny)).issubset({0, 255})
    return maps, stats


def centered(rgb: np.ndarray, fill: tuple[int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    height, width = rgb.shape[:2]
    assert width <= 500 and height <= 500
    x, y = (500 - width) // 2, (500 - height) // 2
    canvas = np.full((500, 500, 3), fill, dtype=np.uint8)
    canvas[y:y + height, x:x + width] = rgb
    assert np.array_equal(canvas[y:y + height, x:x + width], rgb)
    return Image.fromarray(canvas), (x, y)


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact(images: dict[str, Image.Image], title: str, methods: list[str]) -> Image.Image:
    margin, gap = 16, 12
    result = Image.new("RGB", (margin * 2 + 500 * len(methods) + gap * (len(methods) - 1), 586), (245, 245, 245))
    draw = ImageDraw.Draw(result)
    draw.text((margin, 12), title, fill=(20, 20, 20), font=font(20))
    for index, method in enumerate(methods):
        x = margin + index * (500 + gap)
        draw.text((x, 46), LABELS[method], fill=(20, 20, 20), font=font(18))
        result.paste(images[method], (x, 76))
    return result


def self_test() -> dict:
    flat = np.full((64, 64), 127, np.uint8)
    flat_maps, _ = extract(flat)
    assert all(not value.any() for value in flat_maps.values()), "Uniform input must have zero edges"
    step = flat.copy()
    step[:, 32:] = 255
    step_maps, _ = extract(step)
    assert all(value.any() for value in step_maps.values())
    rgb = np.repeat(step_maps["scharr"][..., None], 3, axis=2)
    canvas, offset = centered(rgb, (0, 0, 0))
    assert canvas.size == (500, 500) and offset == (218, 218)
    assert not np.asarray(canvas)[:218].any()
    return {"uniform_input_zero_response": True, "step_input_nonzero_response": True,
            "canny_binary": True, "native_pixels_unchanged_by_padding": True,
            "gray_maps_replicated_to_three_equal_channels": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.out.exists():
        raise FileExistsError(f"Use a new task-owned output directory: {args.out}")
    tests = self_test()
    image_index = {}
    for line in (args.data / "images.txt").read_text().splitlines():
        key, value = line.split(maxsplit=1)
        image_index[int(key)] = value
    train_ids = {int(line.split()[0]) for line in (args.data / "train_test_split.txt").read_text().splitlines()
                 if line.split()[1] == "1"}
    assert len(train_ids) == 5994
    args.out.mkdir(parents=True)
    manifest = {"purpose": "Visual inspection only; training NOT started",
                "source": str(args.data), "source_split": "official_train_only",
                "bbox_or_segmentation_read": False, "geometric_resize": False,
                "native_pixel_processing_before_padding": True,
                "canvas": [500, 500], "original_padding_rgb": TRAIN_PADDING,
                "edge_padding_rgb": [0, 0, 0], "per_image_minmax_normalization": False,
                "gray_conversion": "PIL RGB.convert('L'); float32 / 255",
                "border": "OpenCV BORDER_REFLECT_101",
                "parameters": {"scharr": {"scale": 1 / 32, "gain": 4, "magnitude": "hypot(gx, gy)"},
                    "canny": {"preblur_sigma": 1, "thresholds": [50, 100], "L2gradient": True},
                    "dog": {"sigmas": [1, 2, 4], "response": "max(abs(G1-G2), abs(G2-G4))", "gain": 8},
                    "gabor": {"wavelengths": [4, 8], "angles_degrees": [0, 45, 90, 135],
                        "sigma_over_wavelength": 0.56, "gamma": 0.5, "phases": [0, math.pi / 2],
                        "kernel": "zero mean, unit L1; radius=ceil(3*sigma)",
                        "response": "max over scale/orientation of hypot(even, odd)", "gain": 8}},
                "quantization": "round(255*clip(fixed_gain*response,0,1)); three equal RGB channels",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "opencv": cv2.__version__, "pillow": PIL.__version__},
                "self_tests": tests, "script_sha256": sha(Path(__file__)), "samples": []}
    rendered = []
    for image_id, stage, relative_path, expected_sha in SAMPLES:
        assert image_id in train_ids and image_index[image_id] == relative_path
        source = args.data / "images" / relative_path
        assert sha(source) == expected_sha, f"Source changed: {relative_path}"
        with Image.open(source) as image:
            original_mode = image.mode
            image = image.convert("RGB")
            rgb = np.asarray(image)
            gray_u8 = np.asarray(image.convert("L"))
        maps, stats = extract(gray_u8)
        folder = args.out / f"{image_id:05d}_{stage}"
        folder.mkdir()
        images = {}
        images["original"], offset = centered(rgb, TRAIN_PADDING)
        images["original"].save(folder / "00_original.png")
        for number, name in enumerate(METHODS[1:], 1):
            expanded = np.repeat(maps[name][..., None], 3, axis=2)
            images[name], _ = centered(expanded, (0, 0, 0))
            images[name].save(folder / f"{number:02d}_{name}.png")
            assert np.array_equal(np.asarray(images[name])[..., 0], np.asarray(images[name])[..., 2])
        contact(images, f"CUB train image {image_id}: native {image.width} x {image.height}; no resizing", METHODS).save(folder / "comparison_all.png")
        contact(images, f"CUB train image {image_id}: original / continuous / binary edges", METHODS[:3]).save(folder / "comparison_edges.png")
        manifest["samples"].append({"image_id": image_id, "stage": stage,
            "relative_path": relative_path, "official_train": True, "source_sha256": expected_sha,
            "source_mode": original_mode, "size_wh": list(image.size), "padding_offset_xy": list(offset),
            "map_statistics_valid_pixels_only": stats,
            "output_sha256": {file.name: sha(file) for file in sorted(folder.glob("*.png"))}})
        rendered.append((image_id, images))
    side, row_height, margin, gap = 244, 292, 16, 12
    overview = Image.new("RGB", (margin * 2 + 5 * side + 4 * gap, 60 + len(rendered) * row_height), (245, 245, 245))
    draw = ImageDraw.Draw(overview)
    draw.text((margin, 10), "CUB training samples: original / Scharr / Canny / DoG / Gabor", font=font(19), fill=(20, 20, 20))
    draw.text((margin, 35), "Thumbnails only; individual PNGs keep native pixels on 500 x 500 canvas", font=font(16), fill=(40, 40, 40))
    for row, (image_id, images) in enumerate(rendered):
        y = 60 + row * row_height
        for col, name in enumerate(METHODS):
            x = margin + col * (side + gap)
            draw.text((x, y + 4), f"{image_id}  {name.upper()}", font=font(17), fill=(20, 20, 20))
            overview.paste(images[name].resize((side, side), Image.Resampling.LANCZOS), (x, y + 33))
    overview.save(args.out / "overview_8_samples.png")
    manifest["overview_sha256"] = sha(args.out / "overview_8_samples.png")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "README.txt").write_text(
        "EDGE/TEXTURE INPUT PREVIEW ONLY - NO TRAINING STARTED\n"
        "Eight official TRAIN images reused from the prior A-E preview. No class label, bbox or segmentation is used by the transforms.\n"
        "00_original: original RGB, native pixels, train-mean padding (124,128,111).\n"
        "01_scharr: continuous gradient magnitude; weak/strong image edges.\n"
        "02_canny: binary thin edge map with fixed thresholds; discards most intensity/texture information.\n"
        "03_dog: max absolute difference across Gaussian scales 1/2 and 2/4; multiscale high-frequency detail.\n"
        "04_gabor: maximum quadrature energy at wavelengths 4/8 and four orientations; directional local texture.\n"
        "All four are grayscale luminance-derived maps copied into three RGB channels, not grayscale photos.\n"
        "They extract background edges too, and discard color. No foreground/background distinction is claimed.\n"
        "Process native valid image BEFORE zero padding to 500x500, avoiding derivatives across artificial padding.\n"
        "No per-image min-max: fixed gains and thresholds recorded in manifest. Contact/overview layouts are for viewing only.\n"
        "Parameters are exploratory preview candidates, not a trained or approved protocol. Source and output SHA256 recorded.\n"
        "Full-resolution individual PNGs and comparison_all.png/comparison_edges.png exist in each sample folder.\n",
        encoding="utf-8")
    print(json.dumps({"status": "preview_verified", "samples": len(rendered),
                      "candidate_maps": len(rendered) * 4, "self_tests": tests,
                      "out": str(args.out), "training_started": False}))


if __name__ == "__main__":
    main()
