"""Compare standalone val JSON against predict output on same frozen image."""

import json
from pathlib import Path

from ultralytics import YOLO

base = Path("/home/ubuntu/rz/progressive_blur_detection_research")
records = json.loads((base / "diagnostics/foveated_region_coverage_20260924/last_val_rect_json_crosscheck/predictions.json").read_text())
manifest = (base / "data_splits/visdrone_official/val.txt").read_text().splitlines()
image = Path(manifest[0])
selected = [r for r in records if r["image_id"] == image.stem]
model = YOLO(str(base / "runs/visdrone_official/yolov8n_scratch_official_960_s2027_e125/weights/last.pt"))
result = model.predict(str(image), imgsz=960, conf=0.001, iou=0.7,
                       max_det=1000, half=False, device="0", verbose=False)[0]
print("VAL_JSON_TOTAL", len(records), flush=True)
print("IMAGE", image.name, flush=True)
print("VAL_IMAGE_N", len(selected), "PRED_IMAGE_N", len(result.boxes), flush=True)
print("VAL_TOP", selected[:5], flush=True)
print("PRED_TOP", [(int(b.cls[0]), float(b.conf[0]), b.xyxy[0].tolist())
                   for b in result.boxes[:5]], flush=True)
