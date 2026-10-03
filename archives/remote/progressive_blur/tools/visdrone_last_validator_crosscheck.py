"""One-off frozen-checkpoint validation cross-check for the crop pilot."""

from ultralytics import YOLO

model = YOLO("/home/ubuntu/rz/progressive_blur_detection_research/runs/visdrone_official/yolov8n_scratch_official_960_s2027_e125/weights/last.pt")
metrics = model.val(
    data="/home/ubuntu/rz/progressive_blur_detection_research/data_splits/visdrone_official/dataset.yaml",
    split="val", imgsz=960, batch=16, workers=4, conf=0.001, iou=0.7,
    max_det=1000, half=False, rect=True, device="0", plots=False,
    save=False, save_json=True, verbose=False,
    project="/home/ubuntu/rz/progressive_blur_detection_research/diagnostics/foveated_region_coverage_20260924",
    name="last_val_rect_json_crosscheck", exist_ok=True,
)
print("CROSSCHECK_MAP50", metrics.box.map50, flush=True)
print("CROSSCHECK_MAP50_95", metrics.box.map, flush=True)
print("CROSSCHECK_CLASS_AP", list(zip(model.names.values(), metrics.box.maps.tolist())), flush=True)
