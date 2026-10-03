# VOC PGB + YOLOv8n research implementation

This directory is the isolated implementation workspace for the new study:
object-protective, distance-aware progressive background blur in front of a
YOLOv8n detector. The original files on the Desktop are treated as read-only
source material.

Current implemented scope:

- a 38,513-parameter image-only blur-map predictor;
- continuous box-distance supervision for training;
- differentiable Gaussian-bank rendering at sigma 0/1/2/4/8;
- protected/outer-region balanced auxiliary loss;
- conversion from an augmented Ultralytics batch to pixel xyxy boxes;
- CPU and CUDA-oriented unit tests for tensor behavior and gradients.

Not yet claimed:

- no VOC accuracy result;
- no claim that blur improves detection;
- no end-to-end YOLO checkpoint, EMA, validation, resume, export or DDP result;
- no novelty claim beyond a hypothesis to be tested against close prior work.

Run core tests with the local conda environment:

```powershell
conda run -n pytorch python -m unittest discover -s tests -v
```

Research decisions and every verified milestone are recorded under
`outputs/voc_pgb_yolov8n/`.

The synchronized remote research root is
`/home/ubuntu/rz/progressive_blur_detection_research`.
