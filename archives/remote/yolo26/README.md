# YOLO26 + PASCAL VOC research project

This directory keeps experiment work separate from the Ultralytics source tree.

```text
yolo26_voc_research/
  scripts/  # Training and evaluation Python scripts.
  configs/  # Copies of model YAML files used for baseline and modified models.
  runs/     # Checkpoints, plots, and CSV results from each run.
  notes/    # Experiment records and paper notes.
```

Related read-only locations:

- YOLO source: `F:/2026_9_10/ultralytics_research`
- VOC data: `E:/model/datasets/VOC`

The current smoke-test script creates a YOLO26n model from YAML with
`pretrained=False`; it does not use pretrained weights for training. On this
RTX 4060 it uses BF16 AMP (`amp="bf16"`), which was verified with a complete
small VOC training-and-validation run.
