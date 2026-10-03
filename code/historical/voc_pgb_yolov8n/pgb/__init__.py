"""Progressive background blur research package."""

from .core import (
    BlurMapNet,
    BlurTargets,
    GaussianBankRenderer,
    ProgressiveBlurFrontEnd,
    balanced_map_loss,
    make_targets,
    yolo_xywh_to_boxes,
)

# Ultralytics integration is intentionally not imported here. Keeping the core
# package import lightweight allows its tensor tests to run independently.

__all__ = [
    "BlurMapNet",
    "BlurTargets",
    "GaussianBankRenderer",
    "ProgressiveBlurFrontEnd",
    "balanced_map_loss",
    "make_targets",
    "yolo_xywh_to_boxes",
]
