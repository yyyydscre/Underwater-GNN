"""Project-local Ultralytics trainers for constrained backbone ablations."""
from __future__ import annotations

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


class SquareValidationDetectionTrainer(DetectionTrainer):
    """Disable rectangular validation batches for fixed-resolution backbones."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        grid_size = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
            stride=grid_size,
        )
