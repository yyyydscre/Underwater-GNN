"""Small helpers for one-keypoint Ultralytics dataset inspection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .io_utils import list_images


@dataclass(frozen=True)
class PoseLabel:
    box_xyxy: np.ndarray
    keypoint_xy: np.ndarray
    visibility: float


def load_dataset_paths(data_path: str | Path, split: str) -> tuple[Path, Path]:
    data_path = Path(data_path)
    with data_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = Path(str(settings["path"]))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    images = root / str(settings[split])
    labels = root / str(settings[split]).replace("images/", "labels/", 1)
    return images, labels


def read_pose_labels(label_path: Path, image_shape: tuple[int, int]) -> list[PoseLabel]:
    """Read normalized YOLO-Pose labels with exactly one keypoint."""
    if not label_path.exists():
        return []
    height, width = image_shape[:2]
    labels: list[PoseLabel] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        values = raw_line.split()
        if len(values) < 8:
            continue
        numeric = np.asarray([float(value) for value in values[1:]], dtype=np.float64)
        x, y, box_width, box_height = numeric[:4]
        kx, ky, visibility = numeric[4:7]
        box = np.array(
            [(x - box_width / 2) * width, (y - box_height / 2) * height, (x + box_width / 2) * width, (y + box_height / 2) * height],
            dtype=np.float64,
        )
        labels.append(PoseLabel(box, np.array([kx * width, ky * height], dtype=np.float64), float(visibility)))
    return labels


def paired_images(data_path: str | Path, split: str, limit: int = 0) -> list[tuple[Path, Path]]:
    images_dir, labels_dir = load_dataset_paths(data_path, split)
    images = list_images(images_dir)
    if limit > 0:
        images = images[:limit]
    return [(image_path, labels_dir / f"{image_path.stem}.txt") for image_path in images]
