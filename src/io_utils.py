"""Unicode-safe image I/O for Windows dataset paths."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path: str | Path) -> np.ndarray:
    """Read an image even when the Windows path contains Chinese characters."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def make_video_writer(path: str | Path, frame_shape: Sequence[int], fps: float = 15.0) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer

