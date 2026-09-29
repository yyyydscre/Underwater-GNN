"""Configuration loaders for the physical lamp-array template."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .schema import LampTemplate


@dataclass
class ArrayConfiguration:
    lights: list[LampTemplate]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    k_neighbors: int
    min_match_confidence: float
    expected_colors: dict[str, str]
    camera_calibration_size: tuple[int, int] | None = None
    camera_image_transform: str = "none"

    @property
    def ids(self) -> list[str]:
        return [lamp.lamp_id for lamp in self.lights]

    def by_id(self, lamp_id: str) -> LampTemplate:
        return next(lamp for lamp in self.lights if lamp.lamp_id == lamp_id)

    def by_layer(self, layer: str) -> list[LampTemplate]:
        return [lamp for lamp in self.lights if lamp.layer == layer]


def load_array_configuration(path: str | Path) -> ArrayConfiguration:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    camera = data["camera"]
    calibration_size = camera.get("calibration_image_size")
    if calibration_size is not None:
        if len(calibration_size) != 2:
            raise ValueError("camera.calibration_image_size must be [width, height].")
        calibration_size = (int(calibration_size[0]), int(calibration_size[1]))
        if min(calibration_size) <= 0:
            raise ValueError("Camera calibration dimensions must be positive.")
    image_transform = str(camera.get("image_transform", "none"))
    if image_transform not in {"none", "resize", "center_crop_resize"}:
        raise ValueError(
            "camera.image_transform must be none, resize, or center_crop_resize."
        )
    matching = data.get("matching", {})
    lights = [
        LampTemplate(
            lamp_id=str(item["id"]),
            layer=str(item["layer"]),
            color=str(item.get("color", "other")),
            xyz=np.asarray(item["xyz_m"], dtype=np.float64),
        )
        for item in data["lights"]
    ]
    if len({light.lamp_id for light in lights}) != len(lights):
        raise ValueError("Each lamp ID must be unique.")
    return ArrayConfiguration(
        lights=lights,
        camera_matrix=np.asarray(camera["camera_matrix"], dtype=np.float64),
        distortion=np.asarray(camera.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64),
        k_neighbors=int(matching.get("k_neighbors", 4)),
        min_match_confidence=float(matching.get("min_match_confidence", 0.2)),
        expected_colors=dict(matching.get("expected_colors", {})),
        camera_calibration_size=calibration_size,
        camera_image_transform=image_transform,
    )
