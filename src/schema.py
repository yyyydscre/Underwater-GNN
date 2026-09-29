"""Shared, serializable data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COLOR_INDEX = {"green": 0, "blue": 1, "other": 2}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class LightDetection:
    xy: np.ndarray
    confidence: float
    color_probs: np.ndarray
    color: str
    radius: float
    brightness: float
    source: str
    center_quality: float = 1.0
    bbox_xyxy: np.ndarray | None = None
    detector_confidence: float | None = None
    center_confidence: float | None = None
    center_method: str = "unknown"
    center_covariance: np.ndarray | None = None
    center_valid: bool | None = None
    center_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "xy": [round(float(value), 3) for value in self.xy],
            "confidence": round(float(self.confidence), 5),
            "color": self.color,
            "color_probs": [round(float(value), 5) for value in self.color_probs],
            "radius": round(float(self.radius), 3),
            "brightness": round(float(self.brightness), 3),
            "source": self.source,
            "center_quality": round(float(self.center_quality), 5),
            "detector_confidence": round(
                float(self.detector_confidence if self.detector_confidence is not None else self.confidence), 5
            ),
            "center_confidence": round(
                float(self.center_confidence if self.center_confidence is not None else self.center_quality), 5
            ),
            "center_method": self.center_method,
        }
        if self.bbox_xyxy is not None:
            payload["bbox_xyxy"] = [round(float(value), 3) for value in self.bbox_xyxy]
        if self.center_covariance is not None:
            payload["center_covariance"] = [
                [round(float(value), 6) for value in row]
                for row in np.asarray(self.center_covariance, dtype=np.float64).reshape(2, 2)
            ]
        if self.center_valid is not None:
            payload["center_valid"] = bool(self.center_valid)
        if self.center_diagnostics is not None:
            payload["center_diagnostics"] = {
                str(key): _json_scalar(value) for key, value in self.center_diagnostics.items()
            }
        return payload


@dataclass
class LampTemplate:
    lamp_id: str
    layer: str
    color: str
    xyz: np.ndarray


@dataclass
class LampMatch:
    lamp: LampTemplate
    detection_index: int
    point: np.ndarray
    confidence: float
    geometric_cost: float
    observation_status: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lamp_id": self.lamp.lamp_id,
            "layer": self.lamp.layer,
            "detection_index": int(self.detection_index),
            "xy": [round(float(value), 3) for value in self.point],
            "confidence": round(float(self.confidence), 5),
            "geometric_cost": round(float(self.geometric_cost), 5),
            "observation_status": self.observation_status,
        }


@dataclass
class PoseEstimate:
    success: bool
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reprojection_error_px: float | None = None
    inlier_count: int = 0
    source: str = "pnp"
    confidence: float | None = None
    approximate: bool = False
    measurement_count: int | None = None
    recovered_count: int = 0
    weighted_refinement: bool = False

    def to_dict(self) -> dict[str, Any]:
        if not self.success:
            return {"success": False, "source": self.source}
        return {
            "success": True,
            "source": self.source,
            "rvec_rad": [round(float(value), 7) for value in self.rvec.reshape(-1)],
            "tvec_m": [round(float(value), 7) for value in self.tvec.reshape(-1)],
            "reprojection_error_px": round(float(self.reprojection_error_px), 5),
            "inlier_count": int(self.inlier_count),
            "confidence": round(float(self.confidence), 5) if self.confidence is not None else None,
            "approximate": bool(self.approximate),
            "measurement_count": int(
                self.measurement_count
                if self.measurement_count is not None
                else self.inlier_count
            ),
            "recovered_count": int(self.recovered_count),
            "weighted_refinement": bool(self.weighted_refinement),
        }
