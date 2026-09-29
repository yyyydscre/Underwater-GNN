"""Deterministic end-to-end check independent of dataset paths and YOLO weights."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import load_array_configuration
from src.matcher import LampArrayMatcher
from src.pose import PnPPoseEstimator
from src.schema import COLOR_INDEX, LightDetection
from src.temporal import PoseContinuityFilter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = load_array_configuration(root / "configs" / "lamp_array_3d.yaml")
    object_points = np.asarray([lamp.xyz for lamp in configuration.lights], dtype=np.float64)
    rvec_true = np.array([[0.06], [-0.10], [0.14]], dtype=np.float64)
    tvec_true = np.array([[0.025], [-0.018], [2.35]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec_true, tvec_true, configuration.camera_matrix, configuration.distortion)
    rng = np.random.default_rng(7)
    projected = projected.reshape(-1, 2) + rng.normal(0.0, 0.35, (len(object_points), 2))
    order = rng.permutation(len(configuration.lights))
    detections = []
    for template_index in order:
        lamp = configuration.lights[template_index]
        color_probs = np.full(3, 0.015, dtype=np.float32)
        color_probs[COLOR_INDEX[lamp.color]] = 0.97
        detections.append(LightDetection(projected[template_index], 0.98, color_probs, lamp.color, 7.0, 230.0, "test"))
    matcher = LampArrayMatcher(configuration, mode="geometry")
    matches = matcher.match(detections)
    pose = PnPPoseEstimator(configuration, reprojection_threshold=4.0).estimate(matches)
    filtered = PoseContinuityFilter().update(pose)
    assert len(matches.matches) >= 10, f"too few matches: {len(matches.matches)}"
    assert pose.success, "PnP must succeed on a complete synthetic pattern"
    assert pose.reprojection_error_px < 3.0, pose.reprojection_error_px
    assert filtered.success
    print({"matches": len(matches.matches), "reprojection_error_px": round(pose.reprojection_error_px, 3), "translation_m": np.round(pose.tvec.reshape(-1), 4).tolist()})


if __name__ == "__main__":
    main()

