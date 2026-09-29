"""Result overlays for qualitative validation."""
from __future__ import annotations

import cv2
import numpy as np

from .matcher import MatchResult
from .schema import LightDetection, PoseEstimate


def draw_overlay(
    image: np.ndarray,
    detections: list[LightDetection],
    match_result: MatchResult,
    pose: PoseEstimate,
    show_candidates: bool = False,
    show_candidate_labels: bool = False,
) -> np.ndarray:
    canvas = image.copy()
    if show_candidates:
        for index, detection in enumerate(detections):
            color = (70, 230, 70) if detection.color == "green" else (255, 150, 30) if detection.color == "blue" else (180, 180, 180)
            point = tuple(np.rint(detection.xy).astype(int))
            cv2.circle(canvas, point, 3 if not show_candidate_labels else max(4, int(detection.radius)), color, 1, cv2.LINE_AA)
            if show_candidate_labels:
                cv2.putText(canvas, f"d{index}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for match in match_result.matches:
        point = tuple(np.rint(match.point).astype(int))
        color = (0, 255, 0) if match.lamp.layer == "front" else (255, 120, 0)
        cv2.circle(canvas, point, 8, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, match.lamp.lamp_id, (point[0] + 7, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    status = f"det={len(detections)} match={len(match_result.matches)} [{match_result.method}]"
    if pose.success:
        confidence = 0.0 if pose.confidence is None else pose.confidence
        text = f"{status}  q={confidence:.2f}  z={pose.tvec[2, 0]:.3f}m  err={pose.reprojection_error_px:.2f}px"
    else:
        text = f"{status}  pose=unavailable"
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 900), 44), (0, 0, 0), -1)
    cv2.putText(canvas, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas
