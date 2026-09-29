"""Regression tests for scale-aware YOLO detection deduplication."""
from __future__ import annotations

import unittest

import numpy as np

from src.detector import _deduplicate_yolo
from src.schema import LightDetection


def _detection(
    x: float,
    y: float,
    radius: float,
    score: float,
    box_center: tuple[float, float] | None = None,
) -> LightDetection:
    bbox = None
    if box_center is not None:
        bbox = np.array(
            [
                box_center[0] - radius,
                box_center[1] - radius,
                box_center[0] + radius,
                box_center[1] + radius,
            ],
            dtype=np.float64,
        )
    return LightDetection(
        xy=np.array([x, y], dtype=np.float64),
        confidence=score,
        color_probs=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        color="green",
        radius=radius,
        brightness=255.0,
        source="yolo_box",
        detector_confidence=score,
        bbox_xyxy=bbox,
    )


class TestYOLODeduplication(unittest.TestCase):
    def test_suppresses_concentric_halo_and_core_boxes(self) -> None:
        detections = [_detection(100.0, 100.0, 12.0, 0.8), _detection(100.2, 100.1, 5.0, 0.6)]
        result = _deduplicate_yolo(detections)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].detector_confidence, 0.8)

    def test_preserves_nearby_tiny_lamps(self) -> None:
        detections = [_detection(100.0, 100.0, 3.0, 0.8), _detection(102.0, 100.0, 3.0, 0.7)]
        result = _deduplicate_yolo(detections)
        self.assertEqual(len(result), 2)

    def test_uses_detector_box_centers_not_refined_white_cores(self) -> None:
        detections = [
            _detection(97.0, 99.0, 12.0, 0.8, box_center=(100.0, 100.0)),
            _detection(103.0, 101.0, 5.0, 0.6, box_center=(100.2, 100.1)),
        ]
        result = _deduplicate_yolo(detections)
        self.assertEqual(len(result), 1)

    def test_suppresses_partially_overlapping_saturated_boxes(self) -> None:
        detections = [
            _detection(100.0, 100.0, 28.0, 0.8, box_center=(100.0, 100.0)),
            _detection(117.0, 104.0, 60.0, 0.5, box_center=(117.0, 104.0)),
        ]
        result = _deduplicate_yolo(detections)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].detector_confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
