from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.white_core import WhiteCoreLocalizer


class WhiteCoreLocalizerTest(unittest.TestCase):
    def assert_valid_covariance(self, covariance: np.ndarray) -> None:
        self.assertEqual(covariance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

    def test_prefers_neutral_compact_core_over_coloured_halo(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        core = (83, 58)
        cv2.circle(image, core, 24, (30, 210, 30), -1)
        cv2.circle(image, core, 8, (180, 255, 180), -1)
        cv2.circle(image, core, 3, (255, 255, 255), -1)

        result = WhiteCoreLocalizer().locate(image, np.array([52.0, 29.0, 114.0, 89.0]))

        self.assertLess(float(np.linalg.norm(result.xy - np.asarray(core))), 1.5)
        self.assertGreater(result.confidence, 0.25)
        self.assert_valid_covariance(result.covariance)

    def test_rejects_a_distant_bright_reflection(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        true_core = (83, 58)
        cv2.circle(image, true_core, 15, (120, 230, 120), -1)
        cv2.circle(image, true_core, 5, (230, 255, 230), -1)
        # This reflection is brighter, but too far from the box-centre seed.
        cv2.circle(image, (60, 35), 5, (255, 255, 255), -1)

        result = WhiteCoreLocalizer().locate(image, np.array([53.0, 28.0, 113.0, 88.0]))

        self.assertLess(float(np.linalg.norm(result.xy - np.asarray(true_core))), 2.5)
        self.assert_valid_covariance(result.covariance)

    def test_returns_a_box_center_for_tiny_boxes(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        result = WhiteCoreLocalizer().locate(image, np.array([5.0, 7.0, 7.0, 9.0]))
        np.testing.assert_allclose(result.xy, np.array([6.0, 8.0]))
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.valid)

    def test_recovers_an_offset_core_on_a_sloping_underwater_background(self) -> None:
        height, width = 120, 170
        y_grid, x_grid = np.indices((height, width), dtype=np.float32)
        background = np.zeros((height, width, 3), dtype=np.float32)
        background[..., 0] = 18.0 + 0.04 * x_grid
        background[..., 1] = 42.0 + 0.11 * x_grid + 0.05 * y_grid
        background[..., 2] = 15.0 + 0.03 * y_grid
        image = np.clip(background, 0, 255).astype(np.uint8)
        core = np.array([93.4, 59.7])
        distance_squared = (x_grid - core[0]) ** 2 + (y_grid - core[1]) ** 2
        halo = np.exp(-distance_squared / (2.0 * 11.0**2))
        compact = np.exp(-distance_squared / (2.0 * 2.3**2))
        for channel, halo_gain in enumerate((55.0, 145.0, 45.0)):
            image[..., channel] = np.clip(
                image[..., channel].astype(np.float32) + halo_gain * halo + 190.0 * compact,
                0,
                255,
            ).astype(np.uint8)

        box = np.array([65.0, 34.0, 111.0, 82.0])
        result = WhiteCoreLocalizer().locate(image, box)

        self.assertGreater(float(np.linalg.norm((box[:2] + box[2:]) / 2.0 - core)), 4.0)
        self.assertLess(float(np.linalg.norm(result.xy - core)), 1.5)
        self.assert_valid_covariance(result.covariance)

    def test_uses_saturation_aware_method_for_a_clipped_core(self) -> None:
        image = np.zeros((120, 170, 3), dtype=np.uint8)
        image[:] = (18, 45, 16)
        core = np.array([91.0, 61.0])
        cv2.circle(image, tuple(core.astype(int)), 18, (35, 170, 30), -1)
        cv2.circle(image, tuple(core.astype(int)), 9, (145, 245, 135), -1)
        cv2.circle(image, tuple(core.astype(int)), 5, (255, 255, 255), -1)
        cv2.ellipse(image, (98, 64), (9, 4), 15, 0, 360, (30, 130, 25), -1)

        result = WhiteCoreLocalizer().locate(image, np.array([61.0, 35.0, 111.0, 87.0]))

        self.assertLess(float(np.linalg.norm(result.xy - core)), 2.0)
        self.assertTrue(result.diagnostics["saturated"])
        self.assertIn(
            result.method.replace("_low_confidence", ""),
            {"moffat_wings_saturated", "radial_symmetry", "multi_isophote_ellipse", "weighted_centroid_tiny"},
        )
        self.assert_valid_covariance(result.covariance)

    def test_neighbor_voronoi_mask_rejects_a_brighter_adjacent_lamp(self) -> None:
        image = np.zeros((100, 150, 3), dtype=np.uint8)
        image[:] = (12, 35, 10)
        first_core = np.array([64.0, 51.0])
        second_core = np.array([83.0, 51.0])
        cv2.circle(image, tuple(first_core.astype(int)), 4, (230, 255, 230), -1)
        cv2.circle(image, tuple(second_core.astype(int)), 6, (255, 255, 255), -1)
        boxes = np.array([[53.0, 40.0, 73.0, 62.0], [73.0, 39.0, 94.0, 63.0]])

        result = WhiteCoreLocalizer(max_roi_scale=1.5).locate(image, boxes[0], neighbor_boxes=boxes)

        self.assertLess(float(np.linalg.norm(result.xy - first_core)), 1.5)
        self.assertGreater(float(np.linalg.norm(result.xy - second_core)), 10.0)

    def test_invalid_large_shift_falls_back_to_box_center(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:] = (8, 35, 8)
        cv2.circle(image, (88, 68), 5, (255, 255, 255), -1)
        box = np.array([54.0, 34.0, 86.0, 66.0])

        result = WhiteCoreLocalizer(maximum_shift_fraction=0.04).locate(image, box)

        seed = np.array([70.0, 50.0])
        np.testing.assert_allclose(result.xy, seed)
        self.assertFalse(result.valid)
        self.assertTrue(result.diagnostics["fallback_to_seed"])
        self.assertGreater(result.diagnostics["candidate_shift_px"], 15.0)


if __name__ == "__main__":
    unittest.main()
