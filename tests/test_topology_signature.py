import unittest

import numpy as np

from src.graph import topology_distance_signature


class TopologySignatureTest(unittest.TestCase):
    def test_is_translation_scale_and_rotation_invariant(self):
        points = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.5, 1.0], [3.0, 2.0]])
        rotated_scaled_shifted = points @ np.asarray([[0.0, -3.0], [3.0, 0.0]]) + 7.0
        np.testing.assert_allclose(
            topology_distance_signature(points),
            topology_distance_signature(rotated_scaled_shifted),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_single_node_has_zero_signature(self):
        np.testing.assert_array_equal(
            topology_distance_signature(np.asarray([[2.0, 4.0]])),
            np.zeros((1, 5), dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
