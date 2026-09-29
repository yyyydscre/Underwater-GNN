from __future__ import annotations

import cv2
import numpy as np

from src.config import load_array_configuration
from src.matcher import MatchResult
from src.pose import PnPPoseEstimator, adapt_camera_matrix
from src.schema import LampMatch, LightDetection, PoseEstimate


def _detection(xy: np.ndarray, confidence: float = 0.92) -> LightDetection:
    return LightDetection(
        xy=np.asarray(xy, dtype=np.float64),
        confidence=confidence,
        color_probs=np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
        color="other",
        radius=7.0,
        brightness=255.0,
        source="synthetic",
        center_quality=confidence,
        detector_confidence=confidence,
        center_confidence=confidence,
        center_covariance=np.eye(2, dtype=np.float64) * 0.16,
        center_valid=True,
    )


def _scene():
    configuration = load_array_configuration("configs/lamp_array_pose_unity_approx.yaml")
    image_shape = (2048, 2448, 3)
    camera = adapt_camera_matrix(
        configuration.camera_matrix,
        configuration.camera_calibration_size,
        image_shape,
        configuration.camera_image_transform,
    )
    rvec = np.asarray([[0.035], [-0.025], [0.018]], dtype=np.float64)
    tvec = np.asarray([[0.2], [-0.1], [34.0]], dtype=np.float64)
    object_points = np.asarray([lamp.xyz for lamp in configuration.lights])
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera, configuration.distortion
    )
    return configuration, image_shape, rvec, tvec, projected.reshape(-1, 2)


def test_weighted_pnp_returns_metric_measurement():
    configuration, image_shape, _, _, points = _scene()
    detections = [_detection(point) for point in points]
    detections[2].xy = detections[2].xy + np.asarray([5.0, -4.0])
    detections[2].center_covariance = np.eye(2) * 16.0
    matches = [
        LampMatch(lamp, index, detection.xy, 0.92, 0.0)
        for index, (lamp, detection) in enumerate(zip(configuration.lights, detections))
    ]
    pose = PnPPoseEstimator(configuration).estimate(
        MatchResult(matches, "synthetic", 0.0),
        image_shape,
        detections=detections,
    )
    assert pose.success
    assert not pose.approximate
    assert pose.weighted_refinement
    assert pose.reprojection_error_px < 1.5


def test_template_projection_recovers_image_evidence():
    configuration, image_shape, rvec, tvec, points = _scene()
    image = np.zeros(image_shape, dtype=np.uint8)
    for point in points:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 5, (255, 255, 255), -1)
        cv2.circle(image, tuple(np.rint(point).astype(int)), 10, (100, 180, 100), 2)

    visible = list(range(8))
    detections = [_detection(points[index]) for index in visible]
    matches = [
        LampMatch(configuration.lights[index], local_index, detections[local_index].xy, 0.92, 0.0)
        for local_index, index in enumerate(visible)
    ]
    result = MatchResult(
        matches,
        "synthetic_missing",
        0.0,
        [configuration.lights[index].lamp_id for index in range(8, 13)],
    )
    coarse_pose = PoseEstimate(
        True,
        rvec=rvec,
        tvec=tvec,
        reprojection_error_px=0.0,
        inlier_count=8,
        source="synthetic_pose",
        confidence=0.95,
        approximate=False,
        measurement_count=8,
    )
    estimator = PnPPoseEstimator(configuration)
    refined = estimator.refine_matches(
        result,
        detections,
        coarse_pose,
        image_shape,
        image=image,
    )
    assert refined is not result
    assert refined.recovered_detections
    assert any(
        match.observation_status == "recovered_projection"
        for match in refined.matches
    )
    all_detections = detections + list(refined.recovered_detections)
    pose = estimator.estimate(
        refined,
        image_shape,
        detections=all_detections,
    )
    assert pose.success
    assert pose.recovered_count > 0
    assert pose.reprojection_error_px < 2.0


def test_multi_hypothesis_pose_reranking_repairs_ambiguous_id_swap():
    configuration, image_shape, _, _, points = _scene()
    detections = [_detection(point) for point in points]
    swapped = {2: 3, 3: 2}
    matches = [
        LampMatch(
            configuration.lights[swapped.get(index, index)],
            index,
            detection.xy,
            0.55 if index in swapped else 0.95,
            0.0,
        )
        for index, detection in enumerate(detections)
    ]
    count = len(configuration.lights)
    pair_scores = np.full((count, count), -4.0, dtype=np.float64)
    for index in range(count):
        pair_scores[index, index] = 0.0
    pair_scores[2, 3] = 0.15
    pair_scores[3, 2] = 0.15
    result = MatchResult(
        matches,
        "synthetic_gnn",
        0.0,
        candidate_log_scores=pair_scores,
        observation_dustbin_scores=np.full(count, -2.0),
        template_dustbin_scores=np.full(count, -2.0),
        candidate_template_ids=list(configuration.ids),
    )
    estimator = PnPPoseEstimator(configuration)
    base_pose = estimator.estimate(
        result,
        image_shape,
        detections=detections,
    )
    reranked, reranked_pose = estimator.rerank_matches_multi_hypothesis(
        result,
        detections,
        base_pose,
        image_shape,
        top_k=2,
        beam_width=32,
        max_changes=2,
        policy="conservative_rerank",
    )
    assigned = {match.detection_index: match.lamp.lamp_id for match in reranked.matches}
    assert reranked is not result
    assert reranked_pose.success
    assert assigned[2] == configuration.lights[2].lamp_id
    assert assigned[3] == configuration.lights[3].lamp_id
    assert reranked.rerank_diagnostics["accepted"]
