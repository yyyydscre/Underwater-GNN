"""A compact pose-continuity filter for video sequences."""
from __future__ import annotations

import numpy as np

from .schema import PoseEstimate


class PoseContinuityFilter:
    """Confidence-aware filtering with short low-accuracy pose prediction."""

    def __init__(
        self,
        alpha: float = 0.35,
        max_reprojection_error: float = 8.0,
        max_prediction_frames: int = 5,
        confidence_decay: float = 0.65,
        velocity_alpha: float = 0.30,
    ) -> None:
        self.alpha = alpha
        self.max_reprojection_error = max_reprojection_error
        self.max_prediction_frames = int(max_prediction_frames)
        self.confidence_decay = float(confidence_decay)
        self.velocity_alpha = float(velocity_alpha)
        self.previous: PoseEstimate | None = None
        self.rotation_velocity = np.zeros((3, 1), dtype=np.float64)
        self.translation_velocity = np.zeros((3, 1), dtype=np.float64)
        self.missed_frames = 0

    def reset(self) -> None:
        self.previous = None
        self.rotation_velocity.fill(0.0)
        self.translation_velocity.fill(0.0)
        self.missed_frames = 0

    def predict_prior(self) -> PoseEstimate | None:
        if self.previous is None or not self.previous.success:
            return None
        return PoseEstimate(
            True,
            rvec=self.previous.rvec + self.rotation_velocity,
            tvec=self.previous.tvec + self.translation_velocity,
            reprojection_error_px=self.previous.reprojection_error_px,
            inlier_count=0,
            source="temporal_motion_prior",
            confidence=self.previous.confidence,
            approximate=True,
            measurement_count=0,
        )

    def update(self, pose: PoseEstimate) -> PoseEstimate:
        if not pose.success:
            if (
                self.previous is None
                or not self.previous.success
                or self.missed_frames >= self.max_prediction_frames
            ):
                return PoseEstimate(False, source="temporal_no_measurement")
            predicted = self.predict_prior()
            self.missed_frames += 1
            predicted.confidence = float(
                (predicted.confidence or 0.25) * self.confidence_decay
            )
            predicted.source = "temporal_approximate_prediction"
            predicted.approximate = True
            self.previous = predicted
            return predicted
        if pose.reprojection_error_px is not None and pose.reprojection_error_px > self.max_reprojection_error:
            return self.update(
                PoseEstimate(False, source="temporal_reprojection_gate")
            )
        if self.previous is None or not self.previous.success:
            self.previous = pose
            self.missed_frames = 0
            return pose
        previous_rvec = self.previous.rvec.copy()
        previous_tvec = self.previous.tvec.copy()
        confidence = float(pose.confidence or 0.25)
        alpha = float(np.clip(self.alpha + 0.25 * confidence, 0.25, 0.70))
        filtered = PoseEstimate(
            True,
            rvec=(1.0 - alpha) * self.previous.rvec + alpha * pose.rvec,
            tvec=(1.0 - alpha) * self.previous.tvec + alpha * pose.tvec,
            reprojection_error_px=pose.reprojection_error_px,
            inlier_count=pose.inlier_count,
            source=(
                "temporal_sparse_ema"
                if pose.approximate
                else "temporal_ema"
            ),
            confidence=pose.confidence,
            approximate=pose.approximate,
            measurement_count=pose.measurement_count,
            recovered_count=pose.recovered_count,
            weighted_refinement=pose.weighted_refinement,
        )
        measured_rotation_velocity = filtered.rvec - previous_rvec
        measured_translation_velocity = filtered.tvec - previous_tvec
        self.rotation_velocity = (
            (1.0 - self.velocity_alpha) * self.rotation_velocity
            + self.velocity_alpha * measured_rotation_velocity
        )
        self.translation_velocity = (
            (1.0 - self.velocity_alpha) * self.translation_velocity
            + self.velocity_alpha * measured_translation_velocity
        )
        self.previous = filtered
        self.missed_frames = 0
        return filtered
