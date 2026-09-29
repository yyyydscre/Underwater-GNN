"""End-to-end frame processor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ArrayConfiguration
from .matcher import LampArrayMatcher
from .pose import PnPPoseEstimator
from .schema import LightDetection
from .temporal import PoseContinuityFilter
from .visualize import draw_overlay


@dataclass
class FrameResult:
    detections: list[LightDetection]
    match_result: Any
    raw_pose: Any
    pose: Any
    overlay: np.ndarray

    def to_dict(self, frame_index: int, frame_name: str) -> dict[str, Any]:
        payload = {
            "frame_index": frame_index,
            "frame_name": frame_name,
            "detections": [item.to_dict() for item in self.detections],
            "matches": [item.to_dict() for item in self.match_result.matches],
            "matcher": self.match_result.method,
            "matching_cost": self.match_result.mean_cost,
            "raw_pose": self.raw_pose.to_dict(),
            "pose": self.pose.to_dict(),
        }
        if self.match_result.missing_lamp_ids is not None:
            payload["missing_ids"] = list(
                self.match_result.missing_lamp_ids
            )
        if self.match_result.rerank_diagnostics is not None:
            payload["matcher_diagnostics"] = dict(
                self.match_result.rerank_diagnostics
            )
        return payload


class GuidingLightPipeline:
    def __init__(
        self,
        configuration: ArrayConfiguration,
        detector,
        matcher: LampArrayMatcher,
        pose_configuration: ArrayConfiguration | None = None,
        enable_pose_refinement: bool = True,
        allow_sparse_pose: bool = True,
        max_pose_prediction_frames: int = 5,
        use_weighted_pnp: bool = True,
        enable_template_recovery: bool = True,
        enable_pose_consistency_pruning: bool = False,
        enable_pose_multihypothesis: bool = False,
        enable_residual_pnp_refinement: bool = False,
        residual_pnp_min_matches: int = 10,
        residual_pnp_max_matches: int = 12,
        residual_pnp_low_confidence: float = 0.58,
        residual_pnp_sigma: float = 2.75,
        residual_pnp_max_rejections: int = 2,
        pose_hypothesis_topk: int = 3,
        pose_hypothesis_beam_width: int = 48,
        pose_hypothesis_max_changes: int = 3,
        pose_hypothesis_policy: str = "rescue_only",
    ) -> None:
        self.detector = detector
        self.matcher = matcher
        self.pose_estimator = PnPPoseEstimator(
            pose_configuration or configuration,
            allow_sparse_pose=allow_sparse_pose,
            minimum_ransac_matches=6 if allow_sparse_pose else 8,
            min_inlier_ratio=0.55 if allow_sparse_pose else 0.65,
            use_weighted_refinement=use_weighted_pnp,
            enable_template_recovery=enable_template_recovery,
        )
        self.temporal_filter = PoseContinuityFilter(
            max_prediction_frames=max_pose_prediction_frames,
        )
        self.enable_pose_refinement = bool(enable_pose_refinement)
        self.enable_pose_consistency_pruning = bool(enable_pose_consistency_pruning)
        self.enable_pose_multihypothesis = bool(enable_pose_multihypothesis)
        self.enable_residual_pnp_refinement = bool(enable_residual_pnp_refinement)
        self.residual_pnp_min_matches = int(residual_pnp_min_matches)
        self.residual_pnp_max_matches = int(residual_pnp_max_matches)
        self.residual_pnp_low_confidence = float(residual_pnp_low_confidence)
        self.residual_pnp_sigma = float(residual_pnp_sigma)
        self.residual_pnp_max_rejections = int(residual_pnp_max_rejections)
        self.pose_hypothesis_topk = int(pose_hypothesis_topk)
        self.pose_hypothesis_beam_width = int(pose_hypothesis_beam_width)
        self.pose_hypothesis_max_changes = int(pose_hypothesis_max_changes)
        self.pose_hypothesis_policy = str(pose_hypothesis_policy)

    def reset_temporal_state(self) -> None:
        self.matcher.reset()
        self.temporal_filter.reset()

    def process(self, image: np.ndarray) -> FrameResult:
        detections = self.detector.infer(image)
        match_result = self.matcher.match(detections)
        prior_pose = self.temporal_filter.predict_prior()
        raw_pose = self.pose_estimator.estimate(
            match_result,
            image.shape,
            prior_pose=prior_pose,
            detections=detections,
        )
        if self.enable_pose_multihypothesis:
            match_result, raw_pose = self.pose_estimator.rerank_matches_multi_hypothesis(
                match_result,
                detections,
                raw_pose,
                image.shape,
                prior_pose=prior_pose,
                top_k=self.pose_hypothesis_topk,
                beam_width=self.pose_hypothesis_beam_width,
                max_changes=self.pose_hypothesis_max_changes,
                policy=self.pose_hypothesis_policy,
            )
        pose_pruned_matches = (
            self.pose_estimator.reject_pose_inconsistent_matches(
                match_result,
                raw_pose,
                image.shape,
            )
            if self.enable_pose_consistency_pruning
            else match_result
        )
        if self.enable_pose_consistency_pruning and pose_pruned_matches is not match_result:
            pruned_pose = self.pose_estimator.estimate(
                pose_pruned_matches,
                image.shape,
                prior_pose=prior_pose,
                detections=detections,
            )
            original_error = float(
                raw_pose.reprojection_error_px
                if raw_pose.reprojection_error_px is not None
                else np.inf
            )
            pruned_error = float(
                pruned_pose.reprojection_error_px
                if pruned_pose.reprojection_error_px is not None
                else np.inf
            )
            improves_geometry = pruned_error + 1.0 < original_error
            improves_confidence = (
                pruned_error + 0.35 < original_error
                and float(pruned_pose.confidence or 0.0) > float(raw_pose.confidence or 0.0) + 0.08
            )
            upgrades_measurement = raw_pose.approximate and not pruned_pose.approximate
            keeps_support = len(pose_pruned_matches.matches) >= max(8, int(np.ceil(0.80 * len(match_result.matches))))
            if pruned_pose.success and keeps_support and (
                improves_geometry or improves_confidence or upgrades_measurement
            ):
                match_result = pose_pruned_matches
                raw_pose = pruned_pose
        if self.enable_residual_pnp_refinement:
            match_result, raw_pose = self.pose_estimator.refine_pose_by_residual_gating(
                match_result,
                raw_pose,
                image.shape,
                prior_pose=prior_pose,
                detections=detections,
                target_min_matches=self.residual_pnp_min_matches,
                target_max_matches=self.residual_pnp_max_matches,
                low_confidence_gate=self.residual_pnp_low_confidence,
                residual_sigma=self.residual_pnp_sigma,
                max_rejections=self.residual_pnp_max_rejections,
            )
        refined_matches = (
            self.pose_estimator.refine_matches(
                match_result,
                detections,
                raw_pose,
                image.shape,
                image=image,
            )
            if self.enable_pose_refinement
            else match_result
        )
        if refined_matches is not match_result:
            candidate_detections = detections + list(
                refined_matches.recovered_detections or []
            )
            refined_pose = self.pose_estimator.estimate(
                refined_matches,
                image.shape,
                prior_pose=prior_pose,
                detections=candidate_detections,
            )
            original_error = float(
                raw_pose.reprojection_error_px
                if raw_pose.reprojection_error_px is not None
                else np.inf
            )
            refined_error = float(
                refined_pose.reprojection_error_px
                if refined_pose.reprojection_error_px is not None
                else np.inf
            )
            improves_support = (
                len(refined_matches.matches) > len(match_result.matches)
                and refined_error <= min(
                    self.pose_estimator.reprojection_threshold,
                    original_error + 0.25,
                )
            )
            improves_geometry = refined_error + 0.25 < original_error
            upgrades_measurement = raw_pose.approximate and not refined_pose.approximate
            added_measurements = len(refined_matches.matches) - len(match_result.matches)
            recovered_inliers = refined_pose.recovered_count
            recovery_supported = (
                added_measurements > 0
                and recovered_inliers >= max(1, int(np.ceil(0.67 * added_measurements)))
            )
            if refined_pose.success and (
                recovery_supported
                and (improves_support or improves_geometry or upgrades_measurement)
            ):
                if refined_matches.recovered_detections:
                    detections.extend(refined_matches.recovered_detections)
                match_result = refined_matches
                raw_pose = refined_pose
        pose = self.temporal_filter.update(raw_pose)
        overlay = draw_overlay(image, detections, match_result, pose)
        return FrameResult(detections, match_result, raw_pose, pose, overlay)
