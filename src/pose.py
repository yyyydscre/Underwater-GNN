"""Robust metric PnP pose estimation from labelled light centers."""
from __future__ import annotations

import cv2
import numpy as np

from .assignment import linear_sum_assignment
from .config import ArrayConfiguration
from .matcher import MatchResult
from .schema import LampMatch, LightDetection, PoseEstimate
from .white_core import WhiteCoreLocalizer


def adapt_camera_matrix(
    camera_matrix: np.ndarray,
    calibration_size: tuple[int, int] | None,
    image_shape: tuple[int, ...] | None,
    image_transform: str,
) -> np.ndarray:
    """Map calibration intrinsics to resized or centre-cropped input pixels."""
    matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
    if calibration_size is None or image_shape is None or image_transform == "none":
        return matrix
    target_height, target_width = int(image_shape[0]), int(image_shape[1])
    reference_width, reference_height = calibration_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if image_transform == "resize":
        scale_x = target_width / reference_width
        scale_y = target_height / reference_height
        matrix[0, 0] *= scale_x
        matrix[0, 2] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[1, 2] *= scale_y
        return matrix
    if image_transform != "center_crop_resize":
        raise ValueError(f"Unsupported camera image transform: {image_transform!r}")

    target_aspect = target_width / target_height
    reference_aspect = reference_width / reference_height
    offset_x = 0.0
    offset_y = 0.0
    if target_aspect >= reference_aspect:
        cropped_width = float(reference_width)
        cropped_height = cropped_width / target_aspect
        offset_y = (reference_height - cropped_height) / 2.0
    else:
        cropped_height = float(reference_height)
        cropped_width = cropped_height * target_aspect
        offset_x = (reference_width - cropped_width) / 2.0
    scale_x = target_width / cropped_width
    scale_y = target_height / cropped_height
    matrix[0, 0] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[0, 2] = (matrix[0, 2] - offset_x) * scale_x
    matrix[1, 2] = (matrix[1, 2] - offset_y) * scale_y
    return matrix


class PnPPoseEstimator:
    def __init__(
        self,
        configuration: ArrayConfiguration,
        reprojection_threshold: float = 8.0,
        min_inlier_ratio: float = 0.55,
        minimum_ransac_matches: int = 6,
        allow_sparse_pose: bool = True,
        use_weighted_refinement: bool = True,
        enable_template_recovery: bool = True,
    ) -> None:
        self.configuration = configuration
        self.reprojection_threshold = reprojection_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.minimum_ransac_matches = int(minimum_ransac_matches)
        self.allow_sparse_pose = bool(allow_sparse_pose)
        self.use_weighted_refinement = bool(use_weighted_refinement)
        self.enable_template_recovery = bool(enable_template_recovery)
        self.white_core_localizer = WhiteCoreLocalizer(
            roi_scale=1.20,
            max_roi_scale=1.35,
            maximum_shift_fraction=0.45,
        )

    @staticmethod
    def _match_weights(
        matches: list[LampMatch],
        detections: list[LightDetection] | None,
    ) -> np.ndarray:
        """Combine GNN confidence and white-core uncertainty into PnP weights."""
        weights = []
        for match in matches:
            quality = float(np.clip(match.confidence, 0.05, 1.0))
            if detections is not None and 0 <= match.detection_index < len(detections):
                detection = detections[match.detection_index]
                if detection.center_covariance is not None:
                    covariance = np.asarray(detection.center_covariance, dtype=np.float64).reshape(2, 2)
                    sigma = float(np.sqrt(max(np.trace(covariance) * 0.5, 0.04)))
                    quality *= float(np.clip(1.5 / sigma, 0.20, 1.0))
                if detection.center_valid is False:
                    quality *= 0.45
            if match.observation_status.startswith("recovered"):
                quality *= 0.75
            weights.append(float(np.clip(quality, 0.03, 1.0)))
        values = np.asarray(weights, dtype=np.float64)
        return values / max(float(np.max(values)), 1e-9)

    def _weighted_refine_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Robust IRLS refinement over the six pose parameters."""
        parameters = np.concatenate((rvec.reshape(3), tvec.reshape(3))).astype(np.float64)
        damping = 1e-3
        for _ in range(15):
            projected, jacobian = cv2.projectPoints(
                object_points,
                parameters[:3].reshape(3, 1),
                parameters[3:].reshape(3, 1),
                camera_matrix,
                self.configuration.distortion,
            )
            residual = (image_points - projected.reshape(-1, 2)).reshape(-1)
            point_error = np.linalg.norm(residual.reshape(-1, 2), axis=1)
            huber_scale = max(1.5, 0.45 * self.reprojection_threshold)
            robust = np.where(
                point_error <= huber_scale,
                1.0,
                huber_scale / np.maximum(point_error, 1e-9),
            )
            row_weights = np.repeat(np.sqrt(np.clip(weights * robust, 1e-6, 1.0)), 2)
            design = np.asarray(jacobian[:, :6], dtype=np.float64) * row_weights[:, None]
            target = residual * row_weights
            normal = design.T @ design + damping * np.eye(6, dtype=np.float64)
            try:
                delta = np.linalg.solve(normal, design.T @ target)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            candidate = parameters + delta
            candidate_projection, _ = cv2.projectPoints(
                object_points,
                candidate[:3].reshape(3, 1),
                candidate[3:].reshape(3, 1),
                camera_matrix,
                self.configuration.distortion,
            )
            old_cost = float(np.sum(weights * np.minimum(point_error, huber_scale) ** 2))
            new_error = np.linalg.norm(candidate_projection.reshape(-1, 2) - image_points, axis=1)
            new_cost = float(np.sum(weights * np.minimum(new_error, huber_scale) ** 2))
            if new_cost <= old_cost:
                parameters = candidate
                damping = max(damping * 0.35, 1e-7)
                if np.linalg.norm(delta) < 1e-7:
                    break
            else:
                damping = min(damping * 10.0, 1e6)
        return parameters[:3].reshape(3, 1), parameters[3:].reshape(3, 1)

    @staticmethod
    def _local_brightness(image: np.ndarray, xy: np.ndarray, radius: int = 4) -> float:
        height, width = image.shape[:2]
        x, y = np.rint(xy).astype(int)
        patch = image[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
        return float(np.max(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))) if patch.size else 0.0

    @staticmethod
    def _rotation_difference(first: np.ndarray, second: np.ndarray) -> float:
        first_matrix, _ = cv2.Rodrigues(first)
        second_matrix, _ = cv2.Rodrigues(second)
        relative = first_matrix @ second_matrix.T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cosine))

    def _candidate_metrics(
        self,
        object_points,
        image_points,
        camera_matrix,
        rvec,
        tvec,
        prior_pose: PoseEstimate | None,
    ):
        camera_points = (
            cv2.Rodrigues(rvec)[0] @ object_points.T + tvec.reshape(3, 1)
        ).T
        if not np.all(np.isfinite(camera_points)) or np.median(camera_points[:, 2]) <= 0.0:
            return None
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            self.configuration.distortion,
        )
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - image_points,
            axis=1,
        )
        inliers = np.flatnonzero(
            residuals <= self.reprojection_threshold * 1.5
        )
        if len(inliers) < 3:
            return None
        mean_error = float(np.mean(residuals[inliers]))
        prior_penalty = 0.0
        if prior_pose is not None and prior_pose.success:
            rotation_delta = self._rotation_difference(rvec, prior_pose.rvec)
            translation_delta = float(np.linalg.norm(tvec - prior_pose.tvec))
            translation_gate = max(
                2.0,
                0.22 * abs(float(prior_pose.tvec.reshape(-1)[2])),
            )
            if rotation_delta > np.deg2rad(32.0) or translation_delta > translation_gate:
                return None
            prior_penalty = 1.5 * rotation_delta + translation_delta / translation_gate
        return mean_error + prior_penalty, mean_error, inliers

    def refine_matches(
        self,
        result: MatchResult,
        detections: list[LightDetection],
        pose: PoseEstimate,
        image_shape: tuple[int, ...] | None = None,
        image: np.ndarray | None = None,
    ) -> MatchResult:
        """Use a reliable coarse pose to repair unmatched/high-residual IDs."""
        if (
            not self.enable_template_recovery
            or not pose.success
            or len(result.matches) < 6
            or float(pose.confidence or 0.0) < 0.20
        ):
            return result
        camera_matrix = adapt_camera_matrix(
            self.configuration.camera_matrix,
            self.configuration.camera_calibration_size,
            image_shape,
            self.configuration.camera_image_transform,
        )
        object_points = np.asarray(
            [lamp.xyz for lamp in self.configuration.lights],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(
            object_points,
            pose.rvec,
            pose.tvec,
            camera_matrix,
            self.configuration.distortion,
        )
        projected = projected.reshape(-1, 2)
        spacing = np.linalg.norm(
            projected[:, None, :] - projected[None, :, :],
            axis=2,
        )
        np.fill_diagonal(spacing, np.inf)
        median_spacing = float(np.median(np.min(spacing, axis=1)))
        gate_px = float(np.clip(0.20 * median_spacing, 6.0, 18.0))
        id_to_index = {
            lamp_id: index for index, lamp_id in enumerate(self.configuration.ids)
        }
        # GNN IDs are the primary semantic observation. Template recovery may
        # fill missing IDs but must never relabel an already accepted GNN match.
        locked = list(result.matches)
        locked_detections = {
            match.detection_index
            for match in locked
            if 0 <= match.detection_index < len(detections)
        }
        locked_templates = {
            id_to_index[match.lamp.lamp_id]
            for match in locked
            if match.lamp.lamp_id in id_to_index
        }
        maximum_recoveries = min(3, len(self.configuration.lights) - len(locked_templates))
        if maximum_recoveries <= 0:
            return result

        remaining_detections = [
            index for index in range(len(detections))
            if index not in locked_detections
        ]
        remaining_templates = [
            index for index in range(len(self.configuration.lights))
            if index not in locked_templates
        ]
        repaired: list[LampMatch] = []
        if remaining_detections and remaining_templates:
            observed = np.asarray(
                [detections[index].xy for index in remaining_detections],
                dtype=np.float64,
            )
            expected = projected[np.asarray(remaining_templates)]
            costs = np.linalg.norm(
                observed[:, None, :] - expected[None, :, :],
                axis=2,
            )
            rows, columns = linear_sum_assignment(costs)
            ordered_pairs = sorted(
                zip(rows, columns),
                key=lambda pair: float(costs[pair[0], pair[1]]),
            )
            for row, column in ordered_pairs:
                if len(repaired) >= maximum_recoveries:
                    break
                residual = float(costs[row, column])
                if residual > 0.55 * gate_px:
                    continue
                detection_index = remaining_detections[int(row)]
                template_index = remaining_templates[int(column)]
                detection = detections[detection_index]
                detector_quality = float(
                    detection.detector_confidence
                    if detection.detector_confidence is not None
                    else detection.confidence
                )
                center_quality = float(
                    detection.center_confidence
                    if detection.center_confidence is not None
                    else detection.center_quality
                )
                quality = np.sqrt(
                    np.clip(detector_quality, 0.0, 1.0)
                    * np.clip(center_quality, 0.0, 1.0)
                )
                confidence = float(
                    quality
                    * float(pose.confidence or 0.5)
                    * np.exp(-residual / max(0.55 * gate_px, 1.0))
                )
                if quality < 0.30 or confidence < 0.24:
                    continue
                repaired.append(
                    LampMatch(
                        lamp=self.configuration.lights[template_index],
                        detection_index=detection_index,
                        point=detection.xy,
                        confidence=confidence,
                        geometric_cost=residual,
                        observation_status="recovered_detector",
                    )
                )
        recovered_detections: list[LightDetection] = []
        occupied_templates = locked_templates | {
            id_to_index[match.lamp.lamp_id] for match in repaired
        }
        occupied_points = [match.point for match in locked + repaired]
        if image is not None:
            height, width = image.shape[:2]
            search_side = float(np.clip(0.42 * median_spacing, 10.0, 42.0))
            for template_index in remaining_templates:
                if len(repaired) >= maximum_recoveries:
                    break
                if template_index in occupied_templates:
                    continue
                seed = projected[template_index]
                if not (0 <= seed[0] < width and 0 <= seed[1] < height):
                    continue
                half = 0.5 * search_side
                box = np.array([seed[0] - half, seed[1] - half, seed[0] + half, seed[1] + half])
                core = self.white_core_localizer.locate(image, box)
                residual = float(np.linalg.norm(core.xy - seed))
                snr = float(core.diagnostics.get("snr", 0.0))
                separation = min(
                    [float(np.linalg.norm(core.xy - point)) for point in occupied_points]
                    or [np.inf]
                )
                if (
                    not core.valid
                    or core.confidence < 0.24
                    or snr < 5.0
                    or residual > 0.50 * gate_px
                    or separation < max(3.0, 0.28 * gate_px)
                ):
                    continue
                confidence = float(
                    np.sqrt(core.confidence * np.clip(float(pose.confidence or 0.35), 0.10, 1.0))
                    * np.exp(-residual / max(gate_px, 1.0))
                )
                if confidence < 0.30:
                    continue
                lamp = self.configuration.lights[template_index]
                detection_index = len(detections) + len(recovered_detections)
                recovered_detections.append(
                    LightDetection(
                        xy=core.xy,
                        confidence=confidence,
                        color_probs=np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
                        color=lamp.color,
                        radius=max(3.0, half),
                        brightness=self._local_brightness(image, core.xy),
                        source="template_projection_recovery",
                        center_quality=core.confidence,
                        bbox_xyxy=box,
                        detector_confidence=confidence,
                        center_confidence=core.confidence,
                        center_method=core.method,
                        center_covariance=core.covariance,
                        center_valid=core.valid,
                        center_diagnostics={**core.diagnostics, "projection_residual_px": residual},
                    )
                )
                repaired.append(
                    LampMatch(
                        lamp=lamp,
                        detection_index=detection_index,
                        point=core.xy,
                        confidence=confidence,
                        geometric_cost=residual,
                        observation_status="recovered_projection",
                    )
                )
                occupied_points.append(core.xy)
        matches = locked + repaired
        if len(matches) < max(4, len(result.matches) - 2):
            return result
        mean_cost = float(
            np.mean([match.geometric_cost for match in matches])
        ) if matches else float("inf")
        matched_ids = {match.lamp.lamp_id for match in matches}
        return MatchResult(
            matches,
            f"{result.method}_pose_refined",
            mean_cost,
            [lamp_id for lamp_id in self.configuration.ids if lamp_id not in matched_ids],
            recovered_detections,
            result.candidate_log_scores,
            result.observation_dustbin_scores,
            result.template_dustbin_scores,
            result.candidate_template_ids,
            result.rerank_diagnostics,
        )

    def reject_pose_inconsistent_matches(
        self,
        result: MatchResult,
        pose: PoseEstimate,
        image_shape: tuple[int, ...] | None = None,
    ) -> MatchResult:
        """Remove GNN assignments that are inconsistent with the current 3-D pose."""
        if (
            not pose.success
            or pose.rvec is None
            or pose.tvec is None
            or len(result.matches) < 9
            or float(pose.reprojection_error_px or 0.0) < 3.0
        ):
            return result
        camera_matrix = adapt_camera_matrix(
            self.configuration.camera_matrix,
            self.configuration.camera_calibration_size,
            image_shape,
            self.configuration.camera_image_transform,
        )
        object_points = np.asarray(
            [self.configuration.by_id(match.lamp.lamp_id).xyz for match in result.matches],
            dtype=np.float64,
        )
        image_points = np.asarray([match.point for match in result.matches], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_points,
            pose.rvec,
            pose.tvec,
            camera_matrix,
            self.configuration.distortion,
        )
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = 1.4826 * mad
        gate = max(
            self.reprojection_threshold * 2.4,
            median + 4.0 * max(robust_sigma, 1.0),
            float(pose.reprojection_error_px or 0.0) + 8.0,
        )
        keep: list[LampMatch] = []
        rejected: list[LampMatch] = []
        for match, residual in zip(result.matches, residuals):
            low_semantic_confidence = match.confidence < 0.55
            extreme_residual = residual > max(gate, self.reprojection_threshold * 3.0)
            if extreme_residual and low_semantic_confidence:
                rejected.append(match)
            else:
                keep.append(match)
        max_rejections = max(1, min(2, len(result.matches) // 6))
        if not rejected or len(rejected) > max_rejections:
            return result
        if len(keep) < max(4, min(8, len(result.matches) - max_rejections)):
            return result
        missing_ids = [lamp_id for lamp_id in self.configuration.ids if lamp_id not in {m.lamp.lamp_id for m in keep}]
        return MatchResult(
            keep,
            f"{result.method}_pose_pruned",
            float(np.mean([match.geometric_cost for match in keep])) if keep else float("inf"),
            missing_ids,
            result.recovered_detections,
            result.candidate_log_scores,
            result.observation_dustbin_scores,
            result.template_dustbin_scores,
            result.candidate_template_ids,
            result.rerank_diagnostics,
        )

    def refine_pose_by_residual_gating(
        self,
        result: MatchResult,
        pose: PoseEstimate,
        image_shape: tuple[int, ...] | None = None,
        prior_pose: PoseEstimate | None = None,
        detections: list[LightDetection] | None = None,
        target_min_matches: int = 10,
        target_max_matches: int = 12,
        low_confidence_gate: float = 0.58,
        residual_sigma: float = 2.75,
        max_rejections: int = 2,
        preserve_matches: bool = True,
    ) -> tuple[MatchResult, PoseEstimate]:
        """Refit PnP after dropping only low-confidence, high-residual matches.

        This keeps GNN identities fixed. The pose module may reject a doubtful
        observation, but it never rewrites the lamp ID assigned by the matcher.
        """
        diagnostics = {
            "enabled": True,
            "accepted": False,
            "reason": "not_evaluated",
            "initial_matches": len(result.matches),
            "rejected_matches": 0,
        }
        if (
            not pose.success
            or pose.rvec is None
            or pose.tvec is None
            or len(result.matches) < 8
        ):
            diagnostics["reason"] = "invalid_initial_pose"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose
        if target_min_matches > 0 and len(result.matches) < target_min_matches:
            diagnostics["reason"] = "below_target_match_count"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose
        if target_max_matches > 0 and len(result.matches) > target_max_matches:
            diagnostics["reason"] = "above_target_match_count"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose

        camera_matrix = adapt_camera_matrix(
            self.configuration.camera_matrix,
            self.configuration.camera_calibration_size,
            image_shape,
            self.configuration.camera_image_transform,
        )
        object_points = np.asarray(
            [self.configuration.by_id(match.lamp.lamp_id).xyz for match in result.matches],
            dtype=np.float64,
        )
        image_points = np.asarray([match.point for match in result.matches], dtype=np.float64)
        try:
            projected, _ = cv2.projectPoints(
                object_points,
                pose.rvec,
                pose.tvec,
                camera_matrix,
                self.configuration.distortion,
            )
        except cv2.error:
            diagnostics["reason"] = "projection_failed"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = max(1.4826 * mad, 0.75)
        adaptive_gate = max(
            self.reprojection_threshold * 1.20,
            median + residual_sigma * robust_sigma,
            float(pose.reprojection_error_px or 0.0) + 1.25,
        )
        diagnostics.update(
            {
                "initial_error_px": float(pose.reprojection_error_px or 0.0),
                "median_residual_px": median,
                "residual_gate_px": float(adaptive_gate),
            }
        )
        candidates: list[tuple[int, LampMatch, float]] = []
        for index, (match, residual) in enumerate(zip(result.matches, residuals)):
            low_confidence = float(match.confidence) < low_confidence_gate
            high_residual = float(residual) > adaptive_gate
            if low_confidence and high_residual:
                candidates.append((index, match, float(residual)))
        if not candidates:
            diagnostics["reason"] = "no_low_confidence_high_residual_match"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose
        candidates = sorted(
            candidates,
            key=lambda item: (item[2], -float(item[1].confidence)),
            reverse=True,
        )[: max(1, max_rejections)]
        rejected_indices = {index for index, _, _ in candidates}
        keep = [
            match
            for index, match in enumerate(result.matches)
            if index not in rejected_indices
        ]
        if len(keep) < max(8, len(result.matches) - max(1, max_rejections)):
            diagnostics["reason"] = "insufficient_support_after_rejection"
            result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return result, pose
        missing_ids = [
            lamp_id
            for lamp_id in self.configuration.ids
            if lamp_id not in {match.lamp.lamp_id for match in keep}
        ]
        refined_result = MatchResult(
            keep,
            f"{result.method}_residual_pnp_refined",
            float(np.mean([match.geometric_cost for match in keep])) if keep else float("inf"),
            missing_ids,
            result.recovered_detections,
            result.candidate_log_scores,
            result.observation_dustbin_scores,
            result.template_dustbin_scores,
            result.candidate_template_ids,
            {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            },
        )
        refined_pose = self.estimate(
            refined_result,
            image_shape,
            prior_pose=prior_pose,
            detections=detections,
        )
        refined_error = float(
            refined_pose.reprojection_error_px
            if refined_pose.reprojection_error_px is not None
            else np.inf
        )
        original_error = float(
            pose.reprojection_error_px
            if pose.reprojection_error_px is not None
            else np.inf
        )
        diagnostics.update(
            {
                "rejected_matches": len(rejected_indices),
                "rejected_ids": [match.lamp.lamp_id for _, match, _ in candidates],
                "refined_error_px": refined_error,
            }
        )
        improves_geometry = refined_error + 0.20 < original_error
        improves_confidence = (
            refined_error + 0.05 < original_error
            and float(refined_pose.confidence or 0.0) >= float(pose.confidence or 0.0) - 0.03
        )
        upgrades_measurement = bool(pose.approximate and not refined_pose.approximate)
        if refined_pose.success and (improves_geometry or improves_confidence or upgrades_measurement):
            diagnostics["accepted"] = True
            diagnostics["reason"] = "accepted"
            if preserve_matches:
                result.rerank_diagnostics = {
                    **(result.rerank_diagnostics or {}),
                    "residual_pnp_refinement": diagnostics,
                }
                return result, refined_pose
            refined_result.rerank_diagnostics = {
                **(result.rerank_diagnostics or {}),
                "residual_pnp_refinement": diagnostics,
            }
            return refined_result, refined_pose
        diagnostics["reason"] = "acceptance_gate_failed"
        result.rerank_diagnostics = {
            **(result.rerank_diagnostics or {}),
            "residual_pnp_refinement": diagnostics,
        }
        return result, pose

    def rerank_matches_multi_hypothesis(
        self,
        result: MatchResult,
        detections: list[LightDetection],
        base_pose: PoseEstimate,
        image_shape: tuple[int, ...] | None = None,
        prior_pose: PoseEstimate | None = None,
        top_k: int = 3,
        beam_width: int = 48,
        max_changes: int = 3,
        policy: str = "rescue_only",
    ) -> tuple[MatchResult, PoseEstimate]:
        """Jointly rerank ambiguous GNN assignments with metric 3-D pose.

        The GNN top-1 solution remains the anchor. Only ambiguous observations
        are varied, reliable IDs are locked, and a candidate is accepted only
        when its robust all-point projection consistency improves enough to pay
        for the loss in GNN semantic score.
        """
        if policy not in {"rescue_only", "conservative_rerank"}:
            raise ValueError(f"Unsupported pose hypothesis policy: {policy!r}")
        if policy == "rescue_only" and base_pose.success:
            result.rerank_diagnostics = {
                "enabled": True,
                "policy": policy,
                "accepted": False,
                "reason": "base_pose_already_valid",
                "evaluated_hypotheses": 0,
            }
            return result, base_pose
        pair_scores = result.candidate_log_scores
        observation_bins = result.observation_dustbin_scores
        template_bins = result.template_dustbin_scores
        template_ids = result.candidate_template_ids
        if (
            pair_scores is None
            or observation_bins is None
            or template_bins is None
            or template_ids is None
            or len(result.matches) < 4
        ):
            return result, base_pose
        pair_scores = np.asarray(pair_scores, dtype=np.float64)
        observation_bins = np.asarray(observation_bins, dtype=np.float64)
        template_bins = np.asarray(template_bins, dtype=np.float64)
        if (
            pair_scores.ndim != 2
            or pair_scores.shape != (len(observation_bins), len(template_bins))
            or pair_scores.shape[0] > len(detections)
            or pair_scores.shape[1] != len(template_ids)
        ):
            return result, base_pose
        valid_pose_ids = set(self.configuration.ids)
        valid_columns = [
            column for column, lamp_id in enumerate(template_ids)
            if lamp_id in valid_pose_ids
        ]
        if len(valid_columns) < 4:
            return result, base_pose

        top_k = int(np.clip(top_k, 2, min(5, len(valid_columns))))
        beam_width = int(np.clip(beam_width, 4, 128))
        max_changes = int(np.clip(max_changes, 1, 5))
        base_by_observation = {
            match.detection_index: match
            for match in result.matches
            if 0 <= match.detection_index < pair_scores.shape[0]
        }
        column_by_id = {lamp_id: index for index, lamp_id in enumerate(template_ids)}
        lamp_by_id = {match.lamp.lamp_id: match.lamp for match in result.matches}
        for lamp in self.configuration.lights:
            lamp_by_id.setdefault(lamp.lamp_id, lamp)

        options_by_observation: dict[int, list[int]] = {}
        ambiguity: list[tuple[float, int]] = []
        for observation_index in range(pair_scores.shape[0]):
            row = pair_scores[observation_index]
            ordered = sorted(valid_columns, key=lambda column: row[column], reverse=True)
            best_score = float(row[ordered[0]])
            base_match = base_by_observation.get(observation_index)
            base_column = (
                column_by_id.get(base_match.lamp.lamp_id)
                if base_match is not None
                else None
            )
            options = [
                column for column in ordered[:top_k]
                if best_score - float(row[column]) <= 9.0
                and float(row[column]) >= float(observation_bins[observation_index]) - 2.0
            ]
            if base_column is not None and base_column not in options:
                options.append(base_column)
            options_by_observation[observation_index] = options
            alternatives = [column for column in options if column != base_column]
            if base_match is not None and alternatives:
                alternative_score = max(float(row[column]) for column in alternatives)
                base_score = float(row[base_column])
                gap = base_score - alternative_score
                # Low confidence and a small top-2 gap indicate an ID that the
                # GNN itself considers ambiguous enough for pose reranking.
                priority = float(base_match.confidence) + 0.30 * max(gap, -1.0)
                if base_match.confidence < 0.62 or gap < 9.0:
                    ambiguity.append((priority, observation_index))
            elif base_match is None and options:
                margin = best_score - float(observation_bins[observation_index])
                if margin > 1.0:
                    ambiguity.append((0.90 - 0.20 * margin, observation_index))

        # At most seven observations are allowed to branch; all other accepted
        # GNN IDs are immutable anchors. This keeps the search small and guards
        # against a globally mirrored but semantically implausible solution.
        variable_observations = {
            observation_index
            for _, observation_index in sorted(ambiguity)[:7]
        }
        locked_assignments: dict[int, str] = {}
        for observation_index, match in base_by_observation.items():
            if observation_index not in variable_observations:
                locked_assignments[observation_index] = match.lamp.lamp_id
        locked_ids = set(locked_assignments.values())
        if not variable_observations:
            result.rerank_diagnostics = {
                "enabled": True,
                "accepted": False,
                "reason": "no_ambiguous_gnn_candidates",
                "evaluated_hypotheses": 0,
            }
            return result, base_pose

        # Beam state: semantic score, changed-ID count, assignments, used IDs.
        beam: list[tuple[float, int, dict[int, str], set[str]]] = [
            (0.0, 0, dict(locked_assignments), set(locked_ids))
        ]
        ordered_variables = sorted(
            variable_observations,
            key=lambda index: next(
                (priority for priority, candidate in ambiguity if candidate == index),
                1.0,
            ),
        )
        for observation_index in ordered_variables:
            row_best_score = float(
                np.max(pair_scores[observation_index, valid_columns])
            )
            base_match = base_by_observation.get(observation_index)
            base_id = base_match.lamp.lamp_id if base_match is not None else None
            choices: list[str | None] = [
                template_ids[column]
                for column in options_by_observation[observation_index]
            ]
            if base_id is not None and base_id not in choices:
                choices.append(base_id)
            # Existing GNN observations may be relabelled but never deleted.
            # Deleting a difficult point trivially improves PnP residuals and
            # harmed exact-frame ID accuracy in the earlier pruning ablation.
            if base_id is None:
                choices.append(None)
            expanded: list[tuple[float, int, dict[int, str], set[str]]] = []
            for semantic_score, changes, assignments, used_ids in beam:
                for lamp_id in choices:
                    if lamp_id is not None and lamp_id in used_ids:
                        continue
                    changed = int(lamp_id != base_id)
                    next_changes = changes + changed
                    if next_changes > max_changes:
                        continue
                    next_assignments = dict(assignments)
                    next_used = set(used_ids)
                    if lamp_id is None:
                        choice_score = (
                            float(observation_bins[observation_index]) - row_best_score
                        ) / 8.0
                        if base_id is not None:
                            choice_score -= 0.25
                    else:
                        column = column_by_id[lamp_id]
                        choice_score = (
                            float(pair_scores[observation_index, column]) - row_best_score
                        ) / 8.0
                        next_assignments[observation_index] = lamp_id
                        next_used.add(lamp_id)
                    expanded.append(
                        (
                            semantic_score + choice_score,
                            next_changes,
                            next_assignments,
                            next_used,
                        )
                    )
            expanded.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            beam = expanded[:beam_width]
            if not beam:
                return result, base_pose

        base_count = len(result.matches)
        hypothesis_results: list[tuple[MatchResult, int]] = [(result, 0)]
        signatures = {
            tuple(sorted((match.detection_index, match.lamp.lamp_id) for match in result.matches))
        }
        for _, changes, assignments, _ in beam:
            if len(assignments) < max(4, base_count - 1):
                continue
            signature = tuple(sorted(assignments.items()))
            if signature in signatures:
                continue
            signatures.add(signature)
            matches: list[LampMatch] = []
            for observation_index, lamp_id in sorted(assignments.items()):
                column = column_by_id[lamp_id]
                pair_score = float(pair_scores[observation_index, column])
                dustbin = max(
                    float(observation_bins[observation_index]),
                    float(template_bins[column]),
                )
                margin = float(np.clip(pair_score - dustbin, -12.0, 12.0))
                mutual_confidence = 1.0 / (1.0 + np.exp(-2.5 * margin))
                detection = detections[observation_index]
                detector_quality = float(
                    detection.detector_confidence
                    if detection.detector_confidence is not None
                    else detection.confidence
                )
                center_quality = float(
                    detection.center_confidence
                    if detection.center_confidence is not None
                    else detection.center_quality
                )
                observation_quality = float(
                    np.sqrt(
                        np.clip(detector_quality, 1e-6, 1.0)
                        * np.clip(center_quality, 1e-6, 1.0)
                    )
                )
                confidence = float(mutual_confidence * observation_quality)
                previous = base_by_observation.get(observation_index)
                unchanged = previous is not None and previous.lamp.lamp_id == lamp_id
                if unchanged:
                    confidence = max(confidence, float(previous.confidence))
                matches.append(
                    LampMatch(
                        lamp_by_id[lamp_id],
                        observation_index,
                        detection.xy,
                        confidence,
                        -pair_score,
                        "observed" if unchanged else "pose_reranked",
                    )
                )
            matched_ids = {match.lamp.lamp_id for match in matches}
            hypothesis_results.append(
                (
                    MatchResult(
                        matches,
                        result.method,
                        float(np.mean([match.geometric_cost for match in matches])),
                        [lamp_id for lamp_id in self.configuration.ids if lamp_id not in matched_ids],
                        result.recovered_detections,
                        result.candidate_log_scores,
                        result.observation_dustbin_scores,
                        result.template_dustbin_scores,
                        result.candidate_template_ids,
                    ),
                    changes,
                )
            )

        camera_matrix = adapt_camera_matrix(
            self.configuration.camera_matrix,
            self.configuration.camera_calibration_size,
            image_shape,
            self.configuration.camera_image_transform,
        )

        def score_hypothesis(
            hypothesis: MatchResult,
            pose: PoseEstimate,
        ) -> dict[str, float] | None:
            usable = [match for match in hypothesis.matches if match.confidence >= 0.18]
            if not pose.success or len(usable) < 4:
                return None
            objects = np.asarray(
                [self.configuration.by_id(match.lamp.lamp_id).xyz for match in usable],
                dtype=np.float64,
            )
            images = np.asarray([match.point for match in usable], dtype=np.float64)
            projected, _ = cv2.projectPoints(
                objects,
                pose.rvec,
                pose.tvec,
                camera_matrix,
                self.configuration.distortion,
            )
            residuals = np.linalg.norm(projected.reshape(-1, 2) - images, axis=1)
            gate = max(float(self.reprojection_threshold) * 1.5, 8.0)
            robust_projection = float(np.mean(np.minimum(residuals / gate, 3.0)))
            inlier_ratio = float(np.mean(residuals <= gate))
            semantic_losses = []
            for match in usable:
                column = column_by_id[match.lamp.lamp_id]
                valid_row = pair_scores[match.detection_index, valid_columns]
                semantic_losses.append(
                    float(
                        np.max(valid_row)
                        - pair_scores[match.detection_index, column]
                    ) / 8.0
                )
            semantic_loss = float(np.mean(semantic_losses)) if semantic_losses else 0.0
            support_penalty = max(0.0, (base_count - len(usable)) / max(base_count, 1))
            prior_penalty = 0.0
            if prior_pose is not None and prior_pose.success:
                rotation_delta = self._rotation_difference(pose.rvec, prior_pose.rvec)
                translation_gate = max(
                    2.0,
                    0.22 * abs(float(prior_pose.tvec.reshape(-1)[2])),
                )
                translation_delta = float(np.linalg.norm(pose.tvec - prior_pose.tvec))
                prior_penalty = 0.08 * (
                    rotation_delta / np.deg2rad(15.0)
                    + translation_delta / translation_gate
                )
            objective = (
                robust_projection
                + 0.30 * (1.0 - inlier_ratio)
                + 0.18 * semantic_loss
                + 0.60 * support_penalty
                + prior_penalty
            )
            return {
                "objective": float(objective),
                "robust_projection": robust_projection,
                "inlier_ratio": inlier_ratio,
                "semantic_loss": semantic_loss,
                "support": float(len(usable)),
            }

        evaluated: list[tuple[float, MatchResult, PoseEstimate, int, dict[str, float]]] = []
        for index, (hypothesis, changes) in enumerate(hypothesis_results[: beam_width + 1]):
            pose = base_pose if index == 0 else self.estimate(
                hypothesis,
                image_shape,
                prior_pose=prior_pose,
                detections=detections,
            )
            try:
                metrics = score_hypothesis(hypothesis, pose)
            except (cv2.error, ValueError, KeyError):
                metrics = None
            if metrics is not None:
                evaluated.append((metrics["objective"], hypothesis, pose, changes, metrics))
        diagnostics = {
            "enabled": True,
            "policy": policy,
            "accepted": False,
            "generated_hypotheses": len(hypothesis_results),
            "evaluated_hypotheses": len(evaluated),
        }
        if not evaluated:
            diagnostics["reason"] = "no_valid_pose_hypothesis"
            result.rerank_diagnostics = diagnostics
            return result, base_pose
        base_entry = next((item for item in evaluated if item[1] is result), None)
        best_entry = min(evaluated, key=lambda item: item[0])
        _, best_result, best_pose, best_changes, best_metrics = best_entry
        if base_entry is None:
            base_metrics = None
            objective_gain = float("inf")
        else:
            base_metrics = base_entry[4]
            objective_gain = float(base_entry[0] - best_entry[0])
            diagnostics["base_objective"] = round(float(base_entry[0]), 6)
        diagnostics.update(
            {
                "best_objective": round(float(best_entry[0]), 6),
                "objective_gain": (
                    round(objective_gain, 6) if np.isfinite(objective_gain) else None
                ),
                "changed_ids": int(best_changes),
                "best_semantic_loss": round(best_metrics["semantic_loss"], 6),
                "best_inlier_ratio": round(best_metrics["inlier_ratio"], 6),
            }
        )
        if best_result is result or best_changes == 0:
            diagnostics["reason"] = "gnn_top1_remains_best"
            result.rerank_diagnostics = diagnostics
            return result, base_pose

        support_ok = best_metrics["support"] >= max(4, base_count - 1)
        semantic_ok = best_metrics["semantic_loss"] <= 0.90
        if base_metrics is None:
            accept = (
                support_ok
                and semantic_ok
                and best_metrics["robust_projection"] <= 0.85
                and float(best_pose.confidence or 0.0) >= 0.15
            )
        else:
            projection_gain = (
                base_metrics["robust_projection"]
                - best_metrics["robust_projection"]
            )
            inlier_gain = best_metrics["inlier_ratio"] - base_metrics["inlier_ratio"]
            diagnostics["projection_gain"] = round(float(projection_gain), 6)
            diagnostics["inlier_ratio_gain"] = round(float(inlier_gain), 6)
            accept = (
                support_ok
                and semantic_ok
                and objective_gain >= 0.035
                and (projection_gain >= 0.055 or inlier_gain >= 0.08)
                and float(best_pose.reprojection_error_px or np.inf)
                <= max(
                    self.reprojection_threshold,
                    float(base_pose.reprojection_error_px or np.inf) + 0.50,
                )
            )
        if not accept:
            diagnostics["reason"] = "conservative_acceptance_gate"
            result.rerank_diagnostics = diagnostics
            return result, base_pose
        diagnostics["accepted"] = True
        diagnostics["reason"] = "pose_consistent_hypothesis"
        best_result.method = f"{result.method}_pose_multihypothesis"
        best_result.rerank_diagnostics = diagnostics
        return best_result, best_pose

    def estimate(
        self,
        result: MatchResult,
        image_shape: tuple[int, ...] | None = None,
        prior_pose: PoseEstimate | None = None,
        detections: list[LightDetection] | None = None,
    ) -> PoseEstimate:
        matches = [match for match in result.matches if match.confidence >= 0.18]
        minimum_matches = 3 if self.allow_sparse_pose else 8
        has_usable_prior = (
            self.allow_sparse_pose
            and prior_pose is not None
            and prior_pose.success
            and float(prior_pose.confidence or 0.0) >= 0.12
        )
        if len(matches) < minimum_matches and not (has_usable_prior and len(matches) >= 2):
            return PoseEstimate(False, source="pnp_insufficient_matches")
        object_points = np.asarray(
            [self.configuration.by_id(match.lamp.lamp_id).xyz for match in matches],
            dtype=np.float64,
        )
        image_points = np.asarray([match.point for match in matches], dtype=np.float64)
        match_weights = self._match_weights(matches, detections)
        camera_matrix = adapt_camera_matrix(
            self.configuration.camera_matrix,
            self.configuration.camera_calibration_size,
            image_shape,
            self.configuration.camera_image_transform,
        )
        candidates = []
        if len(matches) >= self.minimum_ransac_matches:
            cv2.setRNGSeed(42)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                self.configuration.distortion,
                flags=cv2.SOLVEPNP_SQPNP,
                reprojectionError=self.reprojection_threshold,
                confidence=0.999,
                iterationsCount=300,
            )
            minimum_inliers = max(
                4 if self.allow_sparse_pose else 8,
                int(np.ceil(self.min_inlier_ratio * len(matches))),
            )
            if success and inliers is not None and len(inliers) >= minimum_inliers:
                inliers = inliers.reshape(-1)
                candidates.append(("pnp_ransac_lm", rvec, tvec, inliers))
        if self.allow_sparse_pose and len(matches) >= 4:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    self.configuration.distortion,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                if success:
                    candidates.append(
                        (
                            "pnp_sparse_sqpnp",
                            rvec,
                            tvec,
                            np.arange(len(matches)),
                        )
                    )
            except cv2.error:
                pass
        if (
            self.allow_sparse_pose
            and prior_pose is not None
            and prior_pose.success
            and len(matches) >= 3
        ):
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    self.configuration.distortion,
                    rvec=prior_pose.rvec.copy(),
                    tvec=prior_pose.tvec.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    candidates.append(
                        (
                            "pnp_sparse_prior",
                            rvec,
                            tvec,
                            np.arange(len(matches)),
                        )
                    )
            except cv2.error:
                pass
        if has_usable_prior and 2 <= len(matches) < 8:
            try:
                all_template_points = np.asarray(
                    [lamp.xyz for lamp in self.configuration.lights],
                    dtype=np.float64,
                )
                projected_template, _ = cv2.projectPoints(
                    all_template_points,
                    prior_pose.rvec,
                    prior_pose.tvec,
                    camera_matrix,
                    self.configuration.distortion,
                )
                projected_template = projected_template.reshape(-1, 2)
                matched_ids = {match.lamp.lamp_id for match in matches}
                image_center = np.asarray(
                    [camera_matrix[0, 2], camera_matrix[1, 2]],
                    dtype=np.float64,
                )
                missing_indices = [
                    index
                    for index, lamp in enumerate(self.configuration.lights)
                    if lamp.lamp_id not in matched_ids
                ]
                missing_indices = sorted(
                    missing_indices,
                    key=lambda index: float(
                        np.linalg.norm(projected_template[index] - image_center)
                    ),
                )[: max(4, 8 - len(matches))]
                virtual_objects = all_template_points[missing_indices]
                virtual_images = projected_template[missing_indices]
                if len(virtual_objects):
                    assisted_objects = np.vstack([object_points, virtual_objects])
                    assisted_images = np.vstack([image_points, virtual_images])
                    virtual_weight = 0.045 if len(matches) <= 3 else 0.075
                    assisted_weights = np.concatenate(
                        [
                            np.clip(match_weights, 0.20, 1.0),
                            np.full(len(virtual_objects), virtual_weight, dtype=np.float64),
                        ]
                    )
                    rvec, tvec = self._weighted_refine_pose(
                        assisted_objects,
                        assisted_images,
                        camera_matrix,
                        prior_pose.rvec.copy(),
                        prior_pose.tvec.copy(),
                        assisted_weights,
                    )
                    camera_points = (
                        cv2.Rodrigues(rvec)[0] @ object_points.T + tvec.reshape(3, 1)
                    ).T
                    if np.all(np.isfinite(camera_points)) and np.median(camera_points[:, 2]) > 0.0:
                        projected_real, _ = cv2.projectPoints(
                            object_points,
                            rvec,
                            tvec,
                            camera_matrix,
                            self.configuration.distortion,
                        )
                        residuals = np.linalg.norm(
                            projected_real.reshape(-1, 2) - image_points,
                            axis=1,
                        )
                        inliers = np.flatnonzero(residuals <= self.reprojection_threshold * 1.75)
                        if len(inliers) >= 2:
                            rotation_delta = self._rotation_difference(rvec, prior_pose.rvec)
                            translation_delta = float(np.linalg.norm(tvec - prior_pose.tvec))
                            translation_gate = max(
                                2.0,
                                0.22 * abs(float(prior_pose.tvec.reshape(-1)[2])),
                            )
                            mean_error = float(np.mean(residuals[inliers]))
                            if (
                                rotation_delta <= np.deg2rad(20.0)
                                and translation_delta <= 0.75 * translation_gate
                                and mean_error <= self.reprojection_threshold * 1.25
                            ):
                                prior_penalty = 1.5 * rotation_delta + translation_delta / translation_gate
                                candidates.append(
                                    (
                                        "pnp_template_assisted",
                                        rvec,
                                        tvec,
                                        inliers,
                                    )
                                )
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                pass
        ranked = []
        for source, rvec, tvec, candidate_inliers in candidates:
            if source == "pnp_template_assisted":
                projected, _ = cv2.projectPoints(
                    object_points[candidate_inliers],
                    rvec,
                    tvec,
                    camera_matrix,
                    self.configuration.distortion,
                )
                residuals = np.linalg.norm(
                    projected.reshape(-1, 2) - image_points[candidate_inliers],
                    axis=1,
                )
                local_inliers = np.flatnonzero(
                    residuals <= self.reprojection_threshold * 1.75
                )
                if len(local_inliers) < 2:
                    continue
                score = float(np.mean(residuals[local_inliers]) + 0.75)
                ranked.append(
                    (
                        score,
                        source,
                        rvec,
                        tvec,
                        candidate_inliers[local_inliers],
                    )
                )
            else:
                metrics = self._candidate_metrics(
                    object_points[candidate_inliers],
                    image_points[candidate_inliers],
                    camera_matrix,
                    rvec,
                    tvec,
                    prior_pose if source != "pnp_ransac_lm" else None,
                )
                if metrics is None:
                    continue
                score, _, local_inliers = metrics
                ranked.append(
                    (
                        score,
                        source,
                        rvec,
                        tvec,
                        candidate_inliers[local_inliers],
                    )
                )
        if not ranked:
            if has_usable_prior and len(matches) >= 2:
                projected, _ = cv2.projectPoints(
                    object_points,
                    prior_pose.rvec,
                    prior_pose.tvec,
                    camera_matrix,
                    self.configuration.distortion,
                )
                residuals = np.linalg.norm(
                    projected.reshape(-1, 2) - image_points,
                    axis=1,
                )
                gate = max(18.0, self.reprojection_threshold * 2.5)
                inliers = np.flatnonzero(residuals <= gate)
                if len(inliers) >= 2:
                    mean_error = float(np.mean(residuals[inliers]))
                    support_factor = min(len(inliers) / 4.0, 1.0)
                    quality = float(np.mean([matches[index].confidence for index in inliers]))
                    confidence = float(
                        np.clip(
                            0.55
                            * support_factor
                            * min(quality / 0.55, 1.0)
                            * np.exp(-mean_error / gate)
                            * float(prior_pose.confidence or 0.25),
                            0.02,
                            0.35,
                        )
                    )
                    return PoseEstimate(
                        True,
                        prior_pose.rvec.copy(),
                        prior_pose.tvec.copy(),
                        mean_error,
                        len(inliers),
                        "pnp_template_assisted_prior",
                        confidence,
                        approximate=True,
                        measurement_count=len(matches),
                        recovered_count=sum(
                            matches[index].observation_status.startswith("recovered")
                            for index in inliers
                        ),
                        weighted_refinement=False,
                    )
            return PoseEstimate(False, source="pnp_sparse_failed")
        _, source, rvec, tvec, inliers = min(ranked, key=lambda item: item[0])
        weighted_refinement = False
        if self.use_weighted_refinement and source != "pnp_template_assisted":
            try:
                rvec, tvec = self._weighted_refine_pose(
                    object_points[inliers],
                    image_points[inliers],
                    camera_matrix,
                    rvec,
                    tvec,
                    match_weights[inliers],
                )
                weighted_refinement = True
                source = f"{source}_weighted"
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                weighted_refinement = False
        if not weighted_refinement:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inliers], image_points[inliers], camera_matrix, self.configuration.distortion, rvec, tvec
                )
            except cv2.error:
                pass
        projected, _ = cv2.projectPoints(
            object_points[inliers],
            rvec,
            tvec,
            camera_matrix,
            self.configuration.distortion,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - image_points[inliers], axis=1).mean()
        inlier_ratio = len(inliers) / len(matches)
        center_quality = float(np.mean([matches[index].confidence for index in inliers]))
        sparse_factor = min(len(inliers) / 8.0, 1.0)
        confidence = float(
            np.clip(
                inlier_ratio
                * sparse_factor
                * min(center_quality / 0.55, 1.0)
                * np.exp(-error / self.reprojection_threshold),
                0.0,
                1.0,
            )
        )
        approximate = len(inliers) < 8
        recovered_count = sum(
            matches[index].observation_status.startswith("recovered")
            for index in inliers
        )
        return PoseEstimate(
            True,
            rvec,
            tvec,
            float(error),
            len(inliers),
            source,
            confidence,
            approximate=approximate,
            measurement_count=len(matches),
            recovered_count=recovered_count,
            weighted_refinement=weighted_refinement,
        )
