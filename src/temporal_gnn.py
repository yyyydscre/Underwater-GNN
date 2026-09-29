"""Online temporal refinement for GNN lamp identity matching."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .graph import log_optimal_transport


@dataclass
class LampTrack:
    position: np.ndarray
    velocity: np.ndarray
    age: int
    confidence: float


class TemporalGraphDecoder:
    """Refine single-frame scores with motion-predicted lamp-ID tracks."""

    def __init__(
        self,
        template_count: int,
        temporal_weight: float = 0.75,
        sigma_ratio: float = 0.08,
        max_age: int = 6,
        minimum_update_confidence: float = 0.35,
        velocity_smoothing: float = 0.65,
        confidence_threshold: float = 0.85,
        confidence_transition_width: float = 0.35,
        consensus_reanchor_ratio: float = 0.75,
        consensus_failure_ratio: float = 0.35,
        hard_failure_bonus: float = 50.0,
        burn_in_frames: int = 2,
        sinkhorn_iterations: int = 60,
        temporal_template_indices: list[int] | None = None,
        recovery_template_indices: list[int] | None = None,
        recovery_max_observations: int = 0,
    ) -> None:
        self.template_count = int(template_count)
        self.temporal_weight = float(temporal_weight)
        self.sigma_ratio = float(sigma_ratio)
        self.max_age = int(max_age)
        self.minimum_update_confidence = float(minimum_update_confidence)
        self.velocity_smoothing = float(velocity_smoothing)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_transition_width = float(
            confidence_transition_width
        )
        self.consensus_reanchor_ratio = float(
            consensus_reanchor_ratio
        )
        self.consensus_failure_ratio = float(
            consensus_failure_ratio
        )
        self.hard_failure_bonus = float(hard_failure_bonus)
        self.burn_in_frames = int(burn_in_frames)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        indices = (
            range(self.template_count)
            if temporal_template_indices is None
            else temporal_template_indices
        )
        self.temporal_template_indices = {
            int(index)
            for index in indices
            if 0 <= int(index) < self.template_count
        }
        self.recovery_template_indices = {
            int(index)
            for index in (recovery_template_indices or [])
            if 0 <= int(index) < self.template_count
        }
        self.recovery_max_observations = int(recovery_max_observations)
        self.tracks: dict[int, LampTrack] = {}

    def _active_template_indices(self, observation_count: int) -> set[int]:
        active = set(self.temporal_template_indices)
        if (
            self.recovery_max_observations > 0
            and observation_count <= self.recovery_max_observations
        ):
            active.update(self.recovery_template_indices)
        return active

    def reset(self) -> None:
        self.tracks.clear()

    @staticmethod
    def _scene_scale(points: np.ndarray) -> float:
        if len(points) < 2:
            return 20.0
        centered = points - np.median(points, axis=0, keepdims=True)
        return max(
            float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
            20.0,
        )

    def _mutual_assignments(
        self,
        transport: torch.Tensor,
    ) -> list[int]:
        observation_choice = transport[:-1].argmax(dim=1)
        template_choice = transport[:, :-1].argmax(dim=0)
        assignments = []
        for observation_index, template_index in enumerate(
            observation_choice.tolist()
        ):
            if (
                template_index < self.template_count
                and int(template_choice[template_index]) == observation_index
            ):
                assignments.append(int(template_index))
            else:
                assignments.append(self.template_count)
        return assignments

    def _assignment_confidence(
        self,
        transport: torch.Tensor,
        observation_index: int,
        template_index: int,
    ) -> float:
        row = transport[observation_index]
        selected = row[template_index]
        alternatives = torch.cat(
            [row[:template_index], row[template_index + 1 :]]
        )
        row_margin = selected - alternatives.max()
        column = transport[:, template_index]
        column_alternatives = torch.cat(
            [
                column[:observation_index],
                column[observation_index + 1 :],
            ]
        )
        column_margin = selected - column_alternatives.max()
        margin = torch.minimum(row_margin, column_margin)
        return float(torch.sigmoid(2.5 * margin).detach().cpu())

    def _predict_track_positions(
        self,
        points: np.ndarray,
        preliminary: list[int],
        scene_scale: float,
    ) -> dict[int, np.ndarray]:
        active = {
            lamp_id: track
            for lamp_id, track in self.tracks.items()
            if track.age <= self.max_age
        }
        if not active:
            return {}

        source = []
        target = []
        for observation_index, lamp_id in enumerate(preliminary):
            track = active.get(lamp_id)
            if track is None:
                continue
            source.append(track.position)
            target.append(points[observation_index])

        affine = None
        if len(source) >= 4:
            affine, inliers = cv2.estimateAffinePartial2D(
                np.asarray(source, dtype=np.float32),
                np.asarray(target, dtype=np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=max(3.0, 0.035 * scene_scale),
                maxIters=300,
                confidence=0.995,
                refineIters=20,
            )
            if affine is not None:
                linear = affine[:, :2]
                scale = float(np.sqrt(abs(np.linalg.det(linear))))
                inlier_count = int(inliers.sum()) if inliers is not None else 0
                if (
                    not np.all(np.isfinite(affine))
                    or not 0.65 <= scale <= 1.55
                    or inlier_count < 3
                ):
                    affine = None

        predictions = {}
        for lamp_id, track in active.items():
            if affine is None:
                predictions[lamp_id] = track.position + track.velocity
            else:
                predictions[lamp_id] = (
                    affine[:, :2] @ track.position + affine[:, 2]
                )
        return predictions

    def _temporal_scores(
        self,
        scores: torch.Tensor,
        base_transport: torch.Tensor,
        points: np.ndarray,
        predicted_positions: dict[int, np.ndarray],
        scene_scale: float,
    ) -> torch.Tensor:
        if not predicted_positions or self.temporal_weight <= 0.0:
            return scores
        sigma = max(3.0, self.sigma_ratio * scene_scale)
        compatibility = np.zeros(
            (len(points), self.template_count),
            dtype=np.float32,
        )
        for lamp_id, predicted in predicted_positions.items():
            distance = np.linalg.norm(points - predicted[None, :], axis=1)
            normalized = distance / sigma
            compatibility[:, lamp_id] = (
                2.0 * np.exp(-0.5 * normalized**2) - 1.0
            )
        active_indices = self._active_template_indices(len(points))
        tracked_ids = np.asarray(
            sorted(
                set(predicted_positions).intersection(
                    active_indices
                )
            ),
            dtype=np.int64,
        )
        if not len(tracked_ids):
            return scores
        temporal = np.zeros_like(compatibility)
        temporal[:, tracked_ids] = (
            self.temporal_weight * compatibility[:, tracked_ids]
        )
        row_choice = base_transport[:-1].argmax(dim=1)
        row_confidence = []
        for observation_index, selected_index in enumerate(
            row_choice.tolist()
        ):
            row = base_transport[observation_index]
            selected = row[selected_index]
            alternatives = torch.cat(
                [row[:selected_index], row[selected_index + 1 :]]
            )
            margin = selected - alternatives.max()
            row_confidence.append(
                float(torch.sigmoid(2.5 * margin).detach().cpu())
            )
        confidence = np.asarray(row_confidence, dtype=np.float32)
        gate = np.clip(
            (self.confidence_threshold - confidence)
            / max(self.confidence_transition_width, 1e-6),
            0.0,
            1.0,
        )
        temporal *= gate[:, None]
        return scores + torch.from_numpy(temporal).to(
            device=scores.device,
            dtype=scores.dtype,
        )

    def _update_tracks(
        self,
        points: np.ndarray,
        assignments: list[int],
        transport: torch.Tensor,
    ) -> None:
        for track in self.tracks.values():
            track.age += 1

        updated_ids = set()
        for observation_index, lamp_id in enumerate(assignments):
            if lamp_id >= self.template_count:
                continue
            confidence = self._assignment_confidence(
                transport,
                observation_index,
                lamp_id,
            )
            if confidence < self.minimum_update_confidence:
                continue
            point = points[observation_index].astype(np.float64)
            previous = self.tracks.get(lamp_id)
            if previous is None:
                velocity = np.zeros(2, dtype=np.float64)
            else:
                displacement = point - previous.position
                velocity = (
                    self.velocity_smoothing * displacement
                    + (1.0 - self.velocity_smoothing) * previous.velocity
                )
            self.tracks[lamp_id] = LampTrack(
                position=point,
                velocity=velocity,
                age=0,
                confidence=confidence,
            )
            updated_ids.add(lamp_id)

        self.tracks = {
            lamp_id: track
            for lamp_id, track in self.tracks.items()
            if track.age <= self.max_age
        }

    def decode(
        self,
        scores: torch.Tensor,
        points: np.ndarray,
        dustbin_score: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        points = np.asarray(points, dtype=np.float64)
        base_transport = log_optimal_transport(
            scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        preliminary = self._mutual_assignments(base_transport)
        scene_scale = self._scene_scale(points)
        predicted_positions = self._predict_track_positions(
            points,
            preliminary,
            scene_scale,
        )
        refined_scores = self._temporal_scores(
            scores,
            base_transport,
            points,
            predicted_positions,
            scene_scale,
        )
        refined_transport = log_optimal_transport(
            refined_scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        assignments = self._mutual_assignments(refined_transport)
        self._update_tracks(points, assignments, refined_transport)
        return assignments, refined_transport


class TrackletGraphDecoder(TemporalGraphDecoder):
    """Propagate identity evidence over detection-to-detection temporal edges."""

    def __init__(
        self,
        template_count: int,
        temporal_weight: float = 0.75,
        association_gate_ratio: float = 0.14,
        minimum_update_confidence: float = 0.35,
        confidence_threshold: float = 0.85,
        confidence_transition_width: float = 0.35,
        consensus_reanchor_ratio: float = 0.75,
        consensus_failure_ratio: float = 0.35,
        hard_failure_bonus: float = 50.0,
        burn_in_frames: int = 2,
        sinkhorn_iterations: int = 60,
        temporal_template_indices: list[int] | None = None,
        recovery_template_indices: list[int] | None = None,
        recovery_max_observations: int = 0,
    ) -> None:
        super().__init__(
            template_count=template_count,
            temporal_weight=temporal_weight,
            sigma_ratio=association_gate_ratio,
            max_age=1,
            minimum_update_confidence=minimum_update_confidence,
            confidence_threshold=confidence_threshold,
            confidence_transition_width=confidence_transition_width,
            consensus_reanchor_ratio=consensus_reanchor_ratio,
            consensus_failure_ratio=consensus_failure_ratio,
            hard_failure_bonus=hard_failure_bonus,
            burn_in_frames=burn_in_frames,
            sinkhorn_iterations=sinkhorn_iterations,
            temporal_template_indices=temporal_template_indices,
            recovery_template_indices=recovery_template_indices,
            recovery_max_observations=recovery_max_observations,
        )
        self.previous_points: np.ndarray | None = None
        self.previous_assignments: list[int] = []
        self.previous_confidences: list[float] = []
        self.stable_frames = 0
        self.temporal_active = burn_in_frames <= 0

    def reset(self) -> None:
        super().reset()
        self.previous_points = None
        self.previous_assignments = []
        self.previous_confidences = []
        self.stable_frames = 0
        self.temporal_active = self.burn_in_frames <= 0

    @staticmethod
    def _greedy_tracklet_edges(
        previous: np.ndarray,
        current: np.ndarray,
        gate_px: float,
    ) -> list[tuple[int, int]]:
        if not len(previous) or not len(current):
            return []
        costs = np.linalg.norm(
            previous[:, None, :] - current[None, :, :],
            axis=-1,
        )
        used_previous = set()
        used_current = set()
        pairs = []
        for flat_index in np.argsort(costs, axis=None):
            previous_index, current_index = np.unravel_index(
                int(flat_index),
                costs.shape,
            )
            if float(costs[previous_index, current_index]) > gate_px:
                break
            if (
                previous_index in used_previous
                or current_index in used_current
            ):
                continue
            used_previous.add(int(previous_index))
            used_current.add(int(current_index))
            pairs.append((int(previous_index), int(current_index)))
        return pairs

    def _row_confidences(
        self,
        transport: torch.Tensor,
        assignments: list[int],
    ) -> list[float]:
        confidences = []
        for observation_index, lamp_id in enumerate(assignments):
            if lamp_id >= self.template_count:
                confidences.append(0.0)
            else:
                confidences.append(
                    self._assignment_confidence(
                        transport,
                        observation_index,
                        lamp_id,
                    )
                )
        return confidences

    def decode(
        self,
        scores: torch.Tensor,
        points: np.ndarray,
        dustbin_score: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        points = np.asarray(points, dtype=np.float64)
        base_transport = log_optimal_transport(
            scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        base_assignments = self._mutual_assignments(base_transport)
        base_confidences = self._row_confidences(
            base_transport,
            base_assignments,
        )

        refined_scores = scores
        active_template_indices = self._active_template_indices(len(points))
        if self.previous_points is not None and self.temporal_weight > 0.0:
            scene_scale = self._scene_scale(points)
            pairs = self._greedy_tracklet_edges(
                self.previous_points,
                points,
                gate_px=max(4.0, self.sigma_ratio * scene_scale),
            )
            temporal = np.zeros(
                (len(points), self.template_count),
                dtype=np.float32,
            )
            propagated = {}
            for previous_index, current_index in pairs:
                lamp_id = self.previous_assignments[previous_index]
                previous_confidence = self.previous_confidences[
                    previous_index
                ]
                if (
                    lamp_id >= self.template_count
                    or lamp_id not in active_template_indices
                    or previous_confidence
                    < self.minimum_update_confidence
                ):
                    continue
                propagated[current_index] = lamp_id
                ambiguity_gate = np.clip(
                    (
                        self.confidence_threshold
                        - base_confidences[current_index]
                    )
                    / max(self.confidence_transition_width, 1e-6),
                    0.0,
                    1.0,
                )
                temporal[current_index, lamp_id] += (
                    self.temporal_weight
                    * ambiguity_gate
                    * previous_confidence
                )
            comparable = [
                current_index
                for current_index in propagated
            ]
            if comparable:
                agreement = np.mean(
                    [
                        base_assignments[current_index]
                        == propagated[current_index]
                        for current_index in comparable
                    ]
                )
                if agreement >= self.consensus_reanchor_ratio:
                    self.stable_frames += 1
                elif not self.temporal_active:
                    self.stable_frames = 0
                if self.stable_frames >= self.burn_in_frames:
                    self.temporal_active = True
                frame_gate = np.clip(
                    (
                        self.consensus_reanchor_ratio - agreement
                    )
                    / max(
                        self.consensus_reanchor_ratio
                        - self.consensus_failure_ratio,
                        1e-6,
                    ),
                    0.0,
                    1.0,
                )
                temporal *= float(frame_gate)
                if not self.temporal_active:
                    temporal.fill(0.0)
                elif (
                    len(comparable) >= 4
                    and agreement <= self.consensus_failure_ratio
                ):
                    for current_index, lamp_id in propagated.items():
                        temporal[current_index, lamp_id] = max(
                            float(temporal[current_index, lamp_id]),
                            self.hard_failure_bonus,
                        )
            refined_scores = scores + torch.from_numpy(temporal).to(
                device=scores.device,
                dtype=scores.dtype,
            )

        refined_transport = log_optimal_transport(
            refined_scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        assignments = self._mutual_assignments(refined_transport)
        confidences = self._row_confidences(
            refined_transport,
            assignments,
        )
        self.previous_points = points.copy()
        self.previous_assignments = assignments
        self.previous_confidences = confidences
        return assignments, refined_transport


class ReliabilityGatedTrackletDecoder(TrackletGraphDecoder):
    """Use temporal evidence only when the previous frame is trustworthy.

    The legacy tracklet decoder deliberately applies a strong fallback when
    most current assignments disagree with history. That repairs isolated
    frame failures, but it can also propagate a bad frame through a sequence.
    This decoder instead estimates a continuous frame reliability from motion
    residual, association coverage, prior confidence and static agreement.
    """

    def __init__(
        self,
        *args,
        reliability_min_coverage: float = 0.35,
        reliability_full_coverage: float = 0.75,
        reliability_agreement_floor: float = 0.45,
        reliability_agreement_full: float = 0.80,
        reliability_residual_scale: float = 0.65,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_min_coverage = float(reliability_min_coverage)
        self.reliability_full_coverage = float(reliability_full_coverage)
        self.reliability_agreement_floor = float(
            reliability_agreement_floor
        )
        self.reliability_agreement_full = float(
            reliability_agreement_full
        )
        self.reliability_residual_scale = float(
            reliability_residual_scale
        )
        self.last_diagnostics: dict[str, float | int | bool] = {}

    def reset(self) -> None:
        super().reset()
        self.last_diagnostics = {}

    @staticmethod
    def _linear_ramp(value: float, lower: float, upper: float) -> float:
        return float(
            np.clip((value - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
        )

    def _motion_normalized_previous(
        self,
        points: np.ndarray,
        base_assignments: list[int],
        base_confidences: list[float],
        scene_scale: float,
    ) -> tuple[np.ndarray, str]:
        """Align the previous frame before building temporal edges."""
        if self.previous_points is None:
            return np.empty((0, 2), dtype=np.float64), "none"

        previous_by_id = {
            lamp_id: index
            for index, lamp_id in enumerate(self.previous_assignments)
            if lamp_id < self.template_count
            and self.previous_confidences[index]
            >= self.minimum_update_confidence
        }
        source = []
        target = []
        for current_index, lamp_id in enumerate(base_assignments):
            previous_index = previous_by_id.get(lamp_id)
            if (
                previous_index is None
                or base_confidences[current_index]
                < self.minimum_update_confidence
            ):
                continue
            source.append(self.previous_points[previous_index])
            target.append(points[current_index])

        previous = self.previous_points.copy()
        if len(source) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                np.asarray(source, dtype=np.float32),
                np.asarray(target, dtype=np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=max(3.0, 0.035 * scene_scale),
                maxIters=300,
                confidence=0.995,
                refineIters=20,
            )
            if matrix is not None:
                scale = float(
                    np.sqrt(abs(np.linalg.det(matrix[:, :2])))
                )
                inlier_count = (
                    int(inliers.sum()) if inliers is not None else 0
                )
                if (
                    np.all(np.isfinite(matrix))
                    and 0.65 <= scale <= 1.55
                    and inlier_count >= 2
                ):
                    return (
                        previous @ matrix[:, :2].T + matrix[:, 2],
                        "similarity",
                    )

        if len(source) >= 2:
            translation = np.median(
                np.asarray(target) - np.asarray(source),
                axis=0,
            )
            return previous + translation, "translation"
        return previous, "identity"

    def _frame_reliability(
        self,
        propagated: dict[int, int],
        pairs: list[tuple[int, int]],
        aligned_previous: np.ndarray,
        points: np.ndarray,
        base_assignments: list[int],
        gate_px: float,
    ) -> tuple[float, dict[str, float | int | bool]]:
        valid_previous = sum(
            lamp_id < self.template_count
            and confidence >= self.minimum_update_confidence
            for lamp_id, confidence in zip(
                self.previous_assignments,
                self.previous_confidences,
            )
        )
        denominator = max(1, min(valid_previous, len(points)))
        coverage = len(propagated) / denominator
        coverage_quality = self._linear_ramp(
            coverage,
            self.reliability_min_coverage,
            self.reliability_full_coverage,
        )

        pair_distance = {
            current_index: float(
                np.linalg.norm(
                    aligned_previous[previous_index]
                    - points[current_index]
                )
            )
            for previous_index, current_index in pairs
        }
        residual_ratio = (
            float(np.median([pair_distance[index] for index in propagated]))
            / max(gate_px, 1e-6)
            if propagated
            else float("inf")
        )
        motion_quality = (
            float(
                np.exp(
                    -0.5
                    * (
                        residual_ratio
                        / max(self.reliability_residual_scale, 1e-6)
                    )
                    ** 2
                )
            )
            if np.isfinite(residual_ratio)
            else 0.0
        )

        prior_confidences = [
            self.previous_confidences[previous_index]
            for previous_index, current_index in pairs
            if current_index in propagated
        ]
        prior_confidence = (
            float(np.median(prior_confidences))
            if prior_confidences
            else 0.0
        )
        confidence_quality = self._linear_ramp(
            prior_confidence,
            self.minimum_update_confidence,
            0.90,
        )
        agreement = (
            float(
                np.mean(
                    [
                        base_assignments[index] == lamp_id
                        for index, lamp_id in propagated.items()
                    ]
                )
            )
            if propagated
            else 0.0
        )
        agreement_quality = self._linear_ramp(
            agreement,
            self.reliability_agreement_floor,
            self.reliability_agreement_full,
        )
        reliability = (
            coverage_quality
            * motion_quality
            * confidence_quality
            * agreement_quality
        )
        diagnostics: dict[str, float | int | bool] = {
            "temporal_reliability": float(reliability),
            "association_count": len(propagated),
            "association_coverage": float(coverage),
            "association_residual_ratio": float(residual_ratio),
            "prior_confidence": float(prior_confidence),
            "static_temporal_agreement": float(agreement),
            "temporal_applied": bool(reliability > 1e-6),
        }
        return float(reliability), diagnostics

    def decode(
        self,
        scores: torch.Tensor,
        points: np.ndarray,
        dustbin_score: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        points = np.asarray(points, dtype=np.float64)
        base_transport = log_optimal_transport(
            scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        base_assignments = self._mutual_assignments(base_transport)
        base_confidences = self._row_confidences(
            base_transport,
            base_assignments,
        )
        refined_scores = scores
        self.last_diagnostics = {
            "temporal_reliability": 0.0,
            "association_count": 0,
            "temporal_applied": False,
            "motion_model": "none",
        }

        if self.previous_points is not None and self.temporal_weight > 0.0:
            scene_scale = self._scene_scale(points)
            gate_px = max(4.0, self.sigma_ratio * scene_scale)
            raw_pairs = self._greedy_tracklet_edges(
                self.previous_points,
                points,
                gate_px=gate_px,
            )
            raw_comparable = [
                (current_index, self.previous_assignments[previous_index])
                for previous_index, current_index in raw_pairs
                if self.previous_assignments[previous_index]
                < self.template_count
            ]
            raw_denominator = max(
                1,
                min(len(self.previous_points), len(points)),
            )
            raw_coverage = len(raw_comparable) / raw_denominator
            raw_agreement = (
                float(
                    np.mean(
                        [
                            base_assignments[current_index] == lamp_id
                            for current_index, lamp_id in raw_comparable
                        ]
                    )
                )
                if raw_comparable
                else 0.0
            )
            aligned_previous, motion_model = self._motion_normalized_previous(
                points,
                base_assignments,
                base_confidences,
                scene_scale,
            )
            pairs = self._greedy_tracklet_edges(
                aligned_previous,
                points,
                gate_px=gate_px,
            )
            active = self._active_template_indices(len(points))
            propagated = {}
            for previous_index, current_index in pairs:
                lamp_id = self.previous_assignments[previous_index]
                confidence = self.previous_confidences[previous_index]
                if (
                    lamp_id >= self.template_count
                    or lamp_id not in active
                    or confidence < self.minimum_update_confidence
                ):
                    continue
                propagated[current_index] = lamp_id

            reliability, diagnostics = self._frame_reliability(
                propagated,
                pairs,
                aligned_previous,
                points,
                base_assignments,
                gate_px,
            )
            # A regular array can make a globally permuted identity assignment
            # look like a valid affine camera motion. Raw local edges provide an
            # independent veto when they already cover enough observations.
            raw_conflict = (
                raw_coverage >= self.reliability_min_coverage
                and raw_agreement < self.reliability_agreement_floor
            )
            if raw_conflict:
                reliability = 0.0
                diagnostics["temporal_reliability"] = 0.0
                diagnostics["temporal_applied"] = False
            diagnostics["raw_association_coverage"] = float(raw_coverage)
            diagnostics["raw_static_temporal_agreement"] = float(raw_agreement)
            diagnostics["raw_conflict_veto"] = bool(raw_conflict)
            diagnostics["motion_model"] = motion_model
            self.last_diagnostics = diagnostics
            temporal = np.zeros(
                (len(points), self.template_count),
                dtype=np.float32,
            )
            for current_index, lamp_id in propagated.items():
                ambiguity = np.clip(
                    (
                        self.confidence_threshold
                        - base_confidences[current_index]
                    )
                    / max(self.confidence_transition_width, 1e-6),
                    0.0,
                    1.0,
                )
                previous_index = next(
                    previous
                    for previous, current in pairs
                    if current == current_index
                )
                temporal[current_index, lamp_id] = (
                    self.temporal_weight
                    * reliability
                    * ambiguity
                    * self.previous_confidences[previous_index]
                )
            if self.temporal_active:
                refined_scores = scores + torch.from_numpy(temporal).to(
                    device=scores.device,
                    dtype=scores.dtype,
                )

        refined_transport = log_optimal_transport(
            refined_scores,
            dustbin_score,
            iterations=self.sinkhorn_iterations,
        )
        assignments = self._mutual_assignments(refined_transport)
        confidences = self._row_confidences(
            refined_transport,
            assignments,
        )
        if self.previous_points is None:
            self.stable_frames = 1
        else:
            self.stable_frames += 1
        self.temporal_active = self.stable_frames >= self.burn_in_frames
        self.previous_points = points.copy()
        self.previous_assignments = assignments
        self.previous_confidences = confidences
        return assignments, refined_transport
