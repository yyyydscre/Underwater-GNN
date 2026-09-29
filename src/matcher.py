"""Layer-aware lamp correspondence with geometry and optional learned GNN scores."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ArrayConfiguration
from .graph import knn_edges, load_gnn_checkpoint, log_optimal_transport, observation_features, template_features
from .schema import COLOR_INDEX, LampMatch, LightDetection
from .temporal_gnn import (
    ReliabilityGatedTrackletDecoder,
    TrackletGraphDecoder,
)


@dataclass
class MatchResult:
    matches: list[LampMatch]
    method: str
    mean_cost: float
    missing_lamp_ids: list[str] | None = None
    recovered_detections: list[LightDetection] | None = None
    # Runtime-only GNN evidence used by pose-guided multi-hypothesis reranking.
    # These arrays are intentionally not serialized into annotation JSONL.
    candidate_log_scores: np.ndarray | None = None
    observation_dustbin_scores: np.ndarray | None = None
    template_dustbin_scores: np.ndarray | None = None
    candidate_template_ids: list[str] | None = None
    rerank_diagnostics: dict | None = None


def _similarity_from_pairs(source_a: np.ndarray, source_b: np.ndarray, target_a: np.ndarray, target_b: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return the direct similarity transform induced by two point correspondences."""
    source_delta = source_b - source_a
    target_delta = target_b - target_a
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))
    if source_length < 1e-7 or target_length < 1e-7:
        return None
    scale = target_length / source_length
    cosine = float(np.dot(source_delta, target_delta) / (source_length * target_length))
    sine = float((source_delta[0] * target_delta[1] - source_delta[1] * target_delta[0]) / (source_length * target_length))
    rotation = scale * np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_a - rotation @ source_a
    return np.column_stack([rotation, translation]), scale


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _minimum_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))


def _greedy_association(predicted: np.ndarray, candidates: np.ndarray, gate_px: float) -> tuple[list[tuple[int, int]], list[float]]:
    """Associate a small template to candidate peaks without reusing observations."""
    if len(predicted) == 0 or len(candidates) == 0:
        return [], []
    costs = np.linalg.norm(predicted[:, None, :] - candidates[None, :, :], axis=-1)
    pairs: list[tuple[int, int]] = []
    errors: list[float] = []
    used_template: set[int] = set()
    used_candidate: set[int] = set()
    for flat_index in np.argsort(costs, axis=None):
        template_index, candidate_index = np.unravel_index(flat_index, costs.shape)
        error = float(costs[template_index, candidate_index])
        if error > gate_px:
            break
        if template_index in used_template or candidate_index in used_candidate:
            continue
        used_template.add(int(template_index))
        used_candidate.add(int(candidate_index))
        pairs.append((int(template_index), int(candidate_index)))
        errors.append(error)
    return pairs, errors


class LampArrayMatcher:
    """Matches 2-D centers to the physical template while retaining a GNN path."""

    def __init__(
        self,
        configuration: ArrayConfiguration,
        mode: str = "geometry",
        weights: str | Path | None = None,
        device: str = "cpu",
        temporal_settings: dict | None = None,
        missing_expert_weights: str | Path | None = None,
        missing_expert_max_detections: int = 8,
    ) -> None:
        self.configuration = configuration
        self.mode = mode
        self.gnn = (
            load_gnn_checkpoint(weights, device)
            if mode in {"gnn", "gnn_temporal"} and weights
            else None
        )
        self.device = device
        self.missing_expert = (
            load_gnn_checkpoint(missing_expert_weights, device)
            if self.gnn is not None and missing_expert_weights
            else None
        )
        self.missing_expert_max_detections = int(
            missing_expert_max_detections
        )
        if mode in {"gnn", "gnn_temporal"} and self.gnn is None:
            raise ValueError(
                "--matcher-weights is required for a GNN matcher."
            )
        self.temporal_decoder = None
        if mode == "gnn_temporal":
            settings = {
                "temporal_weight": 24.0,
                "association_gate_ratio": 0.50,
                "minimum_update_confidence": 0.0,
                "confidence_threshold": 1.0,
                "burn_in_frames": 0,
            }
            settings.update(temporal_settings or {})
            decoder_name = settings.pop("decoder", "tracklets")
            decoder_class = (
                ReliabilityGatedTrackletDecoder
                if decoder_name == "reliability_tracklets"
                else TrackletGraphDecoder
            )
            self.temporal_decoder = decoder_class(
                template_count=len(configuration.lights),
                **settings,
            )

    def reset(self) -> None:
        if self.temporal_decoder is not None:
            self.temporal_decoder.reset()

    def match(self, detections: list[LightDetection]) -> MatchResult:
        if not detections:
            return MatchResult([], self.mode, float("inf"))
        if self.gnn is not None:
            return self._match_gnn(detections)
        return self._match_geometry(detections)

    def _candidate_indices(self, detections: list[LightDetection], layer: str, maximum: int = 24) -> list[int]:
        expected = self.configuration.expected_colors.get(layer, "other")
        column = COLOR_INDEX.get(expected, COLOR_INDEX["other"])
        scores = np.asarray([item.color_probs[column] * item.confidence * item.center_quality for item in detections])
        preferred = [index for index, score in enumerate(scores) if score >= 0.08]
        candidates = preferred if preferred else list(range(len(detections)))
        return sorted(candidates, key=lambda index: scores[index], reverse=True)[:maximum]

    @staticmethod
    def _best_front_hypothesis(template_xy: np.ndarray, candidate_xy: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[float], float] | None:
        """RANSAC over two-point similarities for the asymmetric seven-light U shape."""
        if len(template_xy) < 2 or len(candidate_xy) < 2:
            return None
        spacing = _minimum_spacing(template_xy)
        best: tuple[np.ndarray, list[tuple[int, int]], list[float], float] | None = None
        best_score = (-1, float("inf"))
        for source_a in range(len(template_xy)):
            for source_b in range(source_a + 1, len(template_xy)):
                for target_a in range(len(candidate_xy)):
                    for target_b in range(len(candidate_xy)):
                        if target_a == target_b:
                            continue
                        result = _similarity_from_pairs(template_xy[source_a], template_xy[source_b], candidate_xy[target_a], candidate_xy[target_b])
                        if result is None:
                            continue
                        matrix, scale = result
                        if not 15.0 <= scale <= 3000.0:
                            continue
                        gate_px = max(8.0, 0.34 * scale * spacing)
                        pairs, errors = _greedy_association(_transform(template_xy, matrix), candidate_xy, gate_px)
                        if not errors:
                            continue
                        score = (len(pairs), float(np.mean(errors)))
                        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                            best, best_score = (matrix, pairs, errors, gate_px), score
        return best

    @staticmethod
    def _refine_similarity(template_xy: np.ndarray, candidate_xy: np.ndarray, pairs: list[tuple[int, int]], fallback: np.ndarray) -> np.ndarray:
        if len(pairs) < 2:
            return fallback
        source = np.asarray([template_xy[template_index] for template_index, _ in pairs], dtype=np.float32)
        target = np.asarray([candidate_xy[candidate_index] for _, candidate_index in pairs], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
        return matrix.astype(np.float64) if matrix is not None else fallback

    @staticmethod
    def _best_rear_hypothesis(template_xy: np.ndarray, candidate_xy: np.ndarray, front_matrix: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[float]] | None:
        """Fit the rear 2-by-3 grid with an affine RANSAC and front-layer prior.

        The rear plane is farther from the camera, so its image scale need not
        equal the front U shape's scale. A three-point affine hypothesis models
        that perspective-induced anisotropy while the front prediction resolves
        the otherwise symmetric grid orientation.
        """
        if len(template_xy) < 5 or len(candidate_xy) < 5:
            return None
        anchor_indices = (0, 1, 4)
        source = template_xy[np.asarray(anchor_indices)].astype(np.float32)
        front_prediction = _transform(template_xy, front_matrix)
        best: tuple[np.ndarray, list[tuple[int, int]], list[float]] | None = None
        best_score = (-1, float("inf"))
        candidate_count = len(candidate_xy)
        for first in range(candidate_count):
            for second in range(candidate_count):
                if second == first:
                    continue
                for third in range(candidate_count):
                    if third == first or third == second:
                        continue
                    target = candidate_xy[[first, second, third]].astype(np.float32)
                    matrix = cv2.getAffineTransform(source, target).astype(np.float64)
                    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix[:, :2]))) < 1e-3:
                        continue
                    predicted = _transform(template_xy, matrix)
                    gate_px = max(9.0, 0.34 * _minimum_spacing(predicted))
                    pairs, errors = _greedy_association(predicted, candidate_xy, gate_px)
                    if len(pairs) < 5:
                        continue
                    prior_error = float(np.mean(np.linalg.norm(predicted - front_prediction, axis=1)))
                    score = (len(pairs), float(np.mean(errors)) + 0.12 * prior_error)
                    if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                        best, best_score = (matrix, pairs, errors), score
        return best

    def _match_geometry(self, detections: list[LightDetection]) -> MatchResult:
        front_lamps = self.configuration.by_layer("front")
        rear_lamps = self.configuration.by_layer("rear")
        front_candidates = self._candidate_indices(detections, "front")
        rear_candidates = self._candidate_indices(detections, "rear")
        if len(front_candidates) < 5 or len(rear_candidates) < 4:
            return MatchResult([], "topology_ransac", float("inf"))
        front_xy = np.asarray([lamp.xyz[:2] for lamp in front_lamps], dtype=np.float64)
        front_observations = np.asarray([detections[index].xy for index in front_candidates], dtype=np.float64)
        hypothesis = self._best_front_hypothesis(front_xy, front_observations)
        if hypothesis is None:
            return MatchResult([], "topology_ransac", float("inf"))
        matrix, front_pairs, front_errors, front_gate = hypothesis
        matrix = self._refine_similarity(front_xy, front_observations, front_pairs, matrix)
        front_pairs, front_errors = _greedy_association(_transform(front_xy, matrix), front_observations, front_gate)
        rear_xy = np.asarray([lamp.xyz[:2] for lamp in rear_lamps], dtype=np.float64)
        rear_observations = np.asarray([detections[index].xy for index in rear_candidates], dtype=np.float64)
        rear_hypothesis = self._best_rear_hypothesis(rear_xy, rear_observations, matrix)
        if rear_hypothesis is None:
            return MatchResult([], "topology_ransac", float("inf"))
        _, rear_pairs, rear_errors = rear_hypothesis
        # A valid array must contain most of the front U and a visible rear grid.
        if len(front_pairs) < 5 or len(rear_pairs) < 4 or len(front_pairs) + len(rear_pairs) < 10:
            return MatchResult([], "topology_ransac", float("inf"))
        matches: list[LampMatch] = []
        costs: list[float] = []
        for lamps, candidates, pairs, errors in (
            (front_lamps, front_candidates, front_pairs, front_errors),
            (rear_lamps, rear_candidates, rear_pairs, rear_errors),
        ):
            for (template_index, candidate_index), error in zip(pairs, errors):
                lamp = lamps[template_index]
                detection_index = candidates[candidate_index]
                detection = detections[detection_index]
                color_probability = float(detection.color_probs[COLOR_INDEX.get(lamp.color, 2)])
                confidence = color_probability * detection.confidence * detection.center_quality * np.exp(-error / max(front_gate, 1.0))
                matches.append(LampMatch(lamp, detection_index, detection.xy, float(confidence), float(error)))
                costs.append(float(error))
        return MatchResult(matches, "topology_ransac", float(np.mean(costs)) if costs else float("inf"))

    def _match_gnn(self, detections: list[LightDetection]) -> MatchResult:
        import torch

        if len(detections) < 4:
            return MatchResult(
                [],
                "gnn_insufficient_observations",
                float("inf"),
                list(self.configuration.ids),
            )
        points = np.asarray([item.xy for item in detections], dtype=np.float32)
        use_missing_expert = (
            self.missing_expert is not None
            and len(detections) <= self.missing_expert_max_detections
        )
        active_gnn = self.missing_expert if use_missing_expert else self.gnn
        template_points = np.asarray([lamp.xyz[:2] for lamp in self.configuration.lights], dtype=np.float32)
        obs_edges, obs_attrs = knn_edges(points, self.configuration.k_neighbors)
        tpl_edges, tpl_attrs = knn_edges(template_points, self.configuration.k_neighbors)
        with torch.no_grad():
            logits = active_gnn(
                torch.from_numpy(
                    observation_features(
                        detections,
                        getattr(active_gnn, "observation_color_mode", "original"),
                        getattr(active_gnn, "observation_feature_mode", "legacy"),
                    )
                ).to(self.device),
                torch.from_numpy(obs_edges).to(self.device),
                torch.from_numpy(obs_attrs).to(self.device),
                torch.from_numpy(
                    template_features(
                        self.configuration,
                        getattr(active_gnn, "observation_feature_mode", "legacy"),
                    )
                ).to(self.device),
                torch.from_numpy(tpl_edges).to(self.device),
                torch.from_numpy(tpl_attrs).to(self.device),
            )
            color_compatibility = torch.tensor(
                [[item.color_probs[COLOR_INDEX.get(lamp.color, 2)] for lamp in self.configuration.lights] for item in detections],
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = logits + float(getattr(active_gnn, "color_bias", 1.35)) * torch.log(
                color_compatibility.clamp_min(1e-4)
            )
            if self.temporal_decoder is None:
                transport = log_optimal_transport(
                    logits,
                    active_gnn.bin_score,
                    iterations=60,
                )
            else:
                _, transport = self.temporal_decoder.decode(
                    logits,
                    points,
                    active_gnn.bin_score,
                )
            pair_transport = transport[:-1, :-1]
            observation_choice = torch.argmax(transport[:-1], dim=1).cpu().numpy()
            template_choice = torch.argmax(transport[:, :-1], dim=0).cpu().numpy()
            pair_transport = pair_transport.cpu().numpy()
            transport = transport.cpu().numpy()
        matches = []
        for observation_index, template_index in enumerate(observation_choice):
            if template_index >= len(self.configuration.lights) or template_choice[template_index] != observation_index:
                continue
            # A match must beat both observation and template dustbins.  This
            # rejects reflections/missing lights instead of forcing 13 IDs.
            dustbin_log_probability = max(transport[observation_index, -1], transport[-1, template_index])
            margin = float(pair_transport[observation_index, template_index] - dustbin_log_probability)
            mutual_confidence = 1.0 / (1.0 + np.exp(-2.5 * margin))
            color_probability = float(color_compatibility[observation_index, template_index].cpu())
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
            observation_quality = np.sqrt(
                np.clip(detector_quality, 1e-6, 1.0)
                * np.clip(center_quality, 1e-6, 1.0)
            )
            color_factor = (
                1.0
                if getattr(active_gnn, "observation_color_mode", "original") == "neutral"
                else np.sqrt(color_probability)
            )
            confidence = float(
                mutual_confidence * color_factor * observation_quality
            )
            if confidence >= self.configuration.min_match_confidence:
                matches.append(
                    LampMatch(
                        self.configuration.lights[int(template_index)],
                        int(observation_index),
                        detection.xy,
                        confidence,
                        float(-pair_transport[observation_index, template_index]),
                    )
                )
        matched_ids = {item.lamp.lamp_id for item in matches}
        return MatchResult(
            matches,
            (
                "gnn_temporal_missing_expert"
                if self.temporal_decoder is not None and use_missing_expert
                else "gnn_temporal_tracklets"
                if self.temporal_decoder is not None
                else "gnn_missing_expert"
                if use_missing_expert
                else "gnn_optimal_transport"
            ),
            float(np.mean([item.geometric_cost for item in matches])) if matches else float("inf"),
            [
                lamp_id
                for lamp_id in self.configuration.ids
                if lamp_id not in matched_ids
            ],
            candidate_log_scores=np.asarray(pair_transport, dtype=np.float64),
            observation_dustbin_scores=np.asarray(
                transport[:-1, -1], dtype=np.float64
            ),
            template_dustbin_scores=np.asarray(
                transport[-1, :-1], dtype=np.float64
            ),
            candidate_template_ids=list(self.configuration.ids),
        )
