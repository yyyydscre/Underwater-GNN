"""Graph features and an optimal-transport GNN correspondence model."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import ArrayConfiguration
from .schema import COLOR_INDEX, LightDetection


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Remove image translation and scale while retaining array shape."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.reshape(0, 2)
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered / max(float(scale), 1e-6)


def knn_edges(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return directed KNN edges with normalized distance and bearing."""
    count = len(points)
    if count < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    normalized = normalize_points(points)
    distances = np.linalg.norm(normalized[:, None] - normalized[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, : min(k, count - 1)]
    source = np.repeat(np.arange(count), neighbors.shape[1])
    target = neighbors.reshape(-1)
    delta = normalized[target] - normalized[source]
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    attributes = np.concatenate([length, delta / np.maximum(length, 1e-6)], axis=1)
    return np.vstack([source, target]).astype(np.int64), attributes.astype(np.float32)


def topology_distance_signature(points: np.ndarray) -> np.ndarray:
    """Return a compact, rotation-invariant node topology descriptor.

    A KNN graph is useful for local message passing, but under missed lamps its
    neighbourhood can change abruptly.  These five quantiles summarize each
    node's distances to *all* currently visible nodes, retaining a stable
    multi-scale structural cue without using image colour or frame history.
    """
    normalized = normalize_points(points)
    count = len(normalized)
    if count < 2:
        return np.zeros((count, 5), dtype=np.float32)
    distances = np.linalg.norm(
        normalized[:, None] - normalized[None, :], axis=-1
    ).astype(np.float32)
    np.fill_diagonal(distances, np.nan)
    return np.nanquantile(
        distances,
        (0.0, 0.25, 0.5, 0.75, 1.0),
        axis=1,
    ).T.astype(np.float32)


def observation_features(
    detections: list[LightDetection],
    color_mode: str = "original",
    feature_mode: str = "legacy",
) -> np.ndarray:
    """Build detector-aware node features for the observed image graph."""
    if not detections:
        return np.empty((0, 10), dtype=np.float32)
    points = np.asarray([item.xy for item in detections], dtype=np.float32)
    normalized = normalize_points(points)
    radius = np.asarray([item.radius for item in detections], dtype=np.float32)
    brightness = np.asarray([item.brightness for item in detections], dtype=np.float32)
    confidence = np.asarray([item.confidence for item in detections], dtype=np.float32)
    center_quality = np.asarray([item.center_quality for item in detections], dtype=np.float32)
    radii = np.log1p(radius / max(float(np.median(radius)), 1e-5))[:, None]
    bright = (brightness / max(float(np.quantile(brightness, 0.9)), 1.0))[:, None]
    unknown_layer = np.full((len(detections), 1), 0.5, dtype=np.float32)
    topology_aware = feature_mode == "geometry_uncertainty_topology"
    if feature_mode in {
        "geometry_uncertainty",
        "geometry_uncertainty_relative",
        "geometry_uncertainty_topology",
    }:
        detector_confidence = np.asarray(
            [
                item.detector_confidence
                if item.detector_confidence is not None
                else item.confidence
                for item in detections
            ],
            dtype=np.float32,
        )
        center_confidence = np.asarray(
            [
                item.center_confidence
                if item.center_confidence is not None
                else item.center_quality
                for item in detections
            ],
            dtype=np.float32,
        )
        uncertainty = []
        for item, quality in zip(detections, center_confidence):
            radius_scale = max(float(item.radius), 1.0)
            if item.center_covariance is None:
                normalized_sigma = 0.05 + 0.95 * (1.0 - float(quality))
                sigma_x = sigma_y = normalized_sigma
                correlation = 0.0
            else:
                covariance = np.asarray(item.center_covariance, dtype=np.float64).reshape(2, 2)
                variance_x = max(float(covariance[0, 0]), 1e-8)
                variance_y = max(float(covariance[1, 1]), 1e-8)
                sigma_x = math.sqrt(variance_x) / radius_scale
                sigma_y = math.sqrt(variance_y) / radius_scale
                correlation = float(
                    covariance[0, 1] / math.sqrt(variance_x * variance_y)
                )
            uncertainty.append(
                [
                    math.log1p(min(max(sigma_x, 0.0), 20.0)) / math.log(21.0),
                    math.log1p(min(max(sigma_y, 0.0), 20.0)) / math.log(21.0),
                    float(np.clip(correlation, -1.0, 1.0)),
                ]
            )
        uncertainty_features = np.asarray(uncertainty, dtype=np.float32)
        if feature_mode == "geometry_uncertainty_relative":
            observation_quality = np.sqrt(
                np.clip(detector_confidence, 0.0, 1.0)
                * np.clip(center_confidence, 0.0, 1.0)
            )

            def rank_feature(values: np.ndarray) -> np.ndarray:
                if len(values) < 2:
                    return np.ones_like(values, dtype=np.float32)
                order = np.argsort(values, kind="stable")
                ranks = np.empty(len(values), dtype=np.float32)
                ranks[order] = np.linspace(
                    0.0,
                    1.0,
                    len(values),
                    dtype=np.float32,
                )
                return ranks

            quality_rank = rank_feature(observation_quality)
            sigma_x_rank = rank_feature(uncertainty_features[:, 0])
            sigma_y_rank = rank_feature(uncertainty_features[:, 1])
            return np.concatenate(
                [
                    normalized,
                    radii,
                    bright,
                    observation_quality[:, None],
                    quality_rank[:, None],
                    sigma_x_rank[:, None],
                    sigma_y_rank[:, None],
                    uncertainty_features[:, 2:3],
                    unknown_layer,
                ],
                axis=1,
            )
        features = np.concatenate(
            [
                normalized,
                radii,
                bright,
                detector_confidence[:, None],
                center_confidence[:, None],
                uncertainty_features,
                unknown_layer,
            ],
            axis=1,
        )
        if topology_aware:
            return np.concatenate(
                [features, topology_distance_signature(points)],
                axis=1,
            )
        return features
    if feature_mode != "legacy":
        raise ValueError(f"Unsupported observation feature mode: {feature_mode!r}")
    if color_mode == "neutral":
        colors = np.full((len(detections), 3), 1.0 / 3.0, dtype=np.float32)
    else:
        colors = np.asarray([item.color_probs for item in detections], dtype=np.float32)
    # Unknown layer is represented by 0.5; the learned graph uses colour and
    # neighbourhood structure to resolve it rather than assuming it up front.
    return np.concatenate([normalized, radii, bright, colors, confidence[:, None], center_quality[:, None], unknown_layer], axis=1)


def template_features(
    configuration: ArrayConfiguration,
    feature_mode: str = "legacy",
) -> np.ndarray:
    """Build graph features for the fixed physical array template."""
    points = np.asarray([lamp.xyz[:2] for lamp in configuration.lights], dtype=np.float32)
    normalized = normalize_points(points)
    colors = np.zeros((len(configuration.lights), 3), dtype=np.float32)
    layer = np.zeros((len(configuration.lights), 1), dtype=np.float32)
    for index, lamp in enumerate(configuration.lights):
        colors[index, COLOR_INDEX.get(lamp.color, COLOR_INDEX["other"])] = 1.0
        layer[index, 0] = 0.0 if lamp.layer == "front" else 1.0
    # Radius and brightness are not physical template attributes.  Presence and
    # centre quality are fixed at one to distinguish reference nodes from noise.
    zeros = np.zeros((len(points), 2), dtype=np.float32)
    ones = np.ones((len(points), 2), dtype=np.float32)
    topology_aware = feature_mode == "geometry_uncertainty_topology"
    if feature_mode in {
        "geometry_uncertainty",
        "geometry_uncertainty_relative",
        "geometry_uncertainty_topology",
    }:
        uncertainty = np.zeros((len(points), 3), dtype=np.float32)
        features = np.concatenate([normalized, zeros, ones, uncertainty, layer], axis=1)
        if topology_aware:
            return np.concatenate(
                [features, topology_distance_signature(points)],
                axis=1,
            )
        return features
    if feature_mode != "legacy":
        raise ValueError(f"Unsupported template feature mode: {feature_mode!r}")
    return np.concatenate([normalized, zeros, colors, ones, layer], axis=1)


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - config inspection without PyTorch
    torch = None
    nn = object
    F = None


class EdgeConvLayer(nn.Module):
    """EdgeConv message passing implemented with base PyTorch only."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.gated = gated
        message_dim = channels * 2 + 3
        self.message = nn.Sequential(
            nn.Linear(message_dim, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )
        self.gate = (
            nn.Sequential(nn.Linear(message_dim, channels), nn.Sigmoid())
            if gated
            else None
        )
        self.update = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(inplace=True))

    def forward(self, nodes, edges, edge_attrs):
        if edges.numel() == 0:
            return nodes
        source, target = edges[0], edges[1]
        message_input = torch.cat(
            [nodes[source], nodes[target] - nodes[source], edge_attrs],
            dim=-1,
        )
        message = self.message(message_input)
        if self.gate is not None:
            message = message * self.gate(message_input)
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, source, message)
        degree = torch.bincount(source, minlength=nodes.size(0)).clamp_min(1).unsqueeze(1)
        return self.update(torch.cat([nodes, aggregate / degree], dim=-1))


class CrossGraphLayer(nn.Module):
    """Bidirectional cross attention between observed and template graphs."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.observation_attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.template_attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.observation_norm = nn.LayerNorm(channels)
        self.template_norm = nn.LayerNorm(channels)
        self.observation_ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(channels * 2, channels))
        self.template_ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(channels * 2, channels))

    def forward(self, observation, template):
        observed, _ = self.observation_attention(observation[None], template[None], template[None], need_weights=False)
        templated, _ = self.template_attention(template[None], observation[None], observation[None], need_weights=False)
        observation = self.observation_norm(observation + observed[0])
        template = self.template_norm(template + templated[0])
        return observation + self.observation_ffn(observation), template + self.template_ffn(template)


class GlobalContextLayer(nn.Module):
    """Pre-normalized self-attention over all nodes in one small graph."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes):
        normalized = self.attention_norm(nodes)
        context, _ = self.attention(
            normalized[None],
            normalized[None],
            normalized[None],
            need_weights=False,
        )
        nodes = nodes + self.dropout(context[0])
        return nodes + self.dropout(self.ffn(self.ffn_norm(nodes)))


class GuidingLightGNN(nn.Module):
    """GNN with cross-graph attention and a learned unmatched-node score.

    Node messages encode each graph's local lamp geometry.  Cross attention then
    lets an observed node compare itself with the complete 13-light template.
    Matching is finalized by Sinkhorn optimal transport, so nuisance highlights
    can flow to a dustbin rather than being forced to a lamp ID.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        layers: int = 4,
        cross_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.08,
        dustbin_score: float = 0.8,
        edge_gating: bool = False,
        global_context_layers: int = 0,
        symmetric_pair_scoring: bool = False,
        cosine_similarity: bool = False,
        visibility_aware: bool = False,
        visibility_score_weight: float = 0.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for the GNN matcher.")
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.edge_gating = edge_gating
        self.symmetric_pair_scoring = bool(symmetric_pair_scoring)
        self.cosine_similarity = bool(cosine_similarity)
        self.visibility_aware = bool(visibility_aware)
        self.visibility_score_weight = float(visibility_score_weight)
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.LayerNorm(hidden_dim))
        self.edge_layers = nn.ModuleList(
            EdgeConvLayer(hidden_dim, dropout, gated=edge_gating)
            for _ in range(layers)
        )
        self.edge_norms = (
            nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
            if edge_gating
            else None
        )
        self.cross_layers = nn.ModuleList(CrossGraphLayer(hidden_dim, heads, dropout) for _ in range(cross_layers))
        self.global_context_layers = nn.ModuleList(
            GlobalContextLayer(hidden_dim, heads, dropout)
            for _ in range(int(global_context_layers))
        )
        self.observation_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.template_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if self.symmetric_pair_scoring:
            self.pair_bias = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.pair_bias = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        self.log_similarity_scale = (
            nn.Parameter(torch.tensor(math.log(10.0)))
            if self.cosine_similarity
            else None
        )
        self.template_missing_head = (
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if self.visibility_aware
            else None
        )
        self.bin_score = nn.Parameter(torch.tensor(float(dustbin_score)))

    def encode(self, features, edges, edge_attrs):
        nodes = self.input(features)
        for index, layer in enumerate(self.edge_layers):
            nodes = nodes + layer(nodes, edges, edge_attrs)
            if self.edge_norms is not None:
                nodes = self.edge_norms[index](nodes)
        return nodes

    def forward_with_aux(self, observation, observation_edges, observation_attrs, template, template_edges, template_attrs):
        observed = self.encode(observation, observation_edges, observation_attrs)
        templated = self.encode(template, template_edges, template_attrs)
        for context_layer in self.global_context_layers:
            observed = context_layer(observed)
            templated = context_layer(templated)
        for cross_layer in self.cross_layers:
            observed, templated = cross_layer(observed, templated)
        projected_observed = self.observation_projection(observed)
        projected_template = self.template_projection(templated)
        if self.cosine_similarity:
            projected_observed = F.normalize(projected_observed, dim=-1)
            projected_template = F.normalize(projected_template, dim=-1)
            scale = self.log_similarity_scale.exp().clamp(1.0, 100.0)
            similarity = scale * (
                projected_observed @ projected_template.transpose(0, 1)
            )
        else:
            similarity = (
                projected_observed @ projected_template.transpose(0, 1)
                / math.sqrt(projected_observed.shape[-1])
            )
        expanded_observed = observed[:, None, :].expand(-1, templated.size(0), -1)
        expanded_template = templated[None, :, :].expand(observed.size(0), -1, -1)
        pair_parts = [expanded_observed, expanded_template]
        if self.symmetric_pair_scoring:
            pair_parts.extend(
                [
                    torch.abs(expanded_observed - expanded_template),
                    expanded_observed * expanded_template,
                ]
            )
        pairs = torch.cat(pair_parts, dim=-1)
        scores = similarity + self.pair_bias(pairs).squeeze(-1)
        missing_logits = None
        if self.template_missing_head is not None:
            pooled_observation = observed.mean(dim=0, keepdim=True).expand_as(
                templated
            )
            visibility_features = torch.cat(
                [
                    templated,
                    pooled_observation,
                    torch.abs(templated - pooled_observation),
                    templated * pooled_observation,
                ],
                dim=-1,
            )
            # Positive logits mean that a template lamp is absent.  Penalizing
            # the whole template column lets Sinkhorn send it to the dustbin.
            missing_logits = self.template_missing_head(
                visibility_features
            ).squeeze(-1)
            scores = scores - self.visibility_score_weight * missing_logits[None, :]
        return scores, missing_logits

    def forward(self, observation, observation_edges, observation_attrs, template, template_edges, template_attrs):
        scores, _ = self.forward_with_aux(
            observation,
            observation_edges,
            observation_attrs,
            template,
            template_edges,
            template_attrs,
        )
        return scores


def log_optimal_transport(scores, dustbin_score, iterations: int = 50):
    """Compute SuperGlue-style log Sinkhorn transport with a reject bin."""
    observations, templates = scores.shape
    bins_observation = dustbin_score.expand(observations, 1)
    bins_template = dustbin_score.expand(1, templates)
    bins_both = dustbin_score.expand(1, 1)
    coupling = torch.cat(
        [torch.cat([scores, bins_observation], dim=1), torch.cat([bins_template, bins_both], dim=1)],
        dim=0,
    )
    normalizer = -torch.tensor(float(observations + templates), dtype=scores.dtype, device=scores.device).log()
    log_mu = torch.full((observations + 1,), normalizer, dtype=scores.dtype, device=scores.device)
    log_nu = torch.full((templates + 1,), normalizer, dtype=scores.dtype, device=scores.device)
    log_mu[-1] = math.log(float(templates)) + normalizer
    log_nu[-1] = math.log(float(observations)) + normalizer
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(coupling + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(coupling + u.unsqueeze(1), dim=0)
    return coupling + u.unsqueeze(1) + v.unsqueeze(0) - normalizer


def load_gnn_checkpoint(path: str | Path, device: str = "cpu") -> GuidingLightGNN:
    if torch is None:
        raise RuntimeError("PyTorch is required for --matcher gnn.")
    checkpoint = torch.load(str(path), map_location=device)
    settings = checkpoint.get("model", {})
    model = GuidingLightGNN(**settings).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.observation_color_mode = checkpoint.get("observation_color_mode", "original")
    model.observation_feature_mode = checkpoint.get(
        "observation_feature_mode",
        "legacy",
    )
    model.color_bias = float(checkpoint.get("color_bias", 1.35))
    model.eval()
    return model
