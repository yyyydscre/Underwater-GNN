"""Pre-train the reject-aware GNN on incomplete, noisy projected arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import trange

from .config import load_array_configuration
from .graph import GuidingLightGNN, knn_edges, log_optimal_transport, observation_features, template_features
from .schema import COLOR_INDEX, LightDetection


def synthetic_detections(configuration, rng: np.random.Generator) -> tuple[list[LightDetection], np.ndarray]:
    """Simulate occlusion, false highlights, colour drift and keypoint noise."""
    object_points = np.asarray([lamp.xyz for lamp in configuration.lights], dtype=np.float64)
    rvec = rng.normal(0.0, [0.20, 0.20, 0.45]).astype(np.float64).reshape(3, 1)
    tvec = np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.08, 0.08), rng.uniform(1.4, 4.0)], dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, configuration.camera_matrix, configuration.distortion)
    projected = projected.reshape(-1, 2)
    keep = rng.random(len(object_points)) > rng.uniform(0.05, 0.30)
    if keep.sum() < 8:
        keep[rng.choice(len(object_points), size=8, replace=False)] = True
    detections: list[LightDetection] = []
    labels: list[int] = []
    for template_index in np.flatnonzero(keep):
        lamp = configuration.lights[int(template_index)]
        noise = rng.normal(0.0, rng.uniform(0.4, 2.6), 2)
        probs = np.full(3, 0.05, dtype=np.float32)
        probs[COLOR_INDEX[lamp.color]] = rng.uniform(0.72, 0.97)
        probs += rng.normal(0.0, 0.06, 3).astype(np.float32)
        probs = np.clip(probs, 0.001, None)
        probs /= probs.sum()
        center_quality = float(rng.uniform(0.55, 1.0))
        detections.append(
            LightDetection(
                projected[template_index] + noise,
                float(rng.uniform(0.55, 0.99)),
                probs,
                lamp.color,
                float(rng.uniform(4.0, 11.0)),
                float(rng.uniform(130.0, 255.0)),
                "synthetic",
                center_quality,
            )
        )
        labels.append(int(template_index))
    center = projected.mean(axis=0)
    span = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1])), 80.0)
    for _ in range(int(rng.integers(0, 7))):
        xy = center + rng.uniform(-1.3 * span, 1.3 * span, 2)
        probs = rng.dirichlet([1.4, 1.4, 1.2]).astype(np.float32)
        detections.append(
            LightDetection(
                xy,
                float(rng.uniform(0.18, 0.75)),
                probs,
                ("green", "blue", "other")[int(np.argmax(probs))],
                float(rng.uniform(3.0, 14.0)),
                float(rng.uniform(50.0, 240.0)),
                "synthetic_outlier",
                float(rng.uniform(0.05, 0.65)),
            )
        )
        labels.append(-1)
    order = rng.permutation(len(detections))
    return [detections[int(index)] for index in order], np.asarray([labels[int(index)] for index in order], dtype=np.int64)


def transport_loss(log_transport, labels: np.ndarray, template_count: int):
    """NLL for observed matches/outliers plus missed template nodes."""
    device = log_transport.device
    observation_indices = torch.arange(len(labels), device=device)
    target = torch.from_numpy(np.where(labels >= 0, labels, template_count)).to(device)
    loss = -log_transport[observation_indices, target].mean()
    missed = np.setdiff1d(np.arange(template_count), labels[labels >= 0], assume_unique=False)
    if len(missed):
        missed_indices = torch.from_numpy(missed).to(device)
        loss = loss - 0.35 * log_transport[-1, missed_indices].mean()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn_matcher.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=240)
    parser.add_argument("--output", default="models/gnn_matcher.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with Path(args.gnn_config).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    model_settings = settings["model"]
    training = settings["training"]
    configuration = load_array_configuration(args.array_config)
    model = GuidingLightGNN(**model_settings).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    template = torch.from_numpy(template_features(configuration)).to(args.device)
    template_points = np.asarray([lamp.xyz[:2] for lamp in configuration.lights], dtype=np.float32)
    template_edges_np, template_attrs_np = knn_edges(template_points, configuration.k_neighbors)
    template_edges = torch.from_numpy(template_edges_np).to(args.device)
    template_attrs = torch.from_numpy(template_attrs_np).to(args.device)
    rng = np.random.default_rng(args.seed)
    for epoch in trange(args.epochs, desc="GNN epochs"):
        model.train()
        losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            detections, labels = synthetic_detections(configuration, rng)
            points = np.asarray([item.xy for item in detections], dtype=np.float32)
            edges_np, attrs_np = knn_edges(points, configuration.k_neighbors)
            logits = model(
                torch.from_numpy(observation_features(detections)).to(args.device),
                torch.from_numpy(edges_np).to(args.device),
                torch.from_numpy(attrs_np).to(args.device),
                template,
                template_edges,
                template_attrs,
            )
            log_transport = log_optimal_transport(logits, model.bin_score, int(training.get("sinkhorn_iterations", 60)))
            loss = transport_loss(log_transport, labels, len(configuration.lights))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch + 1} loss={np.mean(losses):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": model_settings,
            "array_config": str(args.array_config),
            "training": "synthetic_projected_partial_outlier_ot",
        },
        path,
    )
    print(f"Saved GNN checkpoint: {path}")


if __name__ == "__main__":
    main()
