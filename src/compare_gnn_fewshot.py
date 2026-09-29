"""Pilot comparison for few-shot dual-layer guiding-light graph matching.

This experiment intentionally uses a shifted synthetic target domain.  It does
not replace evaluation on manually labelled real F0-F6/R0-R5 sequences, but it
does provide a reproducible answer to a practical question: whether synthetic
pretraining makes the current matcher easier to adapt with only a few labelled
frames.

Run from the project root:
    python -m src.compare_gnn_fewshot --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import load_array_configuration
from .graph import GuidingLightGNN, knn_edges, log_optimal_transport, observation_features, template_features
from .schema import COLOR_INDEX, LightDetection


@dataclass(frozen=True)
class SyntheticDomain:
    """Appearance/noise parameters for one simulated camera domain."""

    keep_probability: tuple[float, float]
    pixel_noise: tuple[float, float]
    center_quality: tuple[float, float]
    color_confidence: tuple[float, float]
    outliers: tuple[int, int]
    rotation_scale: float
    depth_range: tuple[float, float]


SOURCE_DOMAIN = SyntheticDomain(
    keep_probability=(0.78, 0.98),
    pixel_noise=(0.35, 1.8),
    center_quality=(0.68, 1.0),
    color_confidence=(0.78, 0.98),
    outliers=(0, 4),
    rotation_scale=0.22,
    depth_range=(1.5, 3.8),
)

# More missed lamps, less certain white cores, ambiguous colours, and clutter.
TARGET_DOMAIN = SyntheticDomain(
    keep_probability=(0.55, 0.9),
    pixel_noise=(1.4, 5.2),
    center_quality=(0.28, 0.86),
    color_confidence=(0.48, 0.88),
    outliers=(2, 9),
    rotation_scale=0.42,
    depth_range=(1.25, 5.2),
)


def _colour_probabilities(colour: str, confidence: float) -> np.ndarray:
    remainder = (1.0 - confidence) / 2.0
    probabilities = np.full(3, remainder, dtype=np.float32)
    probabilities[COLOR_INDEX.get(colour, COLOR_INDEX["other"])] = confidence
    return probabilities


def sample_domain(
    configuration,
    domain: SyntheticDomain,
    rng: np.random.Generator,
) -> tuple[list[LightDetection], list[int]]:
    """Create a partially observed lamp graph with known correspondence IDs."""

    template = np.asarray([lamp.xyz for lamp in configuration.lights], dtype=np.float64)
    rvec = rng.normal(0.0, domain.rotation_scale, 3).astype(np.float32)
    tvec = np.array(
        [
            rng.uniform(-0.16, 0.16),
            rng.uniform(-0.12, 0.12),
            rng.uniform(*domain.depth_range),
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(template, rvec, tvec, configuration.camera_matrix, configuration.distortion)
    projected = projected.reshape(-1, 2)

    detections: list[LightDetection] = []
    labels: list[int] = []
    for index, (point, lamp) in enumerate(zip(projected, configuration.lights)):
        if rng.random() > rng.uniform(*domain.keep_probability):
            continue

        center_quality = float(rng.uniform(*domain.center_quality))
        base_noise = float(rng.uniform(*domain.pixel_noise))
        # Lower white-core confidence explicitly means a noisier observed centre.
        point_noise = base_noise * (1.05 + 1.6 * (1.0 - center_quality))
        xy = point + rng.normal(0.0, point_noise, 2)
        colour_confidence = float(rng.uniform(*domain.color_confidence))
        radius = float(rng.uniform(3.0, 10.0))
        detections.append(
            LightDetection(
                xy=(float(xy[0]), float(xy[1])),
                confidence=float(0.45 * center_quality + 0.55 * colour_confidence),
                color_probs=_colour_probabilities(lamp.color, colour_confidence),
                color=lamp.color,
                radius=radius,
                brightness=float(rng.uniform(0.45, 1.0)),
                source="synthetic_target_or_source",
                center_quality=center_quality,
            )
        )
        labels.append(index)

    for _ in range(int(rng.integers(domain.outliers[0], domain.outliers[1] + 1))):
        wrong_colour = str(rng.choice(["green", "blue", "other"]))
        detections.append(
            LightDetection(
                xy=(float(rng.uniform(60.0, 1220.0)), float(rng.uniform(40.0, 680.0))),
                confidence=float(rng.uniform(0.2, 0.65)),
                color_probs=_colour_probabilities(wrong_colour, float(rng.uniform(0.36, 0.72))),
                color=wrong_colour,
                radius=float(rng.uniform(2.0, 12.0)),
                brightness=float(rng.uniform(0.15, 0.9)),
                source="synthetic_outlier",
                center_quality=float(rng.uniform(0.08, 0.62)),
            )
        )
        labels.append(-1)

    order = rng.permutation(len(detections))
    return [detections[int(index)] for index in order], [labels[int(index)] for index in order]


def _build_observation_graph(
    detections: list[LightDetection],
    k_neighbors: int,
    device: torch.device,
    feature_mode: str = "legacy",
):
    points = np.asarray([detection.xy for detection in detections], dtype=np.float32)
    node_features = observation_features(detections, feature_mode=feature_mode)
    edge_index, edge_attr = knn_edges(points, k_neighbors)
    return (
        torch.from_numpy(node_features).to(device),
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_attr).to(device),
    )


def _template_graph(
    configuration,
    device: torch.device,
    feature_mode: str = "legacy",
):
    template = np.asarray([lamp.xyz[:2] for lamp in configuration.lights], dtype=np.float32)
    node_features = template_features(configuration, feature_mode=feature_mode)
    edge_index, edge_attr = knn_edges(template, configuration.k_neighbors)
    return (
        torch.from_numpy(node_features).to(device),
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_attr).to(device),
    )


def _transport_loss(
    log_transport: torch.Tensor,
    labels: list[int],
    template_count: int,
    layer_loss_weight: float = 0.0,
    front_count: int = 7,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 2.0,
    missing_template_weight: float = 0.3,
    missing_logits: torch.Tensor | None = None,
    visibility_loss_weight: float = 0.0,
    frame_hard_weight: float = 0.0,
    frame_hard_temperature: float = 0.5,
) -> torch.Tensor:
    target = torch.tensor(
        [label if label >= 0 else template_count for label in labels],
        dtype=torch.long,
        device=log_transport.device,
    )
    row_losses = F.nll_loss(
        log_transport[:-1, :],
        target,
        reduction="none",
    )
    loss = row_losses.mean()
    if frame_hard_weight > 0.0 and row_losses.numel() > 1:
        temperature = max(float(frame_hard_temperature), 1e-3)
        hard_weights = torch.softmax(
            row_losses.detach() / temperature,
            dim=0,
        )
        loss = loss + frame_hard_weight * torch.sum(
            hard_weights * row_losses
        )
    matched = {label for label in labels if label >= 0}
    if matched:
        missing = torch.tensor(
            [index for index in range(template_count) if index not in matched],
            dtype=torch.long,
            device=log_transport.device,
        )
        if missing.numel():
            loss = loss + missing_template_weight * (
                -log_transport[-1, missing].mean()
            )
    if missing_logits is not None and visibility_loss_weight > 0.0:
        missing_target = torch.ones(
            template_count,
            dtype=missing_logits.dtype,
            device=missing_logits.device,
        )
        if matched:
            present_indices = torch.tensor(
                sorted(matched),
                dtype=torch.long,
                device=missing_logits.device,
            )
            missing_target[present_indices] = 0.0
        # Missing lamps are much rarer than visible lamps in this dataset.
        # Balanced BCE prevents the auxiliary head from predicting all visible.
        missing_count = int(missing_target.sum().item())
        present_count = template_count - missing_count
        positive_weight = torch.tensor(
            present_count / max(missing_count, 1),
            dtype=missing_logits.dtype,
            device=missing_logits.device,
        )
        visibility_loss = F.binary_cross_entropy_with_logits(
            missing_logits,
            missing_target,
            pos_weight=positive_weight,
        )
        loss = loss + visibility_loss_weight * visibility_loss
    if layer_loss_weight > 0.0:
        labelled_rows = [
            row_index
            for row_index, label in enumerate(labels)
            if label >= 0
        ]
        if labelled_rows:
            row_indices = torch.tensor(
                labelled_rows,
                dtype=torch.long,
                device=log_transport.device,
            )
            layer_targets = torch.tensor(
                [
                    int(labels[row_index] >= front_count)
                    for row_index in labelled_rows
                ],
                dtype=torch.long,
                device=log_transport.device,
            )
            layer_logits = torch.stack(
                [
                    torch.logsumexp(
                        log_transport[row_indices, :front_count],
                        dim=1,
                    ),
                    torch.logsumexp(
                        log_transport[
                            row_indices,
                            front_count:template_count,
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            )
            loss = loss + layer_loss_weight * F.cross_entropy(
                layer_logits,
                layer_targets,
            )
    if hard_negative_weight > 0.0:
        margin_losses = []
        for row_index, label in enumerate(labels):
            if label < 0:
                continue
            layer_start, layer_end = (
                (0, front_count)
                if label < front_count
                else (front_count, template_count)
            )
            competitors = torch.cat(
                [
                    log_transport[row_index, layer_start:label],
                    log_transport[row_index, label + 1 : layer_end],
                ]
            )
            if competitors.numel():
                margin_losses.append(
                    F.softplus(
                        competitors.max()
                        - log_transport[row_index, label]
                        + hard_negative_margin
                    )
                )
        if margin_losses:
            loss = loss + hard_negative_weight * torch.stack(
                margin_losses
            ).mean()
    return loss


def _train_samples(
    model: GuidingLightGNN,
    samples: Iterable[tuple[list[LightDetection], list[int]]],
    template_graph,
    template_count: int,
    device: torch.device,
    k_neighbors: int,
    epochs: int,
    learning_rate: float,
) -> None:
    fixed_samples = list(samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        random.shuffle(fixed_samples)
        for detections, labels in fixed_samples:
            observation_graph = _build_observation_graph(detections, k_neighbors, device)
            logits = model(*observation_graph, *template_graph)
            log_transport = log_optimal_transport(logits, model.bin_score, iterations=60)
            loss = _transport_loss(log_transport, labels, template_count)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def _pretrain(
    model: GuidingLightGNN,
    configuration,
    template_graph,
    device: torch.device,
    k_neighbors: int,
    seed: int,
    steps: int,
    learning_rate: float,
) -> None:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        detections, labels = sample_domain(configuration, SOURCE_DOMAIN, rng)
        observation_graph = _build_observation_graph(detections, k_neighbors, device)
        logits = model(*observation_graph, *template_graph)
        log_transport = log_optimal_transport(logits, model.bin_score, iterations=60)
        loss = _transport_loss(log_transport, labels, len(configuration.lights))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


@torch.no_grad()
def evaluate(
    model: GuidingLightGNN,
    samples: Iterable[tuple[list[LightDetection], list[int]]],
    template_graph,
    template_count: int,
    device: torch.device,
    k_neighbors: int,
) -> dict[str, float]:
    model.eval()
    correct_ids = total_ids = 0
    outlier_tp = outlier_fp = outlier_fn = 0
    all_correct = total = 0
    for detections, labels in samples:
        observation_graph = _build_observation_graph(detections, k_neighbors, device)
        logits = model(*observation_graph, *template_graph)
        transport = log_optimal_transport(logits, model.bin_score, iterations=60)
        prediction = transport[:-1, :].argmax(dim=1).cpu().tolist()
        for predicted, label in zip(prediction, labels):
            target = label if label >= 0 else template_count
            all_correct += int(predicted == target)
            total += 1
            if label >= 0:
                correct_ids += int(predicted == label)
                total_ids += 1
            else:
                outlier_tp += int(predicted == template_count)
                outlier_fn += int(predicted != template_count)
            if label >= 0 and predicted == template_count:
                outlier_fp += 1

    precision = outlier_tp / max(outlier_tp + outlier_fp, 1)
    recall = outlier_tp / max(outlier_tp + outlier_fn, 1)
    return {
        "id_accuracy": correct_ids / max(total_ids, 1),
        "outlier_precision": precision,
        "outlier_recall": recall,
        "outlier_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "overall_assignment_accuracy": all_correct / max(total, 1),
    }


def _make_model(settings: dict, device: torch.device) -> GuidingLightGNN:
    model = GuidingLightGNN(**settings)
    return model.to(device)


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["method"], record["shots"]), []).append(record["metrics"])
    summary = []
    for (method, shots), metrics in grouped.items():
        row = {"method": method, "shots": shots, "runs": len(metrics)}
        for metric_name in metrics[0]:
            values = np.asarray([metric[metric_name] for metric in metrics], dtype=np.float64)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary.append(row)
    return sorted(summary, key=lambda row: (row["shots"], row["method"]))


def _print_summary(summary: list[dict]) -> None:
    print("\nFew-shot synthetic domain-shift pilot")
    print("method                         shots  ID-Acc       Outlier-F1  Overall")
    for row in summary:
        print(
            f"{row['method']:<30} {row['shots']:>5}  "
            f"{row['id_accuracy_mean']:.3f}+/-{row['id_accuracy_std']:.3f}  "
            f"{row['outlier_f1_mean']:.3f}      "
            f"{row['overall_assignment_accuracy_mean']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn_matcher.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shots", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--fewshot-epochs", type=int, default=25)
    parser.add_argument("--test-samples", type=int, default=96)
    parser.add_argument("--output", default="outputs/gnn_fewshot_synthetic_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    device = torch.device(args.device)
    configuration = load_array_configuration(args.array_config)
    with Path(args.gnn_config).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    template_graph = _template_graph(configuration, device)
    model_settings = settings["model"]
    records: list[dict] = []

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        test_rng = np.random.default_rng(seed + 10_000)
        test_samples = [sample_domain(configuration, TARGET_DOMAIN, test_rng) for _ in range(args.test_samples)]
        pretrained = _make_model(model_settings, device)
        _pretrain(
            pretrained,
            configuration,
            template_graph,
            device,
            configuration.k_neighbors,
            seed,
            args.pretrain_steps,
            5e-4,
        )
        records.append(
            {
                "seed": seed,
                "method": "source_pretrained_zero_shot",
                "shots": 0,
                "metrics": evaluate(pretrained, test_samples, template_graph, len(configuration.lights), device, configuration.k_neighbors),
            }
        )

        for shots in args.shots:
            train_rng = np.random.default_rng(seed * 100 + shots)
            fewshot_samples = [sample_domain(configuration, TARGET_DOMAIN, train_rng) for _ in range(shots)]

            scratch = _make_model(model_settings, device)
            _train_samples(
                scratch,
                fewshot_samples,
                template_graph,
                len(configuration.lights),
                device,
                configuration.k_neighbors,
                args.fewshot_epochs,
                5e-4,
            )
            records.append(
                {
                    "seed": seed,
                    "method": "fewshot_from_scratch",
                    "shots": shots,
                    "metrics": evaluate(scratch, test_samples, template_graph, len(configuration.lights), device, configuration.k_neighbors),
                }
            )

            finetuned = copy.deepcopy(pretrained)
            _train_samples(
                finetuned,
                fewshot_samples,
                template_graph,
                len(configuration.lights),
                device,
                configuration.k_neighbors,
                args.fewshot_epochs,
                1e-4,
            )
            records.append(
                {
                    "seed": seed,
                    "method": "source_pretrained_fewshot_finetune",
                    "shots": shots,
                    "metrics": evaluate(finetuned, test_samples, template_graph, len(configuration.lights), device, configuration.k_neighbors),
                }
            )

    summary = _aggregate(records)
    _print_summary(summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "experiment": "synthetic target-domain few-shot adaptation pilot",
                "important_note": "Synthetic result only. Repeat with manually labelled real F0-F6/R0-R5 frames before reporting paper claims.",
                "source_domain": SOURCE_DOMAIN.__dict__,
                "target_domain": TARGET_DOMAIN.__dict__,
                "arguments": vars(args),
                "records": records,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
