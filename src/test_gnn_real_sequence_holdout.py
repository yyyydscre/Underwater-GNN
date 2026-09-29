"""Train and test the guiding-light GNN with a leakage-aware sequence holdout."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .compare_gnn_fewshot import (
    _build_observation_graph,
    _make_model,
    _pretrain,
    _template_graph,
    _train_samples,
    _transport_loss,
    sample_domain,
    SOURCE_DOMAIN,
)
from .config import load_array_configuration
from .graph import log_optimal_transport
from .schema import COLOR_INDEX, LightDetection
from .train_gnn_fewshot_real_pilot import load_completed_examples


def frame_identity(frame_name: str) -> tuple[str, int]:
    stem = Path(frame_name).stem
    frame_match = re.search(r"frame[_ -]?(\d+)", stem, flags=re.IGNORECASE)
    matches = list(re.finditer(r"(\d+)", stem))
    match = frame_match or (matches[-1] if matches else None)
    frame_number = int(match.group(1)) if match else -1
    prefix = stem[: match.start(1)] if match else stem
    prefix = prefix.rstrip("_ ").lower()
    if prefix.startswith("red_"):
        prefix = prefix[4:]
    if prefix.startswith("frame"):
        prefix = "frame"
    return prefix, frame_number


def split_examples(examples, test_sequence: str, validation_fraction: float):
    enriched = [
        (*frame_identity(frame_name), frame_name, detections, labels)
        for frame_name, detections, labels in examples
    ]
    test = [item for item in enriched if item[0] == test_sequence]
    development = [item for item in enriched if item[0] != test_sequence]
    if not test:
        raise ValueError(f"No examples found for test sequence {test_sequence!r}.")
    if not development:
        raise ValueError("No development examples remain after holding out the test sequence.")

    grouped: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for item in development:
        grouped[(item[0], item[1])].append(item)
    ordered_groups = [
        grouped[key] for key in sorted(grouped, key=lambda value: (value[0], value[1]))
    ]
    validation_group_count = max(1, int(np.ceil(len(ordered_groups) * validation_fraction)))
    training_groups = ordered_groups[:-validation_group_count]
    validation_groups = ordered_groups[-validation_group_count:]
    if not training_groups:
        raise ValueError("No training groups remain after the validation split.")

    def flatten(groups):
        return [
            (frame_name, detections, labels)
            for group in groups
            for _, _, frame_name, detections, labels in group
        ]

    training = flatten(training_groups)
    validation = flatten(validation_groups)
    test_examples = [
        (frame_name, detections, labels)
        for _, _, frame_name, detections, labels in sorted(
            test, key=lambda item: (item[1], item[2])
        )
    ]
    return training, validation, test_examples


def split_examples_within_sequences(
    examples,
    validation_fraction: float,
    test_fraction: float,
):
    grouped: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for frame_name, detections, labels in examples:
        sequence, frame_number = frame_identity(frame_name)
        grouped[(sequence, frame_number)].append(
            (frame_name, detections, labels)
        )

    sequence_groups: dict[str, list[list[tuple]]] = defaultdict(list)
    for (sequence, frame_number), items in grouped.items():
        sequence_groups[sequence].append(
            [(frame_number, *item) for item in items]
        )

    training_groups = []
    validation_groups = []
    test_groups = []
    for sequence, groups in sequence_groups.items():
        ordered = sorted(groups, key=lambda group: group[0][0])
        test_count = max(1, int(np.ceil(len(ordered) * test_fraction)))
        validation_count = max(
            1, int(np.ceil(len(ordered) * validation_fraction))
        )
        training_count = len(ordered) - validation_count - test_count
        if training_count < 1:
            raise ValueError(
                f"Sequence {sequence!r} is too small for train/val/test splitting."
            )
        training_groups.extend(ordered[:training_count])
        validation_groups.extend(
            ordered[training_count : training_count + validation_count]
        )
        test_groups.extend(ordered[training_count + validation_count :])

    def flatten(groups):
        return [
            (frame_name, detections, labels)
            for group in groups
            for _, frame_name, detections, labels in group
        ]

    return flatten(training_groups), flatten(validation_groups), flatten(test_groups)


def transform_detections(detections, color_mode: str):
    if color_mode == "original":
        return detections
    neutral = np.full(3, 1.0 / 3.0, dtype=np.float32)
    return [
        replace(
            detection,
            color_probs=neutral.copy(),
            color="other",
        )
        for detection in detections
    ]


def transform_examples(examples, color_mode: str):
    return [
        (frame_name, transform_detections(detections, color_mode), labels)
        for frame_name, detections, labels in examples
    ]


def pretrain_model(
    model,
    configuration,
    template_graph,
    device,
    seed: int,
    steps: int,
    learning_rate: float,
    color_mode: str,
    feature_mode: str,
    layer_loss_weight: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
    missing_template_weight: float,
    visibility_loss_weight: float,
    frame_hard_weight: float,
    frame_hard_temperature: float,
) -> None:
    if (
        color_mode == "original"
        and feature_mode == "legacy"
        and layer_loss_weight <= 0.0
        and hard_negative_weight <= 0.0
        and visibility_loss_weight <= 0.0
        and frame_hard_weight <= 0.0
    ):
        _pretrain(
            model,
            configuration,
            template_graph,
            device,
            configuration.k_neighbors,
            seed,
            steps,
            learning_rate,
        )
        return

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        detections, labels = sample_domain(configuration, SOURCE_DOMAIN, rng)
        detections = transform_detections(detections, color_mode)
        observation_graph = _build_observation_graph(
            detections,
            configuration.k_neighbors,
            device,
            feature_mode,
        )
        logits, missing_logits = model.forward_with_aux(
            *observation_graph,
            *template_graph,
        )
        transport = log_optimal_transport(
            logits, model.bin_score, iterations=60
        )
        loss = _transport_loss(
            transport,
            labels,
            len(configuration.lights),
            layer_loss_weight=layer_loss_weight,
            hard_negative_weight=hard_negative_weight,
            hard_negative_margin=hard_negative_margin,
            missing_template_weight=missing_template_weight,
            missing_logits=missing_logits,
            visibility_loss_weight=visibility_loss_weight,
            frame_hard_weight=frame_hard_weight,
            frame_hard_temperature=frame_hard_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def augment_geometry(detections, rng: np.random.Generator):
    points = np.asarray([detection.xy for detection in detections], dtype=np.float64)
    center = np.median(points, axis=0, keepdims=True)
    centered = points - center

    angle = math.radians(float(rng.uniform(-18.0, 18.0)))
    if rng.random() < 0.12:
        angle += math.pi
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    anisotropic_scale = np.diag(rng.uniform(0.82, 1.18, size=2))
    shear = np.asarray(
        [[1.0, float(rng.uniform(-0.12, 0.12))], [0.0, 1.0]],
        dtype=np.float64,
    )
    transformed = centered @ (rotation @ shear @ anisotropic_scale).T
    transformed += rng.normal(0.0, 0.8, size=transformed.shape)
    transformed += center
    return [
        replace(detection, xy=point)
        for detection, point in zip(detections, transformed)
    ]


def _center_noise_covariance(detection) -> np.ndarray:
    """Return a bounded positive-semidefinite centre covariance."""
    radius = max(float(detection.radius), 1.0)
    if detection.center_covariance is None:
        quality = float(
            detection.center_confidence
            if detection.center_confidence is not None
            else detection.center_quality
        )
        sigma = 0.15 + 0.45 * (1.0 - np.clip(quality, 0.0, 1.0)) * radius
        return np.eye(2, dtype=np.float64) * sigma**2
    covariance = np.asarray(
        detection.center_covariance,
        dtype=np.float64,
    ).reshape(2, 2)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum_variance = (0.4 * radius) ** 2
    eigenvalues = np.clip(eigenvalues, 1e-4, maximum_variance)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def augment_geometry_v2(
    detections,
    labels,
    rng: np.random.Generator,
):
    """Simulate docking geometry, uncertain centres, missed lamps and clutter."""
    if len(detections) < 2:
        return list(detections), list(labels)
    points = np.asarray([detection.xy for detection in detections], dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    spatial_scale = max(
        float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
        8.0,
    )
    normalized = centered / spatial_scale

    angle = math.radians(float(rng.uniform(-20.0, 20.0)))
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    anisotropic_scale = np.diag(rng.uniform(0.84, 1.16, size=2))
    shear = np.asarray(
        [[1.0, float(rng.uniform(-0.10, 0.10))], [0.0, 1.0]],
        dtype=np.float64,
    )
    affine = rotation @ shear @ anisotropic_scale
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :2] = affine
    homography[2, :2] = rng.uniform(-0.055, 0.055, size=2)
    homogeneous = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    warped_h = homogeneous @ homography.T
    denominator = np.where(
        np.abs(warped_h[:, 2]) < 0.35,
        np.sign(warped_h[:, 2]) * 0.35,
        warped_h[:, 2],
    )
    warped = warped_h[:, :2] / denominator[:, None]

    rear_indices = np.asarray(
        [index for index, label in enumerate(labels) if label >= 7],
        dtype=np.int64,
    )
    if len(rear_indices) and rng.random() < 0.9:
        rear_center = warped[rear_indices].mean(axis=0, keepdims=True)
        rear_scale = float(rng.uniform(0.86, 1.06))
        rear_shift = rng.uniform(-0.10, 0.10, size=(1, 2))
        warped[rear_indices] = (
            (warped[rear_indices] - rear_center) * rear_scale
            + rear_center
            + rear_shift
        )

    transformed = warped * spatial_scale + center
    transformed_detections = []
    for detection, point in zip(detections, transformed):
        covariance = _center_noise_covariance(detection)
        noise_scale = float(rng.uniform(0.35, 0.95))
        point = point + rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            covariance * noise_scale**2,
        )
        transformed_detections.append(replace(detection, xy=point))

    drop_probability = float(rng.uniform(0.0, 0.16))
    keep = rng.random(len(transformed_detections)) >= drop_probability
    if int(keep.sum()) < min(4, len(keep)):
        keep[:] = False
        keep[rng.choice(len(keep), size=min(4, len(keep)), replace=False)] = True
    kept_detections = [
        detection
        for detection, is_kept in zip(transformed_detections, keep)
        if is_kept
    ]
    kept_labels = [
        int(label)
        for label, is_kept in zip(labels, keep)
        if is_kept
    ]

    kept_points = np.asarray(
        [detection.xy for detection in kept_detections],
        dtype=np.float64,
    )
    lower = kept_points.min(axis=0) - 0.15 * spatial_scale
    upper = kept_points.max(axis=0) + 0.15 * spatial_scale
    median_radius = float(np.median([item.radius for item in kept_detections]))
    for _ in range(min(int(rng.poisson(1.0)), 3)):
        quality = float(rng.uniform(0.08, 0.45))
        radius = max(1.0, median_radius * float(rng.uniform(0.45, 1.45)))
        sigma = radius * float(rng.uniform(0.18, 0.38))
        kept_detections.append(
            LightDetection(
                xy=rng.uniform(lower, upper),
                confidence=float(rng.uniform(0.15, 0.55)),
                color_probs=np.full(3, 1.0 / 3.0, dtype=np.float32),
                color="other",
                radius=radius,
                brightness=float(rng.uniform(0.15, 0.95)),
                source="docking_v2_synthetic_clutter",
                center_quality=quality,
                detector_confidence=float(rng.uniform(0.15, 0.55)),
                center_confidence=quality,
                center_method="synthetic_clutter",
                center_covariance=np.eye(2, dtype=np.float64) * sigma**2,
                center_valid=False,
            )
        )
        kept_labels.append(-1)

    order = rng.permutation(len(kept_detections))
    return (
        [kept_detections[int(index)] for index in order],
        [kept_labels[int(index)] for index in order],
    )


def augment_structured_occlusion(
    detections,
    labels,
    rng: np.random.Generator,
    probability: float = 0.55,
    maximum_drop: int = 3,
    minimum_visible: int = 7,
):
    """Mask a compact or random subset of visible lamps without inventing points."""
    positive = [index for index, label in enumerate(labels) if label >= 0]
    maximum_allowed = min(
        max(int(maximum_drop), 0),
        max(len(positive) - max(int(minimum_visible), 1), 0),
    )
    if maximum_allowed <= 0 or rng.random() >= probability:
        return list(detections), list(labels)
    drop_count = int(rng.integers(1, maximum_allowed + 1))
    if rng.random() < 0.65:
        anchor = int(rng.choice(positive))
        anchor_point = np.asarray(detections[anchor].xy, dtype=np.float64)
        ranked = sorted(
            positive,
            key=lambda index: float(
                np.linalg.norm(
                    np.asarray(detections[index].xy, dtype=np.float64)
                    - anchor_point
                )
            ),
        )
        dropped = set(ranked[:drop_count])
    else:
        dropped = set(
            int(index)
            for index in rng.choice(
                positive,
                size=drop_count,
                replace=False,
            )
        )
    return (
        [item for index, item in enumerate(detections) if index not in dropped],
        [int(label) for index, label in enumerate(labels) if index not in dropped],
    )


def train_real_model(
    model,
    samples,
    template_graph,
    template_count: int,
    device,
    k_neighbors: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    geometry_augmentation: str,
    feature_mode: str,
    layer_loss_weight: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
    missing_template_weight: float,
    validation_examples=None,
    configuration=None,
    color_bias: float = 0.0,
    select_best_on_validation: bool = False,
    augmentation_close_epochs: int = 0,
    occlusion_probability: float = 0.0,
    occlusion_max_drop: int = 3,
    occlusion_min_visible: int = 7,
    validation_selection_metric: str = "id_accuracy",
    visibility_loss_weight: float = 0.0,
    frame_hard_weight: float = 0.0,
    frame_hard_temperature: float = 0.5,
) -> dict:
    fixed_samples = list(samples)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_epoch = None
    best_metrics = None
    best_score = None
    for epoch in range(epochs):
        model.train()
        use_augmentation = epoch < max(
            epochs - max(int(augmentation_close_epochs), 0),
            0,
        )
        for example_index in rng.permutation(len(fixed_samples)):
            detections, labels = fixed_samples[int(example_index)]
            if use_augmentation and geometry_augmentation == "docking":
                detections = augment_geometry(detections, rng)
            elif use_augmentation and geometry_augmentation == "docking_v2":
                detections, labels = augment_geometry_v2(
                    detections,
                    labels,
                    rng,
                )
            if use_augmentation and occlusion_probability > 0.0:
                detections, labels = augment_structured_occlusion(
                    detections,
                    labels,
                    rng,
                    probability=occlusion_probability,
                    maximum_drop=occlusion_max_drop,
                    minimum_visible=occlusion_min_visible,
                )
            observation_graph = _build_observation_graph(
                detections,
                k_neighbors,
                device,
                feature_mode,
            )
            logits, missing_logits = model.forward_with_aux(
                *observation_graph,
                *template_graph,
            )
            transport = log_optimal_transport(
                logits, model.bin_score, iterations=60
            )
            loss = _transport_loss(
                transport,
                labels,
                template_count,
                layer_loss_weight=layer_loss_weight,
                hard_negative_weight=hard_negative_weight,
                hard_negative_margin=hard_negative_margin,
                missing_template_weight=missing_template_weight,
                missing_logits=missing_logits,
                visibility_loss_weight=visibility_loss_weight,
                frame_hard_weight=frame_hard_weight,
                frame_hard_temperature=frame_hard_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if (
            select_best_on_validation
            and validation_examples is not None
            and configuration is not None
        ):
            validation = evaluate_model(
                model,
                validation_examples,
                template_graph,
                configuration,
                device,
                color_bias,
                feature_mode,
            )["mutual_one_to_one"]
            if validation_selection_metric == "balanced_visibility":
                visibility_scores = [
                    values["id_accuracy"]
                    for values in validation["id_accuracy_by_visibility"].values()
                    if values["frames"] > 0
                ]
                balanced_visibility = float(np.mean(visibility_scores))
                score = (
                    balanced_visibility,
                    validation["missing_id_f1"],
                    validation["id_accuracy"],
                    validation["exact_frame_rate"],
                )
            else:
                if validation_selection_metric == "id_exact":
                    score = (
                        validation["id_accuracy"]
                        + 0.15 * validation["exact_frame_rate"],
                        validation["id_accuracy"],
                        validation["exact_frame_rate"],
                    )
                else:
                    score = (
                        validation["id_accuracy"],
                        validation["exact_frame_rate"],
                        validation["outlier_f1"],
                    )
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_metrics = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "selected_on_validation": best_state is not None,
        "augmentation_close_epochs": int(augmentation_close_epochs),
        "occlusion_probability": float(occlusion_probability),
        "occlusion_max_drop": int(occlusion_max_drop),
        "occlusion_min_visible": int(occlusion_min_visible),
        "validation_selection_metric": validation_selection_metric,
        "visibility_loss_weight": float(visibility_loss_weight),
        "frame_hard_weight": float(frame_hard_weight),
        "frame_hard_temperature": float(frame_hard_temperature),
    }


def predict_assignments(
    model,
    detections,
    template_graph,
    configuration,
    device,
    color_bias: float,
    feature_mode: str,
) -> tuple[list[int], list[int]]:
    observation_graph = _build_observation_graph(
        detections,
        configuration.k_neighbors,
        device,
        feature_mode,
    )
    scores = model(*observation_graph, *template_graph)
    color_compatibility = torch.tensor(
        [
            [
                detection.color_probs[COLOR_INDEX.get(lamp.color, COLOR_INDEX["other"])]
                for lamp in configuration.lights
            ]
            for detection in detections
        ],
        dtype=scores.dtype,
        device=scores.device,
    )
    scores = scores + color_bias * torch.log(color_compatibility.clamp_min(1e-4))
    transport = log_optimal_transport(scores, model.bin_score, iterations=60)
    template_count = len(configuration.lights)

    row_argmax = transport[:-1].argmax(dim=1).cpu().tolist()
    observation_choice = transport[:-1].argmax(dim=1)
    template_choice = transport[:, :-1].argmax(dim=0)
    mutual: list[int] = []
    for observation_index, template_index in enumerate(observation_choice.tolist()):
        if (
            template_index < template_count
            and int(template_choice[template_index]) == observation_index
        ):
            mutual.append(template_index)
        else:
            mutual.append(template_count)
    return row_argmax, mutual


def summarize_predictions(records: list[dict], configuration) -> dict:
    template_count = len(configuration.lights)
    correct_ids = total_ids = 0
    front_correct = front_total = 0
    rear_correct = rear_total = 0
    outlier_tp = outlier_fp = outlier_fn = 0
    overall_correct = overall_total = 0
    exact_frames = 0
    correct_ids_per_frame = []
    four_correct_support = 0
    four_visible_frames = 0
    per_id_correct = [0] * template_count
    per_id_total = [0] * template_count
    missing_tp = missing_fp = missing_fn = 0
    visibility_bins = {
        "complete_13": {"correct": 0, "total": 0, "frames": 0},
        "partial_10_12": {"correct": 0, "total": 0, "frames": 0},
        "severe_le_9": {"correct": 0, "total": 0, "frames": 0},
    }
    fine_visibility_bins = {
        "visible_13": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
        "visible_10_12": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
        "visible_7_9": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
        "visible_4_6": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
        "visible_1_3": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
        "visible_0": {"correct": 0, "total": 0, "frames": 0, "exact": 0},
    }

    for record in records:
        frame_id_correct = True
        frame_correct = frame_total = 0
        for prediction, label in zip(record["predictions"], record["labels"]):
            target = label if label >= 0 else template_count
            overall_correct += int(prediction == target)
            overall_total += 1
            if label >= 0:
                is_correct = prediction == label
                correct_ids += int(is_correct)
                total_ids += 1
                frame_correct += int(is_correct)
                frame_total += 1
                per_id_correct[label] += int(is_correct)
                per_id_total[label] += 1
                frame_id_correct = frame_id_correct and is_correct
                if label < 7:
                    front_correct += int(is_correct)
                    front_total += 1
                else:
                    rear_correct += int(is_correct)
                    rear_total += 1
                if prediction == template_count:
                    outlier_fp += 1
            else:
                outlier_tp += int(prediction == template_count)
                outlier_fn += int(prediction != template_count)
        visible_ids = {
            int(label) for label in record["labels"] if label >= 0
        }
        predicted_ids = {
            int(prediction)
            for prediction in record["predictions"]
            if 0 <= prediction < template_count
        }
        true_missing = set(range(template_count)) - visible_ids
        predicted_missing = set(range(template_count)) - predicted_ids
        missing_tp += len(true_missing & predicted_missing)
        missing_fp += len(predicted_missing - true_missing)
        missing_fn += len(true_missing - predicted_missing)
        visible_count = len(visible_ids)
        correct_ids_per_frame.append(frame_correct)
        if visible_count >= 4:
            four_visible_frames += 1
            four_correct_support += int(frame_correct >= 4)
        bin_name = (
            "complete_13"
            if visible_count == template_count
            else "partial_10_12"
            if visible_count >= 10
            else "severe_le_9"
        )
        visibility_bins[bin_name]["correct"] += frame_correct
        visibility_bins[bin_name]["total"] += frame_total
        visibility_bins[bin_name]["frames"] += 1
        fine_bin_name = (
            "visible_13"
            if visible_count == template_count
            else "visible_10_12"
            if visible_count >= 10
            else "visible_7_9"
            if visible_count >= 7
            else "visible_4_6"
            if visible_count >= 4
            else "visible_1_3"
            if visible_count >= 1
            else "visible_0"
        )
        fine_visibility_bins[fine_bin_name]["correct"] += frame_correct
        fine_visibility_bins[fine_bin_name]["total"] += frame_total
        fine_visibility_bins[fine_bin_name]["frames"] += 1
        fine_visibility_bins[fine_bin_name]["exact"] += int(frame_id_correct)
        exact_frames += int(frame_id_correct)

    precision = outlier_tp / max(outlier_tp + outlier_fp, 1)
    recall = outlier_tp / max(outlier_tp + outlier_fn, 1)
    missing_precision = missing_tp / max(missing_tp + missing_fp, 1)
    missing_recall = missing_tp / max(missing_tp + missing_fn, 1)
    per_id = {
        configuration.ids[index]: {
            "correct": per_id_correct[index],
            "total": per_id_total[index],
            "accuracy": per_id_correct[index] / max(per_id_total[index], 1),
        }
        for index in range(template_count)
    }
    return {
        "id_accuracy": correct_ids / max(total_ids, 1),
        "front_id_accuracy": front_correct / max(front_total, 1),
        "rear_id_accuracy": rear_correct / max(rear_total, 1),
        "overall_assignment_accuracy": overall_correct / max(overall_total, 1),
        "exact_frame_rate": exact_frames / max(len(records), 1),
        "mean_correct_ids_per_frame": float(
            np.mean(correct_ids_per_frame)
        )
        if correct_ids_per_frame
        else 0.0,
        "four_correct_support_rate": (
            four_correct_support / max(four_visible_frames, 1)
        ),
        "four_visible_frames": four_visible_frames,
        "outlier_precision": precision,
        "outlier_recall": recall,
        "outlier_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "missing_id_precision": missing_precision,
        "missing_id_recall": missing_recall,
        "missing_id_f1": 2.0
        * missing_precision
        * missing_recall
        / max(missing_precision + missing_recall, 1e-8),
        "id_accuracy_by_visibility": {
            name: {
                "frames": values["frames"],
                "id_accuracy": values["correct"] / max(values["total"], 1),
            }
            for name, values in visibility_bins.items()
        },
        "metrics_by_visibility_fine": {
            name: {
                "frames": values["frames"],
                "id_accuracy": values["correct"] / max(values["total"], 1),
                "exact_frame_rate": values["exact"] / max(values["frames"], 1),
            }
            for name, values in fine_visibility_bins.items()
        },
        "labelled_detections": total_ids,
        "outlier_detections": overall_total - total_ids,
        "frames": len(records),
        "per_id": per_id,
    }


@torch.no_grad()
def evaluate_model(
    model,
    examples,
    template_graph,
    configuration,
    device,
    color_bias: float,
    feature_mode: str,
) -> dict:
    model.eval()
    row_records = []
    mutual_records = []
    for frame_name, detections, labels in examples:
        row_predictions, mutual_predictions = predict_assignments(
            model,
            detections,
            template_graph,
            configuration,
            device,
            color_bias,
            feature_mode,
        )
        row_records.append(
            {
                "frame_name": frame_name,
                "labels": list(labels),
                "predictions": row_predictions,
            }
        )
        mutual_records.append(
            {
                "frame_name": frame_name,
                "labels": list(labels),
                "predictions": mutual_predictions,
            }
        )
    return {
        "row_argmax": summarize_predictions(row_records, configuration),
        "mutual_one_to_one": summarize_predictions(mutual_records, configuration),
        "test_predictions": mutual_records,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn_matcher.yaml")
    parser.add_argument("--test-sequence", default="v")
    parser.add_argument(
        "--split-mode",
        choices=("sequence_holdout", "within_sequence"),
        default="sequence_holdout",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--color-bias", type=float, default=1.35)
    parser.add_argument(
        "--observation-color-mode",
        choices=("original", "neutral"),
        default="original",
    )
    parser.add_argument(
        "--geometry-augmentation",
        choices=("none", "docking", "docking_v2"),
        default="none",
    )
    parser.add_argument(
        "--feature-mode",
        choices=(
            "legacy",
            "geometry_uncertainty",
            "geometry_uncertainty_relative",
            "geometry_uncertainty_topology",
        ),
        default="legacy",
    )
    parser.add_argument("--scratch-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", default="outputs/gnn_real_sequence_holdout_test.json"
    )
    parser.add_argument(
        "--checkpoint", default="models/gnn_matcher_real_sequence_holdout.pt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5.")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("test-fraction must be between 0 and 0.5.")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation-fraction + test-fraction must be below 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")

    device = torch.device(args.device)
    configuration = load_array_configuration(args.array_config)
    with Path(args.gnn_config).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    layer_loss_weight = float(
        settings.get("training", {}).get("layer_loss_weight", 0.0)
    )
    hard_negative_weight = float(
        settings.get("training", {}).get("hard_negative_weight", 0.0)
    )
    hard_negative_margin = float(
        settings.get("training", {}).get("hard_negative_margin", 2.0)
    )
    missing_template_weight = float(
        settings.get("training", {}).get("missing_template_weight", 0.3)
    )
    visibility_loss_weight = float(
        settings.get("training", {}).get("visibility_loss_weight", 0.0)
    )
    frame_hard_weight = float(
        settings.get("training", {}).get("frame_hard_weight", 0.0)
    )
    frame_hard_temperature = float(
        settings.get("training", {}).get("frame_hard_temperature", 0.5)
    )
    select_best_on_validation = bool(
        settings.get("training", {}).get("select_best_on_validation", False)
    )
    augmentation_close_epochs = int(
        settings.get("training", {}).get("augmentation_close_epochs", 0)
    )
    occlusion_probability = float(
        settings.get("training", {}).get("occlusion_probability", 0.0)
    )
    occlusion_max_drop = int(
        settings.get("training", {}).get("occlusion_max_drop", 3)
    )
    occlusion_min_visible = int(
        settings.get("training", {}).get("occlusion_min_visible", 7)
    )
    validation_selection_metric = str(
        settings.get("training", {}).get(
            "validation_selection_metric",
            "id_accuracy",
        )
    )
    if validation_selection_metric not in {
        "id_accuracy",
        "balanced_visibility",
        "id_exact",
    }:
        raise ValueError(
            "training.validation_selection_metric must be id_accuracy or "
            "balanced_visibility or id_exact."
        )
    examples = load_completed_examples(
        Path(args.annotations),
        configuration,
        include_unreviewed_labelled=True,
    )
    examples = transform_examples(examples, args.observation_color_mode)
    if args.split_mode == "sequence_holdout":
        training, validation, test = split_examples(
            examples, args.test_sequence, args.validation_fraction
        )
    else:
        training, validation, test = split_examples_within_sequences(
            examples,
            args.validation_fraction,
            args.test_fraction,
        )
    template_graph = _template_graph(
        configuration,
        device,
        args.feature_mode,
    )
    template_count = len(configuration.lights)
    train_samples = [(detections, labels) for _, detections, labels in training]

    results = []
    seed_everything(args.seed)
    scratch = _make_model(settings["model"], device)
    scratch_selection = train_real_model(
        scratch,
        train_samples,
        template_graph,
        template_count,
        device,
        configuration.k_neighbors,
        args.finetune_epochs,
        float(settings["training"]["learning_rate"]),
        args.seed,
        args.geometry_augmentation,
        args.feature_mode,
        layer_loss_weight,
        hard_negative_weight,
        hard_negative_margin,
        missing_template_weight,
        validation_examples=validation,
        configuration=configuration,
        color_bias=args.color_bias,
        select_best_on_validation=select_best_on_validation,
        augmentation_close_epochs=augmentation_close_epochs,
        occlusion_probability=occlusion_probability,
        occlusion_max_drop=occlusion_max_drop,
        occlusion_min_visible=occlusion_min_visible,
        validation_selection_metric=validation_selection_metric,
        visibility_loss_weight=visibility_loss_weight,
        frame_hard_weight=frame_hard_weight,
        frame_hard_temperature=frame_hard_temperature,
    )
    results.append(
        {
            "method": "real_from_scratch",
            "training_selection": scratch_selection,
            "validation": evaluate_model(
                scratch,
                validation,
                template_graph,
                configuration,
                device,
                args.color_bias,
                args.feature_mode,
            ),
            "test": evaluate_model(
                scratch,
                test,
                template_graph,
                configuration,
                device,
                args.color_bias,
                args.feature_mode,
            ),
        }
    )

    checkpoint_model = scratch
    checkpoint_training = "real_from_scratch"
    if not args.scratch_only:
        seed_everything(args.seed)
        pretrained = _make_model(settings["model"], device)
        pretrain_model(
            pretrained,
            configuration,
            template_graph,
            device,
            args.seed,
            args.pretrain_steps,
            float(settings["training"]["learning_rate"]),
            args.observation_color_mode,
            args.feature_mode,
            layer_loss_weight,
            hard_negative_weight,
            hard_negative_margin,
            missing_template_weight,
            visibility_loss_weight,
            frame_hard_weight,
            frame_hard_temperature,
        )
        results.append(
            {
                "method": "synthetic_pretrained_zero_shot",
                "validation": evaluate_model(
                    pretrained,
                    validation,
                    template_graph,
                    configuration,
                    device,
                    args.color_bias,
                    args.feature_mode,
                ),
                "test": evaluate_model(
                    pretrained,
                    test,
                    template_graph,
                    configuration,
                    device,
                    args.color_bias,
                    args.feature_mode,
                ),
            }
        )

        finetuned = copy.deepcopy(pretrained)
        finetune_selection = train_real_model(
            finetuned,
            train_samples,
            template_graph,
            template_count,
            device,
            configuration.k_neighbors,
            args.finetune_epochs,
            float(settings["training"]["learning_rate"]) * 0.2,
            args.seed + 1,
            args.geometry_augmentation,
            args.feature_mode,
            layer_loss_weight,
            hard_negative_weight,
            hard_negative_margin,
            missing_template_weight,
            validation_examples=validation,
            configuration=configuration,
            color_bias=args.color_bias,
            select_best_on_validation=select_best_on_validation,
            augmentation_close_epochs=augmentation_close_epochs,
            occlusion_probability=occlusion_probability,
            occlusion_max_drop=occlusion_max_drop,
            occlusion_min_visible=occlusion_min_visible,
            validation_selection_metric=validation_selection_metric,
            visibility_loss_weight=visibility_loss_weight,
            frame_hard_weight=frame_hard_weight,
            frame_hard_temperature=frame_hard_temperature,
        )
        results.append(
            {
                "method": "synthetic_pretrained_real_finetune",
                "training_selection": finetune_selection,
                "validation": evaluate_model(
                    finetuned,
                    validation,
                    template_graph,
                    configuration,
                    device,
                    args.color_bias,
                    args.feature_mode,
                ),
                "test": evaluate_model(
                    finetuned,
                    test,
                    template_graph,
                    configuration,
                    device,
                    args.color_bias,
                    args.feature_mode,
                ),
            }
        )
        checkpoint_model = finetuned
        checkpoint_training = "synthetic_pretrain_real_finetune"

    print(
        f"sequence_holdout train={len(training)} val={len(validation)} "
        f"test={len(test)} split_mode={args.split_mode} "
        f"test_sequence={args.test_sequence}"
    )
    print("method                                  ID-Acc  Front   Rear    Exact")
    for result in results:
        metrics = result["test"]["mutual_one_to_one"]
        print(
            f"{result['method']:<39} {metrics['id_accuracy']:.3f}   "
            f"{metrics['front_id_accuracy']:.3f}   "
            f"{metrics['rear_id_accuracy']:.3f}   "
            f"{metrics['exact_frame_rate']:.3f}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": checkpoint_model.state_dict(),
            "model": settings["model"],
            "array_config": args.array_config,
            "training": checkpoint_training,
            "observation_color_mode": args.observation_color_mode,
            "observation_feature_mode": args.feature_mode,
            "color_bias": args.color_bias,
            "geometry_augmentation": args.geometry_augmentation,
            "layer_loss_weight": layer_loss_weight,
            "hard_negative_weight": hard_negative_weight,
            "hard_negative_margin": hard_negative_margin,
            "missing_template_weight": missing_template_weight,
            "visibility_loss_weight": visibility_loss_weight,
            "frame_hard_weight": frame_hard_weight,
            "frame_hard_temperature": frame_hard_temperature,
            "select_best_on_validation": select_best_on_validation,
            "augmentation_close_epochs": augmentation_close_epochs,
            "occlusion_probability": occlusion_probability,
            "occlusion_max_drop": occlusion_max_drop,
            "occlusion_min_visible": occlusion_min_visible,
            "validation_selection_metric": validation_selection_metric,
            "training_selection": (
                scratch_selection
                if checkpoint_training == "real_from_scratch"
                else finetune_selection
            ),
            "train_frames": [name for name, _, _ in training],
            "validation_frames": [name for name, _, _ in validation],
            "test_frames": [name for name, _, _ in test],
        },
        checkpoint_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "experiment": f"leakage-aware real {args.split_mode}",
                "arguments": vars(args),
                "split": {
                    "training_frames": [name for name, _, _ in training],
                    "validation_frames": [name for name, _, _ in validation],
                    "test_frames": [name for name, _, _ in test],
                },
                "results": results,
                "checkpoint": str(checkpoint_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {checkpoint_path.resolve()}")
    print(f"Wrote results: {output_path.resolve()}")


if __name__ == "__main__":
    main()
