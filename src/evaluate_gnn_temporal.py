"""Tune and evaluate online temporal refinement for a trained GNN checkpoint."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .compare_gnn_fewshot import _build_observation_graph, _template_graph
from .config import load_array_configuration
from .graph import load_gnn_checkpoint, log_optimal_transport
from .sequence import temporal_identity
from .temporal_gnn import (
    ReliabilityGatedTrackletDecoder,
    TemporalGraphDecoder,
    TrackletGraphDecoder,
)
from .test_gnn_real_sequence_holdout import (
    summarize_predictions,
    transform_detections,
)
from .train_gnn_fewshot_real_pilot import load_completed_examples


@dataclass
class ScoredFrame:
    frame_name: str
    detections: list
    labels: list[int]
    scores: np.ndarray
    dustbin_score: float


@torch.no_grad()
def score_examples(
    examples,
    model,
    template_graph,
    configuration,
    device,
    missing_expert=None,
    missing_expert_max_detections: int = 8,
) -> list[ScoredFrame]:
    model.eval()
    feature_mode = getattr(model, "observation_feature_mode", "legacy")
    color_mode = getattr(model, "observation_color_mode", "original")
    scored = []
    for frame_name, detections, labels in examples:
        active_model = (
            missing_expert
            if missing_expert is not None
            and len(detections) <= missing_expert_max_detections
            else model
        )
        active_feature_mode = getattr(
            active_model,
            "observation_feature_mode",
            feature_mode,
        )
        active_color_mode = getattr(
            active_model,
            "observation_color_mode",
            color_mode,
        )
        model_detections = transform_detections(
            detections,
            active_color_mode,
        )
        graph = _build_observation_graph(
            model_detections,
            configuration.k_neighbors,
            device,
            active_feature_mode,
        )
        scores = active_model(*graph, *template_graph)
        scored.append(
            ScoredFrame(
                frame_name=frame_name,
                detections=model_detections,
                labels=list(labels),
                scores=scores.detach().cpu().numpy(),
                dustbin_score=float(active_model.bin_score.detach().cpu()),
            )
        )
    return scored


def ordered_sequences(frames: list[ScoredFrame]) -> list[list[ScoredFrame]]:
    grouped: dict[str, list[ScoredFrame]] = {}
    for frame in frames:
        sequence, _ = temporal_identity(frame.frame_name)
        grouped.setdefault(sequence, []).append(frame)
    return [
        sorted(items, key=lambda item: temporal_identity(item.frame_name)[1])
        for _, items in sorted(grouped.items())
    ]


def static_predictions(
    frames: list[ScoredFrame],
    model,
    configuration,
    device,
) -> tuple[dict, list[dict]]:
    records = []
    template_count = len(configuration.lights)
    for frame in frames:
        scores = torch.from_numpy(frame.scores).to(device)
        transport = log_optimal_transport(
            scores,
            torch.tensor(
                frame.dustbin_score,
                dtype=scores.dtype,
                device=device,
            ),
            iterations=60,
        )
        observation_choice = transport[:-1].argmax(dim=1)
        template_choice = transport[:, :-1].argmax(dim=0)
        predictions = []
        for observation_index, template_index in enumerate(
            observation_choice.tolist()
        ):
            if (
                template_index < template_count
                and int(template_choice[template_index]) == observation_index
            ):
                predictions.append(int(template_index))
            else:
                predictions.append(template_count)
        records.append(
            {
                "frame_name": frame.frame_name,
                "labels": frame.labels,
                "predictions": predictions,
            }
        )
    return summarize_predictions(records, configuration), records


def temporal_predictions(
    frames: list[ScoredFrame],
    model,
    configuration,
    device,
    temporal_weight: float,
    sigma_ratio: float,
    max_age: int,
    minimum_update_confidence: float,
    confidence_threshold: float,
    decoder_name: str,
    burn_in_frames: int,
    temporal_template_indices: list[int] | None = None,
    recovery_template_indices: list[int] | None = None,
    recovery_max_observations: int = 0,
    reliability_settings: dict | None = None,
) -> tuple[dict, list[dict]]:
    records = []
    for sequence in ordered_sequences(frames):
        if decoder_name in {"tracklets", "reliability_tracklets"}:
            decoder_class = (
                ReliabilityGatedTrackletDecoder
                if decoder_name == "reliability_tracklets"
                else TrackletGraphDecoder
            )
            decoder_kwargs = (
                dict(reliability_settings or {})
                if decoder_name == "reliability_tracklets"
                else {}
            )
            decoder = decoder_class(
                template_count=len(configuration.lights),
                temporal_weight=temporal_weight,
                association_gate_ratio=sigma_ratio,
                minimum_update_confidence=minimum_update_confidence,
                confidence_threshold=confidence_threshold,
                burn_in_frames=burn_in_frames,
                temporal_template_indices=temporal_template_indices,
                recovery_template_indices=recovery_template_indices,
                recovery_max_observations=recovery_max_observations,
                **decoder_kwargs,
            )
        else:
            decoder = TemporalGraphDecoder(
                template_count=len(configuration.lights),
                temporal_weight=temporal_weight,
                sigma_ratio=sigma_ratio,
                max_age=max_age,
                minimum_update_confidence=minimum_update_confidence,
                confidence_threshold=confidence_threshold,
                temporal_template_indices=temporal_template_indices,
                recovery_template_indices=recovery_template_indices,
                recovery_max_observations=recovery_max_observations,
            )
        for frame in sequence:
            scores = torch.from_numpy(frame.scores).to(device)
            points = np.asarray(
                [detection.xy for detection in frame.detections],
                dtype=np.float64,
            )
            predictions, _ = decoder.decode(
                scores,
                points,
                torch.tensor(
                    frame.dustbin_score,
                    dtype=scores.dtype,
                    device=device,
                ),
            )
            records.append(
                {
                    "frame_name": frame.frame_name,
                    "labels": frame.labels,
                    "predictions": predictions,
                    "temporal_diagnostics": dict(
                        getattr(decoder, "last_diagnostics", {})
                    ),
                }
            )
    return summarize_predictions(records, configuration), records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--missing-expert-checkpoint")
    parser.add_argument("--missing-expert-max-detections", type=int, default=8)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--temporal-weights", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--sigma-ratios", nargs="+", type=float, default=[0.04, 0.06, 0.08, 0.12, 0.18])
    parser.add_argument("--max-ages", nargs="+", type=int, default=[3, 6, 10])
    parser.add_argument("--minimum-update-confidence", type=float, default=0.35)
    parser.add_argument(
        "--decoder",
        choices=("identity_tracks", "tracklets", "reliability_tracklets"),
        default="tracklets",
    )
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.75, 0.85, 0.95],
    )
    parser.add_argument("--reliability-min-coverage", type=float, default=0.35)
    parser.add_argument("--reliability-full-coverage", type=float, default=0.75)
    parser.add_argument("--reliability-agreement-floor", type=float, default=0.45)
    parser.add_argument("--reliability-agreement-full", type=float, default=0.80)
    parser.add_argument("--reliability-residual-scale", type=float, default=0.65)
    parser.add_argument(
        "--reliability-presets",
        nargs="+",
        choices=("safe", "balanced", "permissive", "custom"),
        default=["safe"],
        help="Validation candidates for the temporal reliability gate.",
    )
    parser.add_argument(
        "--burn-in-frames",
        nargs="+",
        type=int,
        default=[0, 2, 4],
    )
    parser.add_argument(
        "--temporal-layers",
        choices=("all", "front", "rear"),
        default="all",
        help="Apply temporal score propagation to all IDs or only one array layer.",
    )
    parser.add_argument(
        "--adaptive-front-max-detections",
        nargs="+",
        type=int,
        default=[0],
        help="Validation candidates for front-layer temporal recovery; 0 disables it.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/gnn_temporal_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configuration = load_array_configuration(args.array_config)
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    model = load_gnn_checkpoint(args.checkpoint, device=str(device))
    missing_expert = (
        load_gnn_checkpoint(
            args.missing_expert_checkpoint,
            device=str(device),
        )
        if args.missing_expert_checkpoint
        else None
    )
    feature_mode = getattr(model, "observation_feature_mode", "legacy")
    template_graph = _template_graph(
        configuration,
        device,
        feature_mode,
    )
    examples = load_completed_examples(
        Path(args.annotations),
        configuration,
        include_unreviewed_labelled=True,
    )
    by_name = {frame_name: item for item in examples for frame_name in [item[0]]}
    validation = [
        by_name[name]
        for name in checkpoint_payload.get("validation_frames", [])
        if name in by_name
    ]
    test = [
        by_name[name]
        for name in checkpoint_payload.get("test_frames", [])
        if name in by_name
    ]
    if not validation or not test:
        raise ValueError(
            "Checkpoint must contain non-empty validation_frames and test_frames."
        )
    validation_scores = score_examples(
        validation,
        model,
        template_graph,
        configuration,
        device,
        missing_expert,
        args.missing_expert_max_detections,
    )
    test_scores = score_examples(
        test,
        model,
        template_graph,
        configuration,
        device,
        missing_expert,
        args.missing_expert_max_detections,
    )
    temporal_template_indices = (
        None
        if args.temporal_layers == "all"
        else [
            index
            for index, lamp in enumerate(configuration.lights)
            if lamp.layer == args.temporal_layers
        ]
    )
    front_template_indices = [
        index
        for index, lamp in enumerate(configuration.lights)
        if lamp.layer == "front"
    ]
    custom_reliability_settings = {
        "reliability_min_coverage": args.reliability_min_coverage,
        "reliability_full_coverage": args.reliability_full_coverage,
        "reliability_agreement_floor": args.reliability_agreement_floor,
        "reliability_agreement_full": args.reliability_agreement_full,
        "reliability_residual_scale": args.reliability_residual_scale,
    }
    reliability_presets = {
        "safe": custom_reliability_settings,
        "balanced": {
            "reliability_min_coverage": 0.25,
            "reliability_full_coverage": 0.65,
            "reliability_agreement_floor": 0.35,
            "reliability_agreement_full": 0.75,
            "reliability_residual_scale": 0.80,
        },
        "permissive": {
            "reliability_min_coverage": 0.15,
            "reliability_full_coverage": 0.55,
            "reliability_agreement_floor": 0.25,
            "reliability_agreement_full": 0.65,
            "reliability_residual_scale": 1.00,
        },
        "custom": custom_reliability_settings,
    }
    active_reliability_presets = (
        args.reliability_presets
        if args.decoder == "reliability_tracklets"
        else ["custom"]
    )
    static_validation, _ = static_predictions(
        validation_scores,
        model,
        configuration,
        device,
    )
    static_test, static_test_records = static_predictions(
        test_scores,
        model,
        configuration,
        device,
    )

    grid = []
    for temporal_weight in args.temporal_weights:
        for sigma_ratio in args.sigma_ratios:
            for max_age in args.max_ages:
                for confidence_threshold in args.confidence_thresholds:
                    for burn_in_frames in args.burn_in_frames:
                        for recovery_max_observations in args.adaptive_front_max_detections:
                            for reliability_preset in active_reliability_presets:
                                reliability_settings = reliability_presets[
                                    reliability_preset
                                ]
                                metrics, _ = temporal_predictions(
                                    validation_scores,
                                    model,
                                    configuration,
                                    device,
                                    temporal_weight,
                                    sigma_ratio,
                                    max_age,
                                    args.minimum_update_confidence,
                                    confidence_threshold,
                                    args.decoder,
                                    burn_in_frames,
                                    temporal_template_indices,
                                    front_template_indices,
                                    recovery_max_observations,
                                    reliability_settings,
                                )
                                grid.append(
                                    {
                                        "temporal_weight": temporal_weight,
                                        "sigma_ratio": sigma_ratio,
                                        "max_age": max_age,
                                        "confidence_threshold": confidence_threshold,
                                        "burn_in_frames": burn_in_frames,
                                        "recovery_max_observations": recovery_max_observations,
                                        "reliability_preset": reliability_preset,
                                        "reliability_settings": reliability_settings,
                                        "metrics": metrics,
                                    }
                                )
    best = max(
        grid,
        key=lambda item: (
            item["metrics"]["id_accuracy"],
            item["metrics"]["exact_frame_rate"],
            item["metrics"]["outlier_f1"],
            -item["temporal_weight"],
        ),
    )
    temporal_test, temporal_test_records = temporal_predictions(
        test_scores,
        model,
        configuration,
        device,
        best["temporal_weight"],
        best["sigma_ratio"],
        best["max_age"],
        args.minimum_update_confidence,
        best["confidence_threshold"],
        args.decoder,
        best["burn_in_frames"],
        temporal_template_indices,
        front_template_indices,
        best["recovery_max_observations"],
        best["reliability_settings"],
    )

    print(
        "validation selected "
        f"weight={best['temporal_weight']} "
        f"sigma={best['sigma_ratio']} max_age={best['max_age']} "
        f"confidence={best['confidence_threshold']} "
        f"burn_in={best['burn_in_frames']} "
        f"front_recovery<={best['recovery_max_observations']} "
        f"reliability={best['reliability_preset']}"
    )
    print(
        f"static test ID={static_test['id_accuracy']:.4f} "
        f"exact={static_test['exact_frame_rate']:.4f}"
    )
    print(
        f"temporal test ID={temporal_test['id_accuracy']:.4f} "
        f"exact={temporal_test['exact_frame_rate']:.4f}"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "checkpoint_feature_mode": feature_mode,
                "missing_expert_checkpoint": args.missing_expert_checkpoint,
                "missing_expert_max_detections": (
                    args.missing_expert_max_detections
                    if args.missing_expert_checkpoint
                    else None
                ),
                "validation_frames": [item.frame_name for item in validation_scores],
                "test_frames": [item.frame_name for item in test_scores],
                "static_validation": static_validation,
                "static_test": static_test,
                "selected_parameters": {
                    key: best[key]
                    for key in (
                        "temporal_weight",
                        "sigma_ratio",
                        "max_age",
                        "confidence_threshold",
                        "burn_in_frames",
                        "recovery_max_observations",
                        "reliability_preset",
                        "reliability_settings",
                    )
                }
                | {"temporal_layers": args.temporal_layers},
                "selected_validation": best["metrics"],
                "temporal_test": temporal_test,
                "static_test_predictions": static_test_records,
                "temporal_test_predictions": temporal_test_records,
                "validation_grid": grid,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
