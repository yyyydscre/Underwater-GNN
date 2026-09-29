"""Controlled missing-light stress test on the fixed held-out real frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .compare_gnn_fewshot import _template_graph
from .config import load_array_configuration
from .evaluate_gnn_temporal import (
    score_examples,
    static_predictions,
    temporal_predictions,
)
from .graph import load_gnn_checkpoint
from .test_gnn_real_sequence_holdout import split_examples_within_sequences
from .train_gnn_fewshot_real_pilot import load_completed_examples


METRIC_KEYS = (
    "id_accuracy",
    "front_id_accuracy",
    "rear_id_accuracy",
    "exact_frame_rate",
    "missing_id_f1",
    "four_correct_support_rate",
)

T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--visible-counts", nargs="+", type=int, default=[13, 11, 8, 5, 3])
    parser.add_argument("--modes", nargs="+", choices=["random", "block"], default=["random", "block"])
    parser.add_argument("--expert-threshold", type=int, default=8)
    parser.add_argument("--temporal-weight", type=float, default=12.0)
    parser.add_argument("--temporal-gate-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/paper_experiments/occlusion_stress.json")
    return parser.parse_args()


def _checkpoint(directory: str, seed: int) -> Path:
    path = Path(directory) / f"seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _controlled_subset(examples, visible_count: int, rng, mode: str):
    stressed = []
    for frame_name, detections, labels in examples:
        labelled = [index for index, label in enumerate(labels) if label >= 0]
        if len(labelled) < visible_count:
            continue
        drop_count = len(labelled) - visible_count
        dropped: set[int] = set()
        if drop_count:
            if mode == "random":
                dropped = set(
                    int(value)
                    for value in rng.choice(labelled, size=drop_count, replace=False)
                )
            else:
                anchor = int(rng.choice(labelled))
                anchor_xy = np.asarray(detections[anchor].xy, dtype=np.float64)
                ranked = sorted(
                    labelled,
                    key=lambda index: float(
                        np.linalg.norm(
                            np.asarray(detections[index].xy, dtype=np.float64)
                            - anchor_xy
                        )
                    ),
                )
                dropped = set(ranked[:drop_count])
        stressed.append(
            (
                frame_name,
                [item for index, item in enumerate(detections) if index not in dropped],
                [int(label) for index, label in enumerate(labels) if index not in dropped],
            )
        )
    return stressed


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if len(array) else 0.0
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = (
        float(
            T_975.get(len(array) - 1, 1.96)
            * std
            / np.sqrt(len(array))
        )
        if len(array) > 1
        else 0.0
    )
    return {
        "n": int(len(array)),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configuration = load_array_configuration(args.array_config)
    examples = load_completed_examples(
        Path(args.annotations),
        configuration,
        include_unreviewed_labelled=True,
    )
    _, _, test = split_examples_within_sequences(examples, 0.2, 0.2)
    records = []

    for seed in args.seeds:
        baseline = load_gnn_checkpoint(_checkpoint(args.baseline_dir, seed), str(device))
        main_model = load_gnn_checkpoint(_checkpoint(args.main_dir, seed), str(device))
        expert = load_gnn_checkpoint(_checkpoint(args.expert_dir, seed), str(device))
        baseline_template = _template_graph(
            configuration,
            device,
            getattr(baseline, "observation_feature_mode", "legacy"),
        )
        proposed_template = _template_graph(
            configuration,
            device,
            getattr(main_model, "observation_feature_mode", "geometry_uncertainty"),
        )
        for mode_index, mode in enumerate(args.modes):
            for visible_count in args.visible_counts:
                rng = np.random.default_rng(seed * 10_000 + mode_index * 100 + visible_count)
                stressed = _controlled_subset(test, visible_count, rng, mode)
                baseline_scores = score_examples(
                    stressed,
                    baseline,
                    baseline_template,
                    configuration,
                    device,
                )
                baseline_metrics, _ = static_predictions(
                    baseline_scores,
                    baseline,
                    configuration,
                    device,
                )
                proposed_scores = score_examples(
                    stressed,
                    main_model,
                    proposed_template,
                    configuration,
                    device,
                    missing_expert=expert,
                    missing_expert_max_detections=args.expert_threshold,
                )
                proposed_static, _ = static_predictions(
                    proposed_scores,
                    main_model,
                    configuration,
                    device,
                )
                proposed_temporal, _ = temporal_predictions(
                    proposed_scores,
                    main_model,
                    configuration,
                    device,
                    temporal_weight=12.0,
                    sigma_ratio=0.25,
                    max_age=4,
                    minimum_update_confidence=0.0,
                    confidence_threshold=0.9,
                    decoder_name="tracklets",
                    burn_in_frames=0,
                    temporal_template_indices=list(range(7, 13)),
                )
                visibility_scores = score_examples(
                    stressed,
                    expert,
                    proposed_template,
                    configuration,
                    device,
                )
                visibility_static, _ = static_predictions(
                    visibility_scores,
                    expert,
                    configuration,
                    device,
                )
                visibility_temporal, _ = temporal_predictions(
                    visibility_scores,
                    expert,
                    configuration,
                    device,
                    temporal_weight=12.0,
                    sigma_ratio=0.25,
                    max_age=4,
                    minimum_update_confidence=0.0,
                    confidence_threshold=0.9,
                    decoder_name="tracklets",
                    burn_in_frames=0,
                    temporal_template_indices=list(range(7, 13)),
                )
                visibility_reliability_temporal, _ = temporal_predictions(
                    visibility_scores,
                    expert,
                    configuration,
                    device,
                    temporal_weight=args.temporal_weight,
                    sigma_ratio=args.temporal_gate_ratio,
                    max_age=4,
                    minimum_update_confidence=0.35,
                    confidence_threshold=0.9,
                    decoder_name="reliability_tracklets",
                    burn_in_frames=0,
                    temporal_template_indices=list(range(7, 13)),
                )
                for method, metrics in (
                    ("baseline_static", baseline_metrics),
                    ("hard_gated_static", proposed_static),
                    ("hard_gated_temporal", proposed_temporal),
                    ("visibility_static", visibility_static),
                    ("visibility_temporal", visibility_temporal),
                    (
                        "visibility_reliability_temporal",
                        visibility_reliability_temporal,
                    ),
                ):
                    records.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "visible_count": visible_count,
                            "evaluated_frames": len(stressed),
                            "method": method,
                            "metrics": metrics,
                        }
                    )

    summary = []
    for mode in args.modes:
        for visible_count in args.visible_counts:
            for method in (
                "baseline_static",
                "hard_gated_static",
                "hard_gated_temporal",
                "visibility_static",
                "visibility_temporal",
                "visibility_reliability_temporal",
            ):
                selected = [
                    record
                    for record in records
                    if record["mode"] == mode
                    and record["visible_count"] == visible_count
                    and record["method"] == method
                ]
                summary.append(
                    {
                        "mode": mode,
                        "visible_count": visible_count,
                        "method": method,
                        "evaluated_frames_per_seed": selected[0]["evaluated_frames"] if selected else 0,
                        "metrics": {
                            key: _summary([record["metrics"][key] for record in selected])
                            for key in METRIC_KEYS
                        },
                    }
                )

    payload = {
        "protocol": "fixed_real_holdout_with_controlled_observation_removal",
        "notes": [
            "This is a synthetic occlusion stress test on real held-out detections, not an additional real test set.",
            "Random removal and spatially contiguous block removal are reported separately.",
            "The three-visible-light result measures raw correspondence behavior; production inference rejects fewer than four detections unless a temporal prior is available.",
        ],
        "arguments": vars(args),
        "records": records,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summary:
        if row["method"] == "visibility_reliability_temporal":
            metric = row["metrics"]["id_accuracy"]
            print(
                f"{row['mode']:<6} visible={row['visible_count']:>2} "
                f"ID={metric['mean']:.4f}+/-{metric['std']:.4f} "
                "support4="
                f"{row['metrics']['four_correct_support_rate']['mean']:.4f}"
            )
    print(f"Wrote controlled occlusion stress test: {output.resolve()}")


if __name__ == "__main__":
    main()
