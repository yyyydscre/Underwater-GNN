"""Evaluate held-out GNN identities and their PnP impact on identical centres."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_array_configuration
from .evaluate_pipeline_stages import _pose_summary
from .io_utils import read_image
from .matcher import MatchResult
from .pose import PnPPoseEstimator
from .schema import LampMatch
from .test_gnn_real_sequence_holdout import summarize_predictions
from .train_gnn_fewshot_real_pilot import load_completed_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--gnn-evaluation", required=True)
    parser.add_argument(
        "--prediction-fields",
        nargs="+",
        default=["static_test_predictions", "temporal_test_predictions"],
    )
    parser.add_argument(
        "--matching-array-config",
        default="configs/lamp_array_3d.yaml",
    )
    parser.add_argument(
        "--pose-array-config",
        default="configs/lamp_array_pose_unity_approx.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/gnn_holdout_pipeline/metrics.json",
    )
    parser.add_argument(
        "--pose-mode",
        choices=["dense", "sparse"],
        default="sparse",
        help="Use the original >=8-correspondence PnP or the sparse-aware backend.",
    )
    return parser.parse_args()


def matches_from_assignments(
    detections,
    assignments: list[int],
    matching_configuration,
    pose_configuration,
    method: str,
) -> MatchResult:
    matches = []
    template_count = len(matching_configuration.lights)
    for detection_index, (detection, assignment) in enumerate(
        zip(detections, assignments)
    ):
        if assignment < 0 or assignment >= template_count:
            continue
        lamp_id = matching_configuration.lights[assignment].lamp_id
        matches.append(
            LampMatch(
                lamp=pose_configuration.by_id(lamp_id),
                detection_index=detection_index,
                point=detection.xy,
                confidence=1.0,
                geometric_cost=0.0,
            )
        )
    return MatchResult(matches, method, 0.0)


def evaluate_prediction_records(
    records: list[dict],
    examples_by_name: dict,
    images_directory: Path,
    matching_configuration,
    pose_configuration,
    method: str,
    pose_mode: str,
) -> dict:
    sparse = pose_mode == "sparse"
    estimator = PnPPoseEstimator(
        pose_configuration,
        minimum_ransac_matches=6 if sparse else 8,
        min_inlier_ratio=0.55 if sparse else 0.65,
        allow_sparse_pose=sparse,
    )
    predicted_poses = []
    oracle_poses = []
    exact_pose_success: list[bool] = []
    inexact_pose_success: list[bool] = []
    evaluated_records = []

    for record in records:
        frame_name = str(record["frame_name"])
        if frame_name not in examples_by_name:
            continue
        detections, annotation_labels = examples_by_name[frame_name]
        labels = [int(value) for value in record.get("labels", annotation_labels)]
        predictions = [int(value) for value in record["predictions"]]
        if len(detections) != len(labels) or len(labels) != len(predictions):
            raise ValueError(
                f"Detection/label/prediction length mismatch for {frame_name}: "
                f"{len(detections)}/{len(labels)}/{len(predictions)}"
            )
        evaluated_records.append(
            {
                "frame_name": frame_name,
                "labels": labels,
                "predictions": predictions,
            }
        )
        image_shape = read_image(images_directory / frame_name).shape
        predicted_pose = estimator.estimate(
            matches_from_assignments(
                detections,
                predictions,
                matching_configuration,
                pose_configuration,
                method,
            ),
            image_shape,
        )
        oracle_pose = estimator.estimate(
            matches_from_assignments(
                detections,
                labels,
                matching_configuration,
                pose_configuration,
                "oracle_id",
            ),
            image_shape,
        )
        predicted_poses.append(predicted_pose)
        oracle_poses.append(oracle_pose)
        labelled = [index for index, label in enumerate(labels) if label >= 0]
        exact = all(predictions[index] == labels[index] for index in labelled)
        (exact_pose_success if exact else inexact_pose_success).append(
            bool(predicted_pose.success)
        )

    return {
        "gnn_identity": summarize_predictions(
            evaluated_records,
            matching_configuration,
        ),
        "predicted_id_pose": _pose_summary(predicted_poses),
        "oracle_id_pose": _pose_summary(oracle_poses),
        "pose_conditioned_on_gnn": {
            "exact_id_frames": len(exact_pose_success),
            "pose_success_on_exact_id_frames": sum(exact_pose_success),
            "pose_success_rate_on_exact_id_frames": sum(exact_pose_success)
            / max(len(exact_pose_success), 1),
            "inexact_id_frames": len(inexact_pose_success),
            "pose_success_on_inexact_id_frames": sum(inexact_pose_success),
            "pose_success_rate_on_inexact_id_frames": sum(inexact_pose_success)
            / max(len(inexact_pose_success), 1),
        },
    }


def main() -> None:
    args = parse_args()
    matching_configuration = load_array_configuration(args.matching_array_config)
    pose_configuration = load_array_configuration(args.pose_array_config)
    examples = load_completed_examples(
        Path(args.annotations),
        matching_configuration,
        include_unreviewed_labelled=True,
    )
    examples_by_name = {
        frame_name: (detections, labels)
        for frame_name, detections, labels in examples
    }
    evaluation = json.loads(
        Path(args.gnn_evaluation).read_text(encoding="utf-8")
    )
    methods = {}
    for field in args.prediction_fields:
        if field not in evaluation:
            raise KeyError(f"Prediction field {field!r} is absent from the evaluation file.")
        method = field.removesuffix("_predictions")
        methods[method] = evaluate_prediction_records(
            evaluation[field],
            examples_by_name,
            Path(args.images),
            matching_configuration,
            pose_configuration,
            method,
            args.pose_mode,
        )
        identity = methods[method]["gnn_identity"]
        pose = methods[method]["predicted_id_pose"]
        print(
            f"{method:<20} ID={identity['id_accuracy']:.4f} "
            f"exact={identity['exact_frame_rate']:.4f} "
            f"pose={pose['success_rate']:.4f}"
        )

    payload = {
        "protocol": "heldout_fixed_centres_gnn_to_pose",
        "source_evaluation": str(args.gnn_evaluation).replace("\\", "/"),
        "matching_array_config": str(args.matching_array_config).replace("\\", "/"),
        "pose_array_config": str(args.pose_array_config).replace("\\", "/"),
        "pose_mode": args.pose_mode,
        "methods": methods,
        "notes": [
            "All methods use the same manually reviewed detections and centres.",
            "Oracle-ID PnP is a geometric upper-bound diagnostic, not external pose ground truth.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote held-out pipeline evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
