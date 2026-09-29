"""Evaluate detector, ID matching and pose stages under one fixed protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assignment import linear_sum_assignment
from .config import load_array_configuration
from .evaluate_box_results import box_iou, read_box_labels
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
    parser.add_argument("--box-labels")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Repeat NAME=path/to/results.jsonl for each matcher.",
    )
    parser.add_argument(
        "--matching-array-config",
        default="configs/lamp_array_3d.yaml",
    )
    parser.add_argument(
        "--pose-array-config",
        default="configs/lamp_array_pose_unity_approx.yaml",
    )
    parser.add_argument("--center-gate-px", type=float, default=8.0)
    parser.add_argument("--box-iou", type=float, default=0.50)
    parser.add_argument(
        "--output",
        default="outputs/pipeline_stage_evaluation/metrics.json",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_result_spec(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected NAME=PATH for --result, got {specification!r}")
    name, path = specification.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected NAME=PATH for --result, got {specification!r}")
    return name.strip(), Path(path.strip())


def associate_centers(
    reference_xy: np.ndarray,
    observed_xy: np.ndarray,
    maximum_distance: float,
) -> tuple[dict[int, int], list[float]]:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    observed = np.asarray(observed_xy, dtype=np.float64).reshape(-1, 2)
    if len(reference) == 0 or len(observed) == 0:
        return {}, []
    distances = np.linalg.norm(reference[:, None] - observed[None, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    accepted = {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if distances[row, column] <= maximum_distance
    }
    errors = [float(distances[row, column]) for row, column in accepted.items()]
    return accepted, errors


def _pose_summary(poses: list) -> dict:
    successful = [pose for pose in poses if pose.success]
    measured = [pose for pose in successful if not getattr(pose, "approximate", False)]
    approximate = [pose for pose in successful if getattr(pose, "approximate", False)]
    errors = np.asarray(
        [
            pose.reprojection_error_px
            for pose in successful
            if pose.reprojection_error_px is not None
        ],
        dtype=np.float64,
    )
    measured_errors = np.asarray(
        [
            pose.reprojection_error_px
            for pose in measured
            if pose.reprojection_error_px is not None
        ],
        dtype=np.float64,
    )
    return {
        "frames": len(poses),
        "success_frames": len(successful),
        "success_rate": len(successful) / max(len(poses), 1),
        "measured_success_frames": len(measured),
        "approximate_success_frames": len(approximate),
        "mean_reprojection_error_px": float(errors.mean()) if len(errors) else None,
        "mean_measured_reprojection_error_px": (
            float(measured_errors.mean()) if len(measured_errors) else None
        ),
        "median_reprojection_error_px": float(np.median(errors)) if len(errors) else None,
        "p95_reprojection_error_px": float(np.quantile(errors, 0.95)) if len(errors) else None,
    }


def _box_metrics(
    records: list[dict],
    images_directory: Path,
    labels_directory: Path,
    threshold: float,
) -> dict:
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    for record in records:
        image_name = str(record["frame_name"])
        image = read_image(images_directory / image_name)
        truth = read_box_labels(
            labels_directory / f"{Path(image_name).stem}.txt", image.shape
        )
        predictions = np.asarray(
            [
                detection["bbox_xyxy"]
                for detection in record.get("detections", [])
                if "bbox_xyxy" in detection
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        ious = box_iou(predictions, truth)
        rows, columns = linear_sum_assignment(1.0 - ious)
        accepted = [
            float(ious[row, column])
            for row, column in zip(rows, columns)
            if ious[row, column] >= threshold
        ]
        true_positive += len(accepted)
        false_positive += len(predictions) - len(accepted)
        false_negative += len(truth) - len(accepted)
        matched_ious.extend(accepted)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "iou_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _oracle_pose(
    frame_name: str,
    detections,
    labels,
    matching_configuration,
    pose_configuration,
    estimator: PnPPoseEstimator,
    image_shape: tuple[int, ...],
):
    matches = [
        LampMatch(
            lamp=pose_configuration.by_id(
                matching_configuration.lights[label].lamp_id
            ),
            detection_index=index,
            point=detection.xy,
            confidence=1.0,
            geometric_cost=0.0,
        )
        for index, (detection, label) in enumerate(zip(detections, labels))
        if label >= 0
    ]
    return estimator.estimate(MatchResult(matches, "oracle_id", 0.0), image_shape)


def _predicted_pose(
    record: dict,
    pose_configuration,
    estimator: PnPPoseEstimator,
    image_shape: tuple[int, ...],
):
    matches = []
    for item in record.get("matches", []):
        lamp_id = str(item.get("lamp_id", ""))
        if lamp_id not in pose_configuration.ids:
            continue
        matches.append(
            LampMatch(
                lamp=pose_configuration.by_id(lamp_id),
                detection_index=int(item.get("detection_index", -1)),
                point=np.asarray(item["xy"], dtype=np.float64),
                confidence=float(item.get("confidence", 1.0)),
                geometric_cost=float(item.get("geometric_cost", 0.0)),
            )
        )
    mean_cost = float(np.mean([item.geometric_cost for item in matches])) if matches else 0.0
    return estimator.estimate(
        MatchResult(matches, str(record.get("matcher", "pipeline")), mean_cost),
        image_shape,
    )


def evaluate_result(
    records: list[dict],
    annotations_by_name: dict,
    images_directory: Path,
    matching_configuration,
    pose_configuration,
    center_gate_px: float,
    box_labels: Path | None,
    box_iou_threshold: float,
) -> dict:
    template_count = len(matching_configuration.lights)
    id_to_index = {
        lamp_id: index for index, lamp_id in enumerate(matching_configuration.ids)
    }
    metric_records: list[dict] = []
    association_errors: list[float] = []
    associated = reference_count = observed_count = 0
    hard_frames: list[dict] = []
    predicted_poses = []
    oracle_poses = []
    exact_pose_success: list[bool] = []
    inexact_pose_success: list[bool] = []
    estimator = PnPPoseEstimator(pose_configuration)

    for record in records:
        frame_name = str(record["frame_name"])
        if frame_name not in annotations_by_name:
            continue
        reference_detections, reference_labels = annotations_by_name[frame_name]
        observed_detections = record.get("detections", [])
        reference_xy = np.asarray(
            [detection.xy for detection in reference_detections], dtype=np.float64
        )
        observed_xy = np.asarray(
            [item["xy"] for item in observed_detections], dtype=np.float64
        )
        association, errors = associate_centers(
            reference_xy, observed_xy, center_gate_px
        )
        association_errors.extend(errors)
        associated += len(association)
        reference_count += len(reference_detections)
        observed_count += len(observed_detections)

        predicted_by_detection = {
            int(item["detection_index"]): id_to_index[str(item["lamp_id"])]
            for item in record.get("matches", [])
            if str(item.get("lamp_id", "")) in id_to_index
        }
        labels: list[int] = []
        predictions: list[int] = []
        paired_observations = set(association.values())
        for reference_index, label in enumerate(reference_labels):
            observed_index = association.get(reference_index)
            prediction = (
                predicted_by_detection.get(observed_index, template_count)
                if observed_index is not None
                else template_count
            )
            labels.append(int(label))
            predictions.append(int(prediction))
        for observed_index in range(len(observed_detections)):
            if observed_index not in paired_observations:
                labels.append(-1)
                predictions.append(
                    int(predicted_by_detection.get(observed_index, template_count))
                )
        metric_records.append(
            {"frame_name": frame_name, "labels": labels, "predictions": predictions}
        )
        labelled = [index for index, label in enumerate(labels) if label >= 0]
        correct = sum(predictions[index] == labels[index] for index in labelled)
        exact = all(predictions[index] == labels[index] for index in labelled)
        hard_frames.append(
            {
                "frame_name": frame_name,
                "correct_ids": correct,
                "labelled_ids": len(labelled),
                "id_accuracy": correct / max(len(labelled), 1),
                "associated_centers": len(association),
            }
        )

        image_shape = read_image(images_directory / frame_name).shape
        predicted_pose = _predicted_pose(
            record, pose_configuration, estimator, image_shape
        )
        oracle_pose = _oracle_pose(
            frame_name,
            reference_detections,
            reference_labels,
            matching_configuration,
            pose_configuration,
            estimator,
            image_shape,
        )
        predicted_poses.append(predicted_pose)
        oracle_poses.append(oracle_pose)
        (exact_pose_success if exact else inexact_pose_success).append(
            bool(predicted_pose.success)
        )

    metrics = summarize_predictions(metric_records, matching_configuration)
    hard_frames.sort(key=lambda item: (item["id_accuracy"], item["labelled_ids"]))
    result = {
        "evaluated_frames": len(metric_records),
        "detection_to_annotation_association": {
            "reference_detections": reference_count,
            "observed_detections": observed_count,
            "associated": associated,
            "reference_coverage": associated / max(reference_count, 1),
            "observed_coverage": associated / max(observed_count, 1),
            "mean_center_distance_px": float(np.mean(association_errors))
            if association_errors
            else None,
            "p95_center_distance_px": float(np.quantile(association_errors, 0.95))
            if association_errors
            else None,
        },
        "gnn_identity": metrics,
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
        "hardest_frames": hard_frames[:20],
    }
    if box_labels is not None:
        result["yolo_box_detection"] = _box_metrics(
            records,
            images_directory,
            box_labels,
            box_iou_threshold,
        )
    return result


def main() -> None:
    args = parse_args()
    matching_configuration = load_array_configuration(args.matching_array_config)
    pose_configuration = load_array_configuration(args.pose_array_config)
    examples = load_completed_examples(
        Path(args.annotations),
        matching_configuration,
        include_unreviewed_labelled=True,
    )
    annotations_by_name = {
        frame_name: (detections, labels)
        for frame_name, detections, labels in examples
    }
    images_directory = Path(args.images)
    box_labels = Path(args.box_labels) if args.box_labels else None
    methods = {}
    for specification in args.result:
        name, path = _parse_result_spec(specification)
        methods[name] = evaluate_result(
            _load_jsonl(path),
            annotations_by_name,
            images_directory,
            matching_configuration,
            pose_configuration,
            args.center_gate_px,
            box_labels,
            args.box_iou,
        )
        identity = methods[name]["gnn_identity"]
        pose = methods[name]["predicted_id_pose"]
        oracle = methods[name]["oracle_id_pose"]
        print(
            f"{name:<20} ID={identity['id_accuracy']:.4f} "
            f"exact={identity['exact_frame_rate']:.4f} "
            f"pose={pose['success_rate']:.4f} "
            f"oracle_pose={oracle['success_rate']:.4f}"
        )
    payload = {
        "protocol": "fixed_detector_center_stage_evaluation",
        "annotations": str(args.annotations).replace("\\", "/"),
        "matching_array_config": str(args.matching_array_config).replace("\\", "/"),
        "pose_array_config": str(args.pose_array_config).replace("\\", "/"),
        "methods": methods,
        "notes": [
            "GNN metrics over all supplied reviewed frames are diagnostic unless the result file is a held-out split.",
            "Oracle-ID PnP isolates pose geometry from ID errors but is not external pose ground truth.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote stage evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
