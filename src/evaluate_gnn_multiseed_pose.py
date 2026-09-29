"""Compare dense and sparse PnP across multi-seed held-out GNN predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_array_configuration
from .evaluate_gnn_holdout_pipeline import evaluate_prediction_records
from .train_gnn_fewshot_real_pilot import load_completed_examples


T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--matching-array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--pose-array-config", default="configs/lamp_array_pose_unity_approx.yaml")
    parser.add_argument("--output", default="outputs/paper_experiments/multiseed_pose.json")
    return parser.parse_args()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = T_975.get(len(array) - 1, 1.96) * std / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
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
    records = []
    for seed in args.seeds:
        evaluation_path = Path(args.evaluation_dir) / f"seed_{seed}.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        predictions = evaluation["temporal_test_predictions"]
        for pose_mode in ("dense", "sparse"):
            metrics = evaluate_prediction_records(
                predictions,
                examples_by_name,
                Path(args.images),
                matching_configuration,
                pose_configuration,
                f"seed_{seed}_temporal",
                pose_mode,
            )
            records.append(
                {
                    "seed": seed,
                    "pose_mode": pose_mode,
                    "metrics": metrics,
                }
            )

    summary = {}
    for pose_mode in ("dense", "sparse"):
        selected = [row for row in records if row["pose_mode"] == pose_mode]
        summary[pose_mode] = {
            "gnn_id_accuracy": _summary(
                [row["metrics"]["gnn_identity"]["id_accuracy"] for row in selected]
            ),
            "gnn_exact_frame_rate": _summary(
                [row["metrics"]["gnn_identity"]["exact_frame_rate"] for row in selected]
            ),
            "pose_success_rate": _summary(
                [row["metrics"]["predicted_id_pose"]["success_rate"] for row in selected]
            ),
            "measured_success_frames": _summary(
                [float(row["metrics"]["predicted_id_pose"]["measured_success_frames"]) for row in selected]
            ),
            "approximate_success_frames": _summary(
                [float(row["metrics"]["predicted_id_pose"]["approximate_success_frames"]) for row in selected]
            ),
            "mean_reprojection_error_px": _summary(
                [
                    float(row["metrics"]["predicted_id_pose"]["mean_reprojection_error_px"])
                    for row in selected
                    if row["metrics"]["predicted_id_pose"]["mean_reprojection_error_px"] is not None
                ]
            ),
        }

    payload = {
        "protocol": "five_seed_fixed_holdout_identity_to_pose",
        "notes": [
            "The camera/array configuration is approximate; success and reprojection consistency are reported without claiming metric pose accuracy.",
            "Each frame is evaluated independently here, so temporal pose carry-forward is excluded.",
        ],
        "arguments": vars(args),
        "records": records,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, values in summary.items():
        pose = values["pose_success_rate"]
        print(f"{mode:<6} pose={pose['mean']:.4f}+/-{pose['std']:.4f}")
    print(f"Wrote multi-seed pose evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
