"""Run detection, correspondence and pose estimation over an image sequence."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .config import load_array_configuration
from .detector import make_detector
from .io_utils import list_images, make_video_writer, read_image, write_image
from .matcher import LampArrayMatcher
from .pipeline import GuidingLightPipeline
from .sequence import order_temporal_paths, temporal_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image directory or individual image.")
    parser.add_argument("--output", default="outputs/sequence_run")
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument(
        "--pose-array-config",
        help=(
            "Optional physical 3D/camera calibration used only by PnP. "
            "The matcher keeps --array-config, so existing GNN weights remain compatible."
        ),
    )
    parser.add_argument("--detector", choices=("auto", "blob", "yolo"), default="auto")
    parser.add_argument("--weights", help="Ultralytics YOLO lamp-detection .pt weights.")
    parser.add_argument(
        "--matcher",
        choices=("geometry", "gnn", "gnn_temporal"),
        default="geometry",
    )
    parser.add_argument(
        "--matcher-weights",
        help="Trained GNN checkpoint for a GNN matcher.",
    )
    parser.add_argument(
        "--missing-expert-weights",
        help="Optional visibility-aware GNN used only for severe missing-light frames.",
    )
    parser.add_argument(
        "--missing-expert-max-detections",
        type=int,
        default=8,
        help="Use the missing-light expert at or below this detection count.",
    )
    parser.add_argument("--temporal-weight", type=float, default=24.0)
    parser.add_argument("--temporal-gate-ratio", type=float, default=0.50)
    parser.add_argument(
        "--temporal-confidence-threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument("--temporal-burn-in", type=int, default=0)
    parser.add_argument(
        "--temporal-decoder",
        choices=("tracklets", "reliability_tracklets"),
        default="reliability_tracklets",
        help="Reliability gating prevents propagation from an inconsistent history.",
    )
    parser.add_argument(
        "--disable-pose-refinement",
        action="store_true",
        help="Disable pose-guided second-pass correspondence repair.",
    )
    parser.add_argument(
        "--disable-weighted-pnp",
        action="store_true",
        help="Disable confidence/uncertainty-weighted robust PnP refinement.",
    )
    parser.add_argument(
        "--disable-template-recovery",
        action="store_true",
        help="Disable 3-D template projection, local candidate recovery, and second-pass PnP.",
    )
    parser.add_argument(
        "--enable-pose-consistency-pruning",
        action="store_true",
        help="Enable pose-guided rejection of low-confidence GNN assignments before PnP refinement.",
    )
    parser.add_argument(
        "--enable-pose-multihypothesis",
        action="store_true",
        help=(
            "Rerank ambiguous GNN top-k assignments with 3-D pose consistency "
            "before weighted PnP refinement."
        ),
    )
    parser.add_argument("--pose-hypothesis-topk", type=int, default=3)
    parser.add_argument("--pose-hypothesis-beam-width", type=int, default=48)
    parser.add_argument("--pose-hypothesis-max-changes", type=int, default=3)
    parser.add_argument(
        "--pose-hypothesis-policy",
        choices=("rescue_only", "conservative_rerank"),
        default="rescue_only",
        help=(
            "rescue_only preserves valid GNN+PnP output; conservative_rerank "
            "also allows post-hoc ID replacement for ablation."
        ),
    )
    parser.add_argument(
        "--enable-residual-pnp-refinement",
        action="store_true",
        help=(
            "After GNN matching and initial PnP, reject only low-confidence, "
            "high-reprojection-residual matches and refit weighted PnP."
        ),
    )
    parser.add_argument("--residual-pnp-min-matches", type=int, default=10)
    parser.add_argument("--residual-pnp-max-matches", type=int, default=12)
    parser.add_argument("--residual-pnp-low-confidence", type=float, default=0.58)
    parser.add_argument("--residual-pnp-sigma", type=float, default=2.75)
    parser.add_argument("--residual-pnp-max-rejections", type=int, default=2)
    parser.add_argument(
        "--disable-sparse-pose",
        action="store_true",
        help="Require the original dense PnP path for ablation.",
    )
    parser.add_argument(
        "--pose-prediction-frames",
        type=int,
        default=5,
        help="Maximum consecutive frames for low-confidence temporal pose prediction.",
    )
    parser.add_argument(
        "--temporal-layers",
        choices=("all", "front", "rear"),
        default="all",
        help="Apply temporal GNN evidence to all IDs or only one lamp-array layer.",
    )
    parser.add_argument(
        "--temporal-front-max-detections",
        type=int,
        default=0,
        help=(
            "Also propagate front-layer tracks when detections are at or below "
            "this count; 0 disables adaptive front recovery."
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0 for GNN inference.")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Skip per-frame JPEG writes for an algorithm-only throughput benchmark.",
    )
    parser.add_argument(
        "--sequence-mode",
        choices=("filename_groups", "single"),
        default="filename_groups",
        help="Reset temporal state between filename-derived clips or treat all images as one clip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input}")
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if args.sequence_mode == "filename_groups":
        image_paths = order_temporal_paths(image_paths)
    output = Path(args.output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    configuration = load_array_configuration(args.array_config)
    pose_configuration = (
        load_array_configuration(args.pose_array_config)
        if args.pose_array_config
        else configuration
    )
    detector = make_detector(args.detector, args.weights, args.confidence, args.imgsz)
    temporal_settings = {
        "decoder": args.temporal_decoder,
        "temporal_weight": args.temporal_weight,
        "association_gate_ratio": args.temporal_gate_ratio,
        "minimum_update_confidence": 0.0,
        "confidence_threshold": args.temporal_confidence_threshold,
        "burn_in_frames": args.temporal_burn_in,
        "temporal_template_indices": (
            None
            if args.temporal_layers == "all"
            else [
                index
                for index, lamp in enumerate(configuration.lights)
                if lamp.layer == args.temporal_layers
            ]
        ),
        "recovery_template_indices": [
            index
            for index, lamp in enumerate(configuration.lights)
            if lamp.layer == "front"
        ],
        "recovery_max_observations": args.temporal_front_max_detections,
    }
    matcher = LampArrayMatcher(
        configuration,
        args.matcher,
        args.matcher_weights,
        args.device,
        temporal_settings=temporal_settings,
        missing_expert_weights=args.missing_expert_weights,
        missing_expert_max_detections=args.missing_expert_max_detections,
    )
    pipeline = GuidingLightPipeline(
        configuration,
        detector,
        matcher,
        pose_configuration=pose_configuration,
        enable_pose_refinement=not args.disable_pose_refinement,
        allow_sparse_pose=not args.disable_sparse_pose,
        max_pose_prediction_frames=args.pose_prediction_frames,
        use_weighted_pnp=not args.disable_weighted_pnp,
        enable_template_recovery=not args.disable_template_recovery,
        enable_pose_consistency_pruning=args.enable_pose_consistency_pruning,
        enable_pose_multihypothesis=args.enable_pose_multihypothesis,
        enable_residual_pnp_refinement=args.enable_residual_pnp_refinement,
        residual_pnp_min_matches=args.residual_pnp_min_matches,
        residual_pnp_max_matches=args.residual_pnp_max_matches,
        residual_pnp_low_confidence=args.residual_pnp_low_confidence,
        residual_pnp_sigma=args.residual_pnp_sigma,
        residual_pnp_max_rejections=args.residual_pnp_max_rejections,
        pose_hypothesis_topk=args.pose_hypothesis_topk,
        pose_hypothesis_beam_width=args.pose_hypothesis_beam_width,
        pose_hypothesis_max_changes=args.pose_hypothesis_max_changes,
        pose_hypothesis_policy=args.pose_hypothesis_policy,
    )
    writer = None
    records: list[dict] = []
    jsonl_path = output / "results.jsonl"
    previous_sequence = None
    processing_started = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(tqdm(image_paths, desc="Processing frames")):
            if args.sequence_mode == "filename_groups":
                current_sequence, _ = temporal_identity(image_path.name)
                if previous_sequence is not None and current_sequence != previous_sequence:
                    pipeline.reset_temporal_state()
                previous_sequence = current_sequence
            image = read_image(image_path)
            result = pipeline.process(image)
            record = result.to_dict(index, image_path.name)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not args.no_save_frames:
                overlay_path = frames_dir / f"{index:04d}_{image_path.stem}.jpg"
                write_image(overlay_path, result.overlay)
            if not args.no_video:
                if writer is None:
                    writer = make_video_writer(output / "overlay.mp4", result.overlay.shape, args.fps)
                writer.write(result.overlay)
    if writer is not None:
        writer.release()
    elapsed_seconds = time.perf_counter() - processing_started
    successful = [record for record in records if record["pose"]["success"]]
    measured_poses = [
        record for record in successful
        if not record["pose"].get("approximate", False)
    ]
    approximate_poses = [
        record for record in successful
        if record["pose"].get("approximate", False)
    ]
    missing_id_counts = [
        len(record.get("missing_ids", []))
        for record in records
        if record.get("missing_ids")
    ]
    summary = {
        "input": str(args.input).replace("\\", "/"),
        "matching_array_config": str(args.array_config).replace("\\", "/"),
        "pose_array_config": str(
            args.pose_array_config or args.array_config
        ).replace("\\", "/"),
        "detector": args.detector,
        "detector_weights": str(args.weights or "").replace("\\", "/"),
        "matcher": args.matcher,
        "matcher_weights": str(args.matcher_weights or "").replace("\\", "/"),
        "missing_expert": (
            {
                "weights": str(args.missing_expert_weights).replace("\\", "/"),
                "max_detections": args.missing_expert_max_detections,
            }
            if args.missing_expert_weights
            else None
        ),
        "sequence_mode": args.sequence_mode,
        "temporal_settings": (
            {
                "weight": args.temporal_weight,
                "gate_ratio": args.temporal_gate_ratio,
                "confidence_threshold": args.temporal_confidence_threshold,
                "burn_in_frames": args.temporal_burn_in,
                "decoder": args.temporal_decoder,
                "layers": args.temporal_layers,
                "front_recovery_max_detections": (
                    args.temporal_front_max_detections
                ),
            }
            if args.matcher == "gnn_temporal"
            else None
        ),
        "pose_settings": {
            "pose_guided_refinement": not args.disable_pose_refinement,
            "sparse_pose": not args.disable_sparse_pose,
            "prediction_frames": args.pose_prediction_frames,
            "weighted_pnp": not args.disable_weighted_pnp,
            "template_recovery": not args.disable_template_recovery,
            "pose_consistency_pruning": args.enable_pose_consistency_pruning,
            "pose_multihypothesis": args.enable_pose_multihypothesis,
            "pose_hypothesis_topk": args.pose_hypothesis_topk,
            "pose_hypothesis_beam_width": args.pose_hypothesis_beam_width,
            "pose_hypothesis_max_changes": args.pose_hypothesis_max_changes,
            "pose_hypothesis_policy": args.pose_hypothesis_policy,
            "residual_pnp_refinement": args.enable_residual_pnp_refinement,
            "residual_pnp_min_matches": args.residual_pnp_min_matches,
            "residual_pnp_max_matches": args.residual_pnp_max_matches,
            "residual_pnp_low_confidence": args.residual_pnp_low_confidence,
            "residual_pnp_sigma": args.residual_pnp_sigma,
            "residual_pnp_max_rejections": args.residual_pnp_max_rejections,
        },
        "pose_multihypothesis": {
            "evaluated_frames": sum(
                record.get("matcher_diagnostics", {}).get("evaluated_hypotheses", 0) > 0
                for record in records
            ),
            "accepted_frames": sum(
                record.get("matcher_diagnostics", {}).get("accepted", False)
                for record in records
            ),
            "mean_evaluated_hypotheses": (
                sum(
                    record.get("matcher_diagnostics", {}).get("evaluated_hypotheses", 0)
                    for record in records
                ) / max(len(records), 1)
            ),
            "reason_counts": dict(
                Counter(
                    record.get("matcher_diagnostics", {}).get("reason")
                    for record in records
                    if record.get("matcher_diagnostics", {}).get("reason")
                )
            ),
        } if args.enable_pose_multihypothesis else None,
        "residual_pnp_refinement": {
            "evaluated_frames": sum(
                bool(record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("enabled"))
                and record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("reason")
                not in {"below_target_match_count", "above_target_match_count", "invalid_initial_pose"}
                for record in records
            ),
            "accepted_frames": sum(
                record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("accepted", False)
                for record in records
            ),
            "rejected_matches": sum(
                int(record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("rejected_matches", 0))
                for record in records
            ),
            "reason_counts": dict(
                Counter(
                    record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("reason")
                    for record in records
                    if record.get("matcher_diagnostics", {}).get("residual_pnp_refinement", {}).get("reason")
                )
            ),
        } if args.enable_residual_pnp_refinement else None,
        "matcher_method_counts": dict(
            Counter(record["matcher"] for record in records)
        ),
        "frames_with_missing_ids": len(missing_id_counts),
        "mean_missing_ids_when_reported": (
            sum(missing_id_counts) / len(missing_id_counts)
            if missing_id_counts
            else 0.0
        ),
        "frames": len(records),
        "saved_frame_overlays": not args.no_save_frames,
        "saved_video": not args.no_video,
        "elapsed_seconds": elapsed_seconds,
        "throughput_fps": len(records) / max(elapsed_seconds, 1e-9),
        "end_to_end_ms_per_frame": 1000.0 * elapsed_seconds / max(len(records), 1),
        "pose_success_frames": len(successful),
        "measured_pose_frames": len(measured_poses),
        "approximate_pose_frames": len(approximate_poses),
        "pose_source_counts": dict(
            Counter(record["pose"]["source"] for record in successful)
        ),
        "raw_pose_source_counts": dict(
            Counter(
                record["raw_pose"]["source"]
                for record in records
                if record["raw_pose"].get("success")
            )
        ),
        "raw_pose_failed_source_counts": dict(
            Counter(
                record["raw_pose"]["source"]
                for record in records
                if not record["raw_pose"].get("success")
            )
        ),
        "frames_with_recovered_lamps": sum(
            any(match.get("observation_status", "observed").startswith("recovered") for match in record["matches"])
            for record in records
        ),
        "recovered_lamp_observations": sum(
            match.get("observation_status", "observed").startswith("recovered")
            for record in records
            for match in record["matches"]
        ),
        "mean_detections": sum(len(record["detections"]) for record in records) / len(records),
        "mean_matches": sum(len(record["matches"]) for record in records) / len(records),
        "mean_reprojection_error_px": (
            sum(record["pose"].get("reprojection_error_px", 0.0) for record in successful) / len(successful) if successful else None
        ),
        "mean_measured_reprojection_error_px": (
            sum(
                record["pose"].get("reprojection_error_px", 0.0)
                for record in measured_poses
            ) / len(measured_poses)
            if measured_poses
            else None
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
