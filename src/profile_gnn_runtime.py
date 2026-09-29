"""Profile GNN correspondence latency without detector or rendering time."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .compare_gnn_fewshot import _template_graph
from .config import load_array_configuration
from .evaluate_gnn_temporal import score_examples, static_predictions, temporal_predictions
from .graph import load_gnn_checkpoint
from .test_gnn_real_sequence_holdout import split_examples_within_sequences
from .train_gnn_fewshot_real_pilot import load_completed_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="outputs/paper_experiments/gnn_runtime.json")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile(callable_, frame_count: int, device: torch.device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        callable_()
    _synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        _synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - started) / frame_count)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "repeats": repeats,
        "frames_per_repeat": frame_count,
        "mean_ms_per_frame": float(array.mean()),
        "std_ms_per_frame": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "throughput_fps": float(1000.0 / array.mean()),
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
    baseline = load_gnn_checkpoint(args.baseline, str(device))
    main_model = load_gnn_checkpoint(args.main, str(device))
    expert = load_gnn_checkpoint(args.expert, str(device))
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

    def baseline_call():
        frames = score_examples(test, baseline, baseline_template, configuration, device)
        static_predictions(frames, baseline, configuration, device)

    def proposed_static_call():
        frames = score_examples(
            test,
            main_model,
            proposed_template,
            configuration,
            device,
            missing_expert=expert,
            missing_expert_max_detections=8,
        )
        static_predictions(frames, main_model, configuration, device)

    cached_scores = score_examples(
        test,
        main_model,
        proposed_template,
        configuration,
        device,
        missing_expert=expert,
        missing_expert_max_detections=8,
    )

    def temporal_decode_call():
        temporal_predictions(
            cached_scores,
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

    payload = {
        "protocol": "fixed_46_frame_holdout_no_detector_no_rendering",
        "device": str(device),
        "torch_version": torch.__version__,
        "frames": len(test),
        "parameter_counts": {
            "baseline": sum(parameter.numel() for parameter in baseline.parameters()),
            "main": sum(parameter.numel() for parameter in main_model.parameters()),
            "expert": sum(parameter.numel() for parameter in expert.parameters()),
        },
        "baseline_static": _profile(baseline_call, len(test), device, args.warmup, args.repeats),
        "proposed_static": _profile(proposed_static_call, len(test), device, args.warmup, args.repeats),
        "temporal_decode_only": _profile(temporal_decode_call, len(test), device, args.warmup, args.repeats),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote GNN runtime profile: {output.resolve()}")


if __name__ == "__main__":
    main()
