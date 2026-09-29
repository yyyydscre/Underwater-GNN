"""Run a leakage-aware real few-shot GNN pilot from completed identity labels.

Only frames marked reviewed, or legacy frames where every detection has an
explicit F0-F6/R0-R5 identity, are included.  This prevents default nulls in
unfinished annotation frames from being treated as false detections.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .compare_gnn_fewshot import _make_model, _pretrain, _template_graph, _train_samples, evaluate
from .config import load_array_configuration
from .schema import COLOR_INDEX, LightDetection


def load_completed_examples(
    path: Path,
    configuration,
    include_unreviewed_labelled: bool = False,
) -> list[tuple[str, list[LightDetection], list[int]]]:
    id_to_index = {lamp.lamp_id: index for index, lamp in enumerate(configuration.lights)}
    examples: list[tuple[str, list[LightDetection], list[int]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("dataset_partition") == "unused":
            continue
        entries = record.get("detections", [])
        lamp_ids = [entry.get("lamp_id") for entry in entries]
        reviewed = record.get("review_status") == "reviewed"
        legacy_complete = bool(entries) and all(lamp_id is not None for lamp_id in lamp_ids)
        explicitly_labelled = sum(lamp_id is not None for lamp_id in lamp_ids) >= 4
        if not (
            reviewed
            or legacy_complete
            or (include_unreviewed_labelled and explicitly_labelled)
        ):
            continue

        detections: list[LightDetection] = []
        labels: list[int] = []
        assigned_ids: set[str] = set()
        for entry in entries:
            lamp_id = entry.get("lamp_id")
            if lamp_id is not None and lamp_id not in id_to_index:
                raise ValueError(f"Unknown lamp_id {lamp_id!r} at {path}:{line_number}")
            if lamp_id is not None:
                if lamp_id in assigned_ids:
                    raise ValueError(f"Duplicate lamp_id {lamp_id!r} at {path}:{line_number}")
                assigned_ids.add(lamp_id)
            colour = str(entry.get("color", "other"))
            probabilities = np.asarray(entry.get("color_probs", []), dtype=np.float32)
            if probabilities.shape != (3,):
                probabilities = np.full(3, 0.05, dtype=np.float32)
                probabilities[COLOR_INDEX.get(colour, COLOR_INDEX["other"])] = 0.9
            probabilities = np.clip(probabilities, 0.001, None)
            probabilities /= probabilities.sum()
            covariance = entry.get("center_covariance")
            if covariance is not None:
                covariance = np.asarray(covariance, dtype=np.float64)
                if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
                    covariance = None
            bbox = entry.get("bbox_xyxy")
            if bbox is not None:
                bbox = np.asarray(bbox, dtype=np.float64)
                if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                    bbox = None
            detections.append(
                LightDetection(
                    xy=np.asarray(entry["xy"], dtype=np.float64),
                    confidence=float(entry.get("confidence", 0.8)),
                    color_probs=probabilities,
                    color=colour,
                    radius=float(entry.get("radius", 7.0)),
                    brightness=float(entry.get("brightness", 220.0)),
                    source="real_identity_annotation",
                    center_quality=float(entry.get("center_quality", entry.get("center_confidence", 0.8))),
                    bbox_xyxy=bbox,
                    detector_confidence=float(
                        entry.get("detector_confidence", entry.get("confidence", 0.8))
                    ),
                    center_confidence=float(
                        entry.get("center_confidence", entry.get("center_quality", 0.8))
                    ),
                    center_method=str(entry.get("center_method", "unknown")),
                    center_covariance=covariance,
                    center_valid=entry.get("center_valid"),
                )
            )
            labels.append(id_to_index[lamp_id] if lamp_id is not None else -1)
        if len(detections) >= 4 and any(label >= 0 for label in labels):
            examples.append((str(record["frame_name"]), detections, labels))
    if len(examples) < 6:
        raise ValueError("At least six completed labelled frames are required for this pilot.")
    return examples


def metric_row(name: str, model, validation, template_graph, template_count: int, device, k_neighbors: int) -> dict:
    metrics = evaluate(
        model,
        [(detections, labels) for _, detections, labels in validation],
        template_graph,
        template_count,
        device,
        k_neighbors,
    )
    return {"method": name, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations/gnn_daoyin_center_seeded_labelled.jsonl")
    parser.add_argument("--array-config", default="configs/lamp_array_3d.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn_matcher.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/gnn_real_fewshot_pilot.json")
    parser.add_argument("--checkpoint", default="models/gnn_matcher_real_fewshot_pilot.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5.")
    device = torch.device(args.device)
    configuration = load_array_configuration(args.array_config)
    with Path(args.gnn_config).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    examples = load_completed_examples(Path(args.annotations), configuration)
    validation_count = max(1, int(np.ceil(len(examples) * args.validation_fraction)))
    training = examples[:-validation_count]
    validation = examples[-validation_count:]
    if not training:
        raise ValueError("No training frames remain after validation split.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    template_graph = _template_graph(configuration, device)
    template_count = len(configuration.lights)
    train_samples = [(detections, labels) for _, detections, labels in training]
    rows: list[dict] = []

    scratch = _make_model(settings["model"], device)
    _train_samples(
        scratch,
        train_samples,
        template_graph,
        template_count,
        device,
        configuration.k_neighbors,
        args.finetune_epochs,
        float(settings["training"]["learning_rate"]),
    )
    rows.append(metric_row("real_fewshot_from_scratch", scratch, validation, template_graph, template_count, device, configuration.k_neighbors))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pretrained = _make_model(settings["model"], device)
    _pretrain(
        pretrained,
        configuration,
        template_graph,
        device,
        configuration.k_neighbors,
        args.seed,
        args.pretrain_steps,
        float(settings["training"]["learning_rate"]),
    )
    rows.append(metric_row("synthetic_pretrained_zero_shot", pretrained, validation, template_graph, template_count, device, configuration.k_neighbors))

    finetuned = copy.deepcopy(pretrained)
    _train_samples(
        finetuned,
        train_samples,
        template_graph,
        template_count,
        device,
        configuration.k_neighbors,
        args.finetune_epochs,
        float(settings["training"]["learning_rate"]) * 0.2,
    )
    rows.append(metric_row("synthetic_pretrained_real_finetune", finetuned, validation, template_graph, template_count, device, configuration.k_neighbors))

    print("\nReal few-shot temporal hold-out pilot")
    print(f"completed_frames={len(examples)} train={len(training)} val={len(validation)}")
    print("method                                  ID-Acc  Overall  Outlier-F1")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['method']:<39} {metrics['id_accuracy']:.3f}   "
            f"{metrics['overall_assignment_accuracy']:.3f}    {metrics['outlier_f1']:.3f}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": finetuned.state_dict(),
            "model": settings["model"],
            "array_config": args.array_config,
            "training": "synthetic_pretrain_real_fewshot_temporal_holdout",
            "train_frames": [name for name, _, _ in training],
            "validation_frames": [name for name, _, _ in validation],
        },
        checkpoint_path,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "experiment": "real few-shot temporal hold-out pilot",
                "important_note": "Frames selected as complete by explicit labels. The final contiguous 20% of those frames are held out and never used for fine-tuning.",
                "arguments": vars(args),
                "completed_frame_count": len(examples),
                "training_frames": [name for name, _, _ in training],
                "validation_frames": [name for name, _, _ in validation],
                "results": rows,
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
