"""Summarize weighted-PnP and template-recovery sequence ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, summary: dict, identity: dict | None) -> dict:
    frames = int(summary["frames"])
    row = {
        "method": name,
        "frames": frames,
        "pose_success_frames": int(summary["pose_success_frames"]),
        "pose_success_rate": summary["pose_success_frames"] / max(frames, 1),
        "strict_measured_frames": int(summary["measured_pose_frames"]),
        "strict_measured_rate": summary["measured_pose_frames"] / max(frames, 1),
        "approximate_frames": int(summary["approximate_pose_frames"]),
        "mean_reprojection_error_px": summary["mean_reprojection_error_px"],
        "mean_strict_reprojection_error_px": summary["mean_measured_reprojection_error_px"],
        "frames_with_recovered_lamps": int(summary.get("frames_with_recovered_lamps", 0)),
        "recovered_lamp_observations": int(summary.get("recovered_lamp_observations", 0)),
        "throughput_fps": summary.get("throughput_fps"),
    }
    if identity is not None:
        metrics = identity["gnn_identity"]
        row["id_accuracy"] = metrics["id_accuracy"]
        row["exact_frame_rate"] = metrics["exact_frame_rate"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--annotation-metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    methods = {
        "unweighted_no_recovery": "A_unweighted_no_recovery",
        "weighted_no_recovery": "B_weighted_no_recovery",
        "unweighted_template_recovery": "C2_unweighted_template_recovery_strict",
        "weighted_template_recovery": "E_weighted_template_recovery_strict",
    }
    annotation = _load(Path(args.annotation_metrics)) if args.annotation_metrics else {}
    rows = []
    for name, directory in methods.items():
        summary = _load(root / directory / "summary.json")
        metric_key = {
            "unweighted_no_recovery": "A",
            "weighted_no_recovery": "B",
            "unweighted_template_recovery": "C",
            "weighted_template_recovery": "D",
        }[name]
        identity = annotation.get("methods", {}).get(metric_key)
        rows.append(_row(name, summary, identity))

    baseline = rows[0]
    final = rows[-1]
    deltas = {
        "pose_success_pp": 100.0 * (final["pose_success_rate"] - baseline["pose_success_rate"]),
        "strict_measured_pp": 100.0 * (final["strict_measured_rate"] - baseline["strict_measured_rate"]),
        "strict_measured_frames": final["strict_measured_frames"] - baseline["strict_measured_frames"],
        "strict_reprojection_error_px": final["mean_strict_reprojection_error_px"] - baseline["mean_strict_reprojection_error_px"],
        "strict_reprojection_error_relative_percent": 100.0 * (
            final["mean_strict_reprojection_error_px"]
            / baseline["mean_strict_reprojection_error_px"]
            - 1.0
        ),
    }
    if "id_accuracy" in final and "id_accuracy" in baseline:
        deltas["id_accuracy_pp"] = 100.0 * (final["id_accuracy"] - baseline["id_accuracy"])
        deltas["exact_frame_rate_pp"] = 100.0 * (
            final["exact_frame_rate"] - baseline["exact_frame_rate"]
        )
    payload = {
        "protocol": {
            "frames": baseline["frames"],
            "gnn_seed": 43,
            "note": "Recovered lamps require local image evidence; pure template projections are never PnP measurements.",
        },
        "methods": rows,
        "final_vs_baseline": deltas,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    header = "| Method | Pose available | Strict measured | Strict error (px) | ID-Acc | Exact frame | Recovered |\n|---|---:|---:|---:|---:|---:|---:|"
    table = [header]
    for row in rows:
        table.append(
            "| {method} | {pose:.2f}% | {strict:.2f}% | {error:.3f} | {id_acc} | {exact} | {recovered} |".format(
                method=row["method"],
                pose=100.0 * row["pose_success_rate"],
                strict=100.0 * row["strict_measured_rate"],
                error=row["mean_strict_reprojection_error_px"],
                id_acc=(f"{100.0 * row['id_accuracy']:.2f}%" if "id_accuracy" in row else "-"),
                exact=(f"{100.0 * row['exact_frame_rate']:.2f}%" if "exact_frame_rate" in row else "-"),
                recovered=row["recovered_lamp_observations"],
            )
        )
    output.with_suffix(".md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
