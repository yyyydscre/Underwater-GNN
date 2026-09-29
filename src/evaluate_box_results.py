"""Evaluate exported YOLO lamp boxes against 5-column YOLO ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assignment import linear_sum_assignment
from .io_utils import read_image


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    left_top = np.maximum(first[:, None, :2], second[None, :, :2])
    right_bottom = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(right_bottom - left_top, 0.0).prod(axis=2)
    first_area = np.maximum(first[:, 2] - first[:, 0], 0.0) * np.maximum(first[:, 3] - first[:, 1], 0.0)
    second_area = np.maximum(second[:, 2] - second[:, 0], 0.0) * np.maximum(second[:, 3] - second[:, 1], 0.0)
    return intersection / np.maximum(first_area[:, None] + second_area[None, :] - intersection, 1e-6)


def read_box_labels(label_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float64)
    height, width = image_shape[:2]
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        x, y, box_width, box_height = (float(value) for value in values[1:])
        boxes.append(
            [
                (x - box_width / 2.0) * width,
                (y - box_height / 2.0) * height,
                (x + box_width / 2.0) * width,
                (y + box_height / 2.0) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="results.jsonl from src.predict_yolo")
    parser.add_argument("--images", required=True, help="Image directory used during prediction")
    parser.add_argument("--labels", required=True, help="Matching 5-column YOLO label directory")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", help="Defaults to box_metrics.json alongside --results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    per_frame: list[dict[str, int | float | str]] = []
    for record in records:
        image_name = record["frame_name"]
        image = read_image(images_dir / image_name)
        truth = read_box_labels(labels_dir / f"{Path(image_name).stem}.txt", image.shape)
        predictions = np.asarray([item["bbox_xyxy"] for item in record["detections"] if "bbox_xyxy" in item], dtype=np.float64).reshape(-1, 4)
        iou = box_iou(predictions, truth)
        rows, columns = linear_sum_assignment(1.0 - iou)
        accepted = [float(iou[row, column]) for row, column in zip(rows, columns) if iou[row, column] >= args.iou]
        matches = len(accepted)
        true_positive += matches
        false_positive += len(predictions) - matches
        false_negative += len(truth) - matches
        matched_ious.extend(accepted)
        per_frame.append({"frame_name": image_name, "predictions": len(predictions), "ground_truth": len(truth), "matches": matches})
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    summary = {
        "iou_threshold": args.iou,
        "frames": len(records),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(2.0 * precision * recall / max(precision + recall, 1e-9), 5),
        "mean_matched_iou": round(float(np.mean(matched_ious)), 5) if matched_ious else None,
        "note": "This evaluates lamp boxes only. White-core error requires independently annotated white-core coordinates.",
    }
    output = Path(args.output) if args.output else results_path.with_name("box_metrics.json")
    output.write_text(json.dumps({"summary": summary, "per_frame": per_frame}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
