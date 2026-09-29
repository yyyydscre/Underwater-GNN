"""Measure raw and refined YOLO white-core error on a held-out split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .assignment import linear_sum_assignment
from .detector import refine_white_core_center
from .io_utils import read_image
from .yolo_dataset import paired_images, read_pose_labels


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left_top = np.maximum(first[:, None, :2], second[None, :, :2])
    right_bottom = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(right_bottom - left_top, 0.0).prod(axis=2)
    first_area = np.maximum(first[:, 2] - first[:, 0], 0.0) * np.maximum(first[:, 3] - first[:, 1], 0.0)
    second_area = np.maximum(second[:, 2] - second[:, 0], 0.0) * np.maximum(second[:, 3] - second[:, 1], 0.0)
    return intersection / np.maximum(first_area[:, None] + second_area[None, :] - intersection, 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/underwater_light_pose.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", default="outputs/center_evaluation")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    values_array = np.asarray(values)
    return {
        "mean": round(float(values_array.mean()), 4),
        "median": round(float(np.median(values_array)), 4),
        "p90": round(float(np.quantile(values_array, 0.90)), 4),
        "p95": round(float(np.quantile(values_array, 0.95)), 4),
    }


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    raw_errors: list[float] = []
    refined_errors: list[float] = []
    matched = 0
    labelled = 0
    predicted = 0
    records = []
    for image_path, label_path in tqdm(paired_images(args.data, args.split, args.max_images), desc="Evaluating centres"):
        image = read_image(image_path)
        truth = read_pose_labels(label_path, image.shape)
        labelled += len(truth)
        result = model(image, conf=args.confidence, imgsz=args.imgsz, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            records.append({"image": image_path.name, "labels": len(truth), "predictions": 0, "matches": 0})
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        keypoints = result.keypoints.xy.detach().cpu().numpy()[:, 0]
        predicted += len(boxes)
        if not truth:
            continue
        truth_boxes = np.asarray([item.box_xyxy for item in truth])
        iou = box_iou(boxes, truth_boxes)
        rows, columns = linear_sum_assignment(1.0 - iou)
        image_matches = 0
        for prediction_index, truth_index in zip(rows, columns):
            if iou[prediction_index, truth_index] < 0.25 or truth[truth_index].visibility <= 0:
                continue
            raw = keypoints[prediction_index].astype(np.float64)
            box = boxes[prediction_index]
            radius = max(7, int(max(box[2] - box[0], box[3] - box[1]) * 0.55))
            refined, _ = refine_white_core_center(image, raw, radius=radius, max_shift_px=max(2.0, min(4.5, radius * 0.22)))
            ground_truth = truth[truth_index].keypoint_xy
            raw_errors.append(float(np.linalg.norm(raw - ground_truth)))
            refined_errors.append(float(np.linalg.norm(refined - ground_truth)))
            matched += 1
            image_matches += 1
        records.append({"image": image_path.name, "labels": len(truth), "predictions": len(boxes), "matches": image_matches})
    summary = {
        "split": args.split,
        "labels": labelled,
        "predictions": predicted,
        "matched": matched,
        "detection_recall": round(matched / labelled, 4) if labelled else None,
        "detection_precision": round(matched / predicted, 4) if predicted else None,
        "raw_keypoint_error_px": stats(raw_errors),
        "refined_keypoint_error_px": stats(refined_errors),
        "criterion": "The model is ready for pose only after the refined validation median and p95 are both acceptable for the physical lamp spacing.",
    }
    (output / "per_image.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
