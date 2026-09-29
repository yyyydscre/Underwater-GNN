"""Evaluate the final lamp detector on a fixed labelled box test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _test_image_count(data_path: str) -> int:
    yaml_path = Path(data_path)
    settings = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(settings.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    split = settings.get("test", settings.get("val"))
    directories = split if isinstance(split, list) else [split]
    return sum(
        1
        for directory in directories
        for path in (root / directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output", default="outputs/paper_experiments/yolo_box_test.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    run_directory = output.parent / "yolo_validation_artifacts"
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=100,
        plots=True,
        project=str(run_directory.parent.resolve()),
        name=run_directory.name,
        exist_ok=True,
        verbose=True,
    )
    speed = {key: float(value) for key, value in metrics.speed.items()}
    payload = {
        "protocol": "independent_labelled_target_domain_box_test",
        "arguments": vars(args),
        "class_names": model.names,
        "images": _test_image_count(args.data),
        "box": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "speed_ms_per_image": speed,
        "results_dict": {
            key: float(value) for key, value in metrics.results_dict.items()
        },
        "artifacts": str(run_directory).replace("\\", "/"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote YOLO box evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
