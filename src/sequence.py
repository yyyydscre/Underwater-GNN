"""Filename-based temporal stream grouping for extracted image sequences."""
from __future__ import annotations

import re
from pathlib import Path


def temporal_identity(frame_name: str) -> tuple[str, int]:
    stem = Path(frame_name).stem.lower()
    match = re.search(r"(\d+)", stem)
    number = int(match.group(1)) if match else -1
    suffix = stem[match.end() :].strip(" _-") if match else ""
    variant = suffix or "original"
    if stem.startswith("red_v"):
        sequence = f"red_v:{variant}"
    elif stem.startswith("v"):
        sequence = f"v:{variant}"
    elif stem.startswith("frame"):
        sequence = f"frame:{variant}"
    else:
        prefix = stem[: match.start()].rstrip("_ ") if match else stem
        sequence = f"{prefix}:{variant}"
    return sequence, number


def order_temporal_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: temporal_identity(path.name))
