"""YOLO-Pose and color/blob light-center detectors."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import COLOR_INDEX, LightDetection
from .white_core import WhiteCoreLocalizer


def _color_features(image: np.ndarray, xy: np.ndarray, radius: int = 9) -> tuple[np.ndarray, str, float]:
    """Estimate a soft green/blue/other distribution around a light center."""
    h, w = image.shape[:2]
    x, y = np.rint(xy).astype(int)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), "other", 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = (hsv[..., 1] >= 28) & (hsv[..., 2] >= 65)
    hue = hsv[..., 0]
    green = float(np.count_nonzero(valid & (hue >= 38) & (hue <= 78)))
    blue = float(np.count_nonzero(valid & (hue >= 82) & (hue <= 128)))
    other = float(np.count_nonzero(valid)) * 0.18 + 1.0
    probs = np.array([green + 1.0, blue + 1.0, other], dtype=np.float32)
    probs /= probs.sum()
    color = ("green", "blue", "other")[int(np.argmax(probs))]
    return probs, color, float(hsv[..., 2].mean())


def refine_center(image: np.ndarray, xy: np.ndarray, radius: int = 10, color: str | None = None) -> np.ndarray:
    """Compute a brightness-weighted subpixel center around a detector proposal."""
    h, w = image.shape[:2]
    x, y = np.rint(xy).astype(int)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return xy.astype(np.float64)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].astype(np.float64)
    weights = np.maximum(value - np.percentile(value, 55), 0.0)
    if color == "green":
        weights *= ((hsv[..., 0] >= 38) & (hsv[..., 0] <= 78) & (hsv[..., 1] >= 28))
    elif color == "blue":
        weights *= ((hsv[..., 0] >= 82) & (hsv[..., 0] <= 128) & (hsv[..., 1] >= 28))
    total = weights.sum()
    if total < 1e-6:
        return xy.astype(np.float64)
    ys, xs = np.indices(weights.shape)
    return np.array([(xs * weights).sum() / total + x0, (ys * weights).sum() / total + y0], dtype=np.float64)


def _subpixel_peak(response: np.ndarray, peak_x: int, peak_y: int) -> np.ndarray:
    """Quadratic subpixel correction around a response-map maximum."""
    height, width = response.shape
    if not 1 <= peak_x < width - 1 or not 1 <= peak_y < height - 1:
        return np.array([float(peak_x), float(peak_y)])
    center = float(response[peak_y, peak_x])
    left, right = float(response[peak_y, peak_x - 1]), float(response[peak_y, peak_x + 1])
    top, bottom = float(response[peak_y - 1, peak_x]), float(response[peak_y + 1, peak_x])
    denominator_x = left - 2.0 * center + right
    denominator_y = top - 2.0 * center + bottom
    offset_x = 0.0 if abs(denominator_x) < 1e-6 else 0.5 * (left - right) / denominator_x
    offset_y = 0.0 if abs(denominator_y) < 1e-6 else 0.5 * (top - bottom) / denominator_y
    return np.array([peak_x + np.clip(offset_x, -0.75, 0.75), peak_y + np.clip(offset_y, -0.75, 0.75)])


def refine_colored_center(image: np.ndarray, seed_xy: np.ndarray, color: str, radius: int = 12) -> tuple[np.ndarray, float]:
    """Find a subpixel light core from hue-weighted local response, not its bloom.

    The center of a water-scattered light halo is often displaced from the LED
    core. This refinement first finds the compact color response peak, then uses
    only its high-response core for a robust centroid and quadratic correction.
    """
    height, width = image.shape[:2]
    seed_x, seed_y = np.rint(seed_xy).astype(int)
    x0, x1 = max(0, seed_x - radius), min(width, seed_x + radius + 1)
    y0, y1 = max(0, seed_y - radius), min(height, seed_y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return seed_xy.astype(np.float64), 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, saturation, value = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2]
    if color == "green":
        hue_score = np.clip(1.0 - np.abs(hue - 55.0) / 25.0, 0.0, 1.0)
    elif color == "blue":
        hue_score = np.clip(1.0 - np.abs(hue - 96.0) / 25.0, 0.0, 1.0)
    else:
        hue_score = np.ones_like(value)
    # Keep low-saturation white cores while strongly suppressing background haze.
    response = value * (0.20 + 0.80 * hue_score) * (0.35 + 0.65 * saturation)
    response = cv2.GaussianBlur(response, (3, 3), 0)
    ys, xs = np.indices(response.shape, dtype=np.float32)
    local_seed_x = float(seed_xy[0] - x0)
    local_seed_y = float(seed_xy[1] - y0)
    proximity = np.exp(-((xs - local_seed_x) ** 2 + (ys - local_seed_y) ** 2) / max(2.0 * (radius * 0.70) ** 2, 1.0))
    weighted_response = response * proximity
    peak_y, peak_x = np.unravel_index(int(np.argmax(weighted_response)), weighted_response.shape)
    peak_value = float(response[peak_y, peak_x])
    if peak_value <= 1e-6:
        return seed_xy.astype(np.float64), 0.0
    core = response >= 0.78 * peak_value
    core_weights = np.where(core, np.maximum(response - 0.72 * peak_value, 0.0) ** 2, 0.0)
    total = float(core_weights.sum())
    if total <= 1e-6:
        core_center = np.array([float(peak_x), float(peak_y)])
    else:
        core_center = np.array([(xs * core_weights).sum() / total, (ys * core_weights).sum() / total])
    subpixel = _subpixel_peak(response, int(peak_x), int(peak_y))
    # The local maximum is a better proxy for the LED core than a halo centroid.
    center = 0.25 * core_center + 0.75 * subpixel + np.array([x0, y0], dtype=np.float64)
    contrast = (peak_value - float(np.median(response))) / max(peak_value, 1e-6)
    seed_distance = float(np.linalg.norm(center - seed_xy))
    quality = float(np.clip(0.25 + 0.50 * contrast + 0.25 * np.exp(-seed_distance / max(radius, 1)), 0.0, 1.0))
    return center, quality


def refine_white_core_center(
    image: np.ndarray,
    seed_xy: np.ndarray,
    radius: int = 12,
    max_shift_px: float | None = None,
) -> tuple[np.ndarray, float]:
    """Refine a predicted keypoint to the compact high-luminance white core.

    The predictor, not the image heuristic, is the primary source of position.
    A water-scattered halo can contain a brighter-looking coloured patch away
    from its physical LED centre.  The refinement therefore searches for a
    high-Lab-L, low-chroma component near the predicted keypoint and refuses a
    correction that is too large.  This makes the operation safe after a
    trained YOLO-Pose model while keeping a wider fallback mode for blobs.
    """
    height, width = image.shape[:2]
    seed_x, seed_y = np.rint(seed_xy).astype(int)
    x0, x1 = max(0, seed_x - radius), min(width, seed_x + radius + 1)
    y0, y1 = max(0, seed_y - radius), min(height, seed_y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return seed_xy.astype(np.float64), 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    value = hsv[..., 2] / 255.0
    saturation = hsv[..., 1] / 255.0
    luminance = lab[..., 0] / 255.0
    chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2) / 181.0
    # A white LED core has high luminance and substantially lower chroma than
    # the surrounding green/blue bloom. `min(B,G,R)` handles clipped cores.
    neutral = np.clip(1.0 - chroma, 0.0, 1.0)
    neutral_brightness = patch.astype(np.float32).min(axis=2) / 255.0
    core_response = 255.0 * (
        0.46 * luminance + 0.30 * neutral_brightness + 0.17 * neutral + 0.07 * (1.0 - saturation)
    )
    core_response = cv2.GaussianBlur(core_response, (3, 3), 0)
    ys, xs = np.indices(core_response.shape, dtype=np.float32)
    local_seed_x = float(seed_xy[0] - x0)
    local_seed_y = float(seed_xy[1] - y0)
    proximity = np.exp(-((xs - local_seed_x) ** 2 + (ys - local_seed_y) ** 2) / max(2.0 * (radius * 0.75) ** 2, 1.0))
    seeded_response = core_response * proximity
    peak_y, peak_x = np.unravel_index(int(np.argmax(seeded_response)), seeded_response.shape)
    peak = float(core_response[peak_y, peak_x])
    if peak <= 1e-6:
        return seed_xy.astype(np.float64), 0.0
    # Keep only the compact white-core plateau containing the local maximum.
    mask = (core_response >= 0.94 * peak).astype(np.uint8)
    components, labels, _, _ = cv2.connectedComponentsWithStats(mask)
    label = labels[peak_y, peak_x] if components > 1 else 0
    component = labels == label if label > 0 else mask.astype(bool)
    weights = np.where(component, np.maximum(core_response - 0.90 * peak, 0.0) ** 2, 0.0)
    total = float(weights.sum())
    if total > 1e-6:
        core_center = np.array([(xs * weights).sum() / total, (ys * weights).sum() / total])
    else:
        core_center = np.array([float(peak_x), float(peak_y)])
    subpixel = _subpixel_peak(core_response, int(peak_x), int(peak_y))
    candidate = 0.40 * core_center + 0.60 * subpixel + np.array([x0, y0], dtype=np.float64)
    contrast = (peak - float(np.median(core_response))) / max(peak, 1e-6)
    seed_distance = float(np.linalg.norm(candidate - seed_xy))
    if max_shift_px is None:
        max_shift_px = max(3.0, min(7.0, 0.38 * radius))
    # Do not convert a good learned landmark into a local-halo peak.  A low
    # quality candidate is deliberately left at the YOLO coordinate.
    accepted = seed_distance <= max_shift_px and contrast >= 0.08
    center = candidate if accepted else seed_xy.astype(np.float64)
    compactness = float(component.sum()) / max(float(mask.sum()), 1.0)
    quality = float(
        np.clip(
            0.15 + 0.42 * contrast + 0.25 * neutral[peak_y, peak_x] + 0.18 * compactness,
            0.0,
            1.0,
        )
    )
    if not accepted:
        quality *= 0.65
    return center, quality


class ColorBlobDetector:
    """No-weight fallback detector for bright green/blue guiding lights."""

    def __init__(self, minimum_peak_value: int = 140, nonmax_radius: int = 18, max_per_color: int = 24) -> None:
        self.minimum_peak_value = minimum_peak_value
        self.nonmax_radius = nonmax_radius
        self.max_per_color = max_per_color

    @staticmethod
    def _mask(hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "green":
            mask = cv2.inRange(hsv, (38, 28, 90), (78, 255, 255))
        else:
            mask = cv2.inRange(hsv, (82, 28, 70), (128, 255, 255))
        return mask

    def _peak_points(self, hsv: np.ndarray, color: str) -> list[np.ndarray]:
        mask = self._mask(hsv, color)
        score = cv2.GaussianBlur(hsv[..., 2], (5, 5), 0)
        score[mask == 0] = 0
        neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        peaks = (score == cv2.dilate(score, neighborhood)) & (score >= self.minimum_peak_value) & (mask > 0)
        ys, xs = np.where(peaks)
        candidates = sorted(zip(xs, ys), key=lambda point: int(score[point[1], point[0]]), reverse=True)
        accepted: list[np.ndarray] = []
        for x, y in candidates:
            point = np.array([x, y], dtype=np.float64)
            if all(np.linalg.norm(point - previous) >= self.nonmax_radius for previous in accepted):
                accepted.append(point)
            if len(accepted) >= self.max_per_color:
                break
        return accepted

    def infer(self, image: np.ndarray) -> list[LightDetection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections: list[LightDetection] = []
        for color in ("green", "blue"):
            for peak in self._peak_points(hsv, color):
                xy, center_quality = refine_white_core_center(image, peak, radius=14, max_shift_px=7.0)
                probs, sampled_color, brightness = _color_features(image, xy, radius=8)
                confidence = min(0.99, 0.35 + brightness / 500.0 + 0.15 * center_quality)
                detections.append(
                    LightDetection(
                        xy=xy,
                        confidence=float(confidence),
                        color_probs=probs,
                        color=sampled_color if sampled_color != "other" else color,
                        radius=5.0,
                        brightness=brightness,
                        source="blob",
                        center_quality=center_quality,
                    )
                )
        return _deduplicate(detections)


class YOLOBoxDetector:
    """YOLO lamp-box detector followed by a box-constrained white-core locator."""

    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.20,
        image_size: int = 1280,
        test_time_augmentation: bool = False,
    ) -> None:
        try:
            from .yolo_extensions import register_ultralytics_layers
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is required for --detector yolo.") from error
        register_ultralytics_layers()
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.image_size = image_size
        self.test_time_augmentation = test_time_augmentation
        has_fixed_fastervit = any(
            module.__class__.__name__ == "_FasterViTFixedResolutionStage"
            for module in self.model.model.modules()
        )
        if has_fixed_fastervit and test_time_augmentation:
            raise ValueError("FasterViT-0 uses fixed 1280 inputs and cannot use scale-based TTA.")
        self.white_core_localizer = WhiteCoreLocalizer()

    def infer(self, image: np.ndarray) -> list[LightDetection]:
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            augment=self.test_time_augmentation,
            rect=False,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        detections: list[LightDetection] = []
        for box, score in zip(boxes, scores):
            radius = max(3.0, float(max(box[2] - box[0], box[3] - box[1]) / 2))
            white_core = self.white_core_localizer.locate(image, box, neighbor_boxes=boxes)
            probs, color, brightness = _color_features(image, white_core.xy, radius=min(18, int(radius)))
            center_diagnostics = {
                **white_core.diagnostics,
                "seed_xy": [round(float(value), 4) for value in white_core.seed_xy],
                "roi_xyxy": [round(float(value), 3) for value in white_core.roi_xyxy],
                "saturation_ratio": round(float(white_core.saturation_ratio), 6),
                "fit_residual": round(float(white_core.fit_residual), 6),
            }
            detections.append(
                LightDetection(
                    xy=white_core.xy,
                    confidence=float(score) * white_core.confidence,
                    color_probs=probs,
                    color=color,
                    radius=radius,
                    brightness=brightness,
                    source="yolo_box",
                    center_quality=white_core.confidence,
                    bbox_xyxy=box.astype(np.float64),
                    detector_confidence=float(score),
                    center_confidence=white_core.confidence,
                    center_method=white_core.method,
                    center_covariance=white_core.covariance,
                    center_valid=white_core.valid,
                    center_diagnostics=center_diagnostics,
                )
            )
        # Standard box NMS can retain a small box inside a larger halo box
        # because their IoU is low. Merge only nearly concentric predictions
        # using a radius-scaled threshold, never a fixed pixel distance.
        return _deduplicate_yolo(detections)


# Kept as a compatibility alias for callers written before the move from
# pseudo keypoints to detector-guided white-core localization.
YOLOPoseDetector = YOLOBoxDetector


def _deduplicate(detections: list[LightDetection], distance_px: float = 20.0) -> list[LightDetection]:
    accepted: list[LightDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence * item.center_quality, reverse=True):
        if all(np.linalg.norm(detection.xy - existing.xy) > distance_px for existing in accepted):
            accepted.append(detection)
    return accepted


def _deduplicate_yolo(detections: list[LightDetection]) -> list[LightDetection]:
    """Suppress concentric halo/core boxes without merging adjacent small lamps."""
    def box_center(item: LightDetection) -> np.ndarray:
        if item.bbox_xyxy is None:
            return item.xy
        left, top, right, bottom = np.asarray(item.bbox_xyxy, dtype=np.float64)
        return np.array([(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float64)

    accepted: list[LightDetection] = []
    for detection in sorted(detections, key=lambda item: item.detector_confidence, reverse=True):
        duplicate = False
        for existing in accepted:
            # The previous 0.35 factor missed partially overlapping duplicate
            # boxes near image borders and on saturated halos.  A 0.65 radius
            # gate removes those boxes while still preserving adjacent tiny
            # lamps whose centres are separated by at least one diameter.
            adaptive_distance = max(1.5, 0.65 * min(detection.radius, existing.radius))
            if np.linalg.norm(box_center(detection) - box_center(existing)) <= adaptive_distance:
                duplicate = True
                break
        if not duplicate:
            accepted.append(detection)
    return accepted


def make_detector(mode: str, weights: str | None, confidence: float, image_size: int):
    if mode == "blob":
        return ColorBlobDetector()
    if mode == "yolo":
        if not weights:
            raise ValueError("--weights is required when --detector yolo is selected.")
        return YOLOBoxDetector(weights, confidence, image_size)
    if weights:
        return YOLOBoxDetector(weights, confidence, image_size)
    return ColorBlobDetector()
