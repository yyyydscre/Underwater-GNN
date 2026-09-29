"""Exposure-adaptive subpixel localization of an LED white core."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class WhiteCoreResult:
    """Subpixel white-core position with uncertainty and fit diagnostics."""

    xy: np.ndarray
    confidence: float
    method: str = "adaptive_white_core"
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64) * 25.0)
    seed_xy: np.ndarray | None = None
    roi_xyxy: np.ndarray | None = None
    saturation_ratio: float = 0.0
    fit_residual: float = 1.0
    valid: bool = True
    diagnostics: dict[str, float | int | bool | str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    xy: np.ndarray
    covariance: np.ndarray
    method: str
    quality: float
    residual: float


def _subpixel_peak(response: np.ndarray, peak_x: int, peak_y: int) -> np.ndarray:
    """Fit independent parabolas around a local maximum."""
    height, width = response.shape
    if not 1 <= peak_x < width - 1 or not 1 <= peak_y < height - 1:
        return np.array([float(peak_x), float(peak_y)], dtype=np.float64)
    center = float(response[peak_y, peak_x])
    left, right = float(response[peak_y, peak_x - 1]), float(response[peak_y, peak_x + 1])
    top, bottom = float(response[peak_y - 1, peak_x]), float(response[peak_y + 1, peak_x])
    denominator_x = left - 2.0 * center + right
    denominator_y = top - 2.0 * center + bottom
    offset_x = 0.0 if abs(denominator_x) < 1e-8 else 0.5 * (left - right) / denominator_x
    offset_y = 0.0 if abs(denominator_y) < 1e-8 else 0.5 * (top - bottom) / denominator_y
    return np.array(
        [peak_x + np.clip(offset_x, -0.75, 0.75), peak_y + np.clip(offset_y, -0.75, 0.75)],
        dtype=np.float64,
    )


def _regularize_covariance(covariance: np.ndarray, maximum_variance: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64).reshape(2, 2)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        return np.eye(2, dtype=np.float64) * maximum_variance
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.01, maximum_variance)
    return vectors @ np.diag(values) @ vectors.T


def _fit_background_plane(signal: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a robust affine background to dim pixels in the ROI border."""
    height, width = signal.shape
    ys, xs = np.indices(signal.shape, dtype=np.float64)
    border_y = max(1, int(round(0.18 * height)))
    border_x = max(1, int(round(0.18 * width)))
    border = (ys < border_y) | (ys >= height - border_y) | (xs < border_x) | (xs >= width - border_x)
    base_sample = border & valid_mask
    values = signal[base_sample]
    if values.size < 12:
        base_sample = valid_mask.copy()
        values = signal[base_sample]
    if values.size == 0:
        return np.zeros_like(signal, dtype=np.float64), 0.05

    cutoff = float(np.quantile(values, 0.72))
    sample = base_sample & (signal <= cutoff)
    if int(sample.sum()) < 8:
        sample = base_sample
    x_values = xs[sample]
    y_values = ys[sample]
    z_values = signal[sample].astype(np.float64)
    design = np.column_stack((np.ones_like(x_values), x_values, y_values))
    weights = np.ones_like(z_values)
    coefficients = np.array([float(np.median(z_values)), 0.0, 0.0], dtype=np.float64)
    for _ in range(5):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = z_values * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        residuals = z_values - design @ coefficients
        scale = max(1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))), 1e-4)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-6))
    plane = coefficients[0] + coefficients[1] * xs + coefficients[2] * ys
    noise = max(
        1.4826 * float(np.median(np.abs(z_values - design @ coefficients))),
        0.002,
    )
    return plane, noise


def _component_at_peak(mask: np.ndarray, peak_x: int, peak_y: int) -> np.ndarray:
    components, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    peak_label = int(labels[peak_y, peak_x]) if components > 1 else 0
    if peak_label > 0:
        return labels == peak_label
    return mask.astype(bool)


def _weighted_centroid_candidate(
    response: np.ndarray,
    valid_mask: np.ndarray,
    peak_x: int,
    peak_y: int,
    noise: float,
    snr: float,
    maximum_variance: float,
) -> tuple[_Candidate, np.ndarray, int]:
    peak = float(response[peak_y, peak_x])
    positive = response[valid_mask]
    quantile = float(np.quantile(positive, 0.76)) if positive.size else 0.0
    threshold = max(2.5 * noise, 0.16 * peak, quantile)
    component = _component_at_peak((response >= threshold) & valid_mask, peak_x, peak_y)
    if int(component.sum()) < 2:
        component = _component_at_peak((response >= max(1.5 * noise, 0.08 * peak)) & valid_mask, peak_x, peak_y)

    ys, xs = np.indices(response.shape, dtype=np.float64)
    weights = np.where(component, np.maximum(response - threshold, 0.0) ** 1.35, 0.0)
    if float(weights.sum()) <= 1e-10:
        weights = np.where(component, np.maximum(response, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 1e-10:
        xy = np.array([float(peak_x), float(peak_y)], dtype=np.float64)
        covariance = np.eye(2, dtype=np.float64) * maximum_variance
        quality = 0.05
    else:
        centroid = np.array([(xs * weights).sum() / total, (ys * weights).sum() / total], dtype=np.float64)
        peak_xy = _subpixel_peak(response, peak_x, peak_y)
        xy = 0.90 * centroid + 0.10 * peak_xy
        offsets = np.stack((xs - centroid[0], ys - centroid[1]), axis=-1)
        spatial = np.einsum("hwi,hwj,hw->ij", offsets, offsets, weights) / total
        effective_pixels = total**2 / max(float(np.square(weights).sum()), 1e-10)
        localization_scale = max(effective_pixels * max(np.sqrt(snr), 1.0), 1.0)
        covariance = spatial / localization_scale + np.eye(2, dtype=np.float64) * 0.02
        compactness = np.exp(-0.08 * max(int(component.sum()) - 12, 0))
        quality = float(np.clip((1.0 - np.exp(-snr / 5.0)) * compactness, 0.05, 0.92))
    covariance = _regularize_covariance(covariance, maximum_variance)
    component_values = response[component]
    residual = (
        float(np.sqrt(np.mean(np.square(component_values - peak)))) / max(peak, 1e-6)
        if component_values.size
        else 1.0
    )
    return _Candidate(xy, covariance, "weighted_centroid_tiny", quality, residual), component, int(component.sum())


def _moffat_candidate(
    response: np.ndarray,
    valid_mask: np.ndarray,
    saturated_mask: np.ndarray,
    initial_xy: np.ndarray,
    box_size: tuple[float, float],
    noise: float,
    saturated: bool,
    maximum_variance: float,
) -> _Candidate | None:
    """Fit an elliptical Moffat profile, masking clipped pixels when needed."""
    height, width = response.shape
    peak = float(np.max(response[valid_mask])) if np.any(valid_mask) else 0.0
    if peak <= 4.0 * noise:
        return None
    ys, xs = np.indices(response.shape, dtype=np.float64)
    fit_radius = max(4.0, 0.65 * min(box_size))
    radial_mask = np.hypot(xs - initial_xy[0], ys - initial_xy[1]) <= fit_radius
    fit_mask = valid_mask & radial_mask & (response >= max(0.012 * peak, 0.8 * noise))
    if saturated:
        fit_mask &= ~saturated_mask
    if int(fit_mask.sum()) < 16:
        return None

    x_values = xs[fit_mask]
    y_values = ys[fit_mask]
    observations = response[fit_mask].astype(np.float64)
    if observations.size > 900:
        order = np.linspace(0, observations.size - 1, 900).astype(int)
        x_values, y_values, observations = x_values[order], y_values[order], observations[order]

    centered_x = x_values - initial_xy[0]
    centered_y = y_values - initial_xy[1]
    moment_weights = np.maximum(observations - np.quantile(observations, 0.15), 0.0)
    moment_total = max(float(moment_weights.sum()), 1e-9)
    sigma_x = np.sqrt(max(float(np.sum(moment_weights * centered_x**2) / moment_total), 0.8))
    sigma_y = np.sqrt(max(float(np.sum(moment_weights * centered_y**2) / moment_total), 0.8))
    maximum_axis = max(2.0, 0.60 * max(box_size))
    initial = np.array(
        [
            initial_xy[0],
            initial_xy[1],
            np.log(max(peak, 1e-5)),
            np.log(np.clip(sigma_x, 0.8, maximum_axis)),
            np.log(np.clip(sigma_y, 0.8, maximum_axis)),
            0.0,
            np.log(1.5),
            0.0,
        ],
        dtype=np.float64,
    )
    center_margin = max(3.0, 0.38 * np.hypot(*box_size))
    lower = np.array(
        [
            max(0.0, initial_xy[0] - center_margin),
            max(0.0, initial_xy[1] - center_margin),
            np.log(max(0.15 * peak, 1e-6)),
            np.log(0.55),
            np.log(0.55),
            -np.pi / 2.0,
            np.log(0.20),
            -0.10 * peak,
        ]
    )
    upper = np.array(
        [
            min(width - 1.0, initial_xy[0] + center_margin),
            min(height - 1.0, initial_xy[1] + center_margin),
            np.log(max(3.0 * peak, 1e-5)),
            np.log(maximum_axis),
            np.log(maximum_axis),
            np.pi / 2.0,
            np.log(7.0),
            0.20 * peak,
        ]
    )

    def model_and_jacobian(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center_x, center_y, log_amplitude, log_axis_x, log_axis_y, angle, log_beta, offset = parameters
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        delta_x, delta_y = x_values - center_x, y_values - center_y
        rotated_x = cos_angle * delta_x + sin_angle * delta_y
        rotated_y = -sin_angle * delta_x + cos_angle * delta_y
        axis_x, axis_y = np.exp(log_axis_x), np.exp(log_axis_y)
        inverse_axis_x_squared = 1.0 / axis_x**2
        inverse_axis_y_squared = 1.0 / axis_y**2
        radius_squared = rotated_x**2 * inverse_axis_x_squared + rotated_y**2 * inverse_axis_y_squared
        one_plus_radius = 1.0 + radius_squared
        amplitude = np.exp(log_amplitude)
        beta = 1.0 + np.exp(log_beta)
        profile = np.power(one_plus_radius, -beta)
        model = amplitude * profile + offset
        profile_radius_derivative = -amplitude * beta * profile / one_plus_radius

        radius_center_x = (
            -2.0 * rotated_x * cos_angle * inverse_axis_x_squared
            + 2.0 * rotated_y * sin_angle * inverse_axis_y_squared
        )
        radius_center_y = (
            -2.0 * rotated_x * sin_angle * inverse_axis_x_squared
            - 2.0 * rotated_y * cos_angle * inverse_axis_y_squared
        )
        radius_angle = (
            2.0
            * rotated_x
            * rotated_y
            * (inverse_axis_x_squared - inverse_axis_y_squared)
        )
        jacobian = np.column_stack(
            (
                profile_radius_derivative * radius_center_x,
                profile_radius_derivative * radius_center_y,
                amplitude * profile,
                profile_radius_derivative * (-2.0 * rotated_x**2 * inverse_axis_x_squared),
                profile_radius_derivative * (-2.0 * rotated_y**2 * inverse_axis_y_squared),
                profile_radius_derivative * radius_angle,
                -amplitude * profile * np.log(one_plus_radius) * np.exp(log_beta),
                np.ones_like(model),
            )
        )
        return model, jacobian

    parameters = initial.copy()
    damping = 1e-2
    robust_scale = max(1.5 * noise, 0.015 * peak)
    converged = False
    for _ in range(55):
        model, jacobian = model_and_jacobian(parameters)
        residuals = model - observations
        robust_weights = 1.0 / np.sqrt(1.0 + np.square(residuals / robust_scale))
        square_root_weights = np.sqrt(robust_weights)
        weighted_jacobian = jacobian * square_root_weights[:, None]
        weighted_residuals = residuals * square_root_weights
        normal = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residuals
        diagonal = np.maximum(np.diag(normal), 1e-8)
        try:
            step = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(normal + damping * np.diag(diagonal), -gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            return None
        candidate_parameters = np.clip(parameters + step, lower, upper)
        candidate_model, _ = model_and_jacobian(candidate_parameters)
        candidate_residuals = candidate_model - observations
        current_cost = float(np.sum(2.0 * robust_scale**2 * (np.sqrt(1.0 + np.square(residuals / robust_scale)) - 1.0)))
        candidate_cost = float(
            np.sum(2.0 * robust_scale**2 * (np.sqrt(1.0 + np.square(candidate_residuals / robust_scale)) - 1.0))
        )
        if candidate_cost < current_cost:
            parameters = candidate_parameters
            damping = max(damping * 0.35, 1e-7)
            if float(np.linalg.norm(step)) < 1e-5 * (1.0 + float(np.linalg.norm(parameters))):
                converged = True
                break
        else:
            damping = min(damping * 8.0, 1e8)
            if damping >= 1e8:
                break

    model, jacobian = model_and_jacobian(parameters)
    residuals = model - observations
    if not converged and damping >= 1e8:
        return None
    xy = parameters[:2].astype(np.float64)
    normalized_residual = float(np.sqrt(np.mean(np.square(residuals)))) / max(peak, 1e-6)
    degrees_of_freedom = max(len(observations) - len(parameters), 1)
    residual_variance = float(np.sum(np.square(residuals)) / degrees_of_freedom)
    try:
        robust_weights = 1.0 / np.sqrt(1.0 + np.square(residuals / robust_scale))
        weighted_jacobian = jacobian * np.sqrt(robust_weights)[:, None]
        information = weighted_jacobian.T @ weighted_jacobian
        parameter_covariance = np.linalg.pinv(information, rcond=1e-8) * residual_variance
        covariance = parameter_covariance[:2, :2]
        condition = float(np.linalg.cond(information))
    except np.linalg.LinAlgError:
        covariance = np.eye(2, dtype=np.float64) * maximum_variance
        condition = np.inf
    covariance = _regularize_covariance(covariance, maximum_variance)
    boundary_gap = min(
        xy[0] - lower[0],
        upper[0] - xy[0],
        xy[1] - lower[1],
        upper[1] - xy[1],
    )
    condition_score = 0.2 if not np.isfinite(condition) else float(np.exp(-max(np.log10(condition) - 6.0, 0.0)))
    quality = float(
        np.clip(
            np.exp(-5.0 * normalized_residual)
            * condition_score
            * np.clip(boundary_gap / 1.5, 0.25, 1.0),
            0.02,
            0.98,
        )
    )
    method = "moffat_wings_saturated" if saturated else "moffat_unsaturated"
    return _Candidate(xy, covariance, method, quality, normalized_residual)


def _radial_symmetry_candidate(
    response: np.ndarray,
    valid_mask: np.ndarray,
    saturated_mask: np.ndarray,
    initial_xy: np.ndarray,
    noise: float,
    maximum_variance: float,
) -> _Candidate | None:
    """Estimate the intersection of intensity-gradient radial lines."""
    smoothed = cv2.GaussianBlur(response.astype(np.float32), (0, 0), 0.8)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    support = valid_mask & ~saturated_mask & (response > max(1.2 * noise, 0.03 * float(response.max())))
    magnitudes = magnitude[support]
    if magnitudes.size < 10:
        return None
    support &= magnitude >= float(np.quantile(magnitudes, 0.60))
    ys, xs = np.nonzero(support)
    if len(xs) < 8:
        return None
    gx, gy = gradient_x[support], gradient_y[support]
    norms = np.hypot(gx, gy)
    line_x, line_y = gy / np.maximum(norms, 1e-9), -gx / np.maximum(norms, 1e-9)
    design = np.column_stack((line_x, line_y))
    targets = line_x * xs + line_y * ys
    weights = np.square(norms) * np.maximum(response[support], noise)
    weights /= max(float(np.max(weights)), 1e-9)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_targets = targets * np.sqrt(weights)
    regularization = max(float(weights.sum()) * 0.004, 0.02)
    weighted_design = np.vstack((weighted_design, np.eye(2) * np.sqrt(regularization)))
    weighted_targets = np.concatenate((weighted_targets, initial_xy * np.sqrt(regularization)))
    try:
        xy, *_ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
        residuals = design @ xy - targets
        normal = design.T @ (weights[:, None] * design) + regularization * np.eye(2)
        residual_variance = float(np.average(np.square(residuals), weights=weights))
        covariance = np.linalg.pinv(normal, rcond=1e-8) * max(residual_variance, 0.01)
        condition = float(np.linalg.cond(normal))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(xy)):
        return None
    covariance = _regularize_covariance(covariance, maximum_variance)
    normalized_residual = float(np.sqrt(np.average(np.square(residuals), weights=weights))) / max(
        np.hypot(*response.shape), 1.0
    )
    quality = float(
        np.clip(
            np.exp(-10.0 * normalized_residual)
            * np.exp(-max(np.log10(max(condition, 1.0)) - 4.0, 0.0)),
            0.03,
            0.92,
        )
    )
    return _Candidate(xy.astype(np.float64), covariance, "radial_symmetry", quality, normalized_residual)


def _ellipse_candidate(
    response: np.ndarray,
    valid_mask: np.ndarray,
    peak_xy: np.ndarray,
    maximum_variance: float,
) -> _Candidate | None:
    """Aggregate ellipse centers from several high-intensity isophotes."""
    peak = float(response[int(round(peak_xy[1])), int(round(peak_xy[0]))])
    centers: list[np.ndarray] = []
    weights: list[float] = []
    for fraction in (0.32, 0.45, 0.58, 0.72, 0.84):
        mask = ((response >= fraction * peak) & valid_mask).astype(np.uint8)
        component = _component_at_peak(mask, int(round(peak_xy[0])), int(round(peak_xy[1])))
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if len(contour) >= 5 and area >= 2.0:
            ellipse = cv2.fitEllipse(contour)
            centers.append(np.asarray(ellipse[0], dtype=np.float64))
            axis_a, axis_b = ellipse[1]
            shape_score = min(axis_a, axis_b) / max(max(axis_a, axis_b), 1e-6)
            weights.append(max(area * shape_score, 0.1))
        else:
            moments = cv2.moments(component.astype(np.uint8))
            if moments["m00"] > 0:
                centers.append(np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]))
                weights.append(max(float(component.sum()), 0.1))
    if not centers:
        return None
    center_array = np.stack(centers)
    weight_array = np.asarray(weights, dtype=np.float64)
    xy = np.average(center_array, axis=0, weights=weight_array)
    offsets = center_array - xy
    covariance = (
        np.einsum("ni,nj,n->ij", offsets, offsets, weight_array) / max(float(weight_array.sum()), 1e-9)
        + np.eye(2) * 0.08
    )
    covariance = _regularize_covariance(covariance, maximum_variance)
    spread = float(np.sqrt(np.trace(covariance)))
    quality = float(np.clip(np.exp(-spread / 2.5) * min(len(centers) / 3.0, 1.0), 0.04, 0.90))
    return _Candidate(xy, covariance, "multi_isophote_ellipse", quality, spread / max(np.hypot(*response.shape), 1.0))


class WhiteCoreLocalizer:
    """Refine a YOLO box seed using adaptive photometric center models."""

    def __init__(
        self,
        roi_scale: float = 1.30,
        max_roi_scale: float = 1.50,
        min_side_px: int = 5,
        maximum_shift_fraction: float = 0.18,
        saturation_threshold: float = 0.035,
    ) -> None:
        if not 1.0 <= roi_scale <= max_roi_scale:
            raise ValueError("Expected 1.0 <= roi_scale <= max_roi_scale.")
        self.roi_scale = roi_scale
        self.max_roi_scale = max_roi_scale
        self.min_side_px = min_side_px
        self.maximum_shift_fraction = maximum_shift_fraction
        self.saturation_threshold = saturation_threshold

    @staticmethod
    def _neighbor_mask(
        shape: tuple[int, int],
        origin_xy: np.ndarray,
        seed_xy: np.ndarray,
        neighbor_boxes: np.ndarray | None,
    ) -> np.ndarray:
        valid = np.ones(shape, dtype=bool)
        if neighbor_boxes is None or len(neighbor_boxes) <= 1:
            return valid
        ys, xs = np.indices(shape, dtype=np.float64)
        global_x, global_y = xs + origin_xy[0], ys + origin_xy[1]
        current_distance = np.square(global_x - seed_xy[0]) + np.square(global_y - seed_xy[1])
        for neighbor in np.asarray(neighbor_boxes, dtype=np.float64).reshape(-1, 4):
            neighbor_center = np.array(
                [(neighbor[0] + neighbor[2]) / 2.0, (neighbor[1] + neighbor[3]) / 2.0],
                dtype=np.float64,
            )
            if np.linalg.norm(neighbor_center - seed_xy) < 1.0:
                continue
            neighbor_distance = np.square(global_x - neighbor_center[0]) + np.square(global_y - neighbor_center[1])
            valid &= current_distance <= neighbor_distance
        return valid

    def locate(
        self,
        image: np.ndarray,
        box_xyxy: np.ndarray,
        neighbor_boxes: np.ndarray | None = None,
    ) -> WhiteCoreResult:
        height, width = image.shape[:2]
        left, top, right, bottom = np.asarray(box_xyxy, dtype=np.float64)
        box_width = max(right - left, 1.0)
        box_height = max(bottom - top, 1.0)
        seed_xy = np.array(
            [np.clip((left + right) / 2.0, 0, width - 1), np.clip((top + bottom) / 2.0, 0, height - 1)],
            dtype=np.float64,
        )
        maximum_variance = max(4.0, (0.35 * min(box_width, box_height)) ** 2)

        scale = self.roi_scale
        if min(box_width, box_height) <= 12.0:
            scale = self.max_roi_scale
        half_width, half_height = 0.5 * scale * box_width, 0.5 * scale * box_height
        x0 = max(0, int(np.floor(seed_xy[0] - half_width)))
        y0 = max(0, int(np.floor(seed_xy[1] - half_height)))
        x1 = min(width, int(np.ceil(seed_xy[0] + half_width + 1.0)))
        y1 = min(height, int(np.ceil(seed_xy[1] + half_height + 1.0)))
        roi_xyxy = np.array([x0, y0, x1, y1], dtype=np.float64)
        patch = image[y0:y1, x0:x1]
        fallback_covariance = np.eye(2, dtype=np.float64) * maximum_variance
        if patch.shape[0] < self.min_side_px or patch.shape[1] < self.min_side_px:
            return WhiteCoreResult(
                seed_xy,
                0.0,
                "box_center_unresolved_tiny_roi",
                fallback_covariance,
                seed_xy,
                roi_xyxy,
                valid=False,
            )

        origin = np.array([x0, y0], dtype=np.float64)
        seed_local = seed_xy - origin
        valid_mask = self._neighbor_mask(patch.shape[:2], origin, seed_xy, neighbor_boxes)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float64)
        bgr = patch.astype(np.float64) / 255.0
        luminance = lab[..., 0] / 255.0
        chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2) / 181.0
        neutrality = np.clip(1.0 - chroma, 0.0, 1.0)
        minimum_channel = bgr.min(axis=2)
        white_signal = 0.50 * minimum_channel + 0.35 * luminance + 0.15 * luminance * neutrality
        background, noise = _fit_background_plane(white_signal, valid_mask)
        response = np.maximum(white_signal - background, 0.0)
        response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), 0.65).astype(np.float64)
        response[~valid_mask] = 0.0

        if not np.any(valid_mask) or float(response.max()) <= 2.0 * noise:
            return WhiteCoreResult(
                seed_xy,
                0.0,
                "box_center_unresolved_low_snr",
                fallback_covariance,
                seed_xy,
                roi_xyxy,
                valid=False,
                fit_residual=1.0,
                diagnostics={"background_noise": float(noise), "snr": 0.0},
            )

        ys, xs = np.indices(response.shape, dtype=np.float64)
        distance_to_seed = np.hypot(xs - seed_local[0], ys - seed_local[1])
        prior_scale = max(3.0, 0.42 * np.hypot(box_width, box_height))
        center_prior = 0.50 + 0.50 * np.exp(-0.5 * np.square(distance_to_seed / prior_scale))
        seeded_response = np.where(valid_mask, response * center_prior, -np.inf)
        peak_y, peak_x = np.unravel_index(int(np.argmax(seeded_response)), response.shape)
        peak = float(response[peak_y, peak_x])
        snr = max((peak - float(np.median(response[valid_mask]))) / max(noise, 1e-6), 0.0)

        raw_maximum = bgr.max(axis=2)
        top_level = float(np.quantile(white_signal[valid_mask], 0.995))
        raw_gradient_x = cv2.Sobel(white_signal.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        raw_gradient_y = cv2.Sobel(white_signal.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        raw_gradient = np.hypot(raw_gradient_x, raw_gradient_y)
        digital_clip = raw_maximum >= (250.0 / 255.0)
        flat_top = (
            (white_signal >= max(0.82, 0.985 * top_level))
            & (raw_gradient <= max(float(np.quantile(raw_gradient[valid_mask], 0.35)), 0.015))
            & (top_level >= 0.84)
        )
        core_support = (response >= max(0.18 * peak, 2.0 * noise)) & valid_mask
        saturated_mask = (digital_clip | flat_top) & core_support
        saturation_ratio = float(saturated_mask.sum()) / max(float(core_support.sum()), 1.0)
        saturated = saturation_ratio >= self.saturation_threshold or int(saturated_mask.sum()) >= 4

        centroid, component, component_pixels = _weighted_centroid_candidate(
            response,
            valid_mask,
            peak_x,
            peak_y,
            noise,
            snr,
            maximum_variance,
        )
        candidates = [centroid]
        tiny = component_pixels < 8 or min(box_width, box_height) < 8.0
        if not tiny:
            moffat = _moffat_candidate(
                response,
                valid_mask,
                saturated_mask,
                centroid.xy,
                (box_width, box_height),
                noise,
                saturated,
                maximum_variance,
            )
            if moffat is not None:
                candidates.append(moffat)
            radial = _radial_symmetry_candidate(
                response,
                valid_mask,
                saturated_mask,
                centroid.xy,
                noise,
                maximum_variance,
            )
            if radial is not None:
                candidates.append(radial)
            if saturated:
                ellipse = _ellipse_candidate(
                    response,
                    valid_mask,
                    np.array([float(peak_x), float(peak_y)]),
                    maximum_variance,
                )
                if ellipse is not None:
                    candidates.append(ellipse)

        if tiny:
            chosen = centroid
        elif saturated:
            # Saturation can make one-sided PSF wings look deceptively fit-worthy.
            # Let residual, conditioning, and multi-isophote agreement decide.
            chosen = max(candidates, key=lambda candidate: candidate.quality)
        else:
            profile_fits = [candidate for candidate in candidates if candidate.method == "moffat_unsaturated"]
            if profile_fits and profile_fits[0].quality >= 0.30:
                chosen = profile_fits[0]
            else:
                chosen = max(candidates, key=lambda candidate: candidate.quality)

        comparison = [candidate for candidate in candidates if candidate is not chosen and candidate.quality >= 0.15]
        disagreement = (
            float(
                np.average(
                    [np.linalg.norm(candidate.xy - chosen.xy) for candidate in comparison],
                    weights=[candidate.quality for candidate in comparison],
                )
            )
            if comparison
            else 0.0
        )
        covariance = chosen.covariance.copy()
        if comparison:
            offsets = np.stack([candidate.xy - chosen.xy for candidate in comparison])
            weights = np.asarray([candidate.quality for candidate in comparison], dtype=np.float64)
            covariance += np.einsum("ni,nj,n->ij", offsets, offsets, weights) / max(float(weights.sum()), 1e-9)
        covariance = _regularize_covariance(covariance, maximum_variance)

        candidate_global_xy = chosen.xy + origin
        global_xy = candidate_global_xy.copy()
        shift = float(np.linalg.norm(candidate_global_xy - seed_xy))
        maximum_shift = max(3.0, self.maximum_shift_fraction * np.hypot(box_width, box_height))
        shift_score = float(np.exp(-max(shift - 0.35 * maximum_shift, 0.0) / max(0.35 * maximum_shift, 1.0)))
        agreement_score = float(np.exp(-disagreement / max(1.5, 0.12 * min(box_width, box_height))))
        sigma_px = float(np.sqrt(max(np.trace(covariance) / 2.0, 0.0)))
        confidence = float(
            np.clip(
                (0.15 + 0.85 * chosen.quality)
                * (1.0 - np.exp(-snr / 5.0))
                * np.exp(-sigma_px / 3.0)
                * agreement_score
                * shift_score,
                0.01,
                0.99,
            )
        )
        valid = bool(
            confidence >= 0.10
            and shift <= 1.35 * maximum_shift
            and 0.0 <= chosen.xy[0] < response.shape[1]
            and 0.0 <= chosen.xy[1] < response.shape[0]
        )
        method = chosen.method if valid else f"{chosen.method}_low_confidence"
        if not valid:
            global_xy = seed_xy
            covariance = fallback_covariance
            method = f"{chosen.method}_rejected_to_seed"
            confidence = min(confidence, 0.05)

        diagnostics: dict[str, float | int | bool | str] = {
            "background_noise": round(float(noise), 6),
            "snr": round(float(snr), 4),
            "shift_px": round(float(np.linalg.norm(global_xy - seed_xy)), 4),
            "candidate_shift_px": round(float(shift), 4),
            "candidate_xy": [
                round(float(candidate_global_xy[0]), 4),
                round(float(candidate_global_xy[1]), 4),
            ],
            "fallback_to_seed": bool(not valid),
            "candidate_disagreement_px": round(disagreement, 4),
            "effective_core_pixels": int(component_pixels),
            "tiny_target": bool(tiny),
            "saturated": bool(saturated),
            "candidate_count": len(candidates),
            "candidate_summary": ";".join(
                f"{candidate.method}:{candidate.quality:.3f}@{candidate.xy[0]:.2f},{candidate.xy[1]:.2f}"
                for candidate in candidates
            ),
            "localization_sigma_px": round(float(np.sqrt(np.trace(covariance) / 2.0)), 4),
        }
        return WhiteCoreResult(
            xy=global_xy,
            confidence=confidence,
            method=method,
            covariance=covariance,
            seed_xy=seed_xy,
            roi_xyxy=roi_xyxy,
            saturation_ratio=saturation_ratio,
            fit_residual=chosen.residual,
            valid=valid,
            diagnostics=diagnostics,
        )
