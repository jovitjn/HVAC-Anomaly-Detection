"""Prediction/reconstruction anomaly scores with normal-only calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreCalibration:
    center: np.ndarray
    scale: np.ndarray
    threshold: float
    threshold_quantile: float

    def as_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "threshold": float(self.threshold),
            "threshold_quantile": float(self.threshold_quantile),
        }


def raw_room_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_last: np.ndarray,
    x_reconstructed: np.ndarray,
    prediction_weight: float = 0.6,
) -> np.ndarray:
    prediction_error = np.square(y_true - y_pred)
    reconstruction_error = np.mean(np.square(x_last - x_reconstructed), axis=1, keepdims=True)
    return prediction_weight * prediction_error + (1.0 - prediction_weight) * reconstruction_error


def normalize_room_scores(raw_scores: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.maximum((raw_scores - center) / scale, 0.0)


def fit_calibration(validation_raw_scores: np.ndarray, threshold_quantile: float = 0.99) -> ScoreCalibration:
    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0.5 and 1.0")
    center = np.median(validation_raw_scores, axis=0)
    upper = np.quantile(validation_raw_scores, 0.95, axis=0)
    scale = np.maximum(upper - center, 1e-8)
    normalized = normalize_room_scores(validation_raw_scores, center, scale)
    global_scores = normalized.mean(axis=1)
    threshold = float(np.quantile(global_scores, threshold_quantile))
    return ScoreCalibration(center=center, scale=scale, threshold=threshold, threshold_quantile=threshold_quantile)


def apply_calibration(raw_scores: np.ndarray, calibration: ScoreCalibration):
    room_scores = normalize_room_scores(raw_scores, calibration.center, calibration.scale)
    global_scores = room_scores.mean(axis=1)
    predictions = (global_scores > calibration.threshold).astype(np.int8)
    return room_scores, global_scores, predictions

