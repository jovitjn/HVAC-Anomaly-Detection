"""Normal-only reference models for anomaly-detection comparison."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .evaluation import classification_metrics
from .scoring import apply_calibration, fit_calibration, raw_room_scores


def _evaluate_components(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float,
    kind: str,
):
    train_last = train_x[:, -1, :]
    validation_last = validation_x[:, -1, :]
    test_last = test_x[:, -1, :]

    if kind == "mean":
        y_reference = train_y.mean(axis=0, keepdims=True)
        x_reference = train_last.mean(axis=0, keepdims=True)
        validation_prediction = np.repeat(y_reference, len(validation_y), axis=0)
        test_prediction = np.repeat(y_reference, len(test_y), axis=0)
        validation_reconstruction = np.repeat(x_reference, len(validation_y), axis=0)
        test_reconstruction = np.repeat(x_reference, len(test_y), axis=0)
    elif kind == "ridge_pca":
        ridge = Ridge(alpha=1.0).fit(train_last, train_y)
        components = min(train_last.shape[1], train_last.shape[0] - 1, 20)
        pca = PCA(n_components=components, random_state=42).fit(train_last)
        validation_prediction = ridge.predict(validation_last)
        test_prediction = ridge.predict(test_last)
        validation_reconstruction = pca.inverse_transform(pca.transform(validation_last))
        test_reconstruction = pca.inverse_transform(pca.transform(test_last))
    else:
        raise ValueError(f"Unknown baseline: {kind}")

    validation_raw = raw_room_scores(
        validation_y, validation_prediction, validation_last, validation_reconstruction
    )
    test_raw = raw_room_scores(test_y, test_prediction, test_last, test_reconstruction)
    calibration = fit_calibration(validation_raw, threshold_quantile)
    _, test_scores, predictions = apply_calibration(test_raw, calibration)
    metrics = classification_metrics(labels, predictions, test_scores)
    metrics["threshold"] = float(calibration.threshold)
    return metrics


def run_baselines(data, threshold_quantile: float) -> dict[str, dict[str, float]]:
    arguments = (
        data.train.x,
        data.train.y,
        data.validation.x,
        data.validation.y,
        data.test.x,
        data.test.y,
        data.test.labels,
        threshold_quantile,
    )
    return {
        "Mean reference": _evaluate_components(*arguments, kind="mean"),
        "Ridge + PCA": _evaluate_components(*arguments, kind="ridge_pca"),
    }

