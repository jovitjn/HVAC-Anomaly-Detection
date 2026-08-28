import numpy as np

from hvac_anomaly.scoring import apply_calibration, fit_calibration, raw_room_scores


def test_larger_prediction_errors_raise_anomaly_score():
    y_true = np.zeros((8, 2))
    y_pred = np.zeros((8, 2))
    y_pred[-1] = 8.0
    x_last = np.zeros((8, 3))
    reconstruction = np.zeros((8, 3))
    raw = raw_room_scores(y_true, y_pred, x_last, reconstruction)
    calibration = fit_calibration(raw[:-1] + np.linspace(0, 0.1, 7)[:, None], 0.9)
    _, scores, predictions = apply_calibration(raw, calibration)
    assert scores[-1] > scores[:-1].max()
    assert predictions[-1] == 1

