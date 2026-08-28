"""End-to-end experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .baselines import run_baselines
from .data import prepare_data
from .evaluation import (
    classification_metrics,
    per_room_metrics,
    plot_baseline_comparison,
    plot_confusion,
    plot_per_room,
    plot_room_prediction,
    plot_score_distribution,
    plot_threshold_sensitivity,
    plot_timeline,
    plot_training_history,
)
from .model import build_model, set_reproducible, training_callbacks
from .scoring import apply_calibration, fit_calibration, normalize_room_scores, raw_room_scores


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_experiment(
    data_path: str,
    output_dir: str = "results",
    artifacts_dir: str = "artifacts",
    sequence_length: int = 4,
    epochs: int = 30,
    batch_size: int = 256,
    threshold_quantile: float = 0.90,
    seed: int = 42,
) -> dict[str, object]:
    output = Path(output_dir)
    figures = output / "figures"
    artifacts = Path(artifacts_dir)
    figures.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    set_reproducible(seed)
    data = prepare_data(data_path, sequence_length=sequence_length)
    model = build_model(
        sequence_length=sequence_length,
        input_dim=data.train.x.shape[-1],
        target_dim=data.train.y.shape[-1],
    )
    history = model.fit(
        data.train.x,
        {"room_predictions": data.train.y, "reconstruction": data.train.x[:, -1, :]},
        validation_data=(
            data.validation.x,
            {
                "room_predictions": data.validation.y,
                "reconstruction": data.validation.x[:, -1, :],
            },
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks(str(artifacts / "best_model.keras")),
        verbose=2,
        shuffle=True,
    )

    validation_prediction, validation_reconstruction = model.predict(
        data.validation.x, batch_size=batch_size, verbose=0
    )
    test_prediction, test_reconstruction = model.predict(data.test.x, batch_size=batch_size, verbose=0)
    validation_raw = raw_room_scores(
        data.validation.y,
        validation_prediction,
        data.validation.x[:, -1, :],
        validation_reconstruction,
    )
    test_raw = raw_room_scores(
        data.test.y,
        test_prediction,
        data.test.x[:, -1, :],
        test_reconstruction,
    )
    calibration = fit_calibration(validation_raw, threshold_quantile)
    validation_room_scores = normalize_room_scores(validation_raw, calibration.center, calibration.scale)
    test_room_scores, global_scores, predictions = apply_calibration(test_raw, calibration)
    deep_metrics = classification_metrics(data.test.labels, predictions, global_scores)
    deep_metrics["threshold"] = float(calibration.threshold)
    deep_metrics["threshold_quantile"] = threshold_quantile

    baselines = run_baselines(data, threshold_quantile)
    comparison = {**baselines, "Shared LSTM": deep_metrics}
    room_metrics = per_room_metrics(
        validation_room_scores,
        test_room_scores,
        data.test.labels,
        data.target_columns,
        threshold_quantile,
    )
    sensitivity_rows = []
    for quantile in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99):
        candidate_calibration = fit_calibration(validation_raw, quantile)
        _, candidate_scores, candidate_predictions = apply_calibration(test_raw, candidate_calibration)
        candidate_metrics = classification_metrics(
            data.test.labels, candidate_predictions, candidate_scores
        )
        sensitivity_rows.append(
            {
                "quantile": quantile,
                "threshold": candidate_calibration.threshold,
                **candidate_metrics,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    actual_temperature = data.y_scaler.inverse_transform(data.test.y)
    predicted_temperature = data.y_scaler.inverse_transform(test_prediction)
    plot_training_history(history.history, figures / "training_history.png")
    plot_confusion(deep_metrics, figures / "confusion_matrix.png")
    plot_score_distribution(
        data.test.labels, global_scores, calibration.threshold, figures / "score_distribution.png"
    )
    plot_timeline(
        data.test.timestamps,
        data.test.labels,
        global_scores,
        calibration.threshold,
        figures / "anomaly_timeline.png",
    )
    plot_room_prediction(
        data.test.timestamps,
        data.test.labels,
        actual_temperature[:, 0],
        predicted_temperature[:, 0],
        "102",
        figures / "room_102_prediction.png",
    )
    plot_baseline_comparison(comparison, figures / "baseline_comparison.png")
    plot_per_room(room_metrics, figures / "per_room_f1.png")
    plot_threshold_sensitivity(sensitivity, figures / "threshold_sensitivity.png")

    model.save(artifacts / "final_model.keras")
    joblib.dump(
        {
            "x_scaler": data.x_scaler,
            "y_scaler": data.y_scaler,
            "input_columns": data.input_columns,
            "target_columns": data.target_columns,
            "sequence_length": sequence_length,
            "calibration": calibration.as_dict(),
        },
        artifacts / "preprocessing.joblib",
    )

    run_config = {
        "sequence_length": sequence_length,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "threshold_quantile": threshold_quantile,
        "seed": seed,
        "prediction_weight": 0.6,
        "reconstruction_weight": 0.4,
    }
    results = {
        "method": "Shared LSTM",
        "metrics": deep_metrics,
        "baselines": baselines,
        "split": data.split_summary,
        "config": run_config,
    }
    _write_json(output / "metrics.json", results)
    _write_json(output / "split_summary.json", data.split_summary)
    _write_json(output / "run_config.json", run_config)
    _write_json(
        output / "feature_manifest.json",
        {"input_columns": data.input_columns, "target_columns": data.target_columns},
    )
    _write_json(output / "calibration.json", calibration.as_dict())
    room_metrics.to_csv(output / "per_room_metrics.csv", index=False)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(
        [{"model": model_name, **metrics} for model_name, metrics in comparison.items()]
    ).to_csv(output / "model_comparison.csv", index=False)
    return results
