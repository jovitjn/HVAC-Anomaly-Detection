"""Evaluation metrics and result plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_room_metrics(
    validation_room_scores: np.ndarray,
    test_room_scores: np.ndarray,
    labels: np.ndarray,
    target_columns: list[str],
    quantile: float,
) -> pd.DataFrame:
    rows = []
    for index, column in enumerate(target_columns):
        threshold = float(np.quantile(validation_room_scores[:, index], quantile))
        predictions = (test_room_scores[:, index] > threshold).astype(np.int8)
        metrics = classification_metrics(labels, predictions, test_room_scores[:, index])
        room = column.split("Room ", 1)[1].split(" ", 1)[0]
        rows.append({"room": room, "threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def plot_training_history(history: dict[str, list[float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="training")
    ax.plot(history["val_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="Weighted MSE", title="Training history")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusion(metrics: dict[str, float | int], output: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Normal", "Fault"])
    ax.set_yticks([0, 1], ["Normal", "Fault"])
    ax.set(xlabel="Predicted", ylabel="Ground truth", title="Test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray, threshold: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    upper = np.quantile(scores, 0.995)
    ax.hist(np.clip(scores[labels == 0], 0, upper), bins=60, alpha=0.65, label="normal", density=True)
    ax.hist(np.clip(scores[labels == 1], 0, upper), bins=60, alpha=0.55, label="fault", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label="normal-only threshold")
    ax.set(xlabel="Global anomaly score", ylabel="Density", title="Score distribution on the future test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_timeline(timestamps: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    line_scores = scores.astype(float).copy()
    gap_mask = np.r_[False, np.diff(timestamps) > np.timedelta64(5, "m")]
    line_scores[gap_mask] = np.nan
    ax.plot(timestamps, line_scores, linewidth=0.8, color="#315a7d", label="anomaly score")
    ax.axhline(threshold, color="#ba3c3c", linestyle="--", label="threshold")
    fault_mask = (labels == 1) & ~gap_mask
    ax.fill_between(timestamps, 0, np.maximum(scores.max(), threshold), where=fault_mask, color="#e9a3a3", alpha=0.22, label="fault period")
    ax.set(xlabel="Time", ylabel="Score", title="Anomaly score over the held-out period")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_room_prediction(
    timestamps: np.ndarray,
    labels: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    room: str,
    output: Path,
) -> None:
    first_fault = int(np.argmax(labels == 1)) if np.any(labels == 1) else 0
    start = max(first_fault - 240, 0)
    end = min(first_fault + 480, len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps[start:end], actual[start:end], label="measured", linewidth=1.1)
    ax.plot(timestamps[start:end], predicted[start:end], label="predicted", linewidth=1.0, linestyle="--")
    ax.fill_between(
        timestamps[start:end],
        min(actual[start:end].min(), predicted[start:end].min()),
        max(actual[start:end].max(), predicted[start:end].max()),
        where=labels[start:end] == 1,
        color="#e9a3a3",
        alpha=0.2,
        label="fault period",
    )
    ax.set(xlabel="Time", ylabel="Temperature", title=f"Room {room}: measured and predicted temperature")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_baseline_comparison(metrics_by_model: dict[str, dict[str, float]], output: Path) -> None:
    names = list(metrics_by_model)
    f1 = [metrics_by_model[name]["f1"] for name in names]
    pr_auc = [metrics_by_model[name]["pr_auc"] for name in names]
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, f1, width, label="F1")
    ax.bar(x + width / 2, pr_auc, width, label="PR-AUC")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set(ylabel="Score", title="Held-out anomaly-detection comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_per_room(room_metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(room_metrics["room"].astype(str), room_metrics["f1"], color="#4d7898")
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Room", ylabel="F1", title="Room-level anomaly detection")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sensitivity["quantile"], sensitivity["precision"], marker="o", label="Precision")
    ax.plot(sensitivity["quantile"], sensitivity["recall"], marker="o", label="Recall")
    ax.plot(sensitivity["quantile"], sensitivity["f1"], marker="o", label="F1")
    ax.plot(sensitivity["quantile"], sensitivity["specificity"], marker="o", label="Specificity")
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1, label="Default")
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Normal calibration quantile", ylabel="Test metric", title="Threshold sensitivity")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
