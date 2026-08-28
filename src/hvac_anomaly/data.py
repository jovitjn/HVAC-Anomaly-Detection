"""Data loading, temporal splitting, imputation, scaling, and sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import LABEL_COLUMN, TARGET_COLUMNS, TIMESTAMP_COLUMN


@dataclass
class SequenceSplit:
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    row_indices: np.ndarray


@dataclass
class PreparedData:
    train: SequenceSplit
    validation: SequenceSplit
    test: SequenceSplit
    input_columns: list[str]
    target_columns: list[str]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    split_summary: dict[str, object]


def _validate_columns(df: pd.DataFrame) -> None:
    required = {TIMESTAMP_COLUMN, LABEL_COLUMN, *TARGET_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load and standardize the supplied RTU CSV."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    _validate_columns(frame)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise")
    frame = frame.sort_values(TIMESTAMP_COLUMN).drop_duplicates(TIMESTAMP_COLUMN).reset_index(drop=True)
    numeric_columns = [column for column in frame.columns if column != TIMESTAMP_COLUMN]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    minute = result[TIMESTAMP_COLUMN].dt.hour * 60 + result[TIMESTAMP_COLUMN].dt.minute
    day = result[TIMESTAMP_COLUMN].dt.dayofweek
    result["Time: Minute Sine"] = np.sin(2 * np.pi * minute / 1440.0)
    result["Time: Minute Cosine"] = np.cos(2 * np.pi * minute / 1440.0)
    result["Time: Day Sine"] = np.sin(2 * np.pi * day / 7.0)
    result["Time: Day Cosine"] = np.cos(2 * np.pi * day / 7.0)
    return result


def _segment_ids(timestamps: pd.Series) -> np.ndarray:
    gap_minutes = timestamps.diff().dt.total_seconds().div(60)
    return gap_minutes.ne(1.0).cumsum().to_numpy(dtype=np.int32)


def _normal_row_masks(
    labels: np.ndarray,
    segments: np.ndarray,
    timestamps: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    block_size: int,
    seed: int,
):
    normal_indices = np.flatnonzero(labels == 0)
    if len(normal_indices) < 3:
        raise ValueError("At least three normal rows are required")
    train_mask = np.zeros(len(labels), dtype=bool)
    validation_mask = np.zeros(len(labels), dtype=bool)
    test_normal_mask = np.zeros(len(labels), dtype=bool)

    rng = np.random.default_rng(seed)
    blocks_by_hour: dict[int, list[np.ndarray]] = {hour: [] for hour in range(24)}
    for segment in np.unique(segments[labels == 0]):
        indices = np.flatnonzero((labels == 0) & (segments == segment))
        blocks = np.arange(len(indices)) // block_size
        for block_id in np.unique(blocks):
            block_rows = indices[blocks == block_id]
            if len(block_rows) < 3:
                continue
            midpoint = block_rows[len(block_rows) // 2]
            hour = int(pd.Timestamp(timestamps[midpoint]).hour)
            blocks_by_hour[hour].append(block_rows)

    for hour_blocks in blocks_by_hour.values():
        if len(hour_blocks) < 3:
            for rows in hour_blocks:
                train_mask[rows] = True
            continue
        order = rng.permutation(len(hour_blocks))
        train_count = min(int(len(order) * train_fraction), len(order) - 2)
        validation_count = max(1, int(len(order) * validation_fraction))
        validation_count = min(validation_count, len(order) - train_count - 1)
        for position, block_index in enumerate(order):
            rows = hour_blocks[int(block_index)]
            if position < train_count:
                train_mask[rows] = True
            elif position < train_count + validation_count:
                validation_mask[rows] = True
            else:
                test_normal_mask[rows] = True
    return train_mask, validation_mask, test_normal_mask


def _forward_fill_by_segment(values: pd.DataFrame, segments: np.ndarray) -> pd.DataFrame:
    result = values.copy()
    result["__segment__"] = segments
    result = result.groupby("__segment__", sort=False).ffill()
    return result


def _make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    segments: np.ndarray,
    sequence_length: int,
    end_mask: np.ndarray,
    require_all_mask: np.ndarray | None = None,
) -> SequenceSplit:
    x_sequences: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    output_labels: list[int] = []
    output_timestamps: list[np.datetime64] = []
    output_indices: list[int] = []

    for end in np.flatnonzero(end_mask):
        start = end - sequence_length + 1
        if start < 0 or segments[start] != segments[end]:
            continue
        if require_all_mask is not None and not require_all_mask[start : end + 1].all():
            continue
        x_sequences.append(x[start : end + 1])
        y_rows.append(y[end])
        output_labels.append(int(labels[end]))
        output_timestamps.append(timestamps[end])
        output_indices.append(int(end))

    if not x_sequences:
        raise ValueError("No valid sequences were produced for a split")
    return SequenceSplit(
        x=np.asarray(x_sequences, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        labels=np.asarray(output_labels, dtype=np.int8),
        timestamps=np.asarray(output_timestamps),
        row_indices=np.asarray(output_indices, dtype=np.int32),
    )


def _combine_splits(*splits: SequenceSplit) -> SequenceSplit:
    order = np.argsort(np.concatenate([split.row_indices for split in splits]))
    return SequenceSplit(
        x=np.concatenate([split.x for split in splits], axis=0)[order],
        y=np.concatenate([split.y for split in splits], axis=0)[order],
        labels=np.concatenate([split.labels for split in splits], axis=0)[order],
        timestamps=np.concatenate([split.timestamps for split in splits], axis=0)[order],
        row_indices=np.concatenate([split.row_indices for split in splits], axis=0)[order],
    )


def prepare_data(
    path: str | Path,
    sequence_length: int = 4,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    split_block_size: int = 60,
    split_seed: int = 42,
    exclude_proxy_temperatures: bool = True,
) -> PreparedData:
    """Prepare leakage-free normal-only training and a future mixed test set.

    Fault labels define the known normal reference rows and are retained for
    retrospective metrics. They are never passed to the model or used to fit a
    threshold.
    """
    frame = _add_time_features(load_frame(path))
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
    timestamps = frame[TIMESTAMP_COLUMN].to_numpy()
    segments = _segment_ids(frame[TIMESTAMP_COLUMN])
    train_mask, validation_mask, test_normal_mask = _normal_row_masks(
        labels,
        segments,
        timestamps,
        train_fraction,
        validation_fraction,
        split_block_size,
        split_seed,
    )

    excluded = {TIMESTAMP_COLUMN, LABEL_COLUMN, *TARGET_COLUMNS}
    candidate_inputs = [column for column in frame.columns if column not in excluded]
    proxy_temperatures = [
        column
        for column in candidate_inputs
        if column.startswith("VAV Box: Room") and column.endswith("Air Temperature")
    ]
    if exclude_proxy_temperatures:
        candidate_inputs = [column for column in candidate_inputs if column not in proxy_temperatures]
    numeric = frame[candidate_inputs + list(TARGET_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    numeric = _forward_fill_by_segment(numeric, segments)

    train_medians = numeric.loc[train_mask].median(axis=0, skipna=True)
    usable_inputs = [column for column in candidate_inputs if pd.notna(train_medians[column])]
    if not usable_inputs:
        raise ValueError("No usable input features remain after training-only preprocessing")
    if train_medians[list(TARGET_COLUMNS)].isna().any():
        missing_targets = train_medians[list(TARGET_COLUMNS)][train_medians[list(TARGET_COLUMNS)].isna()].index
        raise ValueError(f"Targets contain no training observations: {list(missing_targets)}")

    selected = usable_inputs + list(TARGET_COLUMNS)
    numeric = numeric[selected].fillna(train_medians[selected])
    x_raw = numeric[usable_inputs].to_numpy(dtype=np.float64)
    y_raw = numeric[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)

    x_scaler = StandardScaler().fit(x_raw[train_mask])
    y_scaler = StandardScaler().fit(y_raw[train_mask])
    x_scaled = x_scaler.transform(x_raw)
    y_scaled = y_scaler.transform(y_raw)

    train = _make_sequences(
        x_scaled, y_scaled, labels, timestamps, segments, sequence_length,
        end_mask=train_mask, require_all_mask=train_mask,
    )
    validation = _make_sequences(
        x_scaled, y_scaled, labels, timestamps, segments, sequence_length,
        end_mask=validation_mask, require_all_mask=validation_mask,
    )
    normal_test = _make_sequences(
        x_scaled, y_scaled, labels, timestamps, segments, sequence_length,
        end_mask=test_normal_mask, require_all_mask=test_normal_mask,
    )
    fault_mask = labels == 1
    fault_test = _make_sequences(
        x_scaled, y_scaled, labels, timestamps, segments, sequence_length,
        end_mask=fault_mask, require_all_mask=fault_mask,
    )
    test = _combine_splits(normal_test, fault_test)

    split_summary = {
        "rows": int(len(frame)),
        "normal_rows": int((labels == 0).sum()),
        "fault_rows": int((labels == 1).sum()),
        "input_features": len(usable_inputs),
        "targets": len(TARGET_COLUMNS),
        "excluded_proxy_temperature_features": proxy_temperatures if exclude_proxy_temperatures else [],
        "sequence_length": sequence_length,
        "train_sequences": int(len(train.x)),
        "validation_sequences": int(len(validation.x)),
        "test_sequences": int(len(test.x)),
        "test_normal_sequences": int((test.labels == 0).sum()),
        "test_fault_sequences": int((test.labels == 1).sum()),
        "test_start": str(pd.Timestamp(test.timestamps[0])),
        "test_end": str(pd.Timestamp(test.timestamps[-1])),
        "split_strategy": "disjoint 60-minute normal blocks for train/calibration/test; all contiguous fault windows held out for evaluation",
        "split_seed": split_seed,
        "gap_segments": int(segments.max()),
        "imputation": "forward fill within contiguous periods, then training median",
    }
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        input_columns=usable_inputs,
        target_columns=list(TARGET_COLUMNS),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        split_summary=split_summary,
    )
