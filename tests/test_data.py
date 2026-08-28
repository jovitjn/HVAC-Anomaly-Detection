from pathlib import Path

import numpy as np
import pandas as pd

from hvac_anomaly.constants import LABEL_COLUMN, TARGET_COLUMNS, TIMESTAMP_COLUMN
from hvac_anomaly.data import prepare_data


def test_sequences_do_not_cross_timestamp_gaps(tmp_path: Path):
    rows = 60
    timestamps = list(pd.date_range("2024-01-01", periods=30, freq="min"))
    timestamps += list(pd.date_range("2024-01-03", periods=30, freq="min"))
    frame = pd.DataFrame({TIMESTAMP_COLUMN: timestamps, LABEL_COLUMN: [0] * 45 + [1] * 15})
    for index, target in enumerate(TARGET_COLUMNS):
        frame[target] = np.linspace(index, index + 1, rows)
    frame["Sensor A"] = np.linspace(0, 3, rows)
    frame["Sensor B"] = np.sin(np.arange(rows))
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    data = prepare_data(
        path,
        sequence_length=4,
        train_fraction=0.6,
        validation_fraction=0.2,
        split_block_size=5,
    )
    for split in (data.train, data.validation, data.test):
        timestamps = pd.to_datetime(split.timestamps)
        assert len(timestamps) > 0
    assert data.train.x.shape[1] == 4
    assert data.train.y.shape[1] == len(TARGET_COLUMNS)
