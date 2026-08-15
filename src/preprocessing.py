from __future__ import annotations

import pandas as pd


def load_timeseries(csv_path, datetime_column):
    frame = pd.read_csv(csv_path)
    if datetime_column not in frame.columns:
        raise ValueError(f"Missing datetime column: {datetime_column}")
    frame[datetime_column] = pd.to_datetime(frame[datetime_column], errors="coerce")
    frame = frame.dropna(subset=[datetime_column]).copy()
    return frame.sort_values(datetime_column).reset_index(drop=True)
