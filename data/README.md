# Time-Series Dataset

Place a CSV at:

```text
data/raw/timeseries.csv
```

The default configuration expects:

```text
timestamp
feature_1
feature_2
feature_3
target
```

You can change these names in `configs/default.yaml`.

## Evaluation protocol

The V1 uses a strict chronological split:

```text
Past -------------------------------> Future

TRAIN | VALIDATION | TEST
```

No random row split is used for the time dimension.

This repository is designed as a generic benchmarking lab. You can reuse it for:

- solar irradiance,
- battery health,
- temperature,
- load forecasting,
- financial/non-financial engineering series,
- sensor data,
- multivariate industrial signals.
