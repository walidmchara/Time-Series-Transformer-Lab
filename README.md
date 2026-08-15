# Time-Series Transformer Lab — V1

A reproducible benchmarking environment for multivariate time-series regression.

## Models

- LSTM
- Transformer Encoder

## Protocol

- chronological train/validation/test split
- standardized predictors fitted on training data only
- fixed sequence windows
- early stopping
- MAE / RMSE / MAPE / R²
- actual vs predicted plots

## Run one model

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

## Benchmark both models

```bash
python scripts/run_benchmark.py
```

Change `model.type` in the configuration between:

```text
lstm
transformer
```

Future versions can add Informer, Linformer, PatchTST, TimesNet and other efficient time-series architectures.
