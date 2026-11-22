# Time-Series Transformer Lab

A compact, reproducible benchmark for multivariate time-series regression with LSTM and Transformer models.

## Repository layout

```text
configs/       Model, data, and training settings
data/          Local input directory (the dataset is not included)
scripts/       Benchmark runner
src/           Preprocessing, sequence creation, models, training, and evaluation
tests/         Checkpoint recovery tests
training_recovery.py
results/       Generated checkpoints, metrics, and plots
```

## Data

The dataset is not included. The default configuration expects `data/raw/timeseries.csv` with these columns:

```text
timestamp,target,feature_1,feature_2,feature_3
```

Edit `data.csv_path`, `data.datetime_column`, `data.target_column`, and `features.columns` in `configs/default.yaml` to match another dataset. Rows are sorted chronologically, and feature scaling is fitted on the training period only. The default split is chronological; retain this protocol or document any changes when comparing results.

## Setup and run

From the repository root:

```bash
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell), then install dependencies and run one model:

```bash
python -m pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

Set `model.type` to `lstm` or `transformer` in the configuration. To run both models in sequence:

```bash
python scripts/run_benchmark.py
```

Training saves the selected checkpoint, scaler, test sequences and targets, and training history under `results/`. Evaluation writes model-specific MAE, RMSE, MAPE, and R² metrics and an actual-versus-predicted plot.

To resume a run, use the same configuration and recovery directory:

```bash
python -m src.train --config configs/default.yaml --checkpoint-dir results/recovery --resume
```

## Tests

The recovery tests do not require the project dataset:

```bash
python -m unittest discover -s tests -v
```

## Scope

This repository currently implements LSTM and Transformer baselines. Other architectures mentioned as future research directions are not included in the current implementation.
