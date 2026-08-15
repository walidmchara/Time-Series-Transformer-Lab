from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import SequenceDataset, make_sequences
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.preprocessing import load_timeseries


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, input_size, sequence_length):
    mc = cfg["model"]

    common = dict(
        input_size=input_size,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    )

    if mc["type"].lower() == "lstm":
        return LSTMRegressor(**common)

    if mc["type"].lower() == "transformer":
        return TransformerRegressor(
            **common,
            num_heads=mc["num_heads"],
            max_length=max(1024, sequence_length),
        )

    raise ValueError("model.type must be 'lstm' or 'transformer'")


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    dc = cfg["data"]
    features = list(cfg["features"]["columns"])
    target = dc["target_column"]
    dt_col = dc["datetime_column"]
    seq_len = int(dc["sequence_length"])

    frame = load_timeseries(dc["csv_path"], dt_col)

    required = set(features + [target])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.dropna(subset=features + [target]).copy()

    n = len(frame)
    test_n = int(n * dc["test_fraction"])
    val_n = int(n * dc["validation_fraction"])
    train_n = n - test_n - val_n

    if train_n <= seq_len or val_n <= 0 or test_n <= 0:
        raise ValueError("Dataset is too small for the configured splits.")

    train = frame.iloc[:train_n].copy()
    val = frame.iloc[max(0, train_n - seq_len + 1):train_n + val_n].copy()
    test = frame.iloc[max(0, train_n + val_n - seq_len + 1):].copy()

    scaler = StandardScaler().fit(train[features].to_numpy())

    x_train_raw = scaler.transform(train[features].to_numpy(dtype=np.float32))
    x_val_raw = scaler.transform(val[features].to_numpy(dtype=np.float32))
    x_test_raw = scaler.transform(test[features].to_numpy(dtype=np.float32))

    y_train_raw = train[target].to_numpy(dtype=np.float32)
    y_val_raw = val[target].to_numpy(dtype=np.float32)
    y_test_raw = test[target].to_numpy(dtype=np.float32)

    x_train, y_train = make_sequences(x_train_raw, y_train_raw, seq_len)
    x_val, y_val = make_sequences(x_val_raw, y_val_raw, seq_len)
    x_test, y_test = make_sequences(x_test_raw, y_test_raw, seq_len)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        SequenceDataset(x_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(features), seq_len).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    loss_fn = nn.MSELoss()

    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["type"].lower()
    checkpoint = out / f"{model_name}_best.pt"

    best, stale = float("inf"), 0
    history = []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(loss_fn(model(x), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
        })

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save({
                "model_state": model.state_dict(),
                "features": features,
                "config": cfg,
            }, checkpoint)
        else:
            stale += 1
            if stale >= int(cfg["training"]["patience"]):
                print("Early stopping.")
                break

    joblib.dump(scaler, out / "scaler.joblib")
    np.save(out / "x_test.npy", x_test)
    np.save(out / "y_test.npy", y_test)

    with open(out / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best model: {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
