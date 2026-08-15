from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train import build_model


def safe_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["output"]["directory"])
    model_name = cfg["model"]["type"].lower()

    x_test = np.load(out / "x_test.npy")
    y_test = np.load(out / "y_test.npy")

    ckpt = torch.load(out / f"{model_name}_best.pt", map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        cfg,
        input_size=x_test.shape[-1],
        sequence_length=x_test.shape[1],
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    with torch.no_grad():
        pred = model(
            torch.as_tensor(x_test, dtype=torch.float32, device=device)
        ).cpu().numpy()

    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAPE_percent": safe_mape(y_test, pred),
        "R2": float(r2_score(y_test, pred)),
    }

    print(json.dumps(metrics, indent=2))

    with open(out / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 4.8))
    plt.plot(y_test, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Test sample")
    plt.ylabel("Target")
    plt.title(f"{model_name.upper()}: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{model_name}_actual_vs_predicted.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
