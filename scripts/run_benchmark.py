from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import yaml


CONFIG = Path("configs/default.yaml")


def run(model_type):
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["type"] = model_type

    temp = Path(f"configs/{model_type}.yaml")
    with open(temp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    subprocess.run(["python", "-m", "src.train", "--config", str(temp)], check=True)
    subprocess.run(["python", "-m", "src.evaluate", "--config", str(temp)], check=True)


if __name__ == "__main__":
    for model in ["lstm", "transformer"]:
        run(model)

    print("Benchmark complete. Check results/*_metrics.json")
