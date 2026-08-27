import numpy as np
import pandas as pd
from src.config import load_config

def generate_data(n_samples: int, noise: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1, 30, n_samples)
    passengers = rng.integers(1, 5, n_samples)
    duration = distance * 2.5 + passengers * 1.2 + rng.normal(0, noise, n_samples)
    return pd.DataFrame({"distance_km": distance, "passengers": passengers}), duration

if __name__ == "__main__":
    cfg = load_config()
    X, y = generate_data(cfg["data"]["n_samples"], cfg["data"]["noise"], cfg["data"]["seed"])
    df = X.copy()
    df["duration_min"] = y
    df.to_csv(cfg["data"]["raw_path"], index=False)