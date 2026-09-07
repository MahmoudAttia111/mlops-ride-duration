from __future__ import annotations
from typing import Any, Optional
import json
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.config import load_config

AVG_SPEED_KM_PER_MIN = 0.5
PASSENGER_OVERHEAD_MIN = 0.5
DEFAULT_PARAMS: dict[str, Any] = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
FEATURES = ["distance_km", "passengers"]
TARGET = "duration_min"

def generate_data(n_samples=2_000, noise=1.0, seed=42):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 30.0, size=n_samples)
    passengers = rng.integers(1, 5, size=n_samples)
    duration = (distance / AVG_SPEED_KM_PER_MIN + passengers * PASSENGER_OVERHEAD_MIN
                + rng.normal(0.0, noise, size=n_samples))
    X = np.column_stack([distance, passengers]).astype(float)
    y = duration.astype(float)
    return X, y

def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def train_model(X_train, y_train, params=None):
    model = RandomForestRegressor(**(params or DEFAULT_PARAMS))
    model.fit(X_train, y_train)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }

def main():
    cfg = load_config()
    df = pd.read_parquet("data/processed/train.parquet")
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    model = train_model(X, y, cfg["model"])
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    print(f"train: fitted RandomForest({json.dumps(cfg['model'])}) on {len(df)} rows → models/rf_model.pkl")

if __name__ == "__main__":
    main()