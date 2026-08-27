from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def generate_data(n_samples=100, seed=42):
    from src.prepare import generate_data as gen
    X, y = gen(n_samples, seed=seed)
    return X.values, y


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_model(X_train, y_train, params: dict | None = None):
    params = params or {}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val) -> dict[str, float]:
    preds = model.predict(X_val)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
    }