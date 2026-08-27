import joblib
from src.config import load_config
from src.train import generate_data, split_data, train_model

cfg = load_config()
X, y = generate_data(cfg["data"]["n_samples"], cfg["data"]["seed"])
X_train, X_val, y_train, y_val = split_data(X, y, cfg["data"]["test_size"], cfg["data"]["seed"])
model = train_model(X_train, y_train, cfg["model"])

import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")