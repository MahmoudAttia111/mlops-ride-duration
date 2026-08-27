import json, joblib, os
from src.config import load_config
from src.train import generate_data, split_data, evaluate

cfg = load_config()
X, y = generate_data(cfg["data"]["n_samples"], cfg["data"]["seed"])
_, X_val, _, y_val = split_data(X, y, cfg["data"]["test_size"], cfg["data"]["seed"])
model = joblib.load("models/rf_model.pkl")
metrics = evaluate(model, X_val, y_val)

os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=2)