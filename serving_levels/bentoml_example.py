# serving_levels/bentoml_example.py
import bentoml
from pydantic import BaseModel, Field
import numpy as np
import joblib
from pathlib import Path

class Ride(BaseModel):
    distance_km: float = Field(gt=0, le=500)
    passengers: int = Field(ge=1, le=8)

@bentoml.service(traffic={"concurrency": 64, "timeout": 30})
class RideDuration:
    def __init__(self):
        path = Path(__file__).resolve().parent.parent / "models" / "rf_model.pkl"
        self.model = joblib.load(path)

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=100)
    def predict(self, inputs: list[Ride]) -> list[dict]:
        X = np.array([[r.distance_km, r.passengers] for r in inputs])
        preds = self.model.predict(X)
        return [{"duration_min": round(float(p), 2), "batch_size": len(inputs)} for p in preds]