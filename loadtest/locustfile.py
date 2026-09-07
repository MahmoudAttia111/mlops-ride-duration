# loadtest/locustfile.py
import random
from locust import HttpUser, between, task

def _ride():
    return {"distance_km": round(random.uniform(0.5, 30.0), 2),
            "passengers": random.randint(1, 4)}

class FastAPIBaselineUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.5, 2.0)

    @task(weight=10)
    def predict(self):
        with self.client.post("/predict", json=_ride(), catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            duration = resp.json()["duration_min"]
            if duration < 0 or duration > 600:
                resp.failure(f"Implausible duration: {duration}")

    @task(weight=1)
    def health(self):
        self.client.get("/health")