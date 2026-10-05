import random
from locust import HttpUser, between, task

SAMPLE_REVIEWS = [
    "الفندق كان ممتاز والخدمة رائعة",
    "تجربة سيئة جداً ولن أكرر الزيارة",
    "الغرفة نظيفة والموظفين متعاونين",
    "مستوى الخدمة ضعيف جداً ومخيب للآمال",
]

class SentimentAPIUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.5, 2.0)

    @task(weight=10)
    def predict(self):
        text = random.choice(SAMPLE_REVIEWS)
        with self.client.post("/predict", json={"text": text}, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            label = resp.json()["label"]
            if label not in ("positive", "negative"):
                resp.failure(f"Invalid label: {label}")

    @task(weight=1)
    def health(self):
        self.client.get("/health")
