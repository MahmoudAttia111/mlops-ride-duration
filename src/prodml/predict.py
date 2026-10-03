# src/prodml/predict.py
import time
import joblib
from functools import wraps
from prodml.config import settings
from prodml.features import clean_arabic

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {(time.perf_counter() - t0)*1000:.2f}ms")
        return result
    return wrapper

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load(self):
        self.model = joblib.load(settings.model_path)
        self.vectorizer = joblib.load(settings.vectorizer_path)
        return self

    @timed
    def predict_one(self, text: str) -> dict:
        clean = clean_arabic(text)
        vec = self.vectorizer.transform([clean])
        label = self.model.predict(vec)[0]
        proba = float(self.model.predict_proba(vec).max())
        return {"label": label, "confidence": round(proba, 4)}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict_one(t) for t in texts]