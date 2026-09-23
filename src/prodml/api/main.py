# src/prodml/api/main.py
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from prodml.predict import SentimentPredictor
from prodml.api.schemas import PredictRequest, PredictResponse, BatchRequest, BatchResponse

predictor = SentimentPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield

app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    request.state.correlation_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.correlation_id
    return response

@app.get("/health")
def health():
    return {"status": "ok" if predictor.model is not None else "not_ready"}

@app.get("/metadata")
def metadata():
    return {"model": "TF-IDF + LogisticRegression", "labels": ["positive", "negative"]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    result = predictor.predict_one(req.text)
    return PredictResponse(
        **result,
        correlation_id=request.state.correlation_id,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, request: Request):
    results = predictor.predict_batch(req.texts)
    return BatchResponse(results=[
        PredictResponse(**r, correlation_id=request.state.correlation_id, latency_ms=0)
        for r in results
    ])