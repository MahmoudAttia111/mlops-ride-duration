import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.predict import SentimentPredictor
from prodml.logging_conf import logger, correlation_id_var
from prodml.api.schemas import PredictRequest, PredictResponse, BatchRequest, BatchResponse

predictor = SentimentPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    logger.info("Model loaded successfully")
    yield

app = FastAPI(title="Arabic Sentiment API", version="0.1.0", lifespan=lifespan)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    cid = str(uuid.uuid4())
    correlation_id_var.set(cid)
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Request-ID"] = cid
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

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
    logger.info(f"Prediction made: {result['label']}")
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
