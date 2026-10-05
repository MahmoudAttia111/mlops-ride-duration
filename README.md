# Arabic Sentiment Analysis — MLOps Final Project

End-to-end MLOps pipeline for classifying Arabic hotel reviews as positive/negative,
built progressively across Mini Projects 1-3 following the ITI × MLOps MENA Community handbook.

## Quick Start
```bash
pip install -e ".[dev]"
python -m prodml.train
uvicorn prodml.api.main:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "ممتاز جداً"}'
```

## Results
- Model: TF-IDF (20k features, bigrams) + LogisticRegression
- Dataset: [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) (93,700 real Booking.com reviews)
- Accuracy: 94.2% | F1-score: 94.3%

## Architecture
| Component | Tool |
|---|---|
| Packaging | pyproject.toml, pytest (70%+ coverage) |
| API | FastAPI with Pydantic validation, structured JSON logging |
| Containerization | Docker (multi-stage build) |
| Experiment Tracking | MLflow (sqlite backend) |
| Model Registry | MLflow Registry with @production alias |
| Data Versioning | DVC (Backblaze remote) |
| CI/CD | GitHub Actions (lint → test → build) |
| Orchestration | Airflow DAG (weekly retrain) |
| Load Testing | Locust (0% failure rate @ 294 requests) |

## Project History
- `v0.1.0` — Mini Project 1: Packaging, API, Docker
- `v0.2.0` — Mini Project 2: MLflow, DVC, CI/CD
- `v0.3.0` — Mini Project 3: Airflow, Load Testing

## Load Test Results
294 requests, 0% failure rate, p50=4ms, p95=18ms, p99=100ms (see `reports/module-3.md`)
