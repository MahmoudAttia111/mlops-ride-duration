# Final Project Report — Arabic Sentiment Analysis

**Track:** Deep Learning / NLP — Arabic Hotel Review Sentiment Classification
**Dataset:** [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) — 93,700 real reviews from Booking.com

---

## 1. Code & Packaging (pyproject.toml, src/, OOP)
- `pip install -e .` installs cleanly
- Package structure: `src/prodml/{config,data,features,predict,train,logging_conf}.py`
- Type hints on public functions, Pydantic Settings for config

## 2. API Endpoint (/predict, /health)
- `GET /health` → liveness check
- `GET /metadata` → model info
- `POST /predict` → single prediction with confidence score
- `POST /predict/batch` → batch predictions
- Verified live via public Codespaces URL + Swagger UI (`/docs`)

## 3. Docker (Dockerfile, 3-command README)
- Multi-stage Dockerfile (builder + runtime, non-root user)
- `docker run -p 8000:8000 <image>` works standalone

## 4. MLflow Tracking (≥5 experiments, best model promoted)
- Experiment: `arabic-sentiment`
- Model registered as `ArabicSentimentModel`, alias `@production`
- Accuracy: 0.942 | F1-score: 0.943

## 5. DVC Versioning (dvc repro reproduces result)
- `data/raw/balanced-reviews.txt` tracked via DVC (Backblaze remote)
- `dvc push` / `dvc pull` verified working

## 6. CI/CD (lint→test→build on every PR)
- GitHub Actions: `.github/workflows/ci.yml`
- Stages: lint (ruff, black) → test (pytest, 70%+ coverage) → build (Docker push)

## 7. Production Serving (Locust load test documented)
- Load test: 294 requests, **0% failure rate**
- Latency: p50=4ms, p95=18ms, p99=100ms
- Throughput: 15.35 req/s (20 concurrent users)

## 8. Monitoring
- Structured JSON logging with correlation IDs on every request
- Per-request latency tracked via `@timed` decorator
- *(Grafana/Prometheus dashboard: planned for Module 5 — not yet implemented)*

## 9. Peer Review
- *(Pending — to be completed once a course peer is identified)*

## 10. README & Structure
- 3-command setup verified end-to-end
- Architecture table + project history (v0.1.0 → v0.3.0) documented in README.md

---

## Repository & Tags
- Repo: `mlops-ride-duration` (MahmoudAttia111)
- `v0.1.0` — Mini Project 1 (Packaging, API, Docker)
- `v0.2.0` — Mini Project 2 (MLflow, DVC, CI/CD)
- `v0.3.0` — Mini Project 3 (Airflow, Load Testing)

## Key Learnings & Limitations
- TF-IDF + LogisticRegression baseline achieves strong accuracy (94%) but shows
  low-confidence predictions on context-dependent or ambiguous phrases
  (e.g., negation, sarcasm) — a clear motivation for the planned AraBERT
  comparison in the optimization phase.
- Debugging Git merge conflicts caused by Airflow-generated log files
  (root-owned, permission-denied) was a practical lesson in `.gitignore`
  discipline and container file ownership.
