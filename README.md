# Arabic Sentiment Analysis API

Classifies Arabic hotel reviews as positive or negative, trained on the HARD dataset (93,700 real Booking.com reviews).

## Quick Start
```bash
pip install -e ".[dev]"
python -m prodml.train
uvicorn prodml.api.main:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "ممتاز جداً"}'
```

## Results
- Accuracy: 94%
- F1-score: 94% (macro avg)

## Data
[HARD — Hotel Arabic Reviews Dataset](https://github.com/elnagara/HARD-Arabic-Dataset)

## Docker
```bash
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/arabic-sentiment-api:0.1.0
```
