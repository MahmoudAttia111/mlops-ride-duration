# mlops-ride-duration
 apply mlops
# Arabic Sentiment Analysis API

## Quick Startpip install -e ".[dev]"
uvicorn prodml.api.main:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "ممتاز جداً"}'
\`\`\`

Data: [HARD — Hotel Arabic Reviews Dataset][def]

[def]: https://github.com/elnagara/HARD-Arabic-Dataset