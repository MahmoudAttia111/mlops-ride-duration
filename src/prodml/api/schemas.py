# src/prodml/api/schemas.py
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="Arabic review text")

class PredictResponse(BaseModel):
    label: str
    confidence: float
    correlation_id: str
    latency_ms: float

class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)

class BatchResponse(BaseModel):
    results: list[PredictResponse]