# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from prodml.api.main import app, predictor

@pytest.fixture(scope="session", autouse=True)
def load_model():
    predictor.load()

@pytest.fixture
def client():
    return TestClient(app)