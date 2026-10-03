def test_health(client):
    assert client.get("/health").json()["status"] == "ok"

def test_metadata(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    assert "labels" in resp.json()

def test_predict_happy_path(client):
    resp = client.post("/predict", json={"text": "رائع جداً"})
    assert resp.status_code == 200
    assert "label" in resp.json()

def test_predict_empty_text_rejected(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422

def test_predict_batch_endpoint(client):
    resp = client.post("/predict/batch", json={"texts": ["ممتاز", "سيء"]})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2
