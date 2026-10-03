from prodml.predict import SentimentPredictor

def test_predict_returns_valid_label():
    p = SentimentPredictor().load()
    result = p.predict_one("الخدمة سيئة جداً ومحبطة")
    assert result["label"] in ("positive", "negative")
    assert 0 <= result["confidence"] <= 1

def test_predict_batch():
    p = SentimentPredictor().load()
    results = p.predict_batch(["ممتاز", "سيء جداً"])
    assert len(results) == 2
