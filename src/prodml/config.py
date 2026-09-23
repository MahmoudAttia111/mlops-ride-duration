class Settings(BaseSettings):
    data_path: str = "data/raw/balanced-reviews.txt"
    data_encoding: str = "utf-16"
    model_path: str = "models/sentiment_model.pkl"
    vectorizer_path: str = "models/vectorizer.pkl"
    port: int = 8000