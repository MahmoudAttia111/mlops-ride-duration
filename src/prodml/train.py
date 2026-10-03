# src/prodml/train.py
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from prodml.data import load_data
from prodml.config import settings

def main():
    X_train, X_val, y_train, y_val = load_data()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, settings.model_path)
    joblib.dump(vectorizer, settings.vectorizer_path)
    print(f"trained on {len(X_train)} rows -> {settings.model_path}")

if __name__ == "__main__":
    main()