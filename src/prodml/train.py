import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from prodml.data import load_data
from prodml.config import settings

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic-sentiment")

    X_train, X_val, y_train, y_val = load_data()

    with mlflow.start_run():
        params = {"max_features": 20000, "ngram_range": "(1,2)", "max_iter": 1000}
        mlflow.log_params(params)

        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_val_vec)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, pos_label="positive")
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, settings.model_path)
        joblib.dump(vectorizer, settings.vectorizer_path)
        mlflow.sklearn.log_model(model, "model")

        print(f"train: acc={acc:.3f} f1={f1:.3f} -> {settings.model_path}")

if __name__ == "__main__":
    main()
