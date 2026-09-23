# src/prodml/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from prodml.config import settings
from prodml.features import clean_arabic

def load_data():
    df = pd.read_csv(settings.data_path, sep="\t", names=["rating", "review"], encoding="utf-8")
    df["label"] = df["rating"].apply(lambda r: "positive" if r >= 4 else "negative")
    df["clean_review"] = df["review"].apply(clean_arabic)
    return train_test_split(df["clean_review"], df["label"], test_size=0.2,
                             random_state=42, stratify=df["label"])