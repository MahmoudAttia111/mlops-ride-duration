import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. تحميل الداتا
df = pd.read_csv(
    "data/raw/balanced-reviews.txt",
    sep="\t",
    encoding="utf-16",
    usecols=["rating", "review"],
)
df["rating"] = df["rating"].astype(int)
df["label"] = df["rating"].apply(lambda r: "positive" if r >= 4 else "negative")
print("Shape:", df.shape)
print(df["label"].value_counts())

# 2. تنظيف النص العربي
def clean_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean_review"] = df["review"].apply(clean_arabic)

# 3. Split + Baseline model
X_train, X_val, y_train, y_val = train_test_split(
    df["clean_review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("\n--- Baseline Results ---")
print(classification_report(y_val, model.predict(X_val_vec)))
