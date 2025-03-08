import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "Spam_classification_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "email_classification.csv")


df = pd.read_csv(DATA_PATH)
vec = TfidfVectorizer(ngram_range=(1,2))
x = vec.fit_transform(df["email"])
labels = df["label"]
y = [1 if label == 'spam' else 0 for label in labels]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


smote = SMOTE()
x_train_re, y_train_re = smote.fit_resample(x_train, y_train)


model = LogisticRegression()
model.fit(x_train_re, y_train_re)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc*100:.2f}")


joblib.dump(model, MODEL_PATH)
joblib.dump(vec, VEC_PATH)

print("Model and vectorizer saved successfully!")