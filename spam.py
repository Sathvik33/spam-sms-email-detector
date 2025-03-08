import os
import string
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "Spam_classification_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "email_classification.csv")

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    print("🔹 Loading saved model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
else:
    print("Training new model and saving it...")
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
    
    print(f"Model accuracy: {acc * 100:.2f}%")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)

def predict_spam(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text_vec = vec.transform([text])
    pred = model.predict(text_vec)
    return 'Spam' if pred == 1 else 'Ham'

st.title("SMS Spam Classifier")
st.write("Enter a message and check if it's spam or not!")

user_input = st.text_area("Enter text (email or Message) for classification:")
if st.button("Classify"):
    result = predict_spam(user_input)
    st.write(f"The message is: **{result}**")