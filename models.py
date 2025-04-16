
# models.py
# Trains a topic classifier (e.g. sports, politics, tech) and saves the model pipeline

import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import preprocessor

def main():
    # Load labeled topic data
    try:
        df = pd.read_csv('topics.csv')  # Make sure this file exists
    except FileNotFoundError:
        raise FileNotFoundError("topics.csv not found. Please add your dataset.")

    # Check structure
    if 'text' not in df.columns or 'topic' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'topic' columns.")

    # Define pipeline
    tfidf = TfidfVectorizer(max_features=5000)
    classifier = MultinomialNB()
    pipe = make_pipeline(preprocessor(), tfidf, classifier)

    # Train model
    pipe.fit(df['text'], df['topic'])

    # Save model
    joblib.dump(pipe, open('model.joblib', 'wb'))
    print("🚀 Model trained and saved as model.joblib")

if __name__ == "__main__":
    main()
