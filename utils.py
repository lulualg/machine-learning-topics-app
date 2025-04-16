
# utils.py
# Custom text preprocessor for NLP pipeline

import re
import string

from sklearn.base import BaseEstimator, TransformerMixin
import spacy

# Load small English model
nlp = spacy.load("en_core_web_sm")

# Custom stopwords list (can be expanded)
STOPWORDS = set([
    "the", "and", "is", "in", "it", "of", "to", "a", "an", "on", "for", "with"
])

class preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if token.text not in STOPWORDS and not token.is_space
        ]
        return " ".join(tokens)
