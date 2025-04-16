import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
data = {
    "text": [
        "The stock market is doing well",         # Finance
        "Invest in cryptocurrency and NFTs",      # Finance
        "Apple releases new iPhone",              # Technology
        "Artificial intelligence is booming",     # Technology
        "New vaccine approved by health board",   # Health
        "Meditation helps reduce stress",         # Health
        "Elections coming up this November",      # Politics
        "The president gave a speech today",      # Politics
        "The soccer match was intense",           # Sports
        "Olympics are scheduled next year",       # Sports
        "I love eating pizza and sandwiches",     # Food
        "Burgers and fries are my favorite food", # Food
    ],
    "label": [
        "Finance", "Finance",
        "Technology", "Technology",
        "Health", "Health",
        "Politics", "Politics",
        "Sports", "Sports",
        "Food", "Food"
    ]
}

# Create a simple pipeline
df = pd.DataFrame(data)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df["text"], df["label"])

# Streamlit UI
def run():
    st.title("Topic Classifier")
    st.write("Enter a sentence and I'll try to guess its topic.")
    
    userinput = st.text_input("Your sentence:", placeholder="e.g., I just bought some Bitcoin")
    
    if st.button("Classify"):
        prediction = model.predict([userinput])[0]
        st.success(f"Predicted Topic: **{prediction}**")

if __name__ == "__main__":
    run()
