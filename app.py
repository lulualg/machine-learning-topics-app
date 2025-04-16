import streamlit as st
import joblib
import pandas as pd  # Needed for model input format

# Map of numeric labels to topic names
TOPIC_LABELS = {
    0: "Technology",
    1: "Finance",
    2: "Health",
    3: "Politics",
    4: "Sports",
    5: "Food",
    # Add more if needed
}

def run():
    model = joblib.load(open('model.joblib', 'rb'))

    st.title("Topic Classifier")
    st.text("Enter a sentence and I'll try to guess its topic.")
    st.text("")
    
    userinput = st.text_input('Enter your sentence below:', placeholder='e.g., The stock market crashed today...')
    st.text("")

    if st.button("Classify"):
        input_series = pd.Series([userinput])  # Ensure correct format for model
        label = model.predict(input_series)[0]
        topic = TOPIC_LABELS.get(label, f"Unknown (label: {label})")
        st.success(f'Topic: **{topic}**')

if __name__ == "__main__":
    run()
