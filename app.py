import streamlit as st
import joblib
import pandas as pd  # Needed for model input format

def run():
    model = joblib.load(open('model.joblib', 'rb'))

    st.title("Topic Classifier")
    st.text("Enter a sentence and I'll try to guess its topic.")
    st.text("")
    
    userinput = st.text_input('Enter your sentence below:', placeholder='e.g., The stock market crashed today...')
    st.text("")

    if st.button("Classify"):
        input_series = pd.Series([userinput])  # Ensure correct format for model
        prediction = model.predict(input_series)[0]
        st.success(f'Topic: **{prediction}**')

if __name__ == "__main__":
    run()
