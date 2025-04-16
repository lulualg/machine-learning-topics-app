import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import preprocessor  # assuming utils.py has a function to preprocess input text

# Load the model when the app starts
#model = load_model()
model = joblib.load(open('model.joblib', 'rb'))

# Streamlit interface
st.title('Topic Classifier')

st.write("""
    This app classifies topics based on your input text. 
    Type a sentence or a paragraph, and the app will predict the topic!
""")

# User input
user_input = st.text_area('Enter your text here')

# Process the user input and get the prediction when the user submits
if st.button('Predict Topic'):
    if user_input:
        # Preprocess the input
        processed_text = preprocessor(user_input)
        
        # Get the predicted topic
        topic = preprocessor(processed_text, model)
        
        # Display the predicted topic
        st.write(f'The predicted topic is: {topic}')
    else:
        st.write('Please enter some text to classify.')
