import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import preprocessor  # Import the preprocessor class from utils.py

# Load the model when the app starts
model = joblib.load(open('model.joblib', 'rb'))

# Instantiate the preprocessor class
text_preprocessor = preprocessor()

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
        processed_text = text_preprocessor.transform(pd.Series([user_input]))[0]  # Apply transform and get the first result
        
        # Get the predicted topic
        topic = model.predict([processed_text])[0]  # Use the model to predict the topic
        
        # Display the predicted topic
        st.write(f'The predicted topic is: {topic}')
    else:
        st.write('Please enter some text to classify.')
