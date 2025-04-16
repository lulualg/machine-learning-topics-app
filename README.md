# Topic Classifier – Streamlit App

This is a simple web app built using **Streamlit** that classifies the topic of any sentence you enter. It uses basic **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to identify the topic from a predefined list.

---

## Topics It Can Identify
-  **Technology**
-  **Finance**
-  **Health**
-  **Politics**
-  **Sports**
-  **Food**

---

## How It Works

The app includes a lightweight AI model trained on example sentences for each topic using:

- `TfidfVectorizer` – converts text into numerical features
- `MultinomialNB` – a Naive Bayes classifier for text data

The training data is embedded directly in the app, making it fully **self-contained** — no need to load external models or datasets.

---

## How to Run

1. Make sure Python 3 is installed.
2. Install dependencies:
   ```bash
   pip install streamlit scikit-learn pandas
