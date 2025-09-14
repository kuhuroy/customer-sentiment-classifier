import nlpaug.augmenter.word as naw 
import gradio as gr
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')


# Load trained TF-IDF and model (you’ll upload these files)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Basic text cleaning (same as before)
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def basic_processing(text):
    if isinstance(text, list):
        text = " ".join(text)
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def predict_sentiment(text):
    clean = basic_processing(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    prob_dict = {cls: float(f"{p:.2f}") for cls, p in zip(model.classes_, prob)}
    return pred, prob_dict

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type customer feedback here..."),
    outputs=[gr.Label(label="Predicted Sentiment"), gr.Label(label="Confidence Scores")],
    title="Customer Feedback Sentiment Classifier",
    description="Enter feedback text to predict sentiment (Positive, Neutral, Negative)"
)

if __name__ == "__main__":
    demo.launch()
    
