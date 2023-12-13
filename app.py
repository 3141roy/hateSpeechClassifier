import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

import string
import nltk
import re
import contractions

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(nltk.corpus.stopwords.words('english'))
def stopWords_removal(text):
    words = word_tokenize(text)
    words_filtered = [w.lower() for w in words if w.lower() not in stopwords]
    new_text = " ".join(words_filtered)
    return new_text if new_text else ""

def redundance_removal(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1", text)


def processing(text):

    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)

    text = stopWords_removal(text)


    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z0-9' ]", ' ', text)
    text = re.sub(r' +', ' ', text)

    extract_words = re.compile(r'\W+')
    text = extract_words.sub(' ', text)

    text = re.sub('[^a-zA-Z\s]', '', text)

    text = re.sub(r'user(?:name)?\s', '', text)

    text = redundance_removal(text)

    return text

#@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'model_TEST.h5'
    model = tf.keras.models.load_model(model_path)
    return model

#@st.cache(allow_output_mutation=True)
def preprocess_text(text, tokenizer, max_length):
    text = processing(text)
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    return padded_sequences

#@st.cache(allow_output_mutation=True)
def classify_text(text, model, tokenizer, max_length):
    preprocessed_text = preprocess_text(text, tokenizer, max_length)
    predictions = model.predict(preprocessed_text)
    return predictions.flatten().tolist()

def main():
    st.title("Text Classification App")

    model = load_model()
    tokenizer = Tokenizer()
    max_length = 100

    st.write("Enter text for classification:")
    user_input = st.text_area("Input Text", "")

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text for classification.")
        else:
            predictions = classify_text(user_input, model, tokenizer, max_length)
            result_df = pd.DataFrame({"Class": ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"],
                                      "Probability": predictions})
            st.bar_chart(result_df.set_index("Class"))

if __name__ == "__main__":
    main()
