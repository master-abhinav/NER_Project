import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn_crfsuite

# Load the saved model
crf = joblib.load('ner_crf_model.pkl')

# Feature extraction function
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'prefix[:3]': word[:3],
        'suffix[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Streamlit app
st.title("Named Entity Recognition (NER) with CRF")

st.write("Enter a sentence to get its Named Entity Recognition (NER) tags:")

user_input = st.text_input("Enter sentence here")

if user_input:
    # Tokenize the user input
    user_tokens = [(word, 'O') for word in user_input.split()]
    user_features = sent2features(user_tokens)

    # Predict NER tags
    user_pred = crf.predict_single(user_features)

    # Display results
    result = list(zip(user_input.split(), user_pred))
    st.write("Predicted NER tags:")
    st.write(result)