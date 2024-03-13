import numpy as np
import streamlit as st
import sklearn
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import string
from nltk.stem import PorterStemmer

porter = PorterStemmer()


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()
    for i in text:
        y.append(porter.stem(i))

    return " ".join(y)


tdif = pickle.load(open('vectorizor.pkl' , 'rb'))
model = pickle.load(open('multinomial_naive_bayes.pkl' , 'rb'))
st.title("Email/SMS Classifier")
input_sms = st.text_input("Enter Message / Email")

if st.button("predict"):
    text_trans = text_transform(input_sms)
    vector_input =tdif.transform([text_trans])
    result = model.predict(vector_input)[0]
    print(result)
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")




