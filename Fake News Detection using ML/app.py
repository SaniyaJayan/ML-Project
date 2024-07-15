import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake["class"] = 0
data_true["class"] = 1

data_fake_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace = True)
    
data_true_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace = True)
    
data_fake_testing["class"] = 0
data_true_testing["class"] = 1

data_merge = pd.concat([data_fake, data_true], axis = 0)
data = data_merge.drop(['title','subject','date'], axis=1)
data = data.sample(frac = 1)
data.reset_index(inplace = True)
data.drop(['index'], axis=1, inplace=True)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
df = DecisionTreeClassifier()
df.fit(xv_train, y_train)

st.title('Fake News Detector')
input_text = st.text_input('Enter News Article')

def prediction(input_text):
    input_data = vectorization.transform([input_text])
    prediction = df.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 0:
        st.write("News is Fake")
    else:
        st.write("News is Real")