# -*- coding: utf-8 -*-

import pandas as pd
import os

ham_path = r"ham path"
spam_path = r"spam path"

ham_emails = [os.path.join(ham_path, f) for f in os.listdir(ham_path)]
spam_emails = [os.path.join(spam_path, f) for f in os.listdir(spam_path)]

dataset = {
    "text" : [],
    "label" : []
}

for email in ham_emails:
    with open(email, "r", encoding="utf-8", errors="ignore") as f:
        dataset["text"].append(f.read())
        dataset['label'].append("ham")

for email in spam_emails:
    with open(email, "r", encoding="utf-8", errors="ignore") as f:
        dataset['text'].append(f.read())
        dataset['label'].append("spam")

dataframe = pd.DataFrame(dataset)
dataframe.head()

dataframe.shape

dataframe.info()

dataframe.nunique()

print(dataframe.loc[dataframe['label'] == "ham"].shape)
dataframe.loc[dataframe['label'] == "spam"].shape

import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

dataframe['Processed_Text'] = dataframe['text'].apply(preprocess)

dataframe['Processed_Text'][0]

from sklearn.model_selection import train_test_split

x = dataframe['Processed_Text']
y = dataframe['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

x_train_vex = vectorizer.fit_transform(x_train)
x_test_vex = vectorizer.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_vex, y_train)

from sklearn.metrics import classification_report
y_pred = model.predict(x_test_vex)
y_pred

print(classification_report(y_pred, y_test))

print(classification_report(y_test, y_pred))

def predict_email(text):
    process = preprocess(text)
    vectorize = vectorizer.transform([process])
    prediction = model.predict(vectorize)
    return prediction[0]

new  = dataframe['text'][0]
print(predict_email(new))

dataframe['text'][0]

