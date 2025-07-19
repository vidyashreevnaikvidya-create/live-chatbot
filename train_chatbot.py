import json
import random
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import nltk
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    data = json.load(file)

words = []
classes = []
documents = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        words.extend(tokens)
        documents.append((" ".join(tokens), intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

X_text = [doc[0] for doc in documents]
y = [doc[1] for doc in documents]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)
model = MultinomialNB()
model.fit(X, y)

with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("words.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("✅ Chatbot training complete!")