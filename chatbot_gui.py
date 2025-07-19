import tkinter as tk
import random
import json
import pickle
import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

def predict_class(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(w.lower()) for w in sentence]
    sentence = " ".join(sentence)
    X = vectorizer.transform([sentence])
    tag = model.predict(X)[0]
    return tag

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I didn't understand that."

def send():
    msg = entry_box.get()
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + msg + "\n")
    entry_box.delete(0, tk.END)

    tag = predict_class(msg)
    response = get_response(tag)
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

root = tk.Tk()
root.title("ChatBuddy – Final Year Chatbot")

chat_log = tk.Text(root, bd=1, bg="light yellow", height=20, width=50, font="Arial")
chat_log.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(root, command=chat_log.yview)
chat_log['yscrollcommand'] = scrollbar.set

entry_box = tk.Entry(root, bd=1, width=40, font="Arial")
send_button = tk.Button(root, text="Send", width=12, command=send)

footer = tk.Label(root, text="Developed by Vidyashree V Naik | ID: [Your ID]", font=("Arial", 8), fg="gray")

chat_log.grid(row=0, column=0, columnspan=2)
scrollbar.grid(row=0, column=2, sticky='ns')
entry_box.grid(row=1, column=0)
send_button.grid(row=1, column=1)
footer.grid(row=2, column=0, columnspan=2, pady=5)

root.mainloop()