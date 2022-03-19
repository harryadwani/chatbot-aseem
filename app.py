import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)
# app.config["CORS_HEADERS"] = "Content-Type"

# @cross_origin()


@app.route("/")
def home():
    return "ASEEM CHATBOT API"


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json
    msg = msg["msg"]
    if msg.startswith("my name is"):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith("hi my name is"):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)

    return res


@app.route("/demo", methods=["POST"])
def demo():
    msg = request.form["msg"]

    if msg.startswith("my name is"):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith("hi my name is"):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)

    return res


@app.route("/demo", methods=["GET"])
def demo1():
    return render_template("index.html")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    try:
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result
    except:
        return str(
            "Make sure you are using a supported browser (Chrome/Firefox/Safari) in order to use our services correctly. Also please ensure that you have granted webcam and mic permissions. <a href='https://drive.google.com/file/d/1Gl9fZP3EVxjQvlz3c7OUBtshXiIjEnvp/view?usp=sharing' target='_blank'>Click on this link for reference</a>. If you still can't resolve the problem, please write to the developer at 'harryadwani9@gmail.com'"
        )


if __name__ == "__main__":
    app.run(debug=False)
