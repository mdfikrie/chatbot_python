from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pickle
import numpy as np
import string
from util import JSONParser

# inisiasi flask
app = Flask(__name__)
# inisiasi api
api = Api(app)
CORS(app)


def preprocess(chat):
    # konversi ke non capital
    chat = chat.lower()
    # hilangkan tanda baca
    tandabaca = tuple(string.punctuation)
    chat = "".join(ch for ch in chat if ch not in tandabaca)
    return chat


def bot_response(chat, pipeline, jp):
    chat = preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_proba = max(res[0])
    if max_proba < 0.2:
        return "Maaf kak, aku ngga ngerti :(", None
    else:
        index = np.argmax(res[0])
        pred_tag = pipeline.classes_[index]
        return jp.get_response(pred_tag), pred_tag


# load model
with open("chatbot_model.pkl", "rb") as model_file:
    pipeline = pickle.load(model_file)


# load jsonparser
path = "data/intents.json"
jp = JSONParser()
jp.parse(path)


class ChatResource(Resource):
    def post(self):
        data = request.get_json()
        chat = data.get("chat")

        # validasi input
        if not chat:
            return {"message": "Silahkan masukkan chat"}, 400

        responses = bot_response(chat, pipeline, jp)
        return {"message": responses[0]}, 200


# resource
api.add_resource(ChatResource, "/api/chats", methods=["POST"])

if __name__ == "__main__":
    app.run(debug=True, port=5005)
