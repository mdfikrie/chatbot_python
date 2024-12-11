# import library
import string
import numpy as np
import pickle
from util import JSONParser
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


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


# load data
path = "data/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# praproses data
# case folding -> transform capital to non capital, hilangkan tanda baca
df["text_input_prep"] = df.text_input.apply(preprocess)

# pemodelan
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

# train
print("[INFO] Training data...")
pipeline.fit(df.text_input_prep, df.intents)

# save model
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)
    print("Model berhasil disimpan.")

# interaction with bot
print("[INFO] Anda sudah terhubung dengan bot kami..")
while True:
    chat = input("Anda >> ")
    res, tag = bot_response(chat, pipeline, jp)
    print(f"Bot >> {res}")
    if tag == "bye":
        break
