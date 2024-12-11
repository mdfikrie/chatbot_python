import pickle


with open("chatbot_model.pkl", "rb") as model_file:
    model_chat = pickle.load(model_file)
