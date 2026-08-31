from transformers import pipeline


def load_chatbot():
    return pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")
