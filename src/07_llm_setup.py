from transformers import pipeline


def load_chatbot(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    return pipeline("text-generation", model=model_name)
