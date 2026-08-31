from transformers import pipeline


class RAGGenerator:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.chatbot = pipeline("text-generation", model=model_name)

    def generate(self, question: str, context: str, max_new_tokens: int = 50) -> str:
        prompt = f"""<|user|>

Please use the context and carefully read the question before answering.
Use ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present, reply exactly:
I couldn't find that information.

<|assistant|>
"""
        response = self.chatbot(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return response[0]["generated_text"].strip()
