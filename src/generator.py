from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class AnswerGenerator:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def answer(self, question, context):
        prompt = f"""<|user|>\nUse ONLY the context below.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nIf the answer is not present, reply exactly:\nI couldn't find that information.\n<|assistant|>"""
        response = self.pipeline(prompt, max_new_tokens=120, do_sample=False, return_full_text=False)
        return response[0]["generated_text"].strip()
