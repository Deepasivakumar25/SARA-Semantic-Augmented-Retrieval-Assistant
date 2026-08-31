def generate_answer(chatbot, question: str, retrieved_chunks: list[str], max_new_tokens: int = 50) -> str:
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
<|user|>

1. Please use the context and carefully read the question and before answering.
2. Please ignore the characters \\n\\n
3. The sentence should be meaningful

Use ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present, reply exactly:

I couldn't find that information.

<|assistant|>
"""
    response = chatbot(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    )
    return response[0]["generated_text"].strip()
