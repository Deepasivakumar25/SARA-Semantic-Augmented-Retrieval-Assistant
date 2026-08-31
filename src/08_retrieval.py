import numpy as np


def retrieve_chunks(question: str, embedding_model, index, chunk_list: list[str], top_k: int = 3):
    question_embedding = embedding_model.encode([question])
    distance, index_number = index.search(np.array(question_embedding), k=top_k)
    return [chunk_list[int(idx)] for idx in index_number[0] if 0 <= idx < len(chunk_list)]
