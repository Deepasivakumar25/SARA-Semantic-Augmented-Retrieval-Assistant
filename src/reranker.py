import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=3):
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [chunks[int(i)] for i in ranked], scores
