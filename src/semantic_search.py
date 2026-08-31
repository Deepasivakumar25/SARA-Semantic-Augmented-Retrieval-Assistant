import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.asarray(embeddings))

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.asarray(query_embedding), min(top_k, len(self.chunks)))
        valid = [int(i) for i in indices[0] if 0 <= i < len(self.chunks)]
        return [self.chunks[i] for i in valid], valid
