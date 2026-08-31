import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        self.chunks = chunks
        embeddings = self.embedding_model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.asarray(embeddings))

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        question_embedding = self.embedding_model.encode([question])
        _, indices = self.index.search(np.asarray(question_embedding), top_k)
        return [self.chunks[int(idx)] for idx in indices[0] if 0 <= idx < len(self.chunks)]
