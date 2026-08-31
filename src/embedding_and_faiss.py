import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_faiss_index(chunk_list: list[str]):
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_embeddings = embedding_model.encode(chunk_list)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.asarray(chunk_embeddings))
    return embedding_model, index
