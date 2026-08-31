import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def keyword_search(chunks, query, top_k=5):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix)[0]
    indices = np.argsort(scores)[::-1][:top_k]
    indices = [int(i) for i in indices]
    return [chunks[i] for i in indices], indices
