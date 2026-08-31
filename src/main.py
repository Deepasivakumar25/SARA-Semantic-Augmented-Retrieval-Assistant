from chunking import create_chunks
from document_loader import load_pdf
from generator import AnswerGenerator
from keyword_search import keyword_search
from reranker import CrossEncoderReranker
from semantic_search import SemanticRetriever

PDF_PATH = "Hybrid_Search_Practice.pdf"
QUESTION = "what is mean by hybrid search?"


def main():
    text = load_pdf(PDF_PATH)
    chunks = create_chunks(text, chunk_size=50)

    semantic = SemanticRetriever()
    semantic.build_index(chunks)
    _, semantic_indices = semantic.search(QUESTION, top_k=5)

    _, keyword_indices = keyword_search(chunks, QUESTION, top_k=5)
    unique_indices = list(dict.fromkeys(keyword_indices + semantic_indices))
    candidates = [chunks[i] for i in unique_indices]

    reranker = CrossEncoderReranker()
    best_chunks, _ = reranker.rerank(QUESTION, candidates, top_k=3)
    context = "\n\n".join(best_chunks)

    generator = AnswerGenerator()
    print(generator.answer(QUESTION, context))


if __name__ == "__main__":
    main()
