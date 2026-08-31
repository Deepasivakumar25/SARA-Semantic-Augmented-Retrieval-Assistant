from rag_chunking import chunk_text
from rag_generator import RAGGenerator
from rag_pdf_loader import load_pdf
from rag_retriever import RAGRetriever


def run_rag_pipeline(pdf_path: str, question: str, top_k: int = 3) -> str:
    """Run the complete PDF-based RAG workflow."""
    pdf_text = load_pdf(pdf_path)
    chunks = chunk_text(pdf_text, chunk_size=120)

    retriever = RAGRetriever()
    retriever.build_index(chunks)
    retrieved_chunks = retriever.retrieve(question, top_k=top_k)
    context = "\n\n".join(retrieved_chunks)

    generator = RAGGenerator()
    return generator.generate(question, context)


if __name__ == "__main__":
    question = "What kind of jobs are considered as category A?"
    print(run_rag_pipeline("visa_cheklist_2.pdf", question))
