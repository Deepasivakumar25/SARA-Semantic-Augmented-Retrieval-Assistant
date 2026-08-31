from pathlib import Path
import importlib.util

SRC_DIR = Path(__file__).resolve().parent


def load_module(filename: str, module_name: str):
    path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pdf_extraction = load_module("04_pdf_extraction.py", "pdf_extraction")
chunking = load_module("05_text_chunking.py", "text_chunking")
embedding = load_module("06_embedding_and_faiss.py", "embedding_and_faiss")
llm_setup = load_module("07_llm_setup.py", "llm_setup")
retrieval = load_module("08_retrieval.py", "retrieval")
generation = load_module("09_rag_generation.py", "rag_generation")


def run_rag_pipeline(pdf_path: str, question: str, top_k: int = 3) -> str:
    pdf_text = pdf_extraction.extract_pdf_text(pdf_path)
    chunk_list = chunking.create_chunks(pdf_text, chunk_size=120)
    embedding_model, index = embedding.build_faiss_index(chunk_list)
    chatbot = llm_setup.load_chatbot()
    retrieved_chunks = retrieval.retrieve_chunks(
        question, embedding_model, index, chunk_list, top_k=top_k
    )
    return generation.generate_answer(chatbot, question, retrieved_chunks)


if __name__ == "__main__":
    question = "What kind of jobs are considered as category A?"
    print(run_rag_pipeline("visa_cheklist_2.pdf", question))
