# SARA — Semantic-Augmented Retrieval Assistant

SARA (Semantic-Augmented Retrieval Assistant) is a Retrieval-Augmented Generation (RAG) based AI assistant that combines semantic search with Large Language Models (LLMs) to generate accurate and context-aware responses from a knowledge base.

## Features

- Document ingestion and preprocessing
- Text chunking for efficient retrieval
- Semantic embeddings using Sentence Transformers
- Similarity search using FAISS
- Context-aware response generation with an LLM
- End-to-end Retrieval-Augmented Generation pipeline

## RAG Workflow

```text
Documents
   ↓
Preprocessing
   ↓
Text Chunking
   ↓
Embeddings
   ↓
FAISS Vector Store
   ↓
Semantic Retrieval
   ↓
Relevant Context
   ↓
LLM
   ↓
Generated Answer
```

## Repository Contents

This repository contains the project in **both Jupyter Notebook (`.ipynb`) and Python (`.py`) formats**.

- **`rag_pipeline.ipynb`** — The complete RAG workflow in Jupyter/Google Colab notebook format, useful for learning, experimentation, and step-by-step execution.
- **Python files** — The same workflow organized into separate Python files based on the sequence of the code in the notebook, making the implementation easier to understand and reuse.

The notebook and Python files are maintained together so that the project can be explored interactively in the notebook or executed using modular Python code.

## Technologies

- Python
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

## Purpose

The project demonstrates how semantic retrieval can be integrated with an LLM to build an AI assistant that answers questions using relevant information from a custom knowledge base, helping improve response relevance and reduce hallucinations.

## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Deepasivakumar25/SARA-Semantic-Augmented-Retrieval-Assistant.git
cd SARA-Semantic-Augmented-Retrieval-Assistant
pip install -r requirements.txt
```

Run the notebook or use the Python files according to the instructions provided in the repository.

## Project Status

This repository is part of a hands-on learning project focused on understanding and implementing RAG-based AI applications.

## License

This project is intended for educational and learning purposes.
