from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> str:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
