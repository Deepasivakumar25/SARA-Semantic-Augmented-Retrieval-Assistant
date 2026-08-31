from pypdf import PdfReader


def load_pdf(pdf_path: str, pages: int | None = 1) -> str:
    """Extract text from a PDF. By default, extract the first page."""
    reader = PdfReader(pdf_path)
    selected_pages = reader.pages[:pages] if pages else reader.pages
    return "\n".join(page.extract_text() or "" for page in selected_pages)
