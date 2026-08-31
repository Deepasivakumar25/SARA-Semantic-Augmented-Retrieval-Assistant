from pypdf import PdfReader


def extract_pdf_text(pdf_path: str, first_page_only: bool = True) -> str:
    reader = PdfReader(pdf_path)
    pages = reader.pages[:1] if first_page_only else reader.pages
    return "".join(page.extract_text() or "" for page in pages)
