from pypdf import PdfReader


def extract_pdf_text(pdf_path: str) -> str:
    file_content = PdfReader(pdf_path)
    pdf_text = ""
    for page in file_content.pages[:1]:
        pdf_text += page.extract_text() or ""
    return pdf_text
