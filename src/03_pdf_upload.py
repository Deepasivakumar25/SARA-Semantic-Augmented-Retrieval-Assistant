# Google Colab upload step from rag_pipeline.ipynb
from google.colab import files


def upload_pdf():
    uploaded = files.upload()
    return uploaded
