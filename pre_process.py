# PDF processing imports:
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

from rag import BaseRAG

def process_pdf(pdf_path: str, rag_pipeline: BaseRAG):
    """
    Process a PDF file:
     - Extract text from each page with PyPDF2 and add to the RAG pipeline.
     - Extract page images using pdf2image and add them.
    """
    # --- Extract text ---
    print(f"Processing PDF text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()
        if text and text.strip():
            rag_pipeline.add_text(text.strip(), extra_metadata={"page_number": i+1})
    print(f"Extracted text from {num_pages} pages.")

    # --- Extract images (one image per page in this example) ---
    print("Converting PDF pages to images...")
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print("Error during PDF image conversion:", e)
        images = []

    for j, image in enumerate(images):
        rag_pipeline.add_image(image, extra_metadata={"page_number": j+1})
    print(f"Processed {len(images)} page images.")