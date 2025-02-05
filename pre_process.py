import io
from PIL import Image
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

def extract_figures_from_pdf(pdf_path: str):
    """
    Extracts embedded images (figures) from a PDF file and returns a list of PIL images.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[PIL.Image.Image]: A list of PIL image objects.
    """
    pil_images = []
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    
    # Iterate over pages in the PDF
    for page_index in range(len(doc)):
        page = doc[page_index]
        # Get the list of images on this page
        image_list = page.get_images(full=True)
        
        if not image_list:
            continue
        
        for img in image_list:
            xref = img[0]  # Get the image reference ID
            base_image = doc.extract_image(xref)  # Extract the image data
            image_bytes = base_image["image"]
            
            # Open the image with Pillow
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            pil_images.append(pil_image)
    
    doc.close()
    return pil_images

def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """
    Splits the text into overlapping chunks of `chunk_size` words.
    Overlap helps with contexts that run across boundaries.

    Args:
        text (str): The full text to split.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap between consecutive chunks.

    Yields:
        str: A chunk of text.
    """
    words = text.split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        yield " ".join(chunk)
        start += (chunk_size - overlap)


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 200, overlap: int = 50):
    """
    Extracts text from each page of a PDF file and then splits it into chunks.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): Number of words per chunk.
        overlap (int): Overlap in words between consecutive chunks.

    Returns:
        List[dict]: A list of dictionaries containing:
                    - 'text': the chunk text
                    - 'page_number': the page number
                    - 'chunk_index': which chunk in that page
    """
    reader = PdfReader(pdf_path)
    texts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue

        page_text = page_text.strip()
        # Chunk this page's text
        chunk_counter = 0
        for chunk in _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            texts.append({
                "text": chunk,
                "page_number": i + 1,
                "chunk_index": chunk_counter
            })
            chunk_counter += 1

    return texts
