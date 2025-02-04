import io
from PIL import Image
import fitz  # PyMuPDF

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

def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from each page of a PDF file using PyPDF2.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[dict]: A list of dictionaries containing page text and page number.
    """
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            texts.append({"text": text.strip(), "page_number": i + 1})
    return texts