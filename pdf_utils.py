import cv2
import numpy as np
from typing import List, Dict

import tiktoken
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_from_pdf(pdf_path: str, chunk_size_max: int = 512) -> List[Dict[str, any]]:
    """
    Chunks text from a PDF file, ensuring sentence-level boundaries where possible.
    If a page's text exceeds chunk_size_max tokens, it will be split on the last period 
    before the token limit.
    
    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size_max (int): The maximum number of tokens per chunk (default 512 tokens).
    
    Returns:
        List[Dict[str, any]]: A list of dictionaries where each dictionary contains:
            - "text" (str): The chunked text.
            - "page_number" (int): The page number from which the text was extracted.
    """
    text_chunks = []
    reader = PdfReader(pdf_path)

    for page_i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text = page_text.strip()
            start = 0
            while start < len(text):
                # Binary search for the maximum substring length that fits within chunk_size_max tokens.
                low = start
                high = len(text)
                best = start
                while low <= high:
                    mid = (low + high) // 2
                    candidate = text[start:mid]
                    if num_tokens_from_string(candidate, "cl100k_base") <= chunk_size_max:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1

                # Look for the last period in the candidate substring
                last_period = text.rfind(".", start, best)
                if last_period != -1 and last_period > start:
                    chunk_end = last_period + 1  # Include the period in the chunk
                else:
                    chunk_end = best

                chunk = text[start:chunk_end].strip()
                if chunk:
                    text_chunks.append({
                        "text": chunk,
                        "page_number": page_i + 1  # 1-based indexing for page numbers
                    })
                start = chunk_end
                # Skip any whitespace before processing the next chunk
                while start < len(text) and text[start].isspace():
                    start += 1
    return text_chunks


def extract_images_from_pdf(pdf_path: str, padding: int = 300, xpadding: int = 300) -> List[Dict[str, any]]:
    """
    Extracts images from a PDF by rendering each page as an image and detecting image regions.

    Args:
        pdf_path (str): The path to the PDF file.
        padding (int, optional): Vertical padding (in pixels) around detected image regions. Defaults to 300.
        xpadding (int, optional): Horizontal padding (in pixels) around detected image regions. Defaults to 300.

    Returns:
        List[Dict[str, any]]: A list of dictionaries where each dictionary contains:
            - "pil_image" (PIL.Image): Extracted image as a PIL object.
            - "page_number" (int): The page number where the image was extracted.
    """
    pil_images = []
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=300)  # Render page at 300 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to OpenCV format
        full_image = np.array(img)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale and threshold
        gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours of image regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 50 and h > 50:  # Ignore very small detections
                # Apply padding
                x_padded = max(x - xpadding, 0)
                y_padded = max(y - padding, 0)
                w_padded = min(w + 2 * xpadding, full_image.shape[1] - x_padded)
                h_padded = min(h + 2 * padding, full_image.shape[0] - y_padded)

                # Crop and convert to PIL
                cropped_image = full_image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                pil_images.append({
                    "pil_image": pil_image,
                    "page_number": page_index + 1  # Convert 0-based index to 1-based page number
                })

    doc.close()
    return pil_images

if __name__ == "__main__":
    for chunk in chunk_text_from_pdf("knowledge/subset_riksbanken.pdf"):
        print({"type": "text", "content": chunk["text"], "page_number": chunk["page_number"]}, "\n")