import io
import base64
import cv2
import numpy as np
from typing import List, Dict

from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image

from openai import OpenAI
from config import OPENAI_API_KEY

def encode_image_to_base64(pil_image):
    """
    Encodes a PIL image to base64 (PNG by default).
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, any]]: A list of dictionaries where each dictionary contains:
            - "text" (str): Extracted text from the page.
            - "page_number" (int): The page number where the text was extracted.
    """
    text_data = []
    reader = PdfReader(pdf_path)

    for page_i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text_data.append({
                "text": page_text.strip(),
                "page_number": page_i + 1  # Page numbers should be 1-based
            })

    return text_data


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

def generate_image_summary(image):
    """
    Args:
        image (PIL.Image): A PIL image object.
    Returns:
        str: A text summary of the image content.
    """
    base64_str = encode_image_to_base64(image)
    # Build a message payload that includes the image (as a data URL) and the prompt.
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_str}"}
        },
        {
            "type": "text",
            "text": "Provide a summary of the attached image and try to extract Title, X-Label, Y-Label, X-Tick, Y-Tick, and Legend"
        }
    ]
    summary = call_gpt_4(content)
    return summary.strip()

def call_gpt_4(user_prompt, system_prompt: str = ""):
    """
    Calls GPT-4 (or GPT-4-like) with a list of message dicts.

    Example usage:
        user_prompt = [
                        {"type": "text", "text": "Generate 10 question and answer pairs based on the content of this page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                    ]
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual GPT-4 model name if needed
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content