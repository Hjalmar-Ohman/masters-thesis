# common_utils.py

import os
import io
import base64
import cv2

import torch

import fitz  # PyMuPDF
from PIL import Image

from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from config import OPENAI_API_KEY


# =========================
# 1. GLOBAL SETUP
# =========================

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # If needed, but watch for potential miscalculations.
os.environ["OMP_NUM_THREADS"] = "1" # This is to avoid conflicts with Faiss (for MAC users)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model_id = "openai/clip-vit-base-patch32"

# Load CLIP model/processor
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id)


# =========================
# 2. HELPER FUNCTIONS
# =========================

def encode_image_to_base64(pil_image):
    """
    Encodes a PIL image to base64 (PNG by default).
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def extract_embedded_images_from_pdf(pdf_path):
    """
    Extracts embedded images (figures) from a PDF file and returns a list of PIL images.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[PIL.Image]: A list of PIL image objects.
    """
    pil_images = []
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    
    # Iterate over pages in the PDF
    for page_index in range(len(doc)):
        page = doc[page_index]
        # Get the list of images on this page
        image_list = page.get_images(full=True)
        
        # If no images, move on to the next page.
        if not image_list:
            continue
        
        # Iterate through the images in the page
        for img in image_list:
            xref = img[0]  # Get the image reference ID
            base_image = doc.extract_image(xref)  # Extract the image data
            image_bytes = base_image["image"]  # Get the raw image bytes
            
            # Open the image with Pillow
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB mode if necessary
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # Append the PIL image object to the list
            pil_images.append(pil_image)
    
    doc.close()
    return pil_images

def extract_rasterized_images_from_pdf(pdf_path, output_folder="extracted_data", padding=300, xpadding = 300):
    """
    Extracts images from a PDF by rendering each page as an image and detecting image regions.

    Args:
        pdf_path (str): The path to the PDF file.
        output_folder (str, optional): The folder where extracted images will be saved. 
                                       Defaults to "extracted_data".
        padding (int, optional): Vertical padding (in pixels) added around detected image regions. 
                                 Defaults to 300.
        xpadding (int, optional): Horizontal padding (in pixels) added around detected image regions. 
                                  Defaults to 300.

    Returns:
        List[dict]: A list of dictionaries containing file paths of the extracted images, 
                    where each dictionary has the key `"image_path"`.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder
    image_paths = []
    
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    # Loop through all pages in the PDF
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)  # Load the page
        
        # Rasterize the page to an image
        pix = page.get_pixmap(dpi=300)  # Convert to image with high DPI
        full_image_path = os.path.join(output_folder, f"full_page_{page_index + 1}.png")
        pix.save(full_image_path)
        
        # Convert the image to OpenCV format (numpy array)
        full_image = cv2.imread(full_image_path)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create a binary image (to highlight potential image areas)
        _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (regions that are "boxes" in the image)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop through contours and crop the image regions
        img_index = 0
        for contour in contours:
            # Get the bounding box of each contour (x, y, width, height)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ignore small areas (you can adjust the threshold for min area size)
            if w > 50 and h > 50:
                # Add padding to the bounding box
                x_padded = max(x - padding, 0)  # Ensure x doesn't go below 0
                y_padded = max(y - padding, 0)  # Ensure y doesn't go below 0
                w_padded = min(w + 2 * xpadding, full_image.shape[1] - x_padded)  # Ensure width doesn't exceed image
                h_padded = min(h + 2 * padding, full_image.shape[0] - y_padded)  # Ensure height doesn't exceed image
                
                # Crop the image with padding
                cropped_image = full_image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
                
                # Convert cropped image to PIL format to save it as PNG
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                img_filename = f"page_{page_index + 1}_image_{img_index + 1}.png"
                img_path = os.path.join(output_folder, img_filename)
                pil_image.save(img_path, "PNG")
                
                image_paths.append({"image_path": img_path})
                img_index += 1
    
    doc.close()
    return image_paths

def generate_image_summary(image):
    """
    Given a PIL image, this function:
      1. Converts the image to a base64 string.
      2. Prepares a message payload that sends the image to GPT-4 along with a prompt requesting a detailed summary.
      3. Returns the generated summary text.
    
    Note: This function assumes that your `call_gpt_4` helper can handle both image and text parts.
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
            "text": "Please provide a detailed summary of the above image."
        }
    ]
    summary = call_gpt_4(content)
    return summary.strip()

def search_index(index, query_embedding, top_k=5):
    """
    Search the Faiss index for the top_k nearest neighbors to query_embedding.
    Returns (distances, indices).
    """
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices


def retrieve_context(indices, metadata):
    """
    Given a list of indices from Faiss, return the corresponding metadata (text or image).
    """
    retrieved = []
    for idx in indices[0]:
        retrieved.append(metadata[idx])
    return retrieved


def call_gpt_4(user_prompt, system_prompt: str = ""):
    """
    Calls GPT-4 (or GPT-4-like) with a list of message dicts.

    Example usage:
    user_prompt = [
                    {"type": "text", "text": "Generate 10 question and answer pairs based on the content of this page."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                ]
    """
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

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual GPT-4 model name if needed
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content