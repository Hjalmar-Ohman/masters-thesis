# common_utils.py

import os
import io
import base64

import torch

import fitz  # PyMuPDF
from PIL import Image

from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

# =========================
# 1. GLOBAL SETUP
# =========================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # If needed, but watch for potential miscalculations.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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

def extract_figures_from_pdf(pdf_path):
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

def embed_texts(texts, processor, model):
    """
    Given a list of text strings, return their CLIP embeddings as a NumPy array.
    """
    inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)

    # Normalize embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings.cpu().numpy()


def embed_images(images, processor, model):
    """
    Given a list of PIL images, return their CLIP embeddings as a NumPy array.
    """
    inputs = processor(
        images=images,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    # Normalize embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings.cpu().numpy()


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


def call_gpt_4(user_prompt, system_prompt = ""):
    """
    Calls GPT-4 (or GPT-4-like) with a list of message dicts. 
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

if __name__ == "__main__":
    PDF_FILE = "knowledge/catsanddogs.pdf"
    images_base64 = extract_figures_from_pdf(PDF_FILE)

    decoded_bytes = base64.b64decode(images_base64[0])
    with open("decoded_image.png", "wb") as f:
        f.write(decoded_bytes)
