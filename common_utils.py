# common_utils.py

import os
import io
import json
import base64

import torch
import numpy as np
import faiss

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


def encode_image_to_base64(pil_image):
    """
    Encodes a PIL image to base64 (PNG by default).
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def call_gpt_4(messages):
    """
    Calls GPT-4 (or GPT-4-like) with a list of message dicts. 
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual GPT-4 model name if needed
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content