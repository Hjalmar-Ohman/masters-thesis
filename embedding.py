from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import torch

# =======================================
# Abstract Base Classes for Embedders
# =======================================

class BaseTextEmbedder(ABC):
    """
    Abstract base class for text-only embedders.
    """
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single piece of text and return a vector as a NumPy array.
        """
        pass


class BaseMultimodalEmbedder(BaseTextEmbedder):
    """
    Abstract base class for multimodal embedders that can embed both text and images.
    """
    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed an image (e.g., a PIL.Image) and return a vector as a NumPy array.
        """
        pass


# =======================================
# Example Implementations
# =======================================

# 1. CLIP-based Multimodal Embedder

from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder(BaseMultimodalEmbedder):
    """
    A multimodal embedder using CLIP.
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,       # ensure text does not exceed model limits
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()[0].astype("float32")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()[0].astype("float32")


# 2. OpenAI ada-based Text-only Embedder

# (This is a stub example. In production you would call OpenAI's API or your chosen library.)
class OpenAIAdaTextEmbedder(BaseTextEmbedder):
    """
    A text-only embedder that uses OpenAI's ada embeddings.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize your OpenAI client here

    def embed_text(self, text: str) -> np.ndarray:
        # This is a simplified stub; replace it with an actual call to OpenAI's API.
        # For example, using the openai.Embedding.create() method.
        # Ensure the text is truncated/processed appropriately.
        print(f"Embedding text using OpenAIAdaTextEmbedder: {text[:50]}...")
        # Dummy vector for demonstration (typically you'd get a 768 or 1024-dimensional vector)
        return np.random.rand(768).astype("float32")


# =======================================
# Example Usage
# =======================================

if __name__ == "__main__":
    # Example usage of the CLIPEmbedder (multimodal)
    clip_embedder = CLIPEmbedder()
    sample_text = "This is an example text for CLIP."
    sample_image = Image.new("RGB", (224, 224), color="red")

    text_embedding = clip_embedder.embed_text(sample_text)
    image_embedding = clip_embedder.embed_image(sample_image)
    print("CLIP text embedding shape:", text_embedding.shape)
    print("CLIP image embedding shape:", image_embedding.shape)

    # Example usage of the OpenAIAdaTextEmbedder (text-only)
    openai_text_embedder = OpenAIAdaTextEmbedder(api_key="your-openai-api-key")
    ada_text_embedding = openai_text_embedder.embed_text("This is a sample text for OpenAI ada.")
    print("OpenAI ada text embedding shape:", ada_text_embedding.shape)
