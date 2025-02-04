import numpy as np
import torch
from PIL import Image
from abc import ABC, abstractmethod

###############################
# 1. ABSTRACT EMBEDDERS
###############################

class BaseTextEmbedder(ABC):
    """
    Abstract base class for text-only embedders.
    """
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a piece of text and return a vector as a NumPy array.
        """
        pass


class BaseMultimodalEmbedder(BaseTextEmbedder):
    """
    Abstract base class for multimodal embedders (supporting text and images).
    """
    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed an image (e.g., a PIL Image) and return a vector as a NumPy array.
        """
        pass


###############################
# 2. IMPLEMENTATION: CLIP EMBEDDER (MULTIMODAL)
###############################

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
            truncation=True,  # ensure text does not exceed model limits
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
