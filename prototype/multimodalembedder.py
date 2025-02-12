import torch
from abc import ABC, abstractmethod
from transformers import FlavaProcessor, FlavaModel, CLIPProcessor, CLIPModel
#from lavis.models import load_model_and_preprocess
from PIL import Image

class BaseEmbedder(ABC):
    """Abstract base class for multimodal embedders."""

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_text(self, texts):
        """Returns a 2D numpy array of text embeddings."""
        pass

    @abstractmethod
    def embed_images(self, images):
        """Returns a 2D numpy array of image embeddings."""
        pass
    
    def _l2_normalize(self, tensor):
        """L2-normalizes a tensor along the last dimension."""
        return tensor / tensor.norm(dim=-1, keepdim=True)


class ClipEmbedder(BaseEmbedder):
    """CLIP-based embedder."""

    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        super().__init__(device=device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed_text(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return self._l2_normalize(embeddings).cpu().numpy()

    def embed_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return self._l2_normalize(embeddings).cpu().numpy()

class FlavaEmbedder(BaseEmbedder):
    """FLAVA-based embedder with aligned text and image embeddings."""

    def __init__(self, device=None):
        super().__init__(device=device)
        self.model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    def embed_text(self, texts):
        """
        Generate text embeddings aligned with image embeddings.

        :param texts: List of text descriptions.
        :return: NumPy array of text embeddings.
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Use empty images to force multimodal forward pass
        dummy_image = Image.new("RGB", (224, 224))  # A blank image placeholder
        images = [dummy_image] * len(texts)

        return self._extract_embeddings(texts, images)

    def embed_images(self, images):
        """
        Generate image embeddings aligned with text embeddings.

        :param images: List of PIL Image objects.
        :return: NumPy array of image embeddings.
        """
        if not isinstance(images, list):
            images = [images]

        # Use dummy text to force multimodal forward pass
        dummy_texts = [""] * len(images)  # Empty text placeholder

        return self._extract_embeddings(dummy_texts, images)

    def _extract_embeddings(self, texts, images):
        """
        Helper function to extract embeddings using FLAVA's multimodal forward pass.

        :param texts: List of text descriptions.
        :param images: List of PIL Image objects.
        :return: NumPy array of embeddings.
        """
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            multimodal_embeddings = outputs.multimodal_embeddings  # [Batch, Seq_len, Hidden_size]

        embeddings = multimodal_embeddings[:, 0, :]  # Extract CLS token
        embeddings = self._l2_normalize(embeddings)

        return embeddings.cpu().numpy()