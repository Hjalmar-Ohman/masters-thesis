import os
import torch
import numpy as np
from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModel  # SigLIP
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2_5, ColQwen2_5_Processor
from sentence_transformers import SentenceTransformer  # Salesforce
from openai import OpenAI

from config import OPENAI_API_KEY
from common_utils import generate_image_summary

def create_embedder(model_name="CLIP", device=None):
    """
    Factory function to create an embedder based on model_name.
    """
    model_name = model_name.upper()
    embedders = {
        "SIGLIP": SigLIPEmbedder,
        "OPENAI": OpenAIEmbedder,
        "SFR": SFREmbedder,
        "COLPALI": ColPaliEmbedder,
        "COLQWEN": ColQwenEmbedder,
    }

    if model_name not in embedders:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(embedders.keys())}"
        )

    return embedders[model_name](device=device)

# ----------------------------------------------------------------------
# 1) BaseEmbedder: single base class for BOTH text-only and multimodal
# ----------------------------------------------------------------------
class BaseEmbedder(ABC):
    """
    Abstract base class for embedders. Provides:
      - embed_text(texts) -> 2D NumPy array
      - embed_image(images) -> 2D NumPy array (default is text-only flow)
    """

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_text(self, texts):
        """
        Returns a 2D numpy array of text embeddings.
        Must be implemented by all child classes.
        """
        pass

    def embed_image(self, images):
        """
        By default, treat as "text-only" embedder.  
        Subclasses that are truly multimodal should override this method.

        For text-only:
         - Convert each image to a text summary
         - Delegate to self.embed_text() to get embeddings
        """
        summaries = [generate_image_summary(pil_img) for pil_img in images]
        return self.embed_text(summaries)

    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalizes a tensor along the last dimension."""
        return tensor / tensor.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------
# 2) Multimodal embedders
# ----------------------------------------------------------------------
class SigLIPEmbedder(BaseEmbedder):
    """
    SigLIP multimodal embedder.
    """

    def __init__(self, model_id="C:/huggingface_models/siglip/", device=None):
        """
        :param model_id: Path to the local SigLIP model directory.
        :param device: If None, defaults to "cuda" if available, else "cpu".
        """
        super().__init__(device=device)
        
        # Force loading from the local directory to avoid caching issues
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    def embed_text(self, texts):
        """
        Embeds one or more text queries using SigLIP.

        :param texts: str or list of str
        :return: 2D numpy array of shape (batch_size, embedding_dim)
        """
        # Apply SigLIP's recommended text template
        texts = [f"This is a photo of {t}." for t in texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        return self._l2_normalize(text_emb).cpu().numpy()

    def embed_image(self, images):
        """
        Embeds one or more images using SigLIP.

        :param images: PIL Image or list of PIL Images
        :return: 2D numpy array of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)

        return self._l2_normalize(image_emb).cpu().numpy()
   
class ColPaliEmbedder(BaseEmbedder):
    """
    ColPali multimodal embedder.
    """
    def __init__(self, model_name="vidore/colpali-v1.3", device=None):
        super().__init__(device=device)
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            offload_folder=None,
            low_cpu_mem_usage=False
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def embed_text(self, texts):
        query_inputs = self.processor.process_queries(texts).to(self.device)
        with torch.no_grad():
            query_emb = self.model(**query_inputs)

        query_emb = self._l2_normalize(query_emb)
        query_emb = query_emb.cpu().numpy()

        print(f"Text Embeddings Shape: {query_emb.shape}")  # Should be (N, D)

        return query_emb

    def embed_image(self, images):
        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)

        image_emb = self._l2_normalize(image_emb)
        image_emb = image_emb.cpu().numpy()

        print(f"Image Embeddings Shape: {image_emb.shape}")  # Should be (N, D)

        return image_emb

class ColQwenEmbedder(BaseEmbedder):
    """
    ColQwen2.5-3b multimodal embedder.
    """

    def __init__(self, model_name="Metric-AI/colqwen2.5-3b-multilingual", device=None):
        super().__init__(device=device)

        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # bfloat16 may not be supported on CPU
        ).eval()


        # Load the ColQwen2.5 processor
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    def embed_text(self, texts):
        query_inputs = self.processor.process_queries(texts).to(self.device)
        with torch.no_grad():
            query_emb = self.model(**query_inputs)

        # L2-normalize along last dimension
        query_emb = self._l2_normalize(query_emb)
        query_emb = query_emb.cpu().numpy()

        print(f"Text Embeddings Shape: {query_emb.shape}")  # Should be (N, D)

        return query_emb

    def embed_image(self, images):
        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)


        image_emb = self._l2_normalize(image_emb)
        image_emb = image_emb.cpu().numpy()

        print(f"Image Embeddings Shape: {image_emb.shape}")  # Should be (N, D)

        return image_emb

# ----------------------------------------------------------------------
# 3) A text-only embedders
# ----------------------------------------------------------------------
class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text embedder."""

    def embed_text(self, texts):
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

class SFREmbedder(BaseEmbedder):
    """Salesforce SFR-Embedding text embedder."""

    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral", device=None):
        super().__init__(device=device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)