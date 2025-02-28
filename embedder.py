import os
import torch
import numpy as np
from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModel  # SigLIP
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2_5, ColQwen2_5_Processor
from sentence_transformers import SentenceTransformer  # Salesforce
from openai import OpenAI

from config import OPENAI_API_KEY

# ----------------------------------------------------------------------
# 1) Base Classes: MultimodalEmbedder and TextEmbedder
# ----------------------------------------------------------------------
class MultimodalEmbedder(ABC):
    """
    Abstract base class for multimodal embedders.
    Provides methods for embedding text and images.
    """
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_text(self, texts):
        pass

    @abstractmethod
    def embed_image(self, images):
        pass

    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True)

class TextEmbedder(ABC):
    """
    Abstract base class for text-only embedders.
    """
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_text(self, texts):
        pass

    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True)

# ----------------------------------------------------------------------
# 2) Multimodal embedders
# ----------------------------------------------------------------------
class SigLIPEmbedder(MultimodalEmbedder):
    """
    SigLIP multimodal embedder.
    """
    def __init__(self, model_id="C:/huggingface_models/siglip/", device=None):
        super().__init__(device=device)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    def embed_text(self, texts):
        texts = [f"This is a photo of {t}." for t in texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
        return self._l2_normalize(text_emb).cpu().numpy()

    def embed_image(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
        return self._l2_normalize(image_emb).cpu().numpy()

class ColPaliEmbedder(MultimodalEmbedder):
    """
    ColPali multimodal embedder.
    """
    def __init__(self, model_name="vidore/colpali-v1.3", device=None):
        super().__init__(device=device)
        self.model = ColPali.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def embed_text(self, texts):
        query_inputs = self.processor.process_queries(texts).to(self.device)
        with torch.no_grad():
            query_emb = self.model(**query_inputs)
        return self._l2_normalize(query_emb).cpu().numpy()

    def embed_image(self, images):
        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)
        return self._l2_normalize(image_emb).cpu().numpy()


class ColQwenEmbedder(MultimodalEmbedder):
    """
    ColQwen2.5-3b multimodal embedder.
    Crashing on laptop CPU.
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
# 3) Text-only embedders
# ----------------------------------------------------------------------
class OpenAIEmbedder(TextEmbedder):
    """OpenAI text embedder."""
    def embed_text(self, texts):
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

class SFREmbedder(TextEmbedder):
    """Salesforce SFR-Embedding text embedder."""
    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral", device=None):
        super().__init__(device=device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
