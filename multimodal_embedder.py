import torch
import numpy as np
from abc import ABC, abstractmethod

from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer  # Salesforce
from openai import OpenAI

from config import OPENAI_API_KEY
from common_utils import generate_image_summary


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
        if not isinstance(images, list):
            images = [images]
        summaries = [generate_image_summary(pil_img) for pil_img in images]
        return self.embed_text(summaries)

    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalizes a tensor along the last dimension."""
        return tensor / tensor.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------
# 2) Multimodal embedders
# ----------------------------------------------------------------------
class ColPaliEmbedder(BaseEmbedder):
    """
    A multimodal embedder that uses ColPali for both text and image embeddings.
    """

    def __init__(self, model_name="vidore/colpali-v1.3", device=None):
        super().__init__(device=device)
        # Hypothetical API usage; adapt to your actual code
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            offload_folder=None,
            low_cpu_mem_usage=False
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        query_inputs = self.processor.process_queries(texts).to(self.device)
        with torch.no_grad():
            query_emb = self.model(**query_inputs)

        query_emb = self._l2_normalize(query_emb)
        return query_emb.cpu().numpy()

    def embed_image(self, images):
        """
        Override the default image-to-text method with actual image embeddings.
        """
        if not isinstance(images, list):
            images = [images]

        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)

        image_emb = self._l2_normalize(image_emb)
        return image_emb.cpu().numpy()


# ----------------------------------------------------------------------
# 3) A text-only embedders
# ----------------------------------------------------------------------
class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text embedder. For images, uses generate_image_summary first."""

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

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
        if not isinstance(texts, list):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)