import torch
import numpy as np
from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModel, AutoTokenizer
from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import torch.nn.functional as F

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
        self.has_custom_search = False

    @abstractmethod
    def embed_text(self, texts):
        pass

    @abstractmethod
    def embed_image(self, images):
        pass

class TextEmbedder(ABC):
    """
    Abstract base class for text-only embedders.
    """
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.has_custom_search = False

    @abstractmethod
    def embed_text(self, texts):
        pass

# ----------------------------------------------------------------------
# 2) Multimodal embedders
# ----------------------------------------------------------------------
class ColPaliEmbedder(MultimodalEmbedder):
    def __init__(self, model_name="vidore/colpali-v1.3", device=None):
        super().__init__(device=device)
        self.model = ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.has_custom_search = True

    def embed_text(self, texts):
        query_inputs = self.processor.process_queries(texts).to(self.device)
        with torch.no_grad():
            query_emb = self.model(**query_inputs)
        return query_emb

    def embed_image(self, images):
        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)
        return image_emb

    def search(self, query, candidate_embeddings, top_k=5):
        query_embedding = self.embed_text([query])
        scores = self.processor.score_multi_vector(query_embedding, candidate_embeddings)
        
        # Get top-k scores and their indices
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)  # Ensure it's in the correct dimension

        return top_scores, top_indices

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
