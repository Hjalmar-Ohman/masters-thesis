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
    """
    ColPali multimodal embedder.
    """
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

    def search(
        self,
        query: str,
        candidate_embeddings: np.ndarray,   # shape: (N, d)
        candidate_metadata: list,
        top_k: int = 5
    ) -> list:
        """
        Use ColPali's score_multi_vector for single-vector doc embeddings.
        """
        # 1) Get query embedding from text
        q = self.embed_text([query])  # shape: (1, d)
        
        # 2) Convert to torch and reshape
        q_torch = torch.from_numpy(q)  # shape (1, d)
        qs_tensor = q_torch.unsqueeze(0)  # shape (1, 1, d) if you want to pass a single 3D Tensor

        docs_torch = torch.from_numpy(candidate_embeddings)  # shape (N, d)
        ps_tensor = docs_torch.unsqueeze(1)                  # shape (N, 1, d)
        
        # 3) Score
        scores = self.processor.score_multi_vector(qs_tensor, ps_tensor)  # shape (1, N)
        scores = scores[0]  # shape (N,)

        # Transform scores to match FAISS logic (lower is better)
        scores = -scores  # Option 1: Make scores negative
        # scores = 1 - scores  # Option 2: Normalize (TODO: check if scores range [0,1])

        # 4) Sort, pick top_k, build output
        top_indices = scores.argsort(descending=True)[:top_k]
        results = []
        for idx in top_indices:
            item = candidate_metadata[idx].copy()
            item["distance"] = float(scores[idx].item())
            results.append(item)
        return results

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

if __name__ == "__main__":
    pass