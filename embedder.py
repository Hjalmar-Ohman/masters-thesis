import torch
import numpy as np
from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModel, AutoTokenizer
from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer
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

    @abstractmethod
    def embed_text(self, texts):
        pass

def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)

# ----------------------------------------------------------------------
# 2) Multimodal embedders
# ----------------------------------------------------------------------
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
        return _l2_normalize(query_emb).cpu().numpy()

    def embed_image(self, images):
        image_inputs = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_emb = self.model(**image_inputs)
        return _l2_normalize(image_emb).cpu().numpy()

class VisRAGEmbedder(MultimodalEmbedder):
    """VisRAG multimodal embedder."""
    def __init__(self, model_name="openbmb/VisRAG-Ret", device=None):
        super().__init__(device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device).eval()

    def _weighted_mean_pooling(self, hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        return s / d

    def encode(self, text_or_image_list):
        if isinstance(text_or_image_list[0], str):
            inputs = {
                "text": text_or_image_list,
                "image": [None] * len(text_or_image_list),
                "tokenizer": self.tokenizer
            }
        else:
            inputs = {
                "text": [''] * len(text_or_image_list),
                "image": text_or_image_list,
                "tokenizer": self.tokenizer
            }
        
        outputs = self.model(**inputs)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state
        reps = self._weighted_mean_pooling(hidden, attention_mask)
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings

    def embed_text(self, texts):
        return self.encode(["Represent this query for retrieving relevant documents: " + t for t in texts])

    def embed_image(self, images):
        return self.encode(images)

class SigLIPEmbedder(MultimodalEmbedder):
    """
    SigLIP multimodal embedder.
    """
    def __init__(self, model_id="google/siglip-base-patch16-224", device=None):
        super().__init__(device=device)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    def embed_text(self, texts):
        texts = [f"This is a photo of {t}." for t in texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
        return _l2_normalize(text_emb).cpu().numpy()

    def embed_image(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
        return _l2_normalize(image_emb).cpu().numpy()

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
