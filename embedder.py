import torch
import torch.nn.functional as F

import numpy as np
from abc import ABC, abstractmethod

from colpali_engine.models import ColPali, ColPaliProcessor
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

    def embed_image(self, images, batch_size=2):  # Added batch_size parameter
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            image_inputs = self.processor.process_images(batch).to(self.device)
            with torch.no_grad():
                image_emb = self.model(**image_inputs)
            embeddings.append(image_emb)
        return torch.cat(embeddings, dim=0)  # Concatenate all batch embeddings
    
    def search(self, query, candidate_embeddings, top_k=5):
        query_embedding = self.embed_text([query])
        scores = self.processor.score_multi_vector(query_embedding, candidate_embeddings)
        
        # Get top-k scores and their indices
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        return top_scores, top_indices

# ----------------------------------------------------------------------
# 3) Text-only embedders
# ----------------------------------------------------------------------
class OpenAIEmbedder(TextEmbedder):
    """OpenAI text embedder."""
    def embed_text(self, texts):
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def search(self, query, candidate_embeddings, top_k=5):
        query_embedding = self.embed_text([query])


        # Convert to tensors
        query_embedding = torch.tensor(query_embedding)  # Shape: (1, d)
        candidate_embeddings = torch.tensor(candidate_embeddings)  # Shape: (N, d)

        # Ensure both tensors are on the same device
        device = candidate_embeddings.device  # Use the device of candidate_embeddings
        query_embedding = query_embedding.to(device)

        # Compute similarity using dot product (since OpenAI embeddings are normalized)
        scores = query_embedding @ candidate_embeddings.T  # Shape: (1, N)

        # Get top-k scores and their indices
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        return top_scores.tolist(), top_indices.tolist()
    
if __name__ == "__main__":
    for embedder in [OpenAIEmbedder(), ColPaliEmbedder()]:
        embeddings = embedder.embed_text(["Hello, world!"])

        text_data = ["Hello, world!", "Goodbye, world!", "Pizza Party :OOO", "7123909dsfasdv¤#2537#", "The quick brown fox jumps over the lazy dog."]
        candidate_metadata = [{"type": "text", "content": text, "page_number": idx} for idx, text in enumerate(text_data)]
        candidate_embeddings = embedder.embed_text(text_data)
        
        distances, indices = embedder.search("Hello, world!", candidate_embeddings)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            item = candidate_metadata[idx].copy()
            item["score"] = float(dist)
            results.append(item)
        print(results)