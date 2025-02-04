import numpy as np
import faiss
from abc import ABC, abstractmethod

class BaseRetrieval(ABC):
    """Abstract retrieval interface."""
    @abstractmethod
    def add(self, embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int):
        pass


class FaissRetrieval(BaseRetrieval):
    """A FAISS-based retrieval backend using inner-product (IP) search."""
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int):
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]