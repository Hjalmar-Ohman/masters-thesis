import os
import io
import base64
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


###############################
# 2. RETRIEVAL CLASSES
###############################

class BaseRetrieval:
    """Abstract retrieval interface."""
    def add(self, embeddings: np.ndarray):
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int):
        raise NotImplementedError


class FaissRetrieval(BaseRetrieval):
    """A simple FAISS-based retrieval backend using inner-product (IP) search."""
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int):
        # query_embedding should be shape (1, D)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]


# (You could define a MilvusRetrieval class with a similar interface if needed.)

