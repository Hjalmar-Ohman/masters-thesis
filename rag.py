import os
import io
import base64
import numpy as np
import torch
import faiss
from PIL import Image


class BaseRAG:
    """
    Parent RAG class that stores content, builds a retrieval index, and answers queries.
    """
    def __init__(self, embedder: BaseMultimodalEmbedder, retrieval: BaseRetrieval, generator: BaseGenerator):
        self.embedder = embedder
        self.retrieval = retrieval
        self.generator = generator
        self.metadata = []    # List of dicts with metadata for each chunk
        self.embeddings = []  # List of embedding vectors (as numpy arrays)

    def add_text(self, text: str, extra_metadata: dict = None):
        extra_metadata = extra_metadata or {}
        embedding = self.embedder.embed_text(text)
        self.embeddings.append(embedding)
        meta = {"type": "text", "content": text}
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def add_image(self, image: Image.Image, extra_metadata: dict = None):
        """
        For multimodal RAG, add the image by embedding it and also store its base64 encoding.
        """
        extra_metadata = extra_metadata or {}
        embedding = self.embedder.embed_image(image)
        self.embeddings.append(embedding)
        # Convert image to base64 so that it can be passed to the generator.
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        meta = {"type": "image", "content": img_base64}
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def build_index(self):
        """Builds the retrieval index from all stored embeddings."""
        embeddings_np = np.array(self.embeddings).astype("float32")
        self.retrieval.add(embeddings_np)

    def retrieve(self, query: str, top_k: int = 3):
        """Embed the query and return the top_k matching metadata items."""
        query_emb = self.embedder.embed_text(query)
        query_emb = np.expand_dims(query_emb, axis=0).astype("float32")
        distances, indices = self.retrieval.search(query_emb, top_k)
        retrieved = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["distance"] = float(dist)
                retrieved.append(item)
        return retrieved

    def generate_answer(self, query: str, retrieved_items: list):
        """
        Build a prompt that includes the query and the retrieved items,
        then generate an answer using the generator.
        """
        prompt = f"Answer the following question: {query}\n\n"
        for item in retrieved_items:
            if item["type"] == "text":
                prompt += f"Text snippet: {item['content']}\n\n"
            elif item["type"] == "image":
                # Include the image's base64 data (or a reference to it)
                prompt += f"Image (base64): {item['content']}\n\n"
        return self.generator.generate(prompt)

    def answer_query(self, query: str, top_k: int = 3):
        retrieved_items = self.retrieve(query, top_k)
        return self.generate_answer(query, retrieved_items)