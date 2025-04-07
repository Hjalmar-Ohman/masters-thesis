import os
import json
import torch
from document_processor import DocumentProcessor
from typing import List, Dict, Any
from common_utils import call_gpt_4

class MultimodalRAG:
    """
    Orchestrates multiple document processors on the same PDF.
    Adds caching of embeddings to avoid redundant processing.
    """

    CACHE_DIR = "embedding_cache"  # Directory to store cached embeddings

    def __init__(self, processors: List[DocumentProcessor], pdf_file: str):
        self.processors = processors
        self.pdf_file = pdf_file
        self.name = processors[0].name if len(processors) == 1 else "dual_storage"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        if not self.load_cached_embeddings():
            for processor in self.processors:
                processor.process_pdf(pdf_file)
            self.save_embeddings()

    def get_cache_paths(self, processor_name: str) -> Dict[str, str]:
        """Returns file paths for storing metadata and embeddings for a specific processor."""
        base_name = os.path.splitext(os.path.basename(self.pdf_file))[0]
        return {
            "metadata": os.path.join(self.CACHE_DIR, f"{base_name}_{self.name}_metadata.json"),
            "embeddings": os.path.join(self.CACHE_DIR, f"{base_name}_{self.name}_embeddings.pt"),
        }

    def save_embeddings(self):
        """Saves the embeddings and metadata to disk."""
        for processor in self.processors:
            paths = self.get_cache_paths(self.name)
            with open(paths["metadata"], "w") as f:
                json.dump(processor.metadata, f)
            torch.save(processor.embeddings, paths["embeddings"])

    def load_cached_embeddings(self) -> bool:
        """Loads embeddings if they exist. Returns True if successful."""
        all_loaded = True
        for processor in self.processors:
            paths = self.get_cache_paths(self.name)
            if os.path.exists(paths["metadata"]) and os.path.exists(paths["embeddings"]):
                with open(paths["metadata"], "r") as f:
                    processor.metadata = json.load(f)
                processor.embeddings = torch.load(paths["embeddings"])
            else:
                all_loaded = False
        return all_loaded

    def get_most_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        all_results = []
        for processor in self.processors:
            results = processor.search(query, top_k=top_k // len(self.processors))
            all_results.extend(results)

        sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    def generate_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        user_content = [{"type": "text", "text": f"User query: {query}"}]
        
        for doc in relevant_docs:
            if doc["type"] == "text":
                snippet = doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"]
                user_content.append({
                    "type": "text",
                    "text": f"(page {doc['page_number']}) {snippet}"
                })
            elif doc["type"] in ["image", "page_image"]:
                base64_str = doc["content"]
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_str}"}
                })
            else:
                raise ValueError(f"Unknown doc type: {doc['type']}")
        
        return call_gpt_4(user_content)