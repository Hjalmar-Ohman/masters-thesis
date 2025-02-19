import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Adjust as needed for your environment
# os.environ["OMP_NUM_THREADS"] = "1"        # For MAC users to avoid Faiss conflicts

import numpy as np
import faiss

from common_utils import (
    encode_image_to_base64,
    extract_images_from_pdf,
    extract_text_from_pdf,
    call_gpt_4,
)

from multimodal_embedder import BaseEmbedder, ClipEmbedder

class MultimodalRAG:
    def __init__(self, pdf_file: str, embedder: BaseEmbedder = ClipEmbedder):
        """
        1. Creates an embedder.
        2. Extracts text and images from the PDF.
        3. Embeds both text and images.
        4. Builds a Faiss index for retrieval.
        """
        self.pdf_file = pdf_file
        self.embedder = embedder

        # Extract text
        text_data = extract_text_from_pdf(self.pdf_file)
        self.texts_list = [td["text"] for td in text_data]

        # Extract images
        self.image_data = extract_images_from_pdf(self.pdf_file)
        self.pil_images_list = [img_info["pil_image"] for img_info in self.image_data]

        # Prepare containers for all metadata and embeddings
        all_embeddings = []
        self.all_metadata = []

        # --- 1) Embed text ---
        if len(self.texts_list) > 0:
            text_embeddings = self.embedder.embed_text(self.texts_list)
            for i, emb in enumerate(text_embeddings):
                self.all_metadata.append({
                    "type": "text",
                    "content": text_data[i]["text"],
                    "page_number": text_data[i]["page_number"]
                })
                all_embeddings.append(emb)

        # --- 2) Embed images ---
        base64_images_list = [encode_image_to_base64(pil_img) for pil_img in self.pil_images_list]
        if len(base64_images_list) > 0:
            image_embeddings = self.embedder.embed_image(self.pil_images_list)
            for i, emb in enumerate(image_embeddings):
                self.all_metadata.append({
                    "type": "image",
                    "content": base64_images_list[i],
                    "page_number": self.image_data[i]["page_number"]
                })
                all_embeddings.append(emb)

        # Convert embeddings to NumPy float32 for Faiss
        all_embeddings = np.array(all_embeddings).astype('float32')
        embedding_dimension = all_embeddings.shape[1]

        # Build a Faiss index
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.index.add(all_embeddings)

    def _search_index(self, query_embedding, top_k=5):
        """
        Search the Faiss index for the top_k nearest neighbors to query_embedding.
        Returns (distances, indices).
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def _retrieve_context(self, indices):
        """
        Given a list of indices from Faiss, return the corresponding metadata (text or image).
        """
        retrieved = []
        for idx in indices[0]:
            retrieved.append(self.all_metadata[idx])
        return retrieved

    def get_most_relevant_docs(self, query, top_k=5, debug = False):
        """
        1. Embed the user query (as text).
        2. Retrieve top_k similar items from the PDF (text or image).
        3. Return the list of relevant documents (metadata).
        """
        # Step 1: Embed user query
        query_emb = self.embedder.embed_text([query])  # shape: (1, D)

        # Step 2: Retrieve from Faiss
        distances, faiss_indices = self._search_index(query_emb, top_k=top_k)
        retrieved_items = self._retrieve_context(faiss_indices)

        if debug:
        # Inspect the top results with their distances:
            for distance, item in zip(distances[0], retrieved_items):
                 print(f"Distance: {distance:.4f}, Item: {item}")

        return retrieved_items

    def generate_answer(self, query, relevant_docs):
        """
        1. Build a "messages" or "context" list using the query and the retrieved docs.
        2. Call GPT-4 (or another LLM) with this context.
        3. Return the final model response.
        """
        # Construct the user/content payload
        user_content = []
        
        # Add user query
        user_content.append({"type": "text", "text": f"User query: {query}"})

        # Add relevant docs to the context
        for item in relevant_docs:
            if item["type"] == "text":
                snippet = item["content"][:500] + "..." if len(item["content"]) > 500 else item["content"]
                user_content.append({
                    "type": "text",
                    "text": f"(page {item['page_number']}) {snippet}"
                })
            elif item["type"] == "image":
                # Provide the base64 image data as a data URI
                base64_str = item["content"]
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_str}"
                    }
                })

        # Call GPT-4 (or any other LLM)
        gpt_response = "call_gpt_4(user_content)"

        return gpt_response