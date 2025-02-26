import os
import numpy as np
import faiss
from PIL import Image
from pdf2image import convert_from_path

# Local imports in your project
from common_utils import (
    encode_image_to_base64,
    extract_images_from_pdf,
    extract_text_from_pdf,
    call_gpt_4,
)
from multimodal_embedder import BaseEmbedder, ClipEmbedder


class MultimodalRAG:
    def __init__(
        self,
        pdf_file: str,
        embedder: BaseEmbedder = ClipEmbedder,
        page_mode: str = "text_and_images",
        dpi: int = 200,
    ):
        """
        :param pdf_file: Path to the PDF file
        :param embedder: Any embedder that implements BaseEmbedder
        :param page_mode: One of:
            - "text_and_images": regular approach (extract + embed text chunks + inline images)
            - "image_only": convert entire pages to images, embed them only
        :param dpi: Dots per inch for pdf2image
        """

        self.pdf_file = pdf_file
        self.embedder = embedder
        self.page_mode = page_mode.lower()
        self.dpi = dpi

        # Prepare containers for all embeddings and metadata
        self.all_embeddings = []
        self.all_metadata = []

        if self.page_mode == "text_and_images":
            self._process_text_and_inline_images()
        elif self.page_mode == "image_only":
            self._process_pdf_as_pages()
        else:
            raise ValueError(f"Unsupported page_mode={self.page_mode!r}")

        # Build Faiss index
        self._build_faiss_index()

    def _process_text_and_inline_images(self):
        """
          1) Extract text chunks
          2) Extract inline images
          3) Embed both sets of data
        """
        # --- 1) Extract text data ---
        text_data = extract_text_from_pdf(self.pdf_file)
        self.texts_list = [td["text"] for td in text_data]

        # --- 2) Extract inline images ---
        self.image_data = extract_images_from_pdf(self.pdf_file)
        self.pil_images_list = [img_info["pil_image"] for img_info in self.image_data]

        # --- 3) Embed text ---
        if len(self.texts_list) > 0:
            text_embeddings = self.embedder.embed_text(self.texts_list)
            for i, emb in enumerate(text_embeddings):
                self.all_metadata.append({
                    "type": "text",
                    "content": text_data[i]["text"],
                    "page_number": text_data[i]["page_number"]
                })
                self.all_embeddings.append(emb)

        # --- 4) Embed inline images ---
        if len(self.pil_images_list) > 0:
            base64_images_list = [encode_image_to_base64(pil_img) for pil_img in self.pil_images_list]
            image_embeddings = self.embedder.embed_image(self.pil_images_list)
            for i, emb in enumerate(image_embeddings):
                self.all_metadata.append({
                    "type": "image",
                    "content": base64_images_list[i],
                    "page_number": self.image_data[i]["page_number"]
                })
                self.all_embeddings.append(emb)

    def _process_pdf_as_pages(self):
        """
          1) Use pdf2image to convert each entire page to a PIL Image
          2) Embed those page images
          3) Store as metadata in self.all_metadata
        """
        # Convert entire pages to images
        pages = convert_from_path(self.pdf_file, dpi=self.dpi, poppler_path=r'poppler-24.08.0/Library/bin')

        # Embed each page image
        page_embeddings = self.embedder.embed_image(pages)

        for i, emb in enumerate(page_embeddings):
            # Base64-encode if you want to store the image in metadata
            base64_str = encode_image_to_base64(pages[i])

            self.all_metadata.append({
                "type": "page_image",
                "content": base64_str,
                "page_number": i + 1,  # 1-based indexing
            })
            self.all_embeddings.append(emb)

    def _build_faiss_index(self):
        """
        Builds the in-memory Faiss index from self.all_embeddings
        """
        # Convert embeddings to NumPy float32 for Faiss
        all_embeddings = np.array(self.all_embeddings).astype('float32')
        embedding_dimension = all_embeddings.shape[1]

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

    def get_most_relevant_docs(self, query, top_k=5, debug=False):
        """
        1. Embed the user query (as text).
        2. Retrieve top_k similar items from the PDF (text or images).
        3. Return the list of relevant documents (metadata).
        """
        query_emb = self.embedder.embed_text([query])  # shape: (1, D)
        distances, faiss_indices = self._search_index(query_emb, top_k=top_k)
        retrieved_items = self._retrieve_context(faiss_indices)

        if debug:
            for distance, item in zip(distances[0], retrieved_items):
                print(f"Distance: {distance:.4f}, Item: {item}")

        return retrieved_items

    def generate_answer(self, query, relevant_docs):
        """
        1. Build a "messages" or "context" list using the query and the retrieved docs.
        2. Call GPT-4 (or any other LLM) with this context.
        3. Return the final model response.
        """
        user_content = []
        user_content.append({"type": "text", "text": f"User query: {query}"})

        for item in relevant_docs:
            if item["type"] == "text":
                snippet = item["content"][:500] + "..." if len(item["content"]) > 500 else item["content"]
                user_content.append({
                    "type": "text",
                    "text": f"(page {item['page_number']}) {snippet[:500]}..."
                })
            elif item["type"] in ("image", "page_image"):
                base64_str = item["content"]
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_str}"}
                })

        # Example GPT-4 call (replace with your actual call):
        gpt_response = "call_gpt_4(user_content)"
        return gpt_response