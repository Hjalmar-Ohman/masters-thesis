import abc
import faiss

from typing import List, Dict, Any
from pdf2image import convert_from_path

from pdf_utils import extract_text_from_pdf, extract_images_from_pdf
from common_utils import encode_image_to_base64, call_gpt_4
from embedder import MultimodalEmbedder, TextEmbedder


class DocumentProcessor(abc.ABC):
    def __init__(self, embedder):
        self.embedder = embedder
        self.faiss_index = None
        self.embeddings = None
        self.metadata: List[Dict[str, Any]] = []

    @abc.abstractmethod
    def process_pdf(self, pdf_file: str) -> None:
        pass

    def build_faiss_index(self):
        embedding_dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(self.embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embedder.has_custom_search:
            return self.embedder.search(query, self.embeddings, self.metadata, top_k)
        else:
            # Use FAISS
            query_emb = self.embedder.embed_text([query]).astype("float32")  # shape: (1, d)
            distances, indices = self.faiss_index.search(query_emb, top_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                item = self.metadata[idx].copy()
                item["distance"] = float(dist)
                results.append(item)
            return results
        
class TextProcessor(DocumentProcessor):
    def __init__(self, embedder: TextEmbedder):
        super().__init__(embedder)

    def process_pdf(self, pdf_file: str):
        text_data = extract_text_from_pdf(pdf_file)
        texts_list = [td["text"] for td in text_data]
        
        self.metadata = [{"type": "text", "content": td["text"], "page_number": td["page_number"]} for td in text_data]
        self.embeddings = self.embedder.embed_text(texts_list).astype("float32")
        if not self.embedder.has_custom_search:
            self.build_faiss_index()

class ImageProcessor(DocumentProcessor):
    def __init__(self, embedder: MultimodalEmbedder):
        super().__init__(embedder)

    def process_pdf(self, pdf_file: str):
        image_data = extract_images_from_pdf(pdf_file)
        pil_images_list = [info["pil_image"] for info in image_data]
        
        self.metadata = [{"type": "image", "content": encode_image_to_base64(info["pil_image"]), "page_number": info["page_number"]} for info in image_data]
        self.embeddings = self.embedder.embed_image(pil_images_list).astype("float32")
        if not self.embedder.has_custom_search:
            self.build_faiss_index()

class PageImageProcessor(DocumentProcessor):
    def __init__(self, embedder: MultimodalEmbedder, dpi=200):
        super().__init__(embedder)
        self.dpi = dpi

    def process_pdf(self, pdf_file: str):
        pages = convert_from_path(pdf_file, dpi=self.dpi, poppler_path=r'poppler-24.08.0/Library/bin')

        self.metadata = [{"type": "page_image", "content": encode_image_to_base64(page_img), "page_number": i + 1} for i, page_img in enumerate(pages)]
        self.embeddings = self.embedder.embed_image(pages).astype("float32")
        if not self.embedder.has_custom_search:
            self.build_faiss_index()

class TextAndInlineImageProcessor(DocumentProcessor):
    def __init__(self, embedder: TextEmbedder):
        super().__init__(embedder)

    def process_pdf(self, pdf_file: str):
        text_data = extract_text_from_pdf(pdf_file)
        texts_list = [td["text"] for td in text_data]
        image_data = extract_images_from_pdf(pdf_file)
        pil_images_list = [img_info["pil_image"] for img_info in image_data]
        base64_images_list = [encode_image_to_base64(pil_img) for pil_img in pil_images_list]

        image_summaries = []
        for b64_str in base64_images_list:
            prompt_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                {"type": "text", "text": "Summarize the above image in plain text."}
            ]
            summary = call_gpt_4(prompt_content)
            image_summaries.append(summary)

        all_texts = texts_list + image_summaries
        self.embeddings = self.embedder.embed_text(all_texts).astype("float32")

        self.metadata = []
        for td in text_data:
            self.metadata.append({"type": "text", "content": td["text"], "page_number": td["page_number"]})
        for i, img_info in enumerate(image_data):
            self.metadata.append({"type": "image", "content": base64_images_list[i], "page_number": img_info["page_number"]})

        if not self.embedder.has_custom_search:
            self.build_faiss_index()