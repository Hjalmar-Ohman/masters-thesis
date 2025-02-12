import torch
import numpy as np
import faiss
from PyPDF2 import PdfReader
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from common_utils import embed_texts, embed_images, encode_image_to_base64, search_index, retrieve_context, call_gpt_4, extract_figures_from_pdf

class RAG:
    def __init__(self, openai_api_key):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_id)
        self.index = None
        self.all_metadata = []
        self.all_embeddings = []
        self.embedding_dimension = None
    
    def load_documents(self, documents):
        text_data = []
        image_data = []
        
        for doc in documents:
            if doc.endswith(".pdf"):
                reader = PdfReader(doc)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_data.append({"text": page_text.strip(), "page_number": i + 1})
                
                all_images = extract_figures_from_pdf(doc)
                for i, pil_img in enumerate(all_images):
                    image_data.append({"image": pil_img, "image_number": i + 1})
        
        texts_list = [td["text"] for td in text_data]
        if texts_list:
            text_embeddings = embed_texts(texts_list, self.clip_processor, self.clip_model)
            for i, emb in enumerate(text_embeddings):
                self.all_metadata.append({"type": "text", "content": text_data[i]["text"], "page_number": text_data[i]["page_number"]})
                self.all_embeddings.append(emb)
        
        pil_images_list = [id_["image"] for id_ in image_data]
        if pil_images_list:
            image_embeddings = embed_images(pil_images_list, self.clip_processor, self.clip_model)
            for i, emb in enumerate(image_embeddings):
                base64_str = encode_image_to_base64(image_data[i]["image"])
                self.all_metadata.append({"type": "image", "content": base64_str, "image_number": image_data[i]["image_number"]})
                self.all_embeddings.append(emb)
        
        self.all_embeddings = np.array(self.all_embeddings).astype("float32")
        self.embedding_dimension = self.all_embeddings.shape[1]
        self._build_faiss()
    
    def _build_faiss(self):
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(self.all_embeddings)
    
    def get_most_relevant_docs(self, user_query, top_k=3):
        query_emb = embed_texts([user_query], self.clip_processor, self.clip_model)
        distances, faiss_indices = search_index(self.index, query_emb, top_k=top_k)
        return retrieve_context(faiss_indices, self.all_metadata)
    
    def generate_answer(self, user_query, retrieved_docs):
        user_content = [{"type": "text", "text": f"User query: {user_query}"}]
        for doc in retrieved_docs:
            if doc["type"] == "text":
                user_content.append({"type": "text", "text": f"(page {doc['page_number']}) {doc['content'][:500]}..."})
            elif doc["type"] == "image":
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{doc['content']}"}})
        return call_gpt_4(user_content)
