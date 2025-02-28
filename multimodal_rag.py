from document_processor import DocumentProcessor
from typing import List, Dict, Any
from common_utils import call_gpt_4

class MultimodalRAG:
    """
    Orchestrates multiple document processors on the same PDF.
    Each processor has its own FAISS index. We'll query them all.
    """

    def __init__(self, processors: List[DocumentProcessor], pdf_file: str):
        self.processors = processors
        self.pdf_file = pdf_file

        # Process the PDF in every document processor (builds the FAISS index and metadata)
        for processor in self.processors:
            processor.process_pdf(pdf_file)

    def get_most_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        1. For each document processor, use its embedder and FAISS index to search for the most relevant documents.
        2. Combine results from all processors.
        3. Sort the results by distance (score) in descending order.
        4. Return only the top_k overall.
        """
        all_results = []
        # Query each processor’s FAISS index directly
        for processor in self.processors:
            # Embed the query using the processor's embedder
            query_emb = processor.embedder.embed_text([query]).astype("float32")
            distances, indices = processor.index.search(query_emb, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                # Copy the metadata and add the distance score
                item = processor.metadata[idx].copy()
                item["distance"] = float(dist)
                all_results.append(item)
        # Sort the aggregated results and return only the top_k overall
        sorted_results = sorted(all_results, key=lambda x: x["distance"], reverse=True)
        return sorted_results[:top_k]

    def generate_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Build a prompt with the user query + relevant docs, then call GPT.
        Return the GPT response as a string.
        """
        user_content = []
        user_content.append({"type": "text", "text": f"User query: {query}"})

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

        # Here you would call GPT-4 with the constructed prompt.
        gpt_response = call_gpt_4(user_content)
        return gpt_response