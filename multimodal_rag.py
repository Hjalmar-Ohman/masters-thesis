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
        all_results = []
        for processor in self.processors:
            results = processor.search(query, top_k=top_k)
            all_results.extend(results)

        sorted_results = sorted(all_results, key=lambda x: x["distance"], reverse=False)

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