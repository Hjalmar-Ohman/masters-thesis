import os
import torch

from rag import MultiModalRAG, SummaryRAG
from embedding import CLIPEmbedder
from generator import GPTGenerator
from retrieval import FaissRetrieval
from pre_process import extract_text_from_pdf, extract_figures_from_pdf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Adjust this as needed for your environment.

if __name__ == "__main__":
    # Set your PDF file path
    PDF_FILE = "knowledge/catsanddogs.pdf"

    # Set up models and components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_embedder = CLIPEmbedder(device=device)

    # Create separate retrieval backends for each pipeline (if needed)
    retrieval_backend_multimodal = FaissRetrieval(embedding_dim=512)  # adjust dimension as needed
    retrieval_backend_summary = FaissRetrieval(embedding_dim=512)

    # Instantiate generators.
    # The main generator is used to produce final answers.
    generator = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    # For SummaryRAG, we also need an image summarizer.
    image_summarizer = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))  

    # Create the two RAG pipelines:
    # 1. Multimodal RAG: embeds images directly.
    multimodal_rag = MultiModalRAG(
        embedder=clip_embedder,
        retrieval=retrieval_backend_multimodal,
        generator=generator
    )
    # 2. Summary RAG: generates a text summary from images before embedding.
    summary_rag = SummaryRAG(
        embedder=clip_embedder,  # using the text embedding part of CLIP, for example
        retrieval=retrieval_backend_summary,
        generator=generator,
        image_summarizer=image_summarizer
    )

    # --- Process the PDF: Extract text from each page and add it to both pipelines ---
    texts = extract_text_from_pdf(PDF_FILE)
    for item in texts:
        multimodal_rag.add_text(item["text"], extra_metadata={"page_number": item["page_number"]})
        summary_rag.add_text(item["text"], extra_metadata={"page_number": item["page_number"]})
    print(f"Extracted text from {len(texts)} pages.")

    # --- Process the PDF: Extract figures (individual images) and add them to both pipelines ---
    figures = extract_figures_from_pdf(PDF_FILE)
    for i, figure in enumerate(figures):
        multimodal_rag.add_image(figure, extra_metadata={"figure_number": i + 1})
        summary_rag.add_image(figure, extra_metadata={"figure_number": i + 1})
    print(f"Extracted {len(figures)} figures from the PDF.")

    # --- Build the retrieval indices ---
    multimodal_rag.build_index()
    summary_rag.build_index()

    # --- Query the RAG systems ---
    user_query = "What does the document say about recent advances in machine learning?"

    print("\n=== MultiModal RAG Answer ===")
    answer_mm = multimodal_rag.answer_query(user_query, top_k=3)
    print(answer_mm)

    print("\n=== Summary RAG Answer ===")
    answer_summary = summary_rag.answer_query(user_query, top_k=3)
    print(answer_summary)
