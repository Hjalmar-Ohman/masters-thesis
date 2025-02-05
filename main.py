import os
import torch

from rag import MultiModalRAG, SummaryRAG
from embedding import CLIPEmbedder
from generator import GPTGenerator
from retrieval import FaissRetrieval
from pre_process import extract_text_from_pdf, extract_figures_from_pdf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Adjust this as needed for your environment.

if __name__ == "__main__":
    # -- 1. PDF file path --
    PDF_FILE = "knowledge/catsanddogs.pdf"

    # -- 2. Set up models/components --
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_embedder = CLIPEmbedder(device=device)

    # Create separate retrieval backends if desired
    retrieval_backend_multimodal = FaissRetrieval(embedding_dim=512)
    retrieval_backend_summary = FaissRetrieval(embedding_dim=512)

    # Instantiate generator(s)
    main_generator = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    image_summarizer = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))  # used by SummaryRAG

    # -- 3. Create RAG pipelines --
    multimodal_rag = MultiModalRAG(
        embedder=clip_embedder,
        retrieval=retrieval_backend_multimodal,
        generator=main_generator
    )
    summary_rag = SummaryRAG(
        embedder=clip_embedder,    # using CLIP's text embedding for the summaries
        retrieval=retrieval_backend_summary,
        generator=main_generator,
        image_summarizer=image_summarizer
    )

    # -- 4. Extract and add text (already chunked in pre_process) --
    # Here we specify chunk_size and overlap for the text extraction
    texts = extract_text_from_pdf(pdf_path=PDF_FILE, chunk_size=50, overlap=5)
    print(f"PDF text returned {len(texts)} total chunks.")

    for chunk_data in texts:
        text_chunk = chunk_data["text"]
        page_num = chunk_data["page_number"]
        chunk_idx = chunk_data["chunk_index"]

        # Add to each RAG pipeline, preserving metadata
        multimodal_rag.add_text(
            text_chunk,
            extra_metadata={
                "page_number": page_num,
                "chunk_index": chunk_idx
            }
        )
        summary_rag.add_text(
            text_chunk,
            extra_metadata={
                "page_number": page_num,
                "chunk_index": chunk_idx
            }
        )

    # -- 5. Extract figures/images from the PDF --
    figures = extract_figures_from_pdf(PDF_FILE)
    print(f"Extracted {len(figures)} figures from the PDF.")

    for i, figure in enumerate(figures):
        multimodal_rag.add_image(figure, extra_metadata={"figure_number": i + 1})
        summary_rag.add_image(figure, extra_metadata={"figure_number": i + 1})

    # -- 6. Build the retrieval indices --
    multimodal_rag.build_index()
    summary_rag.build_index()

    # -- 7. Query the RAG systems --
    user_query = "What does the document say about recent advances in machine learning?"

    print("\n=== MultiModal RAG Answer ===")
    answer_mm = multimodal_rag.answer_query(user_query, top_k=3)
    print(answer_mm)

    print("\n=== Summary RAG Answer ===")
    answer_summary = summary_rag.answer_query(user_query, top_k=3)
    print(answer_summary)
