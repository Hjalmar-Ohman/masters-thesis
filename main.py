if __name__ == "__main__":
    # Set paths and parameters
    pdf_file = "path/to/your/document.pdf"  # Update with your PDF file path

    # Set up the models (swap these out as needed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_embedder = CLIPEmbedder(device=device)
    # Assume the CLIP model outputs 512-d embeddings (adjust if needed)
    retrieval_backend = FaissRetrieval(embedding_dim=512)
    generator = GPTGenerator(api_key="your-openai-api-key")

    # Create the RAG pipeline
    rag_pipeline = BaseRAG(
        embedder=clip_embedder,
        retrieval=retrieval_backend,
        generator=generator
    )

    # Process the PDF (adding both text and image data)
    process_pdf(pdf_file, rag_pipeline)

    # Build the retrieval index
    rag_pipeline.build_index()

    # Query the RAG system
    user_query = "What information does the document provide about recent research trends?"
    answer = rag_pipeline.answer_query(user_query, top_k=3)
    print("\n=== Answer ===")
    print(answer)