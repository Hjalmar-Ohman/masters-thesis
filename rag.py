import os
import io
import base64
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from embedding import BaseEmbedder, CLIPEmbedder
from retrieval import BaseRetrieval, FaissRetrieval
from generator import BaseGenerator, GPTGenerator

###############################
# 4. BASE RAG CLASS
###############################

class BaseRAG:
    """
    Parent RAG class that stores content, builds a retrieval index, and answers queries.
    The embedder, retrieval backend, and generator are passed in (allowing easy swapping).
    """
    def __init__(self, embedder: BaseEmbedder, retrieval: BaseRetrieval, generator: BaseGenerator):
        self.embedder = embedder
        self.retrieval = retrieval
        self.generator = generator
        self.metadata = []    # List of dicts with metadata about each chunk
        self.embeddings = []  # List of embedding vectors (as numpy arrays)

    def add_text(self, text: str, extra_metadata: dict = None):
        extra_metadata = extra_metadata or {}
        embedding = self.embedder.embed_text(text)
        self.embeddings.append(embedding)
        meta = {"type": "text", "content": text}
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def add_image(self, image: Image.Image, extra_metadata: dict = None):
        """
        Abstract: implement in child classes.
        (Some children may embed the image directly; others might summarize first.)
        """
        raise NotImplementedError("Implement this method in the subclass")

    def build_index(self):
        """Builds the retrieval index from all stored embeddings."""
        embeddings_np = np.array(self.embeddings).astype("float32")
        self.retrieval.add(embeddings_np)

    def retrieve(self, query: str, top_k: int = 3):
        """Embeds the query and returns the top_k matching metadata items."""
        query_emb = self.embedder.embed_text(query)
        query_emb = np.expand_dims(query_emb, axis=0).astype("float32")
        distances, indices = self.retrieval.search(query_emb, top_k)
        retrieved = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["distance"] = float(dist)
                retrieved.append(item)
        return retrieved

    def generate_answer(self, query: str, retrieved_items: list):
        """
        Build a prompt that includes the query and the retrieved items, then use the generator.
        (Child classes may override this to change how items are incorporated.)
        """
        prompt = f"Answer the following question: {query}\n\n"
        for item in retrieved_items:
            if item["type"] == "text":
                prompt += f"Text snippet: {item['content']}\n"
            elif item["type"] == "image":
                prompt += f"Image (original): {item['content']}\n"
            elif item["type"] == "image_summary":
                # For image summaries, you might choose to include the summary
                prompt += f"Image summary: {item['content']}\n"
        return self.generator.generate(prompt)

    def answer_query(self, query: str, top_k: int = 3):
        retrieved_items = self.retrieve(query, top_k)
        return self.generate_answer(query, retrieved_items)


###############################
# 5. CHILD CLASSES
###############################

class MultiModalRAG(BaseRAG):
    """
    Child RAG class that uses a multimodal embedder to directly embed images.
    """
    def add_image(self, image: Image.Image, extra_metadata: dict = None):
        extra_metadata = extra_metadata or {}
        # Use the embedder’s image encoder (e.g. CLIP) to get an embedding.
        embedding = self.embedder.embed_image(image)
        self.embeddings.append(embedding)
        # Convert the PIL image to a base64 string (so that the original image can be
        # passed along to the generator if needed).
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        meta = {"type": "image", "content": img_base64}
        meta.update(extra_metadata)
        self.metadata.append(meta)


class SummaryRAG(BaseRAG):
    """
    Child RAG class that, for images, first calls an LLM to generate a summary,
    then embeds the summary text (using a text embedder). At generation time, if an
    image summary is the top match, the original image (in base64) is used.
    """
    def __init__(self, embedder: BaseEmbedder, retrieval: BaseRetrieval,
                 generator: BaseGenerator, image_summarizer: BaseGenerator):
        super().__init__(embedder, retrieval, generator)
        self.image_summarizer = image_summarizer  # used to generate image summaries

    def add_image(self, image: Image.Image, extra_metadata: dict = None):
        extra_metadata = extra_metadata or {}
        # Convert image to base64 (so we can include it in the prompt if needed).
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        # Create a prompt for the image summarizer LLM.
        prompt = f"Please provide a concise summary of the content of this image: data:image/png;base64,{img_base64}"
        summary = self.image_summarizer.generate(prompt)
        # Now embed the summary text.
        embedding = self.embedder.embed_text(summary)
        self.embeddings.append(embedding)
        # Store metadata including both the summary and the original image.
        meta = {
            "type": "image_summary",
            "content": summary,
            "original_image": img_base64
        }
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def generate_answer(self, query: str, retrieved_items: list):
        """
        If the best match is an image_summary, then include the original image.
        Otherwise, include the text summary.
        """
        prompt = f"Answer the following question: {query}\n\n"
        for i, item in enumerate(retrieved_items):
            if item["type"] == "text":
                prompt += f"Text snippet: {item['content']}\n"
            elif item["type"] == "image_summary":
                # For example, if this is the top match, include the original image data.
                if i == 0:
                    prompt += f"Image (original): {item['original_image']}\n"
                else:
                    prompt += f"Image summary: {item['content']}\n"
        return self.generator.generate(prompt)


###############################
# 6. EXAMPLE USAGE
###############################

if __name__ == "__main__":
    # Set up the models – these can be swapped easily.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_embedder = CLIPEmbedder(device=device)
    # (For text embedding in SummaryRAG you might use a dedicated text embedder; here we reuse CLIP's text encoder.)
    text_embedder = clip_embedder

    # Choose the retrieval backend. (Swap FaissRetrieval for, e.g., Milvus-based retrieval if needed.)
    embedding_dim = 512  # Adjust to match your embedder’s output dimension.
    faiss_retrieval_for_multimodal = FaissRetrieval(embedding_dim)
    faiss_retrieval_for_summary = FaissRetrieval(embedding_dim)

    # Set up the generator – here we use a stub GPT generator.
    # Replace "your-openai-key" with your actual API key when integrating a real generator.
    gpt_generator = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    # For summarizing images, we can use the same (or a different) generator.
    image_summarizer = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create two RAG pipelines: one for direct multimodal embeddings and one for summary-based images.
    multimodal_rag = MultiModalRAG(
        embedder=clip_embedder,
        retrieval=faiss_retrieval_for_multimodal,
        generator=gpt_generator
    )

    summary_rag = SummaryRAG(
        embedder=text_embedder,
        retrieval=faiss_retrieval_for_summary,
        generator=gpt_generator,
        image_summarizer=image_summarizer
    )

    # --- Add some sample text ---
    sample_text = "This is a sample text snippet about cats and their playful behavior."
    multimodal_rag.add_text(sample_text, extra_metadata={"page_number": 1})
    summary_rag.add_text(sample_text, extra_metadata={"page_number": 1})

    # --- Add an image ---
    # For demonstration, we create a dummy image.
    dummy_image = Image.new("RGB", (200, 200), color="blue")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(dummy_image)
    draw.text((50, 90), "Cat", fill="white")

    multimodal_rag.add_image(dummy_image, extra_metadata={"image_number": 1})
    summary_rag.add_image(dummy_image, extra_metadata={"image_number": 1})

    # --- Build retrieval indices ---
    multimodal_rag.build_index()
    summary_rag.build_index()

    # --- Query the pipelines ---
    user_query = "What do you know about cats?"
    print("=== MultiModalRAG Answer ===")
    answer1 = multimodal_rag.answer_query(user_query, top_k=2)
    print(answer1)

    print("\n=== SummaryRAG Answer ===")
    answer2 = summary_rag.answer_query(user_query, top_k=2)
    print(answer2)
