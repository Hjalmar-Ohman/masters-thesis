import io
import base64
import numpy as np

class BaseRAG:
    """
    Parent Retrieval-Augmented Generation (RAG) class that stores content,
    builds a retrieval index, and generates answers based on retrieved context.
    
    Attributes:
        embedder: An embedding model that supports text (and optionally image) embedding.
        retrieval: A retrieval backend (e.g., FaissRetrieval) for indexing and searching embeddings.
        generator: A language model generator used to produce final answers.
        metadata (list): A list of dictionaries storing metadata for each added content chunk.
        embeddings (list): A list of embedding vectors (NumPy arrays) corresponding to each content chunk.
    """
    def __init__(self, embedder, retrieval, generator):
        """
        Initialize the BaseRAG pipeline with the provided components.
        """
        self.embedder = embedder         # Multimodal embedder or text-only embedder (for summaries)
        self.retrieval = retrieval       # Retrieval backend (e.g., FaissRetrieval)
        self.generator = generator       # LLM to generate final answers
        self.metadata = []               # List of metadata dictionaries for each content chunk
        self.embeddings = []             # List of embedding vectors (as NumPy arrays)

    def add_text(self, text: str, extra_metadata: dict = None):
        """
        Add a text chunk to the RAG pipeline.
        
        This method embeds the text, stores its embedding, and saves associated metadata.
        
        Args:
            text (str): The text content to add.
            extra_metadata (dict, optional): Additional metadata (e.g., page number).
        """
        extra_metadata = extra_metadata or {}
        embedding = self.embedder.embed_text(text)
        self.embeddings.append(embedding)
        meta = {"type": "text", "content": text}
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def build_index(self):
        """
        Build the retrieval index using all stored embeddings.
        
        This method converts the list of embeddings into a NumPy array of type float32
        and adds them to the retrieval backend.
        """
        embeddings_np = np.array(self.embeddings).astype("float32")
        self.retrieval.add(embeddings_np)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve the top-k matching content for a given query.
        
        The method embeds the query text and uses the retrieval backend to find
        the most similar content chunks.
        
        Args:
            query (str): The query string.
            top_k (int, optional): The number of top results to retrieve (default is 3).
        
        Returns:
            list: A list of metadata dictionaries for the retrieved content, each including a 'distance' key.
        """
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
        Build a list of content blocks that includes the user's query
        plus each retrieved text/image snippet in a structured way.
        Then call self.generator.generate_blocks().
        """
        blocks = []

        # 1) Add the user query as a text block
        blocks.append({
            "type": "text",
            "text": f"Answer the following question: {query}"
        })

        # 2) Add each retrieved item as either a text block or an image block
        for item in retrieved_items:
            if item["type"] == "text":
                snippet_text = f"Text snippet (page {item.get('page_number', '?')}): {item['content']}"
                blocks.append({"type": "text", "text": snippet_text})

            elif item["type"] in ("image", "image_summary"):
                # For an image with base64 data
                # If you are calling Gemini, you can do:
                blocks.append({
                    "type": "image",        # or "type": "image_url" if your model wants that
                    "mime_type": "image/png",
                    "data": item["content"] # e.g. base64 of the original image
                })
                # If you also want to display the summary text, you can add another text block:
                if item["type"] == "image_summary":
                    blocks.append({"type": "text", "text": f"Summary: {item['content']}"})
        
        # 3) Pass blocks to the generator
        return self.generator.generate_blocks(blocks)


    def answer_query(self, query: str, top_k: int = 3):
        """
        Process a query end-to-end: retrieve relevant content and generate an answer.
        
        Args:
            query (str): The query string.
            top_k (int, optional): The number of top matching items to retrieve (default is 3).
        
        Returns:
            str: The final answer generated by the language model.
        """
        retrieved_items = self.retrieve(query, top_k)
        return self.generate_answer(query, retrieved_items)


class MultiModalRAG(BaseRAG):
    """
    RAG pipeline that handles multimodal content by embedding images directly.
    """
    def add_image(self, image, extra_metadata: dict = None):
        """
        Add an image to the pipeline by directly embedding it.
        
        The image is embedded using the embedder's image encoder and stored along with
        its base64 representation (for potential inclusion in prompts).
        
        Args:
            image: A PIL Image object to add.
            extra_metadata (dict, optional): Additional metadata (e.g., figure number).
        """
        extra_metadata = extra_metadata or {}
        # Directly embed the image.
        embedding = self.embedder.embed_image(image)
        self.embeddings.append(embedding)
        # Convert the image to base64 for potential inclusion in prompts.
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        meta = {"type": "image", "content": img_base64}
        meta.update(extra_metadata)
        self.metadata.append(meta)


class SummaryRAG(BaseRAG):
    """
    RAG pipeline that handles images by first generating a concise text summary
    (using an image summarizer) and then embedding that summary.
    """
    def __init__(self, embedder, retrieval, generator, image_summarizer):
        """
        Initialize the SummaryRAG pipeline.
        
        Args:
            embedder: An embedder instance (typically text-only) to embed summaries.
            retrieval: A retrieval backend (e.g., FaissRetrieval).
            generator: A generator to produce the final answer.
            image_summarizer: A generator instance (e.g., GPTGenerator) that produces a text summary from an image.
        """
        super().__init__(embedder, retrieval, generator)
        self.image_summarizer = image_summarizer

    def add_image(self, image, extra_metadata: dict = None):
        """
        Add an image to the pipeline by first summarizing it into text and then embedding the summary.
        
        The method converts the image to a base64 string, creates a prompt for the image summarizer,
        obtains a concise summary, and then embeds the summary text. Both the summary and the original image
        (in base64) are stored in the metadata.
        
        Args:
            image: A PIL Image object to add.
            extra_metadata (dict, optional): Additional metadata (e.g., figure number).
        """
        extra_metadata = extra_metadata or {}
        # Convert image to base64.
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Create a prompt for a concise summary of the image.
        summarizer_prompt = f"Provide a concise summary (in less than 70 tokens) of the following image:\n data:image/png;base64,{img_base64}"
        summary = self.image_summarizer.generate(summarizer_prompt)
        
        # Embed the generated summary text.
        embedding = self.embedder.embed_text(summary)
        self.embeddings.append(embedding)
        
        # Store both the summary and the original image.
        meta = {"type": "image_summary", "content": summary, "original_image": img_base64}
        meta.update(extra_metadata)
        self.metadata.append(meta)

    def generate_answer(self, query: str, retrieved_items: list):
        """
        Generate an answer from the retrieved content, optionally including the original image for the top result.
        
        For image summaries, the top retrieved item may have its original image included in the prompt.
        
        Args:
            query (str): The user query.
            retrieved_items (list): A list of retrieved metadata dictionaries.
        
        Returns:
            str: The generated answer from the language model.
        """
        prompt = f"Answer the following question: {query}\n\n"
        for i, item in enumerate(retrieved_items):
            if item["type"] == "text":
                prompt += f"Text snippet (page {item.get('page_number', '?')}): {item['content']}\n\n"
            elif item["type"] == "image_summary":
                if i == 0:
                    prompt += f"Image (original): {item['original_image']}\n\n"
                else:
                    prompt += f"Image summary: {item['content']}\n\n"
        return self.generator.generate(prompt)
