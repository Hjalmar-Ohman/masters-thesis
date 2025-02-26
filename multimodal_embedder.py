import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod

from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,  # For SigLIP, BLIP, etc.
    FlavaProcessor, FlavaModel,
    BlipModel,
)
from colpali_engine.models import ColPali, ColPaliProcessor

from sentence_transformers import SentenceTransformer  # Salesforce, Snowflake
from openai import OpenAI

from config import OPENAI_API_KEY
from common_utils import generate_image_summary


# ----------------------------------------------------------------------
# 1) BaseEmbedder: shared interface for any embedder
# ----------------------------------------------------------------------
class BaseEmbedder(ABC):
    """Abstract base class for both multimodal and text-only embedders."""

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_text(self, texts):
        """Returns a 2D numpy array of text embeddings."""
        pass

    @abstractmethod
    def embed_image(self, images):
        """Returns a 2D numpy array of image embeddings."""
        pass
    
    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalizes a tensor along the last dimension."""
        return tensor / tensor.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------
# 2) BaseHFMultiModalEmbedder: common logic for HF models that have:
#    - get_text_features(...)
#    - get_image_features(...)
# ----------------------------------------------------------------------
class BaseHFMultiModalEmbedder(BaseEmbedder):
    """
    Base class for Hugging Face multimodal embedders that implement
    get_text_features(**inputs) and get_image_features(**inputs).
    """

    def __init__(self, model_id: str, processor_cls, model_cls, device=None):
        """
        :param model_id: Hugging Face model checkpoint or path
        :param processor_cls: the Processor class to instantiate (e.g. CLIPProcessor, AutoProcessor)
        :param model_cls: the Model class to instantiate (e.g. CLIPModel, BlipModel)
        """
        super().__init__(device=device)
        self.model_id = model_id
        # Load the model and processor
        self.model = model_cls.from_pretrained(model_id).to(self.device).eval()
        self.processor = processor_cls.from_pretrained(model_id)

    def embed_text(self, texts):
        """Common text embedding logic for HF models with get_text_features."""
        if not isinstance(texts, list):
            texts = [texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        text_emb = self._l2_normalize(text_emb).cpu().numpy()
        return text_emb

    def embed_image(self, images):
        """Common image embedding logic for HF models with get_image_features."""
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)

        image_emb = self._l2_normalize(image_emb).cpu().numpy()
        return image_emb


# ----------------------------------------------------------------------
# 3) BaseTextOnlyEmbedder: for models/APIs that only embed text
#    but must embed images by first generating a summary.
# ----------------------------------------------------------------------
class BaseTextOnlyEmbedder(BaseEmbedder, ABC):
    """Base class for embedders that only provide text embeddings natively."""

    def embed_image(self, images):
        """Converts images to text (via a summary) then calls self.embed_text."""
        if not isinstance(images, list):
            images = [images]
        summaries = [generate_image_summary(pil_img) for pil_img in images]
        return self.embed_text(summaries)


# ----------------------------------------------------------------------
# 4) Multimodal implementations
# ----------------------------------------------------------------------

class ClipEmbedder(BaseHFMultiModalEmbedder):
    """CLIP-based embedder."""
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        super().__init__(
            model_id=model_id,
            processor_cls=CLIPProcessor,
            model_cls=CLIPModel,
            device=device
        )


class SigLIPEmbedder(BaseHFMultiModalEmbedder):
    """SigLIP-based embedder. Uses custom text prompts."""

    def __init__(self, model_id="C:/huggingface_models/siglip/", device=None):
        # Note: SigLIP uses AutoProcessor & AutoModel
        super().__init__(
            model_id=model_id,
            processor_cls=AutoProcessor,
            model_cls=AutoModel,
            device=device
        )

    def embed_text(self, texts):
        """Override to apply SigLIP's recommended prompt template."""
        if not isinstance(texts, list):
            texts = [texts]

        texts = [f"This is a photo of {t}." for t in texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)

        return self._l2_normalize(embeddings).cpu().numpy()


class BlipEmbedder(BaseHFMultiModalEmbedder):
    """BLIP-based embedder."""

    def __init__(self, model_id="Salesforce/blip-image-captioning-base", device=None):
        # BLIP uses BlipModel + AutoProcessor
        super().__init__(
            model_id=model_id,
            processor_cls=AutoProcessor,
            model_cls=BlipModel,
            device=device
        )


class FlavaEmbedder(BaseEmbedder):
    """FLAVA-based embedder with aligned text and image embeddings."""

    def __init__(self, model_id="facebook/flava-full", device=None):
        super().__init__(device=device)
        self.model = FlavaModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = FlavaProcessor.from_pretrained(model_id)

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        # Force multimodal forward pass with a dummy image
        dummy_image = Image.new("RGB", (224, 224))
        images = [dummy_image] * len(texts)
        return self._extract_embeddings(texts, images)

    def embed_image(self, images):
        if not isinstance(images, list):
            images = [images]
        # Force multimodal forward pass with dummy text
        dummy_texts = [""] * len(images)
        return self._extract_embeddings(dummy_texts, images)

    def _extract_embeddings(self, texts, images):
        """
        Helper function to extract embeddings using FLAVA's multimodal forward pass.
        We then take the [CLS] token as the image/text representation.
        """
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            mm_embeds = outputs.multimodal_embeddings  # [Batch, Seq_len, Hidden_size]

        # Typically you take the CLS token as the final embedding
        cls_embeds = mm_embeds[:, 0, :]
        cls_embeds = self._l2_normalize(cls_embeds)
        return cls_embeds.cpu().numpy()

class ColPaliEmbedder(BaseEmbedder):
    """
    A multimodal embedder that uses ColPali for both text and image embeddings.
    """

    def __init__(self, model_name="vidore/colpali-v1.3", device=None):
        """
        :param model_name: The ColPali model checkpoint on HF Hub, e.g., "vidore/colpali-v1.3"
        :param device: If None, defaults to "cuda" if available, else "cpu"
        """
        super().__init__(device=device)
        # You can optionally pass device_map="auto" or device_map="cuda:0" etc.
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # or "cuda:0" or "mps"
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def embed_text(self, texts):
        """
        Embeds one or more text queries using ColPali.

        :param texts: str or list of str
        :return: 2D numpy array of shape (batch_size, embedding_dim)
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Prepare ColPali query inputs
        query_inputs = self.processor.process_queries(texts).to(self.device)

        with torch.no_grad():
            # Forward pass -> shape [batch_size, hidden_dim]
            query_emb = self.model(**query_inputs)

        # L2-normalize for consistency with other embedders
        query_emb = self._l2_normalize(query_emb)
        return query_emb.cpu().numpy()

    def embed_image(self, images):
        """
        Embeds one or more images using ColPali.

        :param images: PIL Image or list of PIL Images
        :return: 2D numpy array of shape (batch_size, embedding_dim)
        """
        if not isinstance(images, list):
            images = [images]

        # Process images using the ColPali processor
        image_inputs = self.processor.process_images(images).to(self.device)

        with torch.no_grad():
            # Forward pass -> shape [batch_size, hidden_dim]
            image_emb = self.model(**image_inputs)

        # L2-normalize for consistency with other embedders
        image_emb = self._l2_normalize(image_emb)
        return image_emb.cpu().numpy()
# ----------------------------------------------------------------------
# 5) Text-only implementations (use BaseTextOnlyEmbedder)
# ----------------------------------------------------------------------

class OpenAIEmbedder(BaseTextOnlyEmbedder):
    """OpenAI text embedder. For images, uses generate_image_summary first."""

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings


class SFREmbedder(BaseTextOnlyEmbedder):
    """Salesforce SFR-Embedding text embedder."""

    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral", device=None):
        super().__init__(device=device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)


class SnowflakeEmbedder(BaseTextOnlyEmbedder):
    """Snowflake embedder with text prompting."""

    def __init__(self, model_name='Snowflake/snowflake-arctic-embed-l-v2.0', device=None):
        super().__init__(device=device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        # The prompt_name="query" argument is presumably Snowflake-specific.
        return self.model.encode(texts, normalize_embeddings=True, prompt_name="query")


# ----------------------------------------------------------------------
# 6) Factory function
# ----------------------------------------------------------------------
def create_embedder(model_name="CLIP", device=None):
    """
    Factory function to create an embedder based on model_name.
    """
    model_name = model_name.upper()
    embedders = {
        "CLIP": ClipEmbedder,
        "FLAVA": FlavaEmbedder,
        "SIGLIP": SigLIPEmbedder,
        "BLIP": BlipEmbedder,
        "OPENAI": OpenAIEmbedder,
        "SFR": SFREmbedder,
        "SNOWFLAKE": SnowflakeEmbedder,
        "COLPALI": ColPaliEmbedder,
    }

    if model_name not in embedders:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(embedders.keys())}"
        )

    return embedders[model_name](device=device)