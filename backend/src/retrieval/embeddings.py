"""
AssistIQ: Dense Semantic Embedding Module
Provides lightweight text normalization and generates dense vector embeddings
using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors).
"""

import re
import html
from typing import List, Union, Optional
import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def clean_for_embedding(text: Optional[str]) -> str:
    """
    Lightweight text normalization for semantic embedding generation.
    Preserves all natural linguistic meaning, intent cues, and terminology.
    - Resolves nulls / empty inputs
    - Unescapes HTML entities (e.g., &amp; -> &)
    - Removes Twitter @handles (@SpotifyCares, @115888) so vector search focuses on
      the customer's actual issue rather than matching arbitrary numerical IDs
    - Normalizes URLs (http/https links)
    - Normalizes multiple spaces/newlines into a clean single-space string
    """
    if text is None:
        return ""

    s = str(text)
    # Unescape HTML entities
    s = html.unescape(s)

    # Normalize URLs
    s = re.sub(r"https?://\S+", "", s)

    # Remove Twitter @mentions (e.g. @SpotifyCares, @115888, @user)
    s = re.sub(r"@\w+", "", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


class SentenceEmbeddingModel:
    """
    Thread-safe wrapper around SentenceTransformer with lazy model initialization,
    batched CPU inference, and L2 normalization for exact cosine similarity.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedding_dim = EMBEDDING_DIM
        self._model = None

    @property
    def model(self):
        """
        Lazily loads the SentenceTransformer model on first usage.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encodes a list of texts (or a single text) into dense normalized float32 vectors.
        L2 normalization ensures dot product corresponds directly to cosine similarity.
        """
        if isinstance(texts, str):
            cleaned = clean_for_embedding(texts)
            embeddings = self.model.encode(
                [cleaned],
                batch_size=1,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)[0]

        cleaned_texts = [clean_for_embedding(t) for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Convenience helper to encode a single search query into a 1D float32 vector.
        """
        return self.encode(query, normalize_embeddings=normalize_embeddings)
