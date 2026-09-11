"""
AssistIQ: Retrieval Package
Phase 2 Historical Support Retrieval / Knowledge Base.
"""

from backend.src.retrieval.data import (
    build_historical_corpus,
    load_support_cases
)
from backend.src.retrieval.embeddings import (
    SentenceEmbeddingModel,
    clean_for_embedding,
    EMBEDDING_DIM,
    DEFAULT_MODEL_NAME
)
from backend.src.retrieval.index import (
    FAISSRetrievalIndex,
    get_default_artifacts_dir
)
from backend.src.retrieval.retrieve import (
    retrieve_similar_cases,
    get_retriever
)

__all__ = [
    "build_historical_corpus",
    "load_support_cases",
    "SentenceEmbeddingModel",
    "clean_for_embedding",
    "EMBEDDING_DIM",
    "DEFAULT_MODEL_NAME",
    "FAISSRetrievalIndex",
    "get_default_artifacts_dir",
    "retrieve_similar_cases",
    "get_retriever"
]
