"""
AssistIQ: Historical Case Retrieval Interface
Provides the high-level Python retrieval interface `retrieve_similar_cases` for querying
historical Spotify support cases using semantic search over FAISS.
"""

from typing import List, Dict, Any, Optional
from backend.src.retrieval.index import FAISSRetrievalIndex, get_default_artifacts_dir

_GLOBAL_INDEX: Optional[FAISSRetrievalIndex] = None


def get_retriever(artifacts_dir: Optional[str] = None) -> FAISSRetrievalIndex:
    """
    Returns a cached, singleton instance of FAISSRetrievalIndex.
    Loads the persisted index on first invocation.
    """
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        if artifacts_dir is None:
            artifacts_dir = get_default_artifacts_dir()
        _GLOBAL_INDEX = FAISSRetrievalIndex.load(artifacts_dir)
    return _GLOBAL_INDEX


def retrieve_similar_cases(
    query: str,
    top_k: int = 5,
    artifacts_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the Top-K historically similar Spotify support cases for a new customer query.

    Args:
        query (str): The incoming customer query text.
        top_k (int): Number of top cases to retrieve (default: 5).
        artifacts_dir (Optional[str]): Custom directory path to index artifacts if non-default.

    Returns:
        List[Dict[str, Any]]: A list of structured dictionaries sorted by similarity descending:
        [
            {
                "case_id": "case_00123",
                "customer_tweet_id": 123456,
                "customer_text": "I got charged two times",
                "support_tweet_id": "123457",
                "support_text": "Please send us a DM...",
                "similarity": 0.8245,
                "intent": None,
                "conversation_id": 123450,
                "rank": 1
            },
            ...
        ]
    """
    if not query or not str(query).strip():
        return []

    retriever = get_retriever(artifacts_dir)
    return retriever.search(query=query, top_k=top_k)
