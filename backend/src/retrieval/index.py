"""
AssistIQ: FAISS Vector Index Module
Builds, serializes, loads, and queries a dense FAISS vector index (IndexFlatIP)
over normalized historical customer tweet embeddings.
"""

import os
import sys
import time
import json
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import faiss

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.src.retrieval.data import find_project_root, find_dataset_path, load_support_cases
from backend.src.retrieval.embeddings import SentenceEmbeddingModel, EMBEDDING_DIM, DEFAULT_MODEL_NAME


def get_default_artifacts_dir() -> str:
    """
    Returns the project-relative artifacts directory for the FAISS index and metadata.
    """
    root = find_project_root()
    return os.path.join(root, "backend", "src", "retrieval", "artifacts")


class FAISSRetrievalIndex:
    """
    FAISS Index Manager for historical support retrieval.
    Uses IndexFlatIP on L2-normalized embeddings for exact, exhaustive cosine similarity.
    """

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        embedding_model: Optional[SentenceEmbeddingModel] = None
    ):
        self.index = index
        self.metadata_df = metadata_df
        self.embedding_model = embedding_model or SentenceEmbeddingModel()

    @property
    def is_built(self) -> bool:
        return self.index is not None and self.metadata_df is not None

    def build_from_corpus(
        self,
        corpus_path: Optional[str] = None,
        batch_size: int = 128,
        show_progress: bool = True
    ) -> "FAISSRetrievalIndex":
        """
        Loads the historical support cases, generates dense L2-normalized embeddings
        for all customer texts, and builds an exhaustive FAISS IndexFlatIP.
        """
        print("==================================================================")
        print("BUILDING FAISS RETRIEVAL INDEX")
        print("==================================================================")
        df = load_support_cases(corpus_path)
        total_cases = len(df)
        print(f"Loaded support cases:       {total_cases:,}")
        print(f"Embedding model:            {self.embedding_model.model_name}")
        print(f"Embedding dimensions:       {EMBEDDING_DIM}")
        print("Similarity metric:          Cosine Similarity (faiss.IndexFlatIP on L2-normalized vectors)")

        customer_texts = df["customer_text"].tolist()

        t0 = time.time()
        print(f"\nGenerating embeddings for {total_cases:,} customer messages (batch_size={batch_size})...")
        embeddings = self.embedding_model.encode(
            customer_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        emb_time = time.time() - t0
        print(f"Embeddings generated in {emb_time:.2f}s ({total_cases / emb_time:.1f} vectors/sec)")

        # Validate embedding shape
        assert embeddings.shape == (total_cases, EMBEDDING_DIM), (
            f"Expected shape ({total_cases}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

        t_idx = time.time()
        # faiss.IndexFlatIP computes exact inner product.
        # Since vectors are unit-normalized (L2 norm = 1.0), inner product == cosine similarity.
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        index_time = time.time() - t_idx
        print(f"FAISS index built in {index_time:.4f}s ({index.ntotal:,} vectors indexed)")

        self.index = index
        self.metadata_df = df.copy().reset_index(drop=True)

        return self

    def save(self, artifacts_dir: Optional[str] = None) -> str:
        """
        Serializes FAISS index binary, metadata table, and index configuration.
        """
        if not self.is_built:
            raise ValueError("Cannot save an unbuilt FAISS index.")

        if artifacts_dir is None:
            artifacts_dir = get_default_artifacts_dir()

        os.makedirs(artifacts_dir, exist_ok=True)

        index_file = os.path.join(artifacts_dir, "spotify_cases.faiss")
        metadata_file = os.path.join(artifacts_dir, "case_metadata.csv")
        info_file = os.path.join(artifacts_dir, "index_info.json")

        print(f"Saving FAISS index binary to: {index_file}")
        faiss.write_index(self.index, index_file)

        print(f"Saving metadata mapping to:   {metadata_file}")
        self.metadata_df.to_csv(metadata_file, index=False, encoding="utf-8")

        info = {
            "model_name": self.embedding_model.model_name,
            "embedding_dimension": EMBEDDING_DIM,
            "total_cases": int(self.index.ntotal),
            "similarity_metric": "cosine_similarity_IndexFlatIP",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        print(" Index artifacts saved successfully.")
        return artifacts_dir

    @classmethod
    def load(
        cls,
        artifacts_dir: Optional[str] = None,
        embedding_model: Optional[SentenceEmbeddingModel] = None
    ) -> "FAISSRetrievalIndex":
        """
        Loads persisted FAISS index and metadata mapping from disk.
        """
        if artifacts_dir is None:
            artifacts_dir = get_default_artifacts_dir()

        index_file = os.path.join(artifacts_dir, "spotify_cases.faiss")
        metadata_file = os.path.join(artifacts_dir, "case_metadata.csv")

        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index binary not found at: {index_file}. "
                f"Run 'python -m backend.src.retrieval.index' to build and save it."
            )
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Index metadata file not found at: {metadata_file}."
            )

        index = faiss.read_index(index_file)
        metadata_df = pd.read_csv(metadata_file, low_memory=False)

        if index.ntotal != len(metadata_df):
            raise ValueError(
                f"FAISS index vector count ({index.ntotal}) does not match "
                f"metadata row count ({len(metadata_df)})."
            )

        model = embedding_model or SentenceEmbeddingModel()
        return cls(index=index, metadata_df=metadata_df, embedding_model=model)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search for a new customer query against historical cases.
        Returns Top-K results sorted by similarity descending.
        """
        if not self.is_built:
            raise ValueError("Index is not loaded or built. Call load() or build_from_corpus() first.")

        if top_k <= 0:
            return []

        # 1. Encode query into unit-normalized 384-d vector
        query_vec = self.embedding_model.encode_query(query, normalize_embeddings=True)
        query_vec_2d = np.expand_dims(query_vec, axis=0).astype(np.float32)

        # 2. FAISS Top-K search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec_2d, k)

        scores_row = scores[0]
        indices_row = indices[0]

        # 3. Map indices back to case metadata
        results = []
        for rank, (score, idx) in enumerate(zip(scores_row, indices_row), start=1):
            if idx < 0 or idx >= len(self.metadata_df):
                continue

            row = self.metadata_df.iloc[idx]
            results.append({
                "case_id": str(row["case_id"]),
                "customer_tweet_id": int(row["customer_tweet_id"]),
                "customer_text": str(row["customer_text"]),
                "support_tweet_id": str(row["support_tweet_id"]),
                "support_text": str(row["support_text"]),
                "similarity": round(float(score), 4),
                "intent": row["intent"] if pd.notna(row["intent"]) else None,
                "conversation_id": int(row["conversation_id"]) if pd.notna(row["conversation_id"]) else None,
                "rank": rank
            })

        # Ensure sorted descending by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


def main():
    """
    CLI command to build and persist the FAISS retrieval index.
    Run via: python -m backend.src.retrieval.index
    """
    t_start = time.time()
    retriever_index = FAISSRetrievalIndex()
    retriever_index.build_from_corpus()
    artifacts_path = retriever_index.save()

    print("\n==================================================================")
    print("VERIFYING INDEX WITH SAMPLE QUERIES")
    print("==================================================================")
    test_queries = [
        "I was charged twice for Spotify Premium",
        "Music keeps pausing on my iPhone",
        "How do I recover my hacked account and change password?"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = retriever_index.search(q, top_k=2)
        for r in results:
            print(f"  [Rank {r['rank']} | Sim: {r['similarity']:.4f} | {r['case_id']}]")
            print(f"   Customer: {r['customer_text'][:90]}...")
            print(f"   Spotify:  {r['support_text'][:90]}...")

    total_time = time.time() - t_start
    print("==================================================================")
    print(f"Complete pipeline executed in {total_time:.2f}s (~{total_time/60:.2f} minutes).")
    print("==================================================================")


if __name__ == "__main__":
    main()
