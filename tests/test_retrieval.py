"""
AssistIQ: Unit Tests for FAISS Retrieval Index and Search API
Tests FAISS index building, persistence, reloading, Top-K ordering,
and schema adherence.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import faiss

from backend.src.retrieval.index import FAISSRetrievalIndex
from backend.src.retrieval.retrieve import retrieve_similar_cases


class MockEmbeddingModel:
    """
    Deterministic mock embedding model for fast, offline unit testing.
    Generates 384-dimensional unit-normalized pseudo-embeddings without network or GPU.
    """

    def __init__(self, dim: int = 384):
        self.embedding_dim = dim
        self.model_name = "mock-all-MiniLM-L6-v2"

    def encode(self, texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]

        vecs = []
        for t in texts:
            # Deterministic vector based on hash of text
            seed = abs(hash(t)) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.embedding_dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)

        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query, normalize_embeddings=True):
        return self.encode(query, normalize_embeddings=normalize_embeddings)[0]


class TestRetrievalIndexAndSearch(unittest.TestCase):
    """
    Unit test suite for FAISSRetrievalIndex and search functions.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.mock_model = MockEmbeddingModel(dim=384)

        # Create small test dataset (5 cases)
        cls.test_cases = pd.DataFrame([
            {
                "case_id": "case_00001",
                "customer_tweet_id": 101,
                "support_tweet_id": "201",
                "customer_text": "I was charged twice for Spotify Premium subscription",
                "support_text": "Please DM us your email and we will issue a refund.",
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "intent": None,
                "conversation_id": 101,
                "customer_author_id": 9991
            },
            {
                "case_id": "case_00002",
                "customer_tweet_id": 102,
                "support_tweet_id": "202",
                "customer_text": "Music keeps pausing on my iPhone whenever screen locks",
                "support_text": "Try a clean reinstall of the app following this guide.",
                "created_at": "Tue Oct 31 11:00:00 +0000 2017",
                "intent": None,
                "conversation_id": 102,
                "customer_author_id": 9992
            },
            {
                "case_id": "case_00003",
                "customer_tweet_id": 103,
                "support_tweet_id": "203",
                "customer_text": "Cannot login to my account, password reset link not working",
                "support_text": "Check your spam folder or send us a DM to verify your account.",
                "created_at": "Tue Oct 31 12:00:00 +0000 2017",
                "intent": None,
                "conversation_id": 103,
                "customer_author_id": 9993
            },
            {
                "case_id": "case_00004",
                "customer_tweet_id": 104,
                "support_tweet_id": "204",
                "customer_text": "My playlist disappeared after the latest update",
                "support_text": "You can recover deleted playlists on your account page on web.",
                "created_at": "Tue Oct 31 13:00:00 +0000 2017",
                "intent": None,
                "conversation_id": 104,
                "customer_author_id": 9994
            },
            {
                "case_id": "case_00005",
                "customer_tweet_id": 105,
                "support_tweet_id": "205",
                "customer_text": "Is Taylor Swift new album available in Australia?",
                "support_text": "Content availability depends on licensing agreements with rights holders.",
                "created_at": "Tue Oct 31 14:00:00 +0000 2017",
                "intent": None,
                "conversation_id": 105,
                "customer_author_id": 9995
            }
        ])

        # Build index with mock model
        cls.index_mgr = FAISSRetrievalIndex(embedding_model=cls.mock_model)
        embs = cls.mock_model.encode(cls.test_cases["customer_text"].tolist(), normalize_embeddings=True)
        idx = faiss.IndexFlatIP(384)
        idx.add(embs)
        cls.index_mgr.index = idx
        cls.index_mgr.metadata_df = cls.test_cases.copy()

        # Save to temp directory
        cls.index_mgr.save(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_index_save_and_load(self):
        """Verify FAISS index and metadata can be saved and reloaded with complete fidelity."""
        reloaded = FAISSRetrievalIndex.load(self.temp_dir, embedding_model=self.mock_model)
        self.assertTrue(reloaded.is_built)
        self.assertEqual(reloaded.index.ntotal, len(self.test_cases))
        self.assertEqual(len(reloaded.metadata_df), len(self.test_cases))

    def test_top_k_retrieval_and_ordering(self):
        """Verify top_k search returns requested number of results, sorted by similarity descending."""
        query = "charged twice for subscription"
        results = self.index_mgr.search(query, top_k=3)

        self.assertEqual(len(results), 3)

        # Check descending order
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check score range [-1.0, 1.0]
        for s in scores:
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)

    def test_result_structure_and_metadata(self):
        """Verify returned dictionaries match expected schema with all metadata."""
        query = "cannot log in"
        results = self.index_mgr.search(query, top_k=1)
        self.assertEqual(len(results), 1)
        r = results[0]

        expected_keys = [
            "case_id",
            "customer_tweet_id",
            "customer_text",
            "support_tweet_id",
            "support_text",
            "similarity",
            "intent",
            "conversation_id",
            "rank"
        ]
        for k in expected_keys:
            self.assertIn(k, r, f"Missing key: {k}")

        self.assertIsInstance(r["similarity"], float)
        self.assertEqual(r["rank"], 1)

    def test_empty_query_handling(self):
        """Verify search handles empty or blank queries gracefully."""
        res_empty = self.index_mgr.search("", top_k=5)
        # Empty string encoded via mock
        self.assertIsInstance(res_empty, list)

        # retrieve_similar_cases returns empty list on blank query
        res_api = retrieve_similar_cases("   ", top_k=3)
        self.assertEqual(res_api, [])


if __name__ == "__main__":
    unittest.main()
