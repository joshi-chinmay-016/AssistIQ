"""
AssistIQ: Unit Tests for Historical Support Data Pipeline
Tests corpus integrity, schema adherence, non-null guarantees,
and strict anti-leakage exclusion of the golden set.
"""

import os
import unittest
import pandas as pd

from backend.src.retrieval.data import (
    load_support_cases,
    find_dataset_path,
    find_project_root
)
from backend.src.retrieval.embeddings import clean_for_embedding


class TestRetrievalData(unittest.TestCase):
    """
    Test suite for dataset/spotify_support_cases.csv and preprocessing.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = find_project_root()
        cls.corpus_path = find_dataset_path("spotify_support_cases.csv")
        cls.golden_path = find_dataset_path("golden_set.csv")
        cls.df = load_support_cases(cls.corpus_path)

    def test_corpus_exists_and_non_empty(self):
        """Verify the retrieval corpus file exists and contains cases."""
        self.assertTrue(os.path.exists(self.corpus_path))
        self.assertGreater(len(self.df), 10000, "Corpus should contain tens of thousands of support cases.")

    def test_required_columns_exist(self):
        """Verify all mandatory columns are present in the corpus schema."""
        required = [
            "case_id",
            "customer_tweet_id",
            "support_tweet_id",
            "customer_text",
            "support_text",
            "created_at",
            "intent"
        ]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")

    def test_no_empty_customer_or_support_text(self):
        """Verify that no customer text or support response is null or whitespace."""
        self.assertEqual(self.df["customer_text"].isna().sum(), 0)
        self.assertEqual(self.df["support_text"].isna().sum(), 0)

        empty_cust = (self.df["customer_text"].astype(str).str.strip() == "").sum()
        empty_supp = (self.df["support_text"].astype(str).str.strip() == "").sum()
        self.assertEqual(empty_cust, 0, "Found empty customer text entries.")
        self.assertEqual(empty_supp, 0, "Found empty support text entries.")

    def test_unique_case_ids(self):
        """Verify every case_id is unique."""
        self.assertEqual(self.df["case_id"].nunique(), len(self.df))

    def test_anti_leakage_guarantee(self):
        """
        Verify that none of the 200 golden evaluation set queries exist in the corpus.
        Ensures zero test leakage during retrieval evaluation.
        """
        if os.path.exists(self.golden_path):
            golden_df = pd.read_csv(self.golden_path)
            golden_ids = set(golden_df["tweet_id"].astype(int).unique())
            corpus_ids = set(self.df["customer_tweet_id"].astype(int).unique())

            overlap_ids = golden_ids.intersection(corpus_ids)
            self.assertEqual(
                len(overlap_ids), 0,
                f"LEAKAGE DETECTED: Found {len(overlap_ids)} golden set tweet IDs in retrieval corpus: {overlap_ids}"
            )

            # Text level overlap
            golden_texts = set(golden_df["text"].astype(str).str.strip().str.lower().unique())
            corpus_texts = set(self.df["customer_text"].astype(str).str.strip().str.lower().unique())
            overlap_texts = golden_texts.intersection(corpus_texts)
            self.assertEqual(
                len(overlap_texts), 0,
                f"LEAKAGE DETECTED: Found {len(overlap_texts)} identical customer texts in retrieval corpus."
            )

    def test_clean_for_embedding(self):
        """Verify lightweight text cleaner correctly cleans mentions/URLs while preserving semantics."""
        raw = "@SpotifyCares I was charged twice &amp; need help! https://t.co/xyz123"
        cleaned = clean_for_embedding(raw)
        self.assertEqual(cleaned, "I was charged twice & need help!")

        # None input test
        self.assertEqual(clean_for_embedding(None), "")


if __name__ == "__main__":
    unittest.main()
