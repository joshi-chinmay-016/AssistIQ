"""
AssistIQ: Historical Support Retrieval Data Pipeline
Reconstructs customer -> SpotifyCares response pairs, handles multi-turn threads,
strictly excludes the golden evaluation set to prevent leakage, and deduplicates
customer interactions to build a clean retrieval corpus.
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def find_project_root() -> str:
    """
    Locates the AssistIQ repository root directory.
    """
    if os.path.exists("dataset") and os.path.isdir("dataset"):
        return os.path.abspath(".")
    if os.path.exists(os.path.join("..", "dataset")):
        return os.path.abspath("..")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def find_dataset_path(filename: str = "spotify_support_cases.csv") -> str:
    """
    Resolves the absolute path to a dataset file.
    """
    root = find_project_root()
    return os.path.join(root, "dataset", filename)


def trace_conversation_roots(df: pd.DataFrame) -> Dict[int, int]:
    """
    Deterministically traverses in_response_to_tweet_id backwards to find the
    originating root tweet ID for every tweet in the dataset.
    Handles cycles defensively.
    """
    parent_map: Dict[int, int] = {}
    has_parent = df[df["in_response_to_tweet_id"].notna()]
    for _, row in has_parent.iterrows():
        try:
            parent_map[int(row["tweet_id"])] = int(row["in_response_to_tweet_id"])
        except (ValueError, TypeError):
            continue

    roots: Dict[int, int] = {}

    def get_root(tid: int) -> int:
        visited = set()
        curr = tid
        while curr in parent_map:
            if curr in visited:
                break
            visited.add(curr)
            curr = parent_map[curr]
        return curr

    for tid in df["tweet_id"]:
        try:
            int_id = int(tid)
            roots[int_id] = get_root(int_id)
        except (ValueError, TypeError):
            continue

    return roots


def build_historical_corpus(
    tweets_path: Optional[str] = None,
    golden_path: Optional[str] = None,
    output_path: Optional[str] = None,
    deduplicate_customer_text: bool = True
) -> pd.DataFrame:
    """
    Constructs the historical support retrieval corpus from spotify_tweets.csv:
    1. Filters customer tweets (inbound=True) and Spotify support tweets (inbound=False, author_id=SpotifyCares).
    2. Pairs support tweets back to customer tweets via in_response_to_tweet_id.
    3. Handles multi-part support replies: if SpotifyCares split its answer into multiple tweets (e.g. 1/2, 2/2),
       they are merged chronologically into a complete, unified support response.
    4. Traces conversation root IDs deterministically.
    5. Applies strict ANTI-LEAKAGE filtering: completely excludes any tweet in golden_set.csv
       by tweet_id and by exact normalized customer text.
    6. Optionally deduplicates repeated customer texts, keeping the case with the most complete support answer.
    7. Formats schema: [case_id, customer_tweet_id, support_tweet_id, customer_text, support_text,
                        created_at, intent, conversation_id, customer_author_id].

    Returns the clean DataFrame and writes to output_path.
    """
    root = find_project_root()
    if tweets_path is None:
        tweets_path = os.path.join(root, "dataset", "spotify_tweets.csv")
    if golden_path is None:
        golden_path = os.path.join(root, "dataset", "golden_set.csv")
    if output_path is None:
        output_path = os.path.join(root, "dataset", "spotify_support_cases.csv")

    if not os.path.exists(tweets_path):
        raise FileNotFoundError(f"Input tweets dataset not found at: {tweets_path}")

    print("==================================================================")
    print("HISTORICAL SUPPORT CORPUS PIPELINE")
    print("==================================================================")
    print(f"Loading input tweets: {tweets_path}")
    raw_df = pd.read_csv(tweets_path, low_memory=False)
    total_raw_tweets = len(raw_df)
    print(f"Total raw tweets loaded: {total_raw_tweets:,}")

    # 1. Inbound & author partitioning
    cust_df = raw_df[raw_df["inbound"] == True].copy()
    supp_df = raw_df[(raw_df["inbound"] == False) & (raw_df["author_id"] == "SpotifyCares")].copy()
    print(f"Customer tweets:        {len(cust_df):,}")
    print(f"SpotifyCares tweets:    {len(supp_df):,}")

    # 2. Support replies with valid customer targets
    supp_with_parent = supp_df[supp_df["in_response_to_tweet_id"].notna()].copy()
    supp_with_parent["in_response_to_tweet_id"] = supp_with_parent["in_response_to_tweet_id"].astype(int)

    # Sort support tweets chronologically so multi-part replies (1:, 2:) order correctly
    supp_with_parent["created_at_dt"] = pd.to_datetime(
        supp_with_parent["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce"
    )
    supp_with_parent = supp_with_parent.sort_values(by=["in_response_to_tweet_id", "created_at_dt", "tweet_id"])

    # 3. Group support replies by customer tweet ID
    grouped_supp = supp_with_parent.groupby("in_response_to_tweet_id").agg({
        "tweet_id": lambda ids: ",".join(str(int(i)) for i in ids),
        "text": lambda texts: "\n".join(str(t).strip() for t in texts if pd.notna(t) and str(t).strip()),
        "created_at": "first"
    }).reset_index().rename(columns={
        "in_response_to_tweet_id": "customer_tweet_id",
        "tweet_id": "support_tweet_id",
        "text": "support_text",
        "created_at": "support_created_at"
    })

    # Join back to customer tweet metadata
    cust_subset = cust_df[["tweet_id", "author_id", "created_at", "text"]].copy()
    cust_subset["tweet_id"] = cust_subset["tweet_id"].astype(int)

    paired_df = grouped_supp.merge(
        cust_subset,
        left_on="customer_tweet_id",
        right_on="tweet_id",
        how="inner"
    ).rename(columns={
        "text": "customer_text",
        "author_id": "customer_author_id",
        "created_at": "customer_created_at"
    })

    initial_pairs_count = len(paired_df)
    print(f"Reconstructed customer -> support pairs: {initial_pairs_count:,}")

    # 4. Trace conversation roots
    print("Tracing deterministic conversation threads...")
    roots_map = trace_conversation_roots(raw_df)
    paired_df["conversation_id"] = paired_df["customer_tweet_id"].map(roots_map).fillna(paired_df["customer_tweet_id"]).astype(int)
    print(f"Unique conversation threads: {paired_df['conversation_id'].nunique():,}")

    # 5. ANTI-LEAKAGE FILTERING (Golden Set Exclusion)
    golden_ids = set()
    golden_texts = set()
    if os.path.exists(golden_path):
        golden_df = pd.read_csv(golden_path)
        golden_ids = set(golden_df["tweet_id"].astype(int).unique())
        golden_texts = set(golden_df["text"].astype(str).str.strip().str.lower().unique())
        print(f"Loaded {len(golden_df)} golden evaluation examples for leakage exclusion.")

    leak_id_mask = paired_df["customer_tweet_id"].isin(golden_ids)
    norm_cust_text = paired_df["customer_text"].astype(str).str.strip().str.lower()
    leak_text_mask = norm_cust_text.isin(golden_texts)
    total_leak_mask = leak_id_mask | leak_text_mask
    leak_removed_count = total_leak_mask.sum()

    filtered_df = paired_df[~total_leak_mask].copy()
    print(f"Excluded golden set cases (ANTI-LEAKAGE): {leak_removed_count} pairs removed")
    print(f"Corpus size after leakage filter:       {len(filtered_df):,} pairs")

    # 6. Deduplication
    # If customers sent the exact same text, keep the one with the longest support response
    # to avoid near-identical duplicate hits crowding out diverse Top-K retrieval results.
    removed_duplicates_count = 0
    if deduplicate_customer_text:
        filtered_df["supp_len"] = filtered_df["support_text"].str.len()
        filtered_df["norm_query"] = filtered_df["customer_text"].astype(str).str.strip().str.lower()

        # Sort so the longest, most comprehensive support response is retained
        filtered_df = filtered_df.sort_values(by=["norm_query", "supp_len"], ascending=[True, False])
        before_dedup = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=["norm_query"], keep="first")
        removed_duplicates_count = before_dedup - len(filtered_df)
        filtered_df = filtered_df.drop(columns=["supp_len", "norm_query"])
        print(f"Deduplicated repeated customer texts:   {removed_duplicates_count:,} duplicate cases removed")
        print(f"Final unique customer-support cases:    {len(filtered_df):,}")

    # 7. Final Schema & Formatting
    filtered_df = filtered_df.sort_values(by="customer_tweet_id").reset_index(drop=True)
    filtered_df["case_id"] = [f"case_{i+1:05d}" for i in range(len(filtered_df))]

    # Ground-truth intent is left unavailable (None) because human labels do not exist
    # for raw historical cases. We do NOT fabricate fake ground-truth labels.
    filtered_df["intent"] = None
    filtered_df["created_at"] = filtered_df["customer_created_at"]

    final_columns = [
        "case_id",
        "customer_tweet_id",
        "support_tweet_id",
        "customer_text",
        "support_text",
        "created_at",
        "intent",
        "conversation_id",
        "customer_author_id"
    ]
    final_df = filtered_df[final_columns].copy()

    # Validate output
    assert final_df["customer_text"].isna().sum() == 0, "Null customer texts found!"
    assert final_df["support_text"].isna().sum() == 0, "Null support texts found!"
    assert (final_df["customer_text"].str.strip() == "").sum() == 0, "Empty customer texts found!"
    assert (final_df["support_text"].str.strip() == "").sum() == 0, "Empty support texts found!"

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n Saved clean retrieval corpus to: {output_path}")
    print("------------------------------------------------------------------")
    print("TRANSFORMATION AUDIT TRAIL:")
    print(f"• Input raw Spotify tweets:           {total_raw_tweets:,}")
    print(f"• Customer tweets:                    {len(cust_df):,}")
    print(f"• SpotifyCares support tweets:        {len(supp_df):,}")
    print(f"• Raw customer->support pairs:        {initial_pairs_count:,}")
    print(f"• Golden-set leakage pairs removed:   {leak_removed_count}")
    print(f"• Duplicate customer queries removed: {removed_duplicates_count:,}")
    print(f"• Final clean retrieval corpus cases: {len(final_df):,}")
    print("==================================================================")

    return final_df


def load_support_cases(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates dataset/spotify_support_cases.csv.
    Raises FileNotFoundError or ValueError if missing or invalid.
    """
    if filepath is None:
        filepath = find_dataset_path("spotify_support_cases.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Retrieval corpus not found at: {filepath}. "
            f"Run 'python -m backend.src.retrieval.data' to build it."
        )

    df = pd.read_csv(filepath, low_memory=False)

    required_cols = [
        "case_id",
        "customer_tweet_id",
        "support_tweet_id",
        "customer_text",
        "support_text",
        "created_at",
        "intent"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Retrieval corpus missing required columns: {missing}")

    if df.empty:
        raise ValueError("Retrieval corpus is empty.")

    null_cust = df["customer_text"].isna().sum() + (df["customer_text"].astype(str).str.strip() == "").sum()
    if null_cust > 0:
        raise ValueError(f"Found {null_cust} empty customer texts in retrieval corpus.")

    null_supp = df["support_text"].isna().sum() + (df["support_text"].astype(str).str.strip() == "").sum()
    if null_supp > 0:
        raise ValueError(f"Found {null_supp} empty support texts in retrieval corpus.")

    return df


if __name__ == "__main__":
    build_historical_corpus()
