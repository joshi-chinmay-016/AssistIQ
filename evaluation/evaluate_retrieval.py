"""
AssistIQ: Historical Retrieval Evaluation Harness
Evaluates FAISS semantic search independently of generation.
Measures automated intent consistency @ 1, 3, 5, Hit Rate @ 1, 3, 5,
similarity distribution statistics, and retrieval latency over golden queries.
Incorporates manual review metrics (Strict/Lenient Precision@k, Hit Rate@3)
and strictly differentiates automated heuristic metrics from human relevance.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.retrieval.index import FAISSRetrievalIndex, get_default_artifacts_dir
from backend.src.retrieval.retrieve import get_retriever
from backend.src.intent.data import find_dataset_path
from backend.src.intent.models import build_proposed_model


def verify_anti_leakage(
    golden_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Tuple[bool, int, List[int]]:
    """
    Verifies that zero golden set customer tweet IDs exist in the historical corpus.
    Guarantees strict evaluation isolation and prevents exact-match data leakage.
    """
    golden_ids = set(golden_df["tweet_id"].astype(int).unique())
    corpus_ids = set(metadata_df["customer_tweet_id"].astype(int).unique())
    overlap = golden_ids.intersection(corpus_ids)
    return len(overlap) == 0, len(overlap), list(overlap)


def evaluate_manual_review_dataset(review_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes rigorous retrieval relevance metrics from human-reviewed cases:
    - Strict Precision@1 (proportion of top-1 cases rated 'relevant')
    - Strict Precision@3 (proportion of top-3 cases rated 'relevant')
    - Lenient Precision@1 (proportion rated 'relevant' or 'partially_relevant')
    - Lenient Precision@3 (proportion rated 'relevant' or 'partially_relevant')
    - Manual Hit Rate@3 (proportion of queries with >= 1 'relevant' case in top 3)
    """
    if review_path is None:
        review_path = os.path.join(PROJECT_ROOT, "evaluation", "retrieval_review.csv")

    if not os.path.exists(review_path):
        return {"status": "NO_MANUAL_REVIEW_FILE"}

    df = pd.read_csv(review_path)
    total_queries = df["tweet_id"].nunique()
    total_rows = len(df)

    rank1_df = df[df["rank"] == 1]
    rank3_df = df[df["rank"].isin([1, 2, 3])]

    # Strict (only 'relevant')
    strict_p1 = (rank1_df["relevance_label"] == "relevant").sum() / len(rank1_df) if len(rank1_df) > 0 else 0.0
    strict_p3 = (rank3_df["relevance_label"] == "relevant").sum() / len(rank3_df) if len(rank3_df) > 0 else 0.0

    # Lenient ('relevant' or 'partially_relevant')
    lenient_p1 = (rank1_df["relevance_label"].isin(["relevant", "partially_relevant"])).sum() / len(rank1_df) if len(rank1_df) > 0 else 0.0
    lenient_p3 = (rank3_df["relevance_label"].isin(["relevant", "partially_relevant"])).sum() / len(rank3_df) if len(rank3_df) > 0 else 0.0

    # Hit Rate @ 3 (at least one relevant case in top-3 for query)
    has_relevant = df[df["rank"].isin([1, 2, 3])].groupby("tweet_id")["relevance_label"].apply(
        lambda s: (s == "relevant").any()
    )
    manual_hit_rate_3 = float(has_relevant.sum()) / total_queries if total_queries > 0 else 0.0

    return {
        "manual_review_queries": total_queries,
        "manual_review_cases": total_rows,
        "strict_precision@1": round(strict_p1, 4),
        "strict_precision@3": round(strict_p3, 4),
        "lenient_precision@1": round(lenient_p1, 4),
        "lenient_precision@3": round(lenient_p3, 4),
        "manual_hit_rate@3": round(manual_hit_rate_3, 4)
    }


def run_retrieval_evaluation(
    golden_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    limit: Optional[int] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Executes historical support retrieval evaluation:
    1. Loads golden set and historical FAISS index.
    2. Verifies anti-leakage.
    3. Performs dense vector search for each golden query.
    4. Computes automated Intent Consistency @ 1, 3, 5 and Hit Rate @ 1, 3, 5.
    5. Computes cosine similarity distribution statistics (mean, median, min, max, std).
    6. Measures query latency (mean, median, p95).
    7. Computes manual review metrics.
    8. Exports machine-readable CSV artifacts.
    """
    if golden_path is None:
        golden_path = find_dataset_path("golden_set.csv")
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n==================================================================")
    print("ASSISTIQ HISTORICAL RETRIEVAL EVALUATION HARNESS")
    print("==================================================================")

    golden_df = pd.read_csv(golden_path)
    if limit is not None and limit < len(golden_df):
        golden_df = golden_df.head(limit)
        print(f"Subsample limit applied: {len(golden_df)} queries")

    # Load retriever
    print("Loading FAISS index and SentenceTransformer embedding model...")
    retriever = get_retriever()
    metadata_df = retriever.metadata_df

    total_corpus_cases = len(metadata_df)
    print(f"Total historical support cases indexed: {total_corpus_cases:,}")

    # Step 1: Anti-leakage check
    leakage_passed, overlap_count, overlapping_ids = verify_anti_leakage(golden_df, metadata_df)
    print(f"Anti-leakage check: {'[PASSED]' if leakage_passed else '[FAILED]'} (Overlap count: {overlap_count})")
    if not leakage_passed:
        raise ValueError(f"CRITICAL: Data leakage detected! Golden tweets found in index: {overlapping_ids}")

    # Step 2: Intent model for consistency check
    print("Fitting intent classifier for intent consistency calculation...")
    from backend.src.intent.data import load_and_split_data
    X_train, _, y_train, _, _, _ = load_and_split_data(golden_path, random_state=2026)
    intent_model = build_proposed_model(random_state=2026)
    intent_model.fit(X_train, y_train)

    # Warm up retriever to separate one-time model initialization from query search latency
    print("Warming up embedding model and FAISS index...")
    _ = retriever.search("warmup query", top_k=1)

    # Step 3: Query search and benchmark loop
    print(f"\nSearching Top-{top_k} cases across {len(golden_df)} golden queries...")
    latencies = []
    top1_sims = []
    all_topk_sims = []

    top1_matches = 0
    top3_matches = 0
    top5_matches = 0

    top1_hit = 0
    top3_hit = 0
    top5_hit = 0

    retrieval_records = []
    retrieval_errors = []

    for idx, row in golden_df.iterrows():
        q_id = int(row["tweet_id"])
        q_text = str(row["text"])
        true_intent = str(row["intent"])

        t0 = time.perf_counter()
        cases = retriever.search(q_text, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if not cases:
            continue

        sims = [float(c.get("similarity", 0.0)) for c in cases]
        top1_sim = sims[0] if sims else 0.0
        top1_sims.append(top1_sim)
        all_topk_sims.extend(sims)

        # Predict intent of retrieved cases
        retrieved_texts = [c.get("customer_text", "") for c in cases]
        pred_ret_intents = intent_model.predict(retrieved_texts)

        # Match against true intent
        matches = [pred == true_intent for pred in pred_ret_intents]

        if matches[0]:
            top1_matches += 1
            top1_hit += 1
        top3_matches += sum(matches[:3]) / min(3, len(matches))
        top5_matches += sum(matches[:5]) / min(5, len(matches))

        if any(matches[:3]):
            top3_hit += 1
        if any(matches[:5]):
            top5_hit += 1

        # Track error cases (top-1 similarity low or zero intent match)
        if top1_sim < 0.45 or not any(matches[:3]):
            retrieval_errors.append({
                "query_tweet_id": q_id,
                "query_text": q_text,
                "golden_intent": true_intent,
                "top1_case_id": cases[0].get("case_id"),
                "top1_similarity": round(top1_sim, 4),
                "top1_retrieved_customer": cases[0].get("customer_text", "")[:120],
                "top1_predicted_intent": pred_ret_intents[0],
                "hit_in_top3": any(matches[:3])
            })

    total_q = len(golden_df)
    ic_1 = round(top1_matches / total_q, 4)
    ic_3 = round(top3_matches / total_q, 4)
    ic_5 = round(top5_matches / total_q, 4)
    hr_1 = round(top1_hit / total_q, 4)
    hr_3 = round(top3_hit / total_q, 4)
    hr_5 = round(top5_hit / total_q, 4)

    sim_mean = round(float(np.mean(top1_sims)), 4)
    sim_median = round(float(np.median(top1_sims)), 4)
    sim_min = round(float(np.min(top1_sims)), 4)
    sim_max = round(float(np.max(top1_sims)), 4)
    sim_std = round(float(np.std(top1_sims)), 4)

    lat_mean = round(float(np.mean(latencies)), 2)
    lat_median = round(float(np.median(latencies)), 2)
    lat_p95 = round(float(np.percentile(latencies, 95)), 2)

    # Step 4: Manual review evaluation
    manual_metrics = evaluate_manual_review_dataset()

    # Step 5: Save structured metrics CSV
    metrics_list = [
        {"metric": "total_indexed_cases", "value": total_corpus_cases, "category": "Corpus", "description": "Total historical Spotify support cases indexed in FAISS"},
        {"metric": "embedding_model", "value": "sentence-transformers/all-MiniLM-L6-v2", "category": "Model", "description": "384-dimensional dense semantic embedding model"},
        {"metric": "index_type", "value": "faiss.IndexFlatIP", "category": "Index", "description": "Exhaustive inner product on L2-normalized vectors (exact cosine)"},
        {"metric": "anti_leakage_overlap", "value": overlap_count, "category": "Integrity", "description": "Overlap between golden test queries and index (0 required)"},
        {"metric": "evaluated_queries", "value": total_q, "category": "Evaluation", "description": "Number of golden evaluation queries evaluated"},
        # Automated metrics
        {"metric": "intent_hit_rate@1", "value": hr_1, "category": "Automated", "description": "Fraction of queries with matching intent at rank 1"},
        {"metric": "intent_hit_rate@3", "value": hr_3, "category": "Automated", "description": "Fraction of queries with >=1 matching intent in Top-3"},
        {"metric": "intent_hit_rate@5", "value": hr_5, "category": "Automated", "description": "Fraction of queries with >=1 matching intent in Top-5 (Headline metric)"},
        {"metric": "intent_consistency@1", "value": ic_1, "category": "Automated", "description": "Fraction of Top-1 retrieved cases matching query golden intent"},
        {"metric": "intent_consistency@3", "value": ic_3, "category": "Automated", "description": "Mean fraction of Top-3 retrieved cases matching query golden intent"},
        {"metric": "intent_consistency@5", "value": ic_5, "category": "Automated", "description": "Mean fraction of Top-5 retrieved cases matching query golden intent"},
        # Similarity statistics
        {"metric": "top1_similarity_mean", "value": sim_mean, "category": "Similarity", "description": "Mean cosine similarity of Top-1 retrieved case"},
        {"metric": "top1_similarity_median", "value": sim_median, "category": "Similarity", "description": "Median cosine similarity of Top-1 retrieved case"},
        {"metric": "top1_similarity_min", "value": sim_min, "category": "Similarity", "description": "Minimum Top-1 cosine similarity across evaluation set"},
        {"metric": "top1_similarity_max", "value": sim_max, "category": "Similarity", "description": "Maximum Top-1 cosine similarity across evaluation set"},
        {"metric": "top1_similarity_std", "value": sim_std, "category": "Similarity", "description": "Standard deviation of Top-1 cosine similarity"},
        # Latency
        {"metric": "mean_retrieval_latency_ms", "value": lat_mean, "category": "Latency", "description": "Mean FAISS search latency per query in ms"},
        {"metric": "median_retrieval_latency_ms", "value": lat_median, "category": "Latency", "description": "Median FAISS search latency per query in ms"},
        {"metric": "p95_retrieval_latency_ms", "value": lat_p95, "category": "Latency", "description": "95th percentile FAISS search latency per query in ms"},
        # Manual review metrics
        {"metric": "manual_review_strict_precision@1", "value": manual_metrics.get("strict_precision@1", 0.0), "category": "Manual", "description": "Top-1 strict precision (directly relevant solution) on 35-query human review set"},
        {"metric": "manual_review_strict_precision@3", "value": manual_metrics.get("strict_precision@3", 0.0), "category": "Manual", "description": "Top-3 strict precision (directly relevant solution) on 35-query human review set"},
        {"metric": "manual_review_lenient_precision@1", "value": manual_metrics.get("lenient_precision@1", 0.0), "category": "Manual", "description": "Top-1 lenient precision (relevant or partially relevant) on human review set"},
        {"metric": "manual_review_lenient_precision@3", "value": manual_metrics.get("lenient_precision@3", 0.0), "category": "Manual", "description": "Top-3 lenient precision (relevant or partially relevant) on human review set"},
        {"metric": "manual_review_hit_rate@3", "value": manual_metrics.get("manual_hit_rate@3", 0.0), "category": "Manual", "description": "Fraction of queries with >=1 relevant case in Top-3 on human review set"}
    ]

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(results_dir, "retrieval_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    # Also write legacy retrieval_results.csv
    legacy_path = os.path.join(results_dir, "retrieval_results.csv")
    metrics_df[["metric", "value", "description"]].to_csv(legacy_path, index=False)

    # Step 6: Save retrieval error analysis
    errors_df = pd.DataFrame(retrieval_errors)
    errors_path = os.path.join(results_dir, "retrieval_errors.csv")
    errors_df.to_csv(errors_path, index=False)

    print(f"\n[OK] Saved retrieval metrics to: {metrics_path}")
    print(f"[OK] Saved retrieval errors to:  {errors_path} ({len(errors_df)} queries flagged)")

    # Step 7: Print Human-Readable Comparison Table
    print("\n==================================================================")
    print("RETRIEVAL EVALUATION SUMMARY")
    print("==================================================================")
    print("AUTOMATED METRICS (N=200 Golden Queries):")
    print(f"  - Intent Hit Rate@1:            {hr_1:.4f}")
    print(f"  - Intent Hit Rate@3:            {hr_3:.4f}")
    print(f"  - Intent Hit Rate@5:            {hr_5:.4f}  <-- MISLEADING HEADLINE NUMBER")
    print(f"  - Intent Consistency@1:         {ic_1:.4f}")
    print(f"  - Intent Consistency@3:         {ic_3:.4f}")
    print(f"  - Intent Consistency@5:         {ic_5:.4f}")
    print("------------------------------------------------------------------")
    print("HUMAN RELEVANCE REVIEW (N=35 Queries, 105 Cases):")
    print(f"  - Strict Precision@1:           {manual_metrics.get('strict_precision@1', 0):.4f} (Direct relevant solution)")
    print(f"  - Strict Precision@3:           {manual_metrics.get('strict_precision@3', 0):.4f} (Direct relevant solution)")
    print(f"  - Lenient Precision@1:          {manual_metrics.get('lenient_precision@1', 0):.4f} (Relevant or partial)")
    print(f"  - Lenient Precision@3:          {manual_metrics.get('lenient_precision@3', 0):.4f} (Relevant or partial)")
    print(f"  - Manual Hit Rate@3:            {manual_metrics.get('manual_hit_rate@3', 0):.4f}")
    print("------------------------------------------------------------------")
    print("SIMILARITY & LATENCY STATS:")
    print(f"  - Top-1 Similarity: Mean={sim_mean:.4f}, Median={sim_median:.4f}, Range=[{sim_min:.4f}, {sim_max:.4f}]")
    print(f"  - Search Latency:   Mean={lat_mean:.2f}ms, Median={lat_median:.2f}ms, P95={lat_p95:.2f}ms")
    print("==================================================================\n")

    return {
        "metrics": metrics_df,
        "errors": errors_df,
        "summary": {
            "intent_hit_rate@5": hr_5,
            "strict_precision@3": manual_metrics.get("strict_precision@3", 0.0),
            "mean_latency_ms": lat_mean
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AssistIQ Retrieval Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for fast benchmark")
    args = parser.parse_args()
    run_retrieval_evaluation(limit=args.limit)
