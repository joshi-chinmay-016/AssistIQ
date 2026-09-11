"""
AssistIQ: Phase 3 Grounded Generation Evaluation Pipeline
Evaluates generated replies against a representative 30-sample subset of dataset/golden_set.csv.
Generates evaluation/reply_review.csv for human evaluation and
evaluation/results/reply_examples.csv for qualitative analysis.
Strictly avoids fabricating human scores or running expensive un-budgeted batch generations.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.src.generation.config import GenerationConfig, find_project_root
from backend.src.generation.generate import assist_customer
from backend.src.intent.data import find_dataset_path


def select_evaluation_sample(
    golden_path: Optional[str] = None,
    sample_size: int = 30,
    random_state: int = 2026
) -> pd.DataFrame:
    """
    Selects a stratified/diverse subset of golden_set.csv spanning as many
    intent classes as possible for cost-controlled evaluation.
    """
    if golden_path is None:
        golden_path = find_dataset_path("golden_set.csv")

    df = pd.read_csv(golden_path)

    # Sample proportionally across intents
    intents = df["intent"].unique()
    samples_per_intent = max(1, sample_size // len(intents))

    sampled_dfs = []
    for intent in intents:
        subset = df[df["intent"] == intent]
        n_to_sample = min(len(subset), samples_per_intent)
        sampled_dfs.append(subset.sample(n=n_to_sample, random_state=random_state))

    combined = pd.concat(sampled_dfs).drop_duplicates(subset=["tweet_id"])

    # If short of sample_size, sample remaining randomly
    if len(combined) < sample_size:
        remaining = df[~df["tweet_id"].isin(combined["tweet_id"])]
        needed = min(len(remaining), sample_size - len(combined))
        combined = pd.concat([combined, remaining.sample(n=needed, random_state=random_state)])

    # If slightly over sample_size, take exactly sample_size
    final_sample = combined.head(sample_size).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return final_sample


def run_generation_evaluation(
    sample_size: Optional[int] = None,
    config: Optional[GenerationConfig] = None
) -> Dict[str, Any]:
    """
    Executes the generation evaluation pipeline:
    1. Loads evaluation sample of golden queries.
    2. Runs assist_customer() across the sample.
    3. Writes evaluation/reply_review.csv with blank rating columns for human review.
    4. Writes evaluation/results/reply_examples.csv with 5 distinct output archetypes.
    5. Benchmarks latency and computes grounding status distribution.
    """
    root = find_project_root()
    cfg = config or GenerationConfig.from_env()

    if not cfg.has_valid_api_key() and cfg.provider == "gemini":
        print("Notice: No live GEMINI_API_KEY detected in backend/.env.")
        print("Running evaluation with MockLLMClient for local artifact generation.\n")
        cfg.provider = "mock"

    if sample_size is None:
        sample_size = cfg.eval_limit

    print("==================================================================")
    print("ASSISTIQ PHASE 3: GROUNDED GENERATION EVALUATION")
    print("==================================================================")
    print(f"Evaluation Sample Size:   {sample_size} golden queries")
    print(f"Random State:             2026")
    print(f"LLM Provider:             {cfg.provider}")
    print(f"LLM Model:                {cfg.model_name}")
    print(f"Temperature:              {cfg.temperature}")

    # Select representative golden sample
    golden_sample = select_evaluation_sample(sample_size=sample_size, random_state=2026)
    print(f"Loaded {len(golden_sample)} queries across {golden_sample['intent'].nunique()} intents.")

    review_records = []
    latencies = {
        "intent_ms": [],
        "retrieval_ms": [],
        "generation_ms": [],
        "total_ms": []
    }
    status_counts = {"grounded": 0, "insufficient_evidence": 0, "generation_failed": 0}

    print("\nExecuting end-to-end customer assistance pipeline...")
    for idx, (_, row) in enumerate(golden_sample.iterrows(), start=1):
        tweet_id = int(row["tweet_id"])
        customer_text = str(row["text"])

        res = assist_customer(customer_text, top_k=cfg.top_k, config=cfg)

        latencies["intent_ms"].append(res["latency"]["intent_ms"])
        latencies["retrieval_ms"].append(res["latency"]["retrieval_ms"])
        latencies["generation_ms"].append(res["latency"]["generation_ms"])
        latencies["total_ms"].append(res["latency"]["total_ms"])

        status = res["grounding_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        # Format record for human review (quality rating columns intentionally left BLANK)
        review_records.append({
            "tweet_id": tweet_id,
            "customer_text": customer_text,
            "predicted_intent": res["predicted_intent"],
            "generated_reply": res["reply"],
            "evidence_case_ids": ",".join(res["evidence_case_ids"]),
            "grounding_status": status,
            "correctness": "",    # Blank for human evaluation
            "groundedness": "",   # Blank for human evaluation
            "relevance": "",      # Blank for human evaluation
            "completeness": "",   # Blank for human evaluation
            "tone": "",           # Blank for human evaluation
            "notes": ""           # Blank for human evaluation
        })

        if idx % 10 == 0 or idx == len(golden_sample):
            print(f"  Processed {idx}/{len(golden_sample)} queries... (avg latency: {np.mean(latencies['total_ms']):.1f}ms)")

    # 1. Save Human Review Dataset
    review_df = pd.DataFrame(review_records)
    review_path = os.path.join(root, "evaluation", "reply_review.csv")
    os.makedirs(os.path.dirname(review_path), exist_ok=True)
    review_df.to_csv(review_path, index=False, encoding="utf-8")
    print(f"\n Saved human evaluation template to: {review_path}")

    # 2. Build and Save Qualitative Examples Across Archetypes
    print("\nGenerating qualitative reply examples across 5 archetypes...")
    archetypes = [
        ("archetype_1_strong_grounded",
         115911,
         "I was charged twice for Spotify Premium subscription this month. Can I get a refund?",
         "Strong grounded answer: duplicate billing inquiry with direct historical resolution"),

        ("archetype_2_lexical_paraphrase",
         115866,
         "songs keep stopping on my phone when screen turns off without me touching anything",
         "Good answer with paraphrased evidence: background playback pausing troubleshooting"),

        ("archetype_3_ambiguous_query",
         115899,
         "why does this app always do this every single time i use it",
         "Ambiguous query: customer expresses general frustration without device/platform details"),

        ("archetype_4_insufficient_evidence",
         999999,
         "Can I play Spotify on my microwave with custom firmware?",
         "Insufficient-evidence case: query unsupported by historical knowledge base"),

        ("archetype_5_conversational_edge_case",
         116099,
         "hello??? @SpotifyCares",
         "Edge/boundary case: ultra-short greeting tweet lacking actionable support issue")
    ]

    example_records = []
    for arch_key, q_id, q_text, arch_desc in archetypes:
        res = assist_customer(q_text, top_k=cfg.top_k, config=cfg)
        example_records.append({
            "archetype": arch_key,
            "archetype_description": arch_desc,
            "query_tweet_id": q_id,
            "customer_text": q_text,
            "predicted_intent": res["predicted_intent"],
            "generated_reply": res["reply"],
            "evidence_case_ids": ",".join(res["evidence_case_ids"]),
            "grounding_status": res["grounding_status"],
            "grounding_summary": res["grounding_summary"]
        })

    example_df = pd.DataFrame(example_records)
    results_dir = os.path.join(root, "evaluation", "results")
    examples_path = os.path.join(results_dir, "reply_examples.csv")
    os.makedirs(results_dir, exist_ok=True)
    example_df.to_csv(examples_path, index=False, encoding="utf-8")
    print(f" Saved qualitative reply examples to: {examples_path}")

    # 3. Print Summary Benchmarks
    mean_total = np.mean(latencies["total_ms"])
    p95_total = np.percentile(latencies["total_ms"], 95)
    mean_intent = np.mean(latencies["intent_ms"])
    mean_retrieval = np.mean(latencies["retrieval_ms"])
    mean_gen = np.mean(latencies["generation_ms"])

    print("\n==================================================================")
    print("PHASE 3 EVALUATION SUMMARY")
    print("==================================================================")
    print(f"Total Evaluated:          {len(golden_sample)} queries")
    print(f"Grounding Status Breakdown:")
    for st, cnt in status_counts.items():
        pct = (cnt / len(golden_sample)) * 100
        print(f"  - {st:23s}: {cnt:2d} ({pct:5.1f}%)")
    print("------------------------------------------------------------------")
    print(f"Latency Benchmarks:")
    print(f"  - Mean Intent Latency:   {mean_intent:.2f} ms")
    print(f"  - Mean Retrieval Latency:{mean_retrieval:.2f} ms")
    print(f"  - Mean Generation Latency:{mean_gen:.2f} ms")
    print(f"  - Mean Total Latency:    {mean_total:.2f} ms")
    print(f"  - P95 Total Latency:     {p95_total:.2f} ms")
    print("------------------------------------------------------------------")
    # Approximate token cost estimation
    # Avg prompt: ~400 input tokens; avg response: ~60 output tokens
    est_input_tokens = len(golden_sample) * 400
    est_output_tokens = len(golden_sample) * 60
    # Gemini 1.5 Flash rates: $0.075 / 1M input, $0.30 / 1M output
    est_cost = (est_input_tokens * 0.075 / 1e6) + (est_output_tokens * 0.30 / 1e6)
    print(f"Estimated Token Usage (30 queries): ~{est_input_tokens:,} input, ~{est_output_tokens:,} output")
    print(f"Estimated API Cost (Gemini 1.5 Flash): < ${est_cost:.5f} USD")
    print("==================================================================")

    return {
        "sample_size": len(golden_sample),
        "status_counts": status_counts,
        "latencies": {
            "mean_intent_ms": round(float(mean_intent), 2),
            "mean_retrieval_ms": round(float(mean_retrieval), 2),
            "mean_generation_ms": round(float(mean_gen), 2),
            "mean_total_ms": round(float(mean_total), 2),
            "p95_total_ms": round(float(p95_total), 2)
        },
        "review_file": review_path,
        "examples_file": examples_path
    }


if __name__ == "__main__":
    run_generation_evaluation()
