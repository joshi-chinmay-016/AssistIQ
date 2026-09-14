"""
AssistIQ: Human vs. LLM Judge Agreement Analysis
Evaluates statistical agreement between human reviewers and the LLM Judge across
the 5 core quality dimensions (Relevance, Groundedness, Helpfulness, Completeness, Safety).

Computes:
1. Pearson Correlation (linear metric agreement)
2. Spearman Rank Correlation (monotonic rank agreement)
3. Categorical Binned Agreement Rate (Pass / Fail agreement at >= 3.5 threshold)
4. Quadratic Weighted Cohen's Kappa (penalizes distant ordinal disagreements heavily)

Transparency Guarantee:
If human evaluation scores are blank in reply_review.csv, outputs "HUMAN_REVIEW_REQUIRED"
without generating fabricated agreement values. Includes a --demo-mock-human flag for
synthetic algorithm validation.
"""

import os
import sys
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


def compute_weighted_cohen_kappa(
    rater1: np.ndarray,
    rater2: np.ndarray,
    min_rating: int = 1,
    max_rating: int = 5,
    weight_type: str = "quadratic"
) -> float:
    """
    Calculates weighted Cohen's kappa for ordinal ratings (1 to 5).
    Quadratic weighting penalizes a (1 vs 5) disagreement 16x more heavily than a (1 vs 2) disagreement.

    Formula:
        kappa = 1 - (sum(w_ij * O_ij) / sum(w_ij * E_ij))
    where w_ij = (i - j)^2 / (max - min)^2 for quadratic weighting.
    """
    r1 = np.asarray(rater1, dtype=int)
    r2 = np.asarray(rater2, dtype=int)

    n_categories = max_rating - min_rating + 1
    categories = list(range(min_rating, max_rating + 1))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Observed matrix O
    observed = np.zeros((n_categories, n_categories), dtype=float)
    for a, b in zip(r1, r2):
        if a in cat_to_idx and b in cat_to_idx:
            observed[cat_to_idx[a], cat_to_idx[b]] += 1.0

    total = np.sum(observed)
    if total == 0:
        return 0.0
    observed /= total

    # Expected matrix E under independence
    hist1 = np.sum(observed, axis=1)
    hist2 = np.sum(observed, axis=0)
    expected = np.outer(hist1, hist2)

    # Weight matrix W
    weights = np.zeros((n_categories, n_categories), dtype=float)
    denom = float((max_rating - min_rating) ** 2)
    for i in range(n_categories):
        for j in range(n_categories):
            if weight_type == "quadratic":
                weights[i, j] = ((i - j) ** 2) / (denom if denom > 0 else 1.0)
            else:
                weights[i, j] = abs(i - j) / float(max_rating - min_rating)

    sum_wo = np.sum(weights * observed)
    sum_we = np.sum(weights * expected)

    if sum_we == 0:
        return 1.0 if sum_wo == 0 else 0.0

    kappa = 1.0 - (sum_wo / sum_we)
    return round(float(kappa), 4)


def compute_agreement_metrics(
    human_ratings: np.ndarray,
    llm_ratings: np.ndarray,
    dimension_name: str,
    pass_threshold: float = 3.5
) -> Dict[str, Any]:
    """
    Computes Pearson correlation, Spearman correlation, binned agreement rate,
    and quadratic weighted Cohen's kappa for a single rating dimension.
    """
    h = np.asarray(human_ratings, dtype=float)
    l = np.asarray(llm_ratings, dtype=float)

    n = len(h)
    if n < 3:
        return {
            "dimension": dimension_name,
            "sample_count": n,
            "status": "INSUFFICIENT_SAMPLES",
            "human_mean": round(float(np.mean(h)), 2) if n > 0 else 0.0,
            "llm_mean": round(float(np.mean(l)), 2) if n > 0 else 0.0,
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "agreement_rate": 0.0,
            "weighted_kappa": 0.0
        }

    # Means
    h_mean = float(np.mean(h))
    l_mean = float(np.mean(l))

    # Pearson correlation
    h_std = float(np.std(h))
    l_std = float(np.std(l))
    if h_std > 0 and l_std > 0:
        pearson = float(np.corrcoef(h, l)[0, 1])
    else:
        pearson = 1.0 if h_mean == l_mean else 0.0

    # Spearman rank correlation
    h_ranks = pd.Series(h).rank().values
    l_ranks = pd.Series(l).rank().values
    if np.std(h_ranks) > 0 and np.std(l_ranks) > 0:
        spearman = float(np.corrcoef(h_ranks, l_ranks)[0, 1])
    else:
        spearman = 1.0 if h_mean == l_mean else 0.0

    # Binned agreement rate (e.g. Pass: >= 3.5, Fail: < 3.5)
    h_bin = (h >= pass_threshold).astype(int)
    l_bin = (l >= pass_threshold).astype(int)
    agreement_rate = float(np.mean(h_bin == l_bin))

    # Quadratic weighted Cohen's kappa
    h_rounded = np.clip(np.round(h), 1, 5).astype(int)
    l_rounded = np.clip(np.round(l), 1, 5).astype(int)
    kappa = compute_weighted_cohen_kappa(h_rounded, l_rounded, min_rating=1, max_rating=5, weight_type="quadratic")

    return {
        "dimension": dimension_name,
        "human_mean": round(h_mean, 2),
        "llm_mean": round(l_mean, 2),
        "pearson_correlation": round(pearson, 4),
        "spearman_correlation": round(spearman, 4),
        "agreement_rate": round(agreement_rate, 4),
        "weighted_kappa": round(kappa, 4),
        "sample_count": n,
        "status": "COMPUTED"
    }


def analyze_human_llm_agreement(
    human_csv_path: Optional[str] = None,
    judge_csv_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    demo_mock_human: bool = False
) -> Dict[str, Any]:
    """
    Merges human ratings from reply_review.csv with LLM Judge ratings from reply_llm_judge.csv
    and computes statistical agreement across all 5 dimensions + overall score.

    If human ratings are blank and demo_mock_human is False:
    Saves an honest status report stating HUMAN_REVIEW_REQUIRED.
    """
    if human_csv_path is None:
        human_csv_path = os.path.join(PROJECT_ROOT, "evaluation", "reply_review.csv")
    if judge_csv_path is None:
        judge_csv_path = os.path.join(PROJECT_ROOT, "evaluation", "results", "reply_llm_judge.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(PROJECT_ROOT, "evaluation", "results", "human_llm_agreement.csv")

    dimensions = ["relevance", "groundedness", "helpfulness", "completeness", "safety", "overall"]

    if not os.path.exists(judge_csv_path):
        # Generate judge results first if missing
        from evaluation.llm_judge import run_judge_cli
        run_judge_cli(mode="mock", limit=30)

    judge_df = pd.read_csv(judge_csv_path)

    # Check for demo mode (synthetic human ratings for formula validation)
    if demo_mock_human:
        print("\n[DEMO MODE] Generating synthetic human ratings for agreement formula verification...")
        rng = np.random.RandomState(2026)
        matched_records = []
        for _, row in judge_df.iterrows():
            rec = {
                "tweet_id": row["tweet_id"],
                "relevance_human": np.clip(float(row["relevance_score"]) + rng.choice([-0.5, 0.0, 0.5], p=[0.2, 0.6, 0.2]), 1.0, 5.0),
                "groundedness_human": np.clip(float(row["groundedness_score"]) + rng.choice([-0.5, 0.0, 0.5], p=[0.2, 0.6, 0.2]), 1.0, 5.0),
                "helpfulness_human": np.clip(float(row["helpfulness_score"]) + rng.choice([-0.5, 0.0, 0.5], p=[0.2, 0.6, 0.2]), 1.0, 5.0),
                "completeness_human": np.clip(float(row["completeness_score"]) + rng.choice([-0.5, 0.0, 0.5], p=[0.2, 0.6, 0.2]), 1.0, 5.0),
                "safety_human": np.clip(float(row["safety_score"]) + rng.choice([-0.5, 0.0], p=[0.1, 0.9]), 1.0, 5.0),
                "overall_human": 0.0,
                # LLM ratings
                "relevance_llm": float(row["relevance_score"]),
                "groundedness_llm": float(row["groundedness_score"]),
                "helpfulness_llm": float(row["helpfulness_score"]),
                "completeness_llm": float(row["completeness_score"]),
                "safety_llm": float(row["safety_score"]),
                "overall_llm": float(row["overall_score"])
            }
            rec["overall_human"] = np.mean([rec[f"{d}_human"] for d in ["relevance", "groundedness", "helpfulness", "completeness", "safety"]])
            matched_records.append(rec)
        merged_df = pd.DataFrame(matched_records)

    else:
        # Load real human evaluation template
        if not os.path.exists(human_csv_path):
            print(f"Human review template not found at {human_csv_path}.")
            return {"status": "HUMAN_REVIEW_REQUIRED"}

        human_df = pd.read_csv(human_csv_path)

        # Check if human scores are actually populated
        valid_rows = []
        for _, h_row in human_df.iterrows():
            val = str(h_row.get("relevance", "")).strip()
            if val and val != "nan" and val != "":
                try:
                    float(val)
                    valid_rows.append(h_row)
                except ValueError:
                    pass

        if len(valid_rows) < 5:
            # Human labels are absent / incomplete: maintain absolute honesty!
            print("\n==================================================================")
            print("HUMAN VS. LLM JUDGE AGREEMENT ANALYSIS")
            print("==================================================================")
            print("STATUS: HUMAN_REVIEW_REQUIRED")
            print(f"File: {human_csv_path}")
            print(f"Found {len(valid_rows)} completed human reviews. Minimum required: 5.")
            print("To complete human evaluation, annotate ratings in reply_review.csv")
            print("following evaluation/HUMAN_REPLY_LABELING_GUIDELINES.txt.")
            print("==================================================================\n")

            pending_records = []
            for dim in dimensions:
                llm_col = f"{dim}_score" if f"{dim}_score" in judge_df.columns else "overall_score"
                llm_mean = round(float(judge_df[llm_col].mean()), 2) if llm_col in judge_df.columns else 0.0
                pending_records.append({
                    "dimension": dim,
                    "human_mean": "Awaiting_Human",
                    "llm_mean": llm_mean,
                    "correlation": "HUMAN_REVIEW_REQUIRED",
                    "agreement_rate": "HUMAN_REVIEW_REQUIRED",
                    "kappa": "HUMAN_REVIEW_REQUIRED",
                    "sample_count": 0,
                    "status": "HUMAN_REVIEW_REQUIRED"
                })

            out_df = pd.DataFrame(pending_records)
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            out_df.to_csv(output_csv_path, index=False)
            return {
                "status": "HUMAN_REVIEW_REQUIRED",
                "completed_human_samples": len(valid_rows),
                "summary_table": out_df
            }

        # Human rows are present: match with judge rows on tweet_id
        valid_df = pd.DataFrame(valid_rows)
        merged_df = pd.merge(valid_df, judge_df, on="tweet_id", suffixes=("_human", "_llm"))

    # Compute agreement metrics across dimensions
    agreement_records = []
    for dim in dimensions:
        h_col = f"{dim}_human" if f"{dim}_human" in merged_df.columns else dim
        l_col = f"{dim}_llm" if f"{dim}_llm" in merged_df.columns else f"{dim}_score"

        h_vals = pd.to_numeric(merged_df[h_col], errors="coerce").dropna().values
        l_vals = pd.to_numeric(merged_df[l_col], errors="coerce").dropna().values

        min_len = min(len(h_vals), len(l_vals))
        h_vals = h_vals[:min_len]
        l_vals = l_vals[:min_len]

        res = compute_agreement_metrics(h_vals, l_vals, dimension_name=dim)
        agreement_records.append(res)

    results_df = pd.DataFrame(agreement_records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    print("\n==================================================================")
    print(f"HUMAN VS. LLM JUDGE AGREEMENT METRICS ({'DEMO / SYNTHETIC' if demo_mock_human else 'VERIFIED HUMAN'})")
    print("==================================================================")
    print(results_df.to_string(index=False))
    print("------------------------------------------------------------------")
    print("Why Quadratic Weighted Cohen's Kappa is Appropriate:")
    print("• Ordinal scale 1-5 has natural ordering: a rating of 4 vs 5 is a minor nuance,")
    print("  whereas 1 vs 5 is a catastrophic discrepancy.")
    print("• Quadratic weighting assigns penalty proportional to (i - j)^2, directly capturing")
    print("  clinical and enterprise evaluation standards.")
    print(f"Results saved to: {output_csv_path}")
    print("==================================================================\n")

    return {
        "status": "COMPUTED" if not demo_mock_human else "DEMO_COMPUTED",
        "results": results_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AssistIQ Human vs LLM Agreement Analysis")
    parser.add_argument("--demo-mock-human", action="store_true", help="Run with synthetic human ratings for mathematical formula verification")
    args = parser.parse_args()
    analyze_human_llm_agreement(demo_mock_human=args.demo_mock_human)
