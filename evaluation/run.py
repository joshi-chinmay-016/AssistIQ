"""
AssistIQ: Unified End-to-End Evaluation Runner
Master evaluation CLI to orchestrate the entire evaluation suite:
- Phase 1: Golden Set Validation
- Phase 2: Intent Classification Evaluation (3 models)
- Phase 3: Historical Retrieval Evaluation (FAISS & Manual Review)
- Phase 4: Grounded Reply Generation & LLM Judge Quality Scoring
- Phase 5: Human vs. LLM Agreement Analysis
- Phase 6: Deterministic Escalation Policy & Threshold Sensitivity
- Phase 7: Automated Failure Categorization (10 buckets)

Supports zero-cost offline mock mode by default, and live Gemini API mode on demand.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd

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

from evaluation.validate_golden import run_validation
from evaluation.evaluate_intent import run_intent_evaluation
from evaluation.evaluate_retrieval import run_retrieval_evaluation
from evaluation.llm_judge import run_judge_cli
from evaluation.human_agreement import analyze_human_llm_agreement
from evaluation.evaluate_escalation import run_escalation_evaluation
from evaluation.failure_analysis import build_representative_failures


def run_full_evaluation(
    mode: str = "mock",
    limit: int = 30,
    stage: str = "all"
) -> Dict[str, Any]:
    """
    Orchestrates the AssistIQ evaluation pipeline.
    """
    t_start = time.perf_counter()

    print("\n==================================================================")
    print("ASSISTIQ UNIFIED EVALUATION SUITE")
    print("==================================================================")
    print(f"Target Brand:       @SpotifyCares (Twitter Customer Support)")
    print(f"Execution Mode:     {mode.upper()} ({'Offline zero-cost mock' if mode == 'mock' else 'Live Gemini API'})")
    print(f"Sample Limit:       {limit} queries")
    print(f"Stage Selected:     {stage}")
    print("==================================================================\n")

    results: Dict[str, Any] = {}

    # Stage 1: Golden Set Validation
    if stage in ["all", "golden"]:
        print("\n>>> STAGE 1: Golden Set Validation...")
        results["golden"] = run_validation()

    # Stage 2: Intent Evaluation
    if stage in ["all", "intent"]:
        print("\n>>> STAGE 2: Intent Classification Evaluation (3 Models)...")
        results["intent"] = run_intent_evaluation()

    # Stage 3: Retrieval Evaluation
    if stage in ["all", "retrieval"]:
        print("\n>>> STAGE 3: Historical Support Retrieval Evaluation...")
        results["retrieval"] = run_retrieval_evaluation(limit=limit)

    # Stage 4: Grounded Reply Generation & LLM Judge
    if stage in ["all", "reply", "judge"]:
        print(f"\n>>> STAGE 4: Grounded Generation & LLM Judge Evaluation (Mode: {mode})...")
        results["judge"] = run_judge_cli(mode=mode, limit=limit)

    # Stage 5: Human vs. LLM Agreement
    if stage in ["all", "agreement"]:
        print("\n>>> STAGE 5: Human vs. LLM Judge Agreement Analysis...")
        results["agreement"] = analyze_human_llm_agreement(demo_mock_human=False)

    # Stage 6: Escalation Policy & Threshold Sensitivity
    if stage in ["all", "escalation"]:
        print("\n>>> STAGE 6: Escalation Policy Evaluation & Sensitivity Analysis...")
        results["escalation"] = run_escalation_evaluation(limit=limit)

    # Stage 7: Failure Analysis
    if stage in ["all", "failures"]:
        print("\n>>> STAGE 7: Failure Analysis & Archetype Categorization...")
        results["failures"] = build_representative_failures()

    elapsed = time.perf_counter() - t_start

    print("\n==================================================================")
    print("ASSISTIQ EVALUATION COMPLETE")
    print("==================================================================")
    print(f"Total Execution Time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f"Artifacts Generated in: {os.path.join(PROJECT_ROOT, 'evaluation', 'results')}")
    print("Artifact Files:")
    print("  - golden_set_validation_report.json (Integrity audit & provenance)")
    print("  - intent_metrics.csv                (Majority, TF-IDF+LR, LinearSVC)")
    print("  - intent_confusion_matrix.csv       (Proposed model 11x11 matrix)")
    print("  - intent_errors.csv                 (Top-3 predictions & margin confidence)")
    print("  - retrieval_metrics.csv             (Hit Rate@1/3/5, consistency, latency)")
    print("  - retrieval_errors.csv              (Low similarity & unmatched cases)")
    print("  - reply_llm_judge.csv               (5-dimension ratings & hallucination flags)")
    print("  - human_llm_agreement.csv           (Pearson, Spearman, Weighted Kappa)")
    print("  - escalation_metrics.csv            (Auto-handle rate, safety failure metrics)")
    print("  - threshold_sensitivity.csv         (2D grid over confidence & similarity)")
    print("  - failure_examples.csv              (10 distinct failure buckets)")
    print("==================================================================\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AssistIQ Master Evaluation Runner")
    parser.add_argument("--stage", choices=["all", "golden", "intent", "retrieval", "reply", "judge", "agreement", "escalation", "failures"], default="all", help="Evaluation stage to execute")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="Execution mode (default: mock)")
    parser.add_argument("--limit", type=int, default=30, help="Query limit for retrieval/generation (default: 30)")
    args = parser.parse_args()

    run_full_evaluation(mode=args.mode, limit=args.limit, stage=args.stage)
