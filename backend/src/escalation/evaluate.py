"""
AssistIQ: Phase 4 Escalation Policy Evaluation Harness & Metrics
Evaluates deterministic escalation policy on representative golden set queries.
Generates evaluation/escalation_review.csv for human grading (with human columns left blank),
computes safety metrics (unsafe auto-handle rate, missed auto-handle rate), and provides
threshold trade-off analysis.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Ensure project root in sys.path
def find_project_root() -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(current, "dataset", "golden_set.csv")):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(".")

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.escalation.models import EscalationDecision, EscalationPolicyConfig
from backend.src.escalation.policy import EscalationPolicy, decide_escalation
from backend.src.generation.generate import assist_customer
from backend.src.generation.llm import MockLLMClient, get_llm_client
from backend.src.generation.config import GenerationConfig
from backend.src.intent.data import load_and_split_data, find_dataset_path


HUMAN_LABELING_GUIDELINES = """
================================================================================
ASSISTIQ PHASE 4: HUMAN EVALUATION GUIDELINES FOR AUTO-HANDLE VS ESCALATE
================================================================================
The reviewer must evaluate each query and candidate AI response against the question:
"Would it be safe for an AI support agent to send this response without human review?"

CRITERIA FOR 'auto_handle':
1. Request is clearly understood and intent is unambiguous.
2. Historical evidence retrieved is directly relevant and sufficient.
3. Response is fully grounded in the retrieved evidence (no hallucinated policies).
4. No account-specific intervention is required (e.g. no manual database changes,
   password resets, refund issuances, or identity verification).
5. Response does not make unsupported promises or guarantees.

CRITERIA FOR 'escalate':
1. Ambiguous, contradictory, or underspecified customer request.
2. Insufficient or weak historical evidence (similarity < 0.45 or 0 matching cases).
3. Sensitive account security issues (hacked accounts, password recovery, stolen access).
4. Financial/transactional actions (duplicate charges, refund requests, payment disputes).
5. Generation or grounding failure in the LLM layer.
6. Any situation where sending an autonomous AI reply risks customer harm or brand liability.

COLUMNS TO COMPLETE IN escalation_review.csv:
- human_decision: "auto_handle" | "escalate"
- human_risk:     "low" | "medium" | "high"
- human_notes:    Specific reason or justification for human decision
================================================================================
"""


def generate_review_dataset(
    output_path: Optional[str] = None,
    num_samples: int = 40,
    random_state: int = 2026,
    use_mock_llm: bool = True
) -> pd.DataFrame:
    """
    Generates the evaluation/escalation_review.csv dataset template with ~40 representative
    queries from golden_set.csv across all 11 intents.
    Human decision columns are left BLANK for actual human grading.
    """
    golden_path = find_dataset_path("golden_set.csv")
    df = pd.read_csv(golden_path)

    # Sample representative subset stratified by intent where possible
    sample_df = df.groupby("intent", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(2, int(num_samples * len(x) / len(df)))), random_state=random_state)
    ).reset_index(drop=True)

    # Ensure exactly or approximately target sample count
    if len(sample_df) > num_samples:
        sample_df = sample_df.sample(num_samples, random_state=random_state).reset_index(drop=True)

    records = []
    llm_client = MockLLMClient() if use_mock_llm else None

    print(f"Running pipeline on {len(sample_df)} representative queries...")
    for idx, row in sample_df.iterrows():
        text = str(row["text"])
        tweet_id = str(row["tweet_id"])

        res = assist_customer(
            customer_message=text,
            top_k=3,
            llm_client=llm_client
        )

        records.append({
            "tweet_id": tweet_id,
            "text": text,
            "predicted_intent": res["predicted_intent"],
            "intent_confidence": res["intent_confidence"],
            "top_retrieval_similarity": round(max([float(c.get("similarity", 0.0)) for c in res["retrieved_cases"]], default=0.0), 4),
            "grounding_status": res["grounding_status"],
            "predicted_decision": res["decision"],
            "predicted_risk": res["risk_level"],
            "predicted_reason": res["escalation_reason"],
            "primary_rule": res["primary_rule"],
            # Human grading columns left blank
            "human_decision": "",
            "human_risk": "",
            "human_notes": ""
        })

    review_df = pd.DataFrame(records)

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "evaluation", "escalation_review.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    review_df.to_csv(output_path, index=False)
    print(f"Saved escalation review template to: {output_path}")

    # Also save the human guidelines
    guidelines_path = os.path.join(PROJECT_ROOT, "evaluation", "HUMAN_LABELING_GUIDELINES.txt")
    with open(guidelines_path, "w", encoding="utf-8") as f:
        f.write(HUMAN_LABELING_GUIDELINES.strip() + "\n")
    print(f"Saved human labeling guidelines to: {guidelines_path}")

    return review_df


def compute_escalation_metrics(review_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates safety and performance metrics comparing predicted decisions against human decisions.
    Requires populated 'human_decision' values in the review dataframe.
    """
    total = len(review_df)
    if total == 0:
        return {"error": "Empty dataset"}

    pred_auto = (review_df["predicted_decision"] == "auto_handle").sum()
    pred_esc = (review_df["predicted_decision"] == "escalate").sum()

    auto_handle_rate = pred_auto / total
    escalation_rate = pred_esc / total

    # Check if human labels are present
    valid_human = review_df["human_decision"].astype(str).str.strip().str.lower()
    has_labels = (valid_human.isin(["auto_handle", "escalate"])).sum()

    if has_labels < 5:
        return {
            "total_samples": total,
            "auto_handle_rate": round(auto_handle_rate, 4),
            "escalation_rate": round(escalation_rate, 4),
            "labeled_human_samples": int(has_labels),
            "status": "Awaiting human review annotations to compute precision/recall/safety metrics."
        }

    # Filter to rows where human label is present
    labeled_df = review_df[valid_human.isin(["auto_handle", "escalate"])].copy()
    labeled_total = len(labeled_df)

    y_pred = labeled_df["predicted_decision"].values
    y_true = labeled_df["human_decision"].values

    # Key safety failures
    # Unsafe auto-handle: Predicted auto_handle, but Human says escalate
    unsafe_auto = ((y_pred == "auto_handle") & (y_true == "escalate")).sum()
    unsafe_auto_handle_rate = unsafe_auto / labeled_total

    # Missed auto-handle (unnecessary escalation): Predicted escalate, but Human says auto_handle
    missed_auto = ((y_pred == "escalate") & (y_true == "auto_handle")).sum()
    missed_auto_handle_rate = missed_auto / labeled_total

    # Metrics for class 'escalate'
    # TP: pred escalate, true escalate
    tp = int(((y_pred == "escalate") & (y_true == "escalate")).sum())
    # FP: pred escalate, true auto_handle
    fp = int(((y_pred == "escalate") & (y_true == "auto_handle")).sum())
    # FN: pred auto_handle, true escalate
    fn = int(((y_pred == "auto_handle") & (y_true == "escalate")).sum())
    # TN: pred auto_handle, true auto_handle
    tn = int(((y_pred == "auto_handle") & (y_true == "auto_handle")).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_samples": total,
        "labeled_samples": labeled_total,
        "auto_handle_rate": round(auto_handle_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "unsafe_auto_handle_rate": round(unsafe_auto_handle_rate, 4),
        "missed_auto_handle_rate": round(missed_auto_handle_rate, 4),
        "confusion_matrix": {
            "true_escalate_pred_escalate (TP)": tp,
            "true_auto_pred_escalate (FP)": fp,
            "true_escalate_pred_auto (FN, Unsafe)": fn,
            "true_auto_pred_auto (TN)": tn
        },
        "escalate_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    }


def analyze_threshold_sensitivity(
    review_df: pd.DataFrame,
    confidence_thresholds: Optional[List[float]] = None,
    similarity_thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Simulates different (min_intent_confidence, min_retrieval_similarity) pairs
    and evaluates the resulting auto-handle rate and policy rule distribution.
    Demonstrates the trade-off between automation rate and risk tolerance.
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.15, 0.20, 0.25, 0.30]
    if similarity_thresholds is None:
        similarity_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]

    results = []

    for conf_th in confidence_thresholds:
        for sim_th in similarity_thresholds:
            cfg = EscalationPolicyConfig(
                min_intent_confidence=conf_th,
                min_retrieval_similarity=sim_th
            )
            policy = EscalationPolicy(config=cfg)

            auto_count = 0
            esc_count = 0

            for _, row in review_df.iterrows():
                dec = policy.evaluate(
                    customer_message=str(row["text"]),
                    predicted_intent=str(row["predicted_intent"]),
                    intent_confidence=float(row["intent_confidence"]),
                    top_retrieval_similarity=float(row["top_retrieval_similarity"]),
                    evidence_count=3,
                    grounding_status=str(row["grounding_status"]),
                    reply_text="Mock grounded reply for sensitivity test."
                )
                if dec.decision == "auto_handle":
                    auto_count += 1
                else:
                    esc_count += 1

            total = len(review_df)
            auto_rate = auto_count / total if total > 0 else 0.0
            results.append({
                "min_intent_confidence": conf_th,
                "min_retrieval_similarity": sim_th,
                "auto_handle_count": auto_count,
                "escalate_count": esc_count,
                "auto_handle_rate": round(auto_rate, 4),
                "escalation_rate": round(1.0 - auto_rate, 4)
            })

    return pd.DataFrame(results)


def main():
    """
    CLI execution runner for Phase 4 evaluation:
    Run via: python -m backend.src.escalation.evaluate
    """
    print("==================================================================")
    print("ASSISTIQ PHASE 4: ESCALATION POLICY EVALUATION HARNESS")
    print("==================================================================")

    # 1. Generate Review Dataset (~40 queries)
    review_df = generate_review_dataset(num_samples=40, random_state=2026, use_mock_llm=True)

    # 2. Compute Baseline Policy Distribution
    metrics = compute_escalation_metrics(review_df)
    print("\n------------------------------------------------------------------")
    print("CURRENT POLICY PREDICTION SUMMARY (N=40)")
    print(f"• Auto-Handle Rate: {metrics['auto_handle_rate'] * 100:.1f}%")
    print(f"• Escalation Rate:  {metrics['escalation_rate'] * 100:.1f}%")
    print(f"• Human Labels:     {metrics.get('status', 'Available')}")
    print("------------------------------------------------------------------")

    # 3. Rule Breakdown
    rule_counts = review_df["primary_rule"].value_counts()
    print("\nPRIMARY RULES TRIGGERED BREAKDOWN:")
    for rule, count in rule_counts.items():
        pct = (count / len(review_df)) * 100
        print(f"  - {rule:18s}: {count:2d} ({pct:5.1f}%)")

    # 4. Threshold Sensitivity Simulation
    print("\n------------------------------------------------------------------")
    print("THRESHOLD SENSITIVITY GRID SIMULATION")
    print("------------------------------------------------------------------")
    sensitivity_df = analyze_threshold_sensitivity(review_df)
    # Pivot for clean visualization
    pivot_table = sensitivity_df.pivot(
        index="min_intent_confidence",
        columns="min_retrieval_similarity",
        values="auto_handle_rate"
    )
    print("Auto-Handle Rate Grid (Intent Confidence vs Retrieval Similarity):")
    print(pivot_table.to_string())
    print("\nNote: Threshold tuning is explicitly documented as initial policy.")
    print("Empirical threshold optimization is deferred until human review annotations exist.")
    print("==================================================================")


if __name__ == "__main__":
    main()
