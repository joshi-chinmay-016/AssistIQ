"""
AssistIQ: Escalation Policy Evaluation & Sensitivity Analysis Harness
Evaluates the deterministic escalation policy (E0–E8, A1) on representative golden queries.
Calculates safety metrics (unsafe auto-handle rate, missed auto-handle rate, escalation precision/recall),
performs 2D threshold trade-off sensitivity analysis across confidence and similarity thresholds,
and produces machine-readable evaluation artifacts.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
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

from backend.src.escalation.models import EscalationDecision, EscalationPolicyConfig
from backend.src.escalation.policy import EscalationPolicy
from backend.src.generation.generate import assist_customer
from backend.src.generation.llm import MockLLMClient
from backend.src.intent.data import find_dataset_path


def evaluate_escalation_policy(
    review_df: pd.DataFrame,
    config: Optional[EscalationPolicyConfig] = None
) -> Dict[str, Any]:
    """
    Evaluates policy decisions and compares them against human ground-truth if available.
    """
    policy = EscalationPolicy(config=config or EscalationPolicyConfig())
    total = len(review_df)

    pred_auto = (review_df["predicted_decision"] == "auto_handle").sum()
    pred_esc = (review_df["predicted_decision"] == "escalate").sum()

    auto_handle_rate = round(float(pred_auto / total), 4) if total > 0 else 0.0
    escalation_rate = round(float(pred_esc / total), 4) if total > 0 else 0.0

    # Rule distribution
    rule_counts = review_df["primary_rule"].value_counts().to_dict()

    # Check for human labels
    valid_human = review_df["human_decision"].astype(str).str.strip().str.lower()
    has_labels = int(valid_human.isin(["auto_handle", "escalate"]).sum())

    res: Dict[str, Any] = {
        "total_samples": total,
        "auto_handle_count": int(pred_auto),
        "escalate_count": int(pred_esc),
        "auto_handle_rate": auto_handle_rate,
        "escalation_rate": escalation_rate,
        "primary_rule_distribution": rule_counts,
        "labeled_human_samples": has_labels
    }

    if has_labels < 5:
        res["status"] = "Awaiting human review annotations in escalation_review.csv"
        res["unsafe_auto_handle_rate"] = "HUMAN_REVIEW_REQUIRED"
        res["missed_auto_handle_rate"] = "HUMAN_REVIEW_REQUIRED"
        res["escalate_precision"] = "HUMAN_REVIEW_REQUIRED"
        res["escalate_recall"] = "HUMAN_REVIEW_REQUIRED"
        res["escalate_f1"] = "HUMAN_REVIEW_REQUIRED"
        return res

    labeled_df = review_df[valid_human.isin(["auto_handle", "escalate"])].copy()
    y_pred = labeled_df["predicted_decision"].values
    y_true = labeled_df["human_decision"].values
    labeled_total = len(labeled_df)

    # Safety-critical failures
    # Unsafe auto-handle: Predicted auto_handle, Human says escalate (CRITICAL)
    unsafe_auto = int(((y_pred == "auto_handle") & (y_true == "escalate")).sum())
    # Missed auto-handle: Predicted escalate, Human says auto_handle (Efficiency loss)
    missed_auto = int(((y_pred == "escalate") & (y_true == "auto_handle")).sum())

    tp = int(((y_pred == "escalate") & (y_true == "escalate")).sum())
    fp = int(((y_pred == "escalate") & (y_true == "auto_handle")).sum())
    fn = int(((y_pred == "auto_handle") & (y_true == "escalate")).sum())
    tn = int(((y_pred == "auto_handle") & (y_true == "auto_handle")).sum())

    prec = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
    rec = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
    f1 = round(float(2 * prec * rec / (prec + rec)), 4) if (prec + rec) > 0 else 0.0

    res.update({
        "status": "COMPUTED",
        "unsafe_auto_handle_count": unsafe_auto,
        "unsafe_auto_handle_rate": round(float(unsafe_auto / labeled_total), 4),
        "missed_auto_handle_count": missed_auto,
        "missed_auto_handle_rate": round(float(missed_auto / labeled_total), 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "escalate_precision": prec,
        "escalate_recall": rec,
        "escalate_f1": f1
    })

    return res


def run_threshold_sensitivity_grid(
    review_df: pd.DataFrame,
    conf_thresholds: Optional[List[float]] = None,
    sim_thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Simulates policy decisions across a 2D parameter grid:
    - min_intent_confidence: [0.15, 0.20, 0.25, 0.30]
    - min_retrieval_similarity: [0.35, 0.40, 0.45, 0.50, 0.55]

    Demonstrates the trade-off between automation rate and risk posture.
    """
    if conf_thresholds is None:
        conf_thresholds = [0.15, 0.20, 0.25, 0.30]
    if sim_thresholds is None:
        sim_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]

    records = []
    total = len(review_df)

    for conf_th in conf_thresholds:
        for sim_th in sim_thresholds:
            cfg = EscalationPolicyConfig(
                min_intent_confidence=conf_th,
                min_retrieval_similarity=sim_th,
                strict_account_security=True,
                strict_billing_actions=True,
                auto_handle_greetings=True,
                auto_handle_feature_requests=True
            )
            policy = EscalationPolicy(config=cfg)

            auto_count = 0
            esc_count = 0
            e1_count = 0
            e2_count = 0

            for _, row in review_df.iterrows():
                dec = policy.evaluate(
                    customer_message=str(row.get("text", "")),
                    predicted_intent=str(row.get("predicted_intent", "")),
                    intent_confidence=float(row.get("intent_confidence", 0.30)),
                    top_retrieval_similarity=float(row.get("top_retrieval_similarity", 0.50)),
                    evidence_count=1,
                    grounding_status=str(row.get("grounding_status", "grounded")),
                    reply_text="Representative candidate reply"
                )
                if dec.decision == "auto_handle":
                    auto_count += 1
                else:
                    esc_count += 1
                if dec.primary_rule == "E1":
                    e1_count += 1
                elif dec.primary_rule == "E2":
                    e2_count += 1

            records.append({
                "min_intent_confidence": conf_th,
                "min_retrieval_similarity": sim_th,
                "auto_handle_rate": round(auto_count / total, 4) if total > 0 else 0.0,
                "escalation_rate": round(esc_count / total, 4) if total > 0 else 0.0,
                "e1_low_confidence_escalations": e1_count,
                "e2_weak_retrieval_escalations": e2_count,
                "is_default_threshold": (conf_th == 0.20 and sim_th == 0.45)
            })

    return pd.DataFrame(records)


def run_escalation_evaluation(
    limit: int = 40,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs end-to-end escalation evaluation on representative sample:
    1. Loads or generates escalation_review.csv.
    2. Computes baseline escalation metrics.
    3. Runs 2D threshold sensitivity grid.
    4. Saves escalation_metrics.csv and threshold_sensitivity.csv.
    """
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)

    review_path = os.path.join(PROJECT_ROOT, "evaluation", "escalation_review.csv")

    if not os.path.exists(review_path):
        # Generate representative sample
        golden_path = find_dataset_path("golden_set.csv")
        df = pd.read_csv(golden_path).head(limit)
        records = []
        llm = MockLLMClient()
        print(f"Generating representative escalation records for {len(df)} queries...")
        for _, row in df.iterrows():
            text = str(row["text"])
            res = assist_customer(text, top_k=3, llm_client=llm)
            sim = max([float(c.get("similarity", 0.0)) for c in res["retrieved_cases"]], default=0.0)
            records.append({
                "tweet_id": row["tweet_id"],
                "text": text,
                "predicted_intent": res["predicted_intent"],
                "intent_confidence": res["intent_confidence"],
                "top_retrieval_similarity": round(sim, 4),
                "grounding_status": res["grounding_status"],
                "predicted_decision": res["decision"],
                "predicted_risk": res["risk_level"],
                "predicted_reason": res["escalation_reason"],
                "primary_rule": res["primary_rule"],
                "human_decision": "",
                "human_risk": "",
                "human_notes": ""
            })
        review_df = pd.DataFrame(records)
        review_df.to_csv(review_path, index=False)
    else:
        review_df = pd.read_csv(review_path)

    # 1. Compute metrics under default policy config
    default_config = EscalationPolicyConfig(
        min_intent_confidence=0.20,
        min_retrieval_similarity=0.45,
        strict_account_security=True,
        strict_billing_actions=True,
        auto_handle_greetings=True,
        auto_handle_feature_requests=True
    )
    metrics = evaluate_escalation_policy(review_df, config=default_config)

    metrics_records = [
        {"metric": "total_evaluated_queries", "value": metrics["total_samples"], "description": "Number of representative queries in evaluation set"},
        {"metric": "auto_handle_rate", "value": metrics["auto_handle_rate"], "description": "Fraction of queries auto-handled under default policy"},
        {"metric": "escalation_rate", "value": metrics["escalation_rate"], "description": "Fraction of queries escalated to human agents"},
        {"metric": "default_min_retrieval_similarity", "value": 0.45, "description": "Cosine similarity cutoff for auto-handle eligibility (initial engineering heuristic)"},
        {"metric": "default_min_intent_confidence", "value": 0.20, "description": "Margin-derived softmax confidence cutoff (uncalibrated LinearSVC score)"},
        {"metric": "unsafe_auto_handle_rate", "value": metrics.get("unsafe_auto_handle_rate", "HUMAN_REVIEW_REQUIRED"), "description": "Rate of unsafe auto-handling (False Negatives for escalation)"},
        {"metric": "missed_auto_handle_rate", "value": metrics.get("missed_auto_handle_rate", "HUMAN_REVIEW_REQUIRED"), "description": "Rate of unnecessary escalation (False Positives for escalation)"},
        {"metric": "labeled_human_samples", "value": metrics["labeled_human_samples"], "description": "Count of human-verified escalation annotations"}
    ]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = os.path.join(results_dir, "escalation_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # 2. Compute 2D threshold sensitivity grid
    grid_df = run_threshold_sensitivity_grid(review_df)
    grid_csv = os.path.join(results_dir, "threshold_sensitivity.csv")
    grid_df.to_csv(grid_csv, index=False)

    # Pivot table for display
    pivot_auto = grid_df.pivot(
        index="min_intent_confidence",
        columns="min_retrieval_similarity",
        values="auto_handle_rate"
    )

    print("\n==================================================================")
    print("ESCALATION POLICY EVALUATION SUMMARY")
    print("==================================================================")
    print(f"Total Evaluated Queries:    {metrics['total_samples']}")
    print(f"Auto-Handle Rate (Default): {metrics['auto_handle_rate'] * 100:.1f}%")
    print(f"Escalation Rate (Default):  {metrics['escalation_rate'] * 100:.1f}%")
    print(f"Human Annotations:          {metrics['labeled_human_samples']} verified")
    print("------------------------------------------------------------------")
    print("Primary Policy Rule Breakdown:")
    for rule, count in sorted(metrics["primary_rule_distribution"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / metrics["total_samples"]) * 100
        print(f"  - Rule {rule:18s}: {count:2d} ({pct:5.1f}%)")
    print("------------------------------------------------------------------")
    print("THRESHOLD SENSITIVITY MATRIX (Auto-Handle Rate):")
    print(pivot_auto.to_string())
    print("------------------------------------------------------------------")
    print("Key Engineering Insights:")
    print("• Unsafe auto-handling carries far greater business risk than unnecessary escalation.")
    print("• The 0.45 similarity threshold is an initial engineering heuristic, not a scientifically optimal value.")
    print("• LinearSVC confidence is a margin-derived softmax score, not a calibrated probability.")
    print(f"[OK] Saved metrics to:     {metrics_csv}")
    print(f"[OK] Saved sensitivity to: {grid_csv}")
    print("==================================================================\n")

    return {
        "metrics": metrics_df,
        "sensitivity": grid_df,
        "pivot": pivot_auto
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AssistIQ Escalation Policy Evaluation")
    parser.add_argument("--limit", type=int, default=40, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_escalation_evaluation(limit=args.limit)
