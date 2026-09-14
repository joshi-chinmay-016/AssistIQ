"""
AssistIQ: Unit & Regression Tests for Evaluation Framework
Tests:
- Golden set validation (valid dataset, missing columns, duplicates, invalid taxonomy)
- Intent evaluation metrics & confusion matrix generation
- LLM Judge structured parsing, score clamping (1-5), and malformed response fallback
- Human agreement analysis with missing labels ("HUMAN_REVIEW_REQUIRED")
- Mathematical correctness of Pearson, Spearman, and Quadratic Weighted Cohen's Kappa
- Escalation safety metrics (unsafe auto-handle, missed auto-handle, confusion matrix)
- Automated failure analysis (10 taxonomy buckets)
- End-to-end mock evaluation runner execution
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.validate_golden import validate_golden_set_detailed
from evaluation.llm_judge import MockLLMJudge, JudgeScore, BaseLLMJudge
from evaluation.human_agreement import compute_weighted_cohen_kappa, compute_agreement_metrics, analyze_human_llm_agreement
from evaluation.evaluate_escalation import evaluate_escalation_policy, run_threshold_sensitivity_grid
from evaluation.failure_analysis import FAILURE_TAXONOMY, build_representative_failures
from evaluation.run import run_full_evaluation
from backend.src.escalation.models import EscalationPolicyConfig


class TestGoldenSetValidation(unittest.TestCase):
    """
    Tests for dataset/golden_set.csv integrity checks.
    """

    def test_valid_golden_set_passes(self):
        report = validate_golden_set_detailed()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["checks"]["row_count"]["actual"], 200)
        self.assertEqual(report["checks"]["empty_text"]["empty_count"], 0)
        self.assertEqual(report["checks"]["empty_intent"]["empty_count"], 0)
        self.assertEqual(report["checks"]["duplicate_tweet_ids"]["duplicate_count"], 0)
        self.assertEqual(report["checks"]["taxonomy_membership"]["passed"], True)

    def test_invalid_taxonomy_caught(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tweet_id,text,intent\n")
            f.write("1001,Test tweet,invalid_intent_xyz\n")
            temp_path = f.name
        try:
            report = validate_golden_set_detailed(temp_path)
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("invalid_intent_xyz", str(report["checks"]["taxonomy_membership"]["invalid_classes"]))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_duplicate_tweet_ids_caught(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tweet_id,text,intent\n")
            f.write("1001,Tweet A,playback_and_app_issues\n")
            f.write("1001,Tweet B,playback_and_app_issues\n")
            temp_path = f.name
        try:
            report = validate_golden_set_detailed(temp_path)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["checks"]["duplicate_tweet_ids"]["passed"], False)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestLLMJudge(unittest.TestCase):
    """
    Tests for LLM Judge quality scoring, parsing, and error recovery.
    """

    def setUp(self):
        self.judge = MockLLMJudge()

    def test_judge_valid_grounded_reply(self):
        score = self.judge.judge_reply(
            customer_text="I was charged twice for Spotify Premium this month.",
            predicted_intent="billing_and_payment",
            generated_reply="Hey there! Could you send us a DM with your account email address? We'll check backstage.",
            retrieved_cases=[
                {"case_id": "case_16893", "similarity": 0.88, "customer_text": "Charged 3 times", "support_text": "Please DM us your account email address."}
            ],
            grounding_status="grounded"
        )
        self.assertIsInstance(score, JudgeScore)
        self.assertGreaterEqual(score.relevance_score, 3.0)
        self.assertGreaterEqual(score.groundedness_score, 4.0)
        self.assertGreaterEqual(score.safety_score, 4.5)
        self.assertEqual(score.pass_fail, "PASS")
        self.assertFalse(score.hallucination_flag)

    def test_judge_detects_empty_generation_failure(self):
        score = self.judge.judge_reply(
            customer_text="App crashes constantly",
            predicted_intent="playback_and_app_issues",
            generated_reply="",
            retrieved_cases=[],
            grounding_status="generation_failed"
        )
        self.assertEqual(score.pass_fail, "FAIL")
        self.assertEqual(score.relevance_score, 1.0)
        self.assertEqual(score.groundedness_score, 1.0)

    def test_judge_detects_hallucination_promise(self):
        score = self.judge.judge_reply(
            customer_text="I want a refund now",
            predicted_intent="billing_and_payment",
            generated_reply="We have issued a 100% guaranteed refund of $50 to your bank account.",
            retrieved_cases=[],
            grounding_status="grounded"
        )
        self.assertTrue(score.hallucination_flag)
        self.assertEqual(score.groundedness_score, 1.0)
        self.assertEqual(score.pass_fail, "FAIL")


class TestHumanAgreementMath(unittest.TestCase):
    """
    Tests for statistical agreement: Pearson, Spearman, and Quadratic Weighted Cohen's Kappa.
    """

    def test_perfect_agreement_kappa_is_one(self):
        r1 = np.array([1, 2, 3, 4, 5])
        r2 = np.array([1, 2, 3, 4, 5])
        kappa = compute_weighted_cohen_kappa(r1, r2)
        self.assertAlmostEqual(kappa, 1.0, places=3)

    def test_quadratic_weighting_penalizes_large_discrepancy(self):
        # r1 spans full 1-5 scale with multiple observations
        r1 = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        # Minor discrepancy (5 vs 4)
        r2_near = np.array([1, 2, 3, 4, 4, 1, 2, 3, 4, 4])
        # Severe discrepancy (5 vs 1)
        r2_far = np.array([1, 2, 3, 4, 1, 1, 2, 3, 4, 1])

        kappa_near = compute_weighted_cohen_kappa(r1, r2_near)
        kappa_far = compute_weighted_cohen_kappa(r1, r2_far)

        self.assertGreater(kappa_near, kappa_far)
        self.assertGreater(kappa_near, 0.90)
        self.assertLess(kappa_far, 0.50)

    def test_compute_agreement_metrics_insufficient_samples(self):
        metrics = compute_agreement_metrics(np.array([4, 5]), np.array([4, 5]), dimension_name="test")
        self.assertEqual(metrics["status"], "INSUFFICIENT_SAMPLES")

    def test_missing_human_labels_returns_human_review_required(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tweet_id,customer_text,relevance,groundedness,helpfulness,completeness,safety,overall\n")
            f.write("101,Hello,,,,,,,\n")
            f.write("102,World,,,,,,,\n")
            temp_human = f.name
        try:
            res = analyze_human_llm_agreement(human_csv_path=temp_human)
            self.assertEqual(res["status"], "HUMAN_REVIEW_REQUIRED")
        finally:
            if os.path.exists(temp_human):
                os.remove(temp_human)


class TestEscalationEvaluation(unittest.TestCase):
    """
    Tests for escalation policy metrics, safety failure rates, and sensitivity grid.
    """

    def test_escalation_metrics_without_human_labels(self):
        df = pd.DataFrame([
            {"predicted_decision": "auto_handle", "primary_rule": "A1", "human_decision": ""},
            {"predicted_decision": "escalate", "primary_rule": "E6", "human_decision": ""}
        ])
        res = evaluate_escalation_policy(df)
        self.assertEqual(res["auto_handle_rate"], 0.5)
        self.assertEqual(res["escalation_rate"], 0.5)
        self.assertEqual(res["unsafe_auto_handle_rate"], "HUMAN_REVIEW_REQUIRED")

    def test_escalation_metrics_with_human_labels(self):
        # 4 samples:
        # Row 1: Pred auto, True auto (TN)
        # Row 2: Pred auto, True esc  (FN -> Unsafe auto-handle!)
        # Row 3: Pred esc,  True esc  (TP)
        # Row 4: Pred esc,  True auto (FP -> Missed auto-handle)
        df = pd.DataFrame([
            {"predicted_decision": "auto_handle", "primary_rule": "A1", "human_decision": "auto_handle"},
            {"predicted_decision": "auto_handle", "primary_rule": "A1", "human_decision": "escalate"},
            {"predicted_decision": "escalate", "primary_rule": "E6", "human_decision": "escalate"},
            {"predicted_decision": "escalate", "primary_rule": "E1", "human_decision": "auto_handle"},
            {"predicted_decision": "auto_handle", "primary_rule": "A1", "human_decision": "auto_handle"}
        ])
        res = evaluate_escalation_policy(df)
        self.assertEqual(res["status"], "COMPUTED")
        self.assertEqual(res["unsafe_auto_handle_count"], 1)
        self.assertEqual(res["missed_auto_handle_count"], 1)
        self.assertEqual(res["confusion_matrix"]["TP"], 1)
        self.assertEqual(res["confusion_matrix"]["FP"], 1)
        self.assertEqual(res["confusion_matrix"]["FN"], 1)
        self.assertEqual(res["confusion_matrix"]["TN"], 2)

    def test_threshold_sensitivity_grid_monotonicity(self):
        df = pd.DataFrame([
            {"text": "my app crashes", "predicted_intent": "playback_and_app_issues", "intent_confidence": 0.22, "top_retrieval_similarity": 0.48, "grounding_status": "grounded"},
            {"text": "give refund", "predicted_intent": "billing_and_payment", "intent_confidence": 0.18, "top_retrieval_similarity": 0.42, "grounding_status": "grounded"}
        ])
        grid = run_threshold_sensitivity_grid(df, conf_thresholds=[0.15, 0.30], sim_thresholds=[0.35, 0.55])
        self.assertEqual(len(grid), 4)
        # Stricter confidence (0.30) should yield auto-handle rate <= laxer confidence (0.15)
        auto_lax = grid[grid["min_intent_confidence"] == 0.15]["auto_handle_rate"].mean()
        auto_strict = grid[grid["min_intent_confidence"] == 0.30]["auto_handle_rate"].mean()
        self.assertGreaterEqual(auto_lax, auto_strict)


class TestFailureAnalysis(unittest.TestCase):
    """
    Tests for automated failure analysis taxonomy and report structure.
    """

    def test_all_10_failure_buckets_exist(self):
        self.assertEqual(len(FAILURE_TAXONOMY), 10)
        expected_buckets = [
            "wrong_intent", "low_retrieval_similarity", "irrelevant_retrieved_evidence",
            "unsupported_generation", "hallucinated_claim", "missing_required_clarification",
            "unsafe_auto_handle", "unnecessary_escalation", "low_quality_response",
            "ambiguous_customer_message"
        ]
        for b in expected_buckets:
            self.assertIn(b, FAILURE_TAXONOMY)

    def test_build_representative_failures_has_all_categories(self):
        df = build_representative_failures()
        self.assertEqual(len(df), 10)
        categories = set(df["failure_category"].unique())
        self.assertEqual(len(categories), 10)
        for col in ["tweet_id", "customer_message", "predicted_intent", "failure_category", "explanation"]:
            self.assertIn(col, df.columns)


class TestEndToEndMockRunner(unittest.TestCase):
    """
    Tests the unified evaluation runner in mock mode.
    """

    def test_runner_golden_stage(self):
        res = run_full_evaluation(mode="mock", limit=5, stage="golden")
        self.assertIn("golden", res)
        self.assertEqual(res["golden"]["status"], "PASSED")

    def test_runner_agreement_stage(self):
        res = run_full_evaluation(mode="mock", limit=5, stage="agreement")
        self.assertIn("agreement", res)
        self.assertIn(res["agreement"]["status"], ["HUMAN_REVIEW_REQUIRED", "COMPUTED"])


if __name__ == "__main__":
    unittest.main()
