"""
AssistIQ: Unit & Integration Tests for Phase 4 Escalation Policy
Tests deterministic rules, priority order, boundary conditions, metric calculations,
and end-to-end mocked pipeline execution.
"""

import unittest
import pandas as pd
from pydantic import ValidationError

from backend.src.escalation.models import EscalationDecision, EscalationPolicyConfig
from backend.src.escalation.policy import EscalationPolicy, decide_escalation
from backend.src.escalation.evaluate import compute_escalation_metrics, analyze_threshold_sensitivity
from backend.src.generation.generate import assist_customer
from backend.src.generation.llm import MockLLMClient


class TestEscalationPolicy(unittest.TestCase):
    """
    Test suite for EscalationPolicy deterministic rule engine.
    """

    def setUp(self):
        self.config = EscalationPolicyConfig(
            min_intent_confidence=0.20,
            min_retrieval_similarity=0.45,
            min_evidence_count=1,
            strict_account_security=True,
            strict_billing_actions=True,
            auto_handle_greetings=True,
            auto_handle_feature_requests=True
        )
        self.policy = EscalationPolicy(config=self.config)

    # 1. High-confidence + strong evidence -> AUTO_HANDLE
    def test_01_high_confidence_strong_evidence_auto_handle(self):
        decision = self.policy.evaluate(
            customer_message="Can I control playback from my Apple Watch?",
            predicted_intent="playback_and_app_issues",
            intent_confidence=0.35,
            top_retrieval_similarity=0.78,
            evidence_count=3,
            grounding_status="grounded",
            reply_text="Yes, you can control playback directly using the Spotify app on your Apple Watch."
        )
        self.assertEqual(decision.decision, "auto_handle")
        self.assertEqual(decision.risk_level, "low")
        self.assertEqual(decision.primary_rule, "A1")
        self.assertIn("Auto-handled", decision.reason)

    # 2. Low intent confidence -> ESCALATE (E1)
    def test_02_low_intent_confidence_escalates(self):
        decision = self.policy.evaluate(
            customer_message="something is strange here with my app",
            predicted_intent="playback_and_app_issues",
            intent_confidence=0.14,  # Below 0.20
            top_retrieval_similarity=0.75,
            evidence_count=3,
            grounding_status="grounded",
            reply_text="Can you restart your app?"
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "medium")
        self.assertEqual(decision.primary_rule, "E1")
        self.assertIn("intent confidence is below the configured threshold", decision.reason)

    # 3. Low retrieval similarity -> ESCALATE (E2)
    def test_03_low_retrieval_similarity_escalates(self):
        decision = self.policy.evaluate(
            customer_message="How do I use Spotify on my toaster?",
            predicted_intent="playback_and_app_issues",
            intent_confidence=0.30,
            top_retrieval_similarity=0.32,  # Below 0.45
            evidence_count=3,
            grounding_status="grounded",
            reply_text="We do not officially support smart toasters."
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "medium")
        self.assertEqual(decision.primary_rule, "E2")
        self.assertIn("historical evidence is insufficient", decision.reason)

    # 4. Generation failure -> ESCALATE (E3)
    def test_04_generation_failure_escalates(self):
        decision = self.policy.evaluate(
            customer_message="Why does my song skip?",
            predicted_intent="playback_and_app_issues",
            intent_confidence=0.32,
            top_retrieval_similarity=0.72,
            evidence_count=3,
            grounding_status="generation_failed",
            reply_text=""
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "high")
        self.assertEqual(decision.primary_rule, "E3")
        self.assertIn("response generation failed", decision.reason)

    # 5. Ungrounded response -> ESCALATE (E4)
    def test_05_ungrounded_response_escalates(self):
        decision = self.policy.evaluate(
            customer_message="Can I get lossless audio now?",
            predicted_intent="premium_and_subscription",
            intent_confidence=0.28,
            top_retrieval_similarity=0.65,
            evidence_count=2,
            grounding_status="insufficient_evidence",
            reply_text="We don't have enough details."
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "medium")
        self.assertEqual(decision.primary_rule, "E4")
        self.assertIn("lacks sufficient grounding", decision.reason)

    # 6. Sensitive account/security issue -> ESCALATE (E5)
    def test_06_sensitive_account_security_escalates(self):
        decision = self.policy.evaluate(
            customer_message="Someone hacked my Spotify account and changed my password",
            predicted_intent="account_and_login",
            intent_confidence=0.35,
            top_retrieval_similarity=0.82,
            evidence_count=3,
            grounding_status="grounded",
            reply_text="Please reach out to our team to recover your account."
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "high")
        self.assertEqual(decision.primary_rule, "E5")
        self.assertIn("sensitive account or security credentials/access issue", decision.reason)

    # 7. Billing/refund account-specific issue -> ESCALATE (E6)
    def test_07_billing_refund_action_escalates(self):
        decision = self.policy.evaluate(
            customer_message="I was charged twice for Premium, I need a refund immediately",
            predicted_intent="billing_and_payment",
            intent_confidence=0.32,
            top_retrieval_similarity=0.85,
            evidence_count=3,
            grounding_status="grounded",
            reply_text="We will check your billing history."
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "high")
        self.assertEqual(decision.primary_rule, "E6")
        self.assertIn("financial transactions, refund requests, or disputed charges", decision.reason)

    # 8. Simple feature request + strong evidence -> AUTO_HANDLE (E8)
    def test_08_feature_request_auto_handles(self):
        decision = self.policy.evaluate(
            customer_message="Please add swipe to queue on Android Spotify app",
            predicted_intent="feature_requests",
            intent_confidence=0.28,
            top_retrieval_similarity=0.74,
            evidence_count=2,
            grounding_status="grounded",
            reply_text="Thanks for the suggestion! We'll pass this feature request to our product team."
        )
        self.assertEqual(decision.decision, "auto_handle")
        self.assertEqual(decision.risk_level, "low")
        self.assertEqual(decision.primary_rule, "E8")
        self.assertIn("straightforward feature suggestion", decision.reason)

    # 9. Simple greeting/thanks -> AUTO_HANDLE (E7_GREETING)
    def test_09_simple_greeting_auto_handles(self):
        decision = self.policy.evaluate(
            customer_message="Hi Spotify team, good morning!",
            predicted_intent="other_non_actionable",
            intent_confidence=0.25,
            top_retrieval_similarity=0.60,
            evidence_count=1,
            grounding_status="grounded",
            reply_text="Good morning! How can we help you with Spotify today?"
        )
        self.assertEqual(decision.decision, "auto_handle")
        self.assertEqual(decision.risk_level, "low")
        self.assertEqual(decision.primary_rule, "E7_GREETING")
        self.assertIn("polite greeting or acknowledgment", decision.reason)

    # 10. Ambiguous/unclear request -> ESCALATE (E7_AMBIGUOUS)
    def test_10_ambiguous_request_escalates(self):
        decision = self.policy.evaluate(
            customer_message="ugh this stupid thing why does it do this to me again",
            predicted_intent="other_non_actionable",
            intent_confidence=0.25,
            top_retrieval_similarity=0.55,
            evidence_count=2,
            grounding_status="grounded",
            reply_text="Could you clarify what is happening?"
        )
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "medium")
        self.assertEqual(decision.primary_rule, "E7")
        self.assertIn("ambiguous, non-actionable, or lacks sufficient issue details", decision.reason)

    # 11. Multiple rules trigger -> deterministic priority
    def test_11_multiple_rules_priority_order(self):
        # Case: Generation failed (E3) AND Low confidence (E1) AND Sensitive keyword (E5)
        decision = self.policy.evaluate(
            customer_message="my password was stolen and I can't log in",
            predicted_intent="account_and_login",
            intent_confidence=0.12,  # E1 triggers
            top_retrieval_similarity=0.25,  # E2 triggers
            evidence_count=0,
            grounding_status="generation_failed",  # E3 triggers
            reply_text=""
        )
        # E3 has higher priority than E2, E1, E5
        self.assertEqual(decision.primary_rule, "E3")
        self.assertEqual(decision.decision, "escalate")
        self.assertEqual(decision.risk_level, "high")
        # All triggered rules must be recorded
        self.assertIn("E3", decision.policy_rules_triggered)
        self.assertIn("E2", decision.policy_rules_triggered)
        self.assertIn("E1", decision.policy_rules_triggered)
        self.assertIn("E5", decision.policy_rules_triggered)

    # 12. Exact boundary values for thresholds
    def test_12_threshold_boundaries(self):
        # Boundary on intent_confidence (0.20 threshold)
        # 0.2000 -> passes E1
        d_pass = self.policy.evaluate(
            customer_message="How do I make a playlist?",
            predicted_intent="playlist_and_library",
            intent_confidence=0.2000,
            top_retrieval_similarity=0.4500,
            evidence_count=1,
            grounding_status="grounded",
            reply_text="Tap Create Playlist."
        )
        self.assertEqual(d_pass.decision, "auto_handle")

        # 0.1999 -> fails E1
        d_fail = self.policy.evaluate(
            customer_message="How do I make a playlist?",
            predicted_intent="playlist_and_library",
            intent_confidence=0.1999,
            top_retrieval_similarity=0.4500,
            evidence_count=1,
            grounding_status="grounded",
            reply_text="Tap Create Playlist."
        )
        self.assertEqual(d_fail.decision, "escalate")
        self.assertEqual(d_fail.primary_rule, "E1")

        # Boundary on retrieval_similarity (0.45 threshold)
        # 0.4499 -> fails E2
        d_sim_fail = self.policy.evaluate(
            customer_message="How do I make a playlist?",
            predicted_intent="playlist_and_library",
            intent_confidence=0.2500,
            top_retrieval_similarity=0.4499,
            evidence_count=1,
            grounding_status="grounded",
            reply_text="Tap Create Playlist."
        )
        self.assertEqual(d_sim_fail.decision, "escalate")
        self.assertEqual(d_sim_fail.primary_rule, "E2")

    # 13. Invalid/missing signals handled safely (E0)
    def test_13_invalid_or_empty_input_handled_safely(self):
        decision_empty = self.policy.evaluate(
            customer_message="",
            predicted_intent="other_non_actionable",
            intent_confidence=0.0,
            top_retrieval_similarity=0.0,
            evidence_count=0,
            grounding_status="insufficient_evidence",
            reply_text=""
        )
        self.assertEqual(decision_empty.decision, "escalate")
        self.assertEqual(decision_empty.primary_rule, "E0")
        self.assertEqual(decision_empty.risk_level, "high")

    # 14. Pydantic validation
    def test_14_pydantic_validation(self):
        # Invalid decision string
        with self.assertRaises(ValidationError):
            EscalationDecision(
                decision="maybe",  # Not "auto_handle" or "escalate"
                risk_level="low",
                reason="some reason",
                intent="playback_and_app_issues",
                intent_confidence=0.5,
                top_retrieval_similarity=0.7,
                evidence_count=2,
                grounding_status="grounded",
                primary_rule="A1"
            )

        # Empty reason string
        with self.assertRaises(ValidationError):
            EscalationDecision(
                decision="auto_handle",
                risk_level="low",
                reason="   ",  # Whitespace only
                intent="playback_and_app_issues",
                intent_confidence=0.5,
                top_retrieval_similarity=0.7,
                evidence_count=2,
                grounding_status="grounded",
                primary_rule="A1"
            )


class TestEscalationMetrics(unittest.TestCase):
    """
    Test suite for evaluation metrics (unsafe auto-handle, missed auto-handle, precision/recall/F1).
    """

    def test_metrics_calculation(self):
        # Create a mock review dataset with known ground truth
        data = {
            "predicted_decision": ["auto_handle", "auto_handle", "escalate", "escalate", "auto_handle"],
            "human_decision":     ["auto_handle", "escalate",    "escalate", "auto_handle", "auto_handle"]
        }
        df = pd.DataFrame(data)
        metrics = compute_escalation_metrics(df)

        # Total 5 rows
        # Predicted auto_handle: 3 -> 3/5 = 0.60
        # Predicted escalate: 2 -> 2/5 = 0.40
        self.assertEqual(metrics["auto_handle_rate"], 0.60)
        self.assertEqual(metrics["escalation_rate"], 0.40)

        # Row 1: pred auto_handle, human escalate -> UNSAFE AUTO-HANDLE (1/5 = 0.20)
        self.assertEqual(metrics["unsafe_auto_handle_rate"], 0.20)

        # Row 3: pred escalate, human auto_handle -> MISSED AUTO-HANDLE (1/5 = 0.20)
        self.assertEqual(metrics["missed_auto_handle_rate"], 0.20)

        # For Escalate:
        # TP: pred escalate, true escalate = 1 (row 2)
        # FP: pred escalate, true auto = 1 (row 3)
        # FN: pred auto, true escalate = 1 (row 1)
        # Precision: 1 / (1+1) = 0.50
        # Recall: 1 / (1+1) = 0.50
        # F1: 0.50
        self.assertEqual(metrics["escalate_metrics"]["precision"], 0.50)
        self.assertEqual(metrics["escalate_metrics"]["recall"], 0.50)
        self.assertEqual(metrics["escalate_metrics"]["f1"], 0.50)


class TestEndToEndEscalationPipeline(unittest.TestCase):
    """
    End-to-end integration test:
    Customer Message -> Phase 1 -> Phase 2 -> Phase 3 (MockLLM) -> Phase 4 EscalationDecision.
    """

    def test_e2e_pipeline_with_mock_llm(self):
        query = "I was charged twice for Spotify Premium this month."
        result = assist_customer(
            customer_message=query,
            top_k=3,
            llm_client=MockLLMClient()
        )

        # Check that pipeline completed and populated all required keys
        self.assertIn("customer_message", result)
        self.assertIn("predicted_intent", result)
        self.assertIn("intent_confidence", result)
        self.assertIn("retrieved_cases", result)
        self.assertIn("support_reply", result)
        self.assertIn("escalation_decision", result)
        self.assertIn("decision", result)
        self.assertIn("risk_level", result)
        self.assertIn("escalation_reason", result)

        decision_obj = result["escalation_decision"]
        self.assertIsInstance(decision_obj, EscalationDecision)
        self.assertIn(decision_obj.decision, ["auto_handle", "escalate"])
        self.assertIn(decision_obj.risk_level, ["low", "medium", "high"])
        self.assertTrue(len(decision_obj.reason) > 0)
        self.assertTrue(len(decision_obj.primary_rule) > 0)

        # Since this query explicitly mentions "charged twice", Billing action (E6) should trigger
        self.assertIn("E6", decision_obj.policy_rules_triggered)
        self.assertEqual(decision_obj.decision, "escalate")
        self.assertEqual(decision_obj.risk_level, "high")


if __name__ == "__main__":
    unittest.main()
