"""
AssistIQ: Deterministic Escalation Policy Engine
Implements transparent, reproducible rules (E0-E8, A1) to decide AUTO_HANDLE vs ESCALATE.
Ensures Gemini never makes the final escalation decision.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

from backend.src.escalation.models import EscalationDecision, EscalationPolicyConfig

# Keyword sets for sensitive and action-requiring categories
SENSITIVE_ACCOUNT_KEYWORDS = {
    "hack", "hacked", "hacking",
    "compromise", "compromised",
    "password", "passwords",
    "stolen", "hijack", "hijacked",
    "unauthorized", "unauthorised",
    "locked out", "lockout", "locked",
    "breach", "breached",
    "credentials", "credential",
    "recover", "recovery",
    "stolen account", "reset password",
    "cant log in", "can't log in", "cannot log in",
    "disabled account", "suspended"
}

BILLING_ACTION_KEYWORDS = {
    "refund", "refunds", "refunding",
    "double charge", "charged twice", "charge twice", "two charges",
    "dispute", "disputed", "disputing",
    "overcharge", "overcharged",
    "bank", "bank statement",
    "unauthorized charge", "fraudulent",
    "money back", "billing error",
    "charged again", "cancel subscription",
    "deducted twice", "deducted"
}

GREETING_PATTERNS = [
    r"^(hi|hello|hey|hey there|good morning|good afternoon|good evening|greetings)\b",
    r"^(thanks|thank you|thx|ty|appreciate it|cheers)\b",
    r"^(ok|okay|got it|cool|sounds good|understood)\b"
]


def _match_keywords(text: str, keywords: set) -> Optional[str]:
    """
    Finds the first matching keyword or phrase in normalized text.
    """
    lower_text = text.lower()
    for kw in sorted(keywords, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
            return kw
    return None


def _is_simple_greeting(text: str) -> bool:
    """
    Checks if a message is purely or predominantly a greeting, thanks, or acknowledgment.
    """
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    words = cleaned.split()
    if not words:
        return False

    # Short messages (<= 6 words) matching common greeting/thanks patterns
    for pat in GREETING_PATTERNS:
        if re.search(pat, cleaned):
            if len(words) <= 8:
                return True
    return False


class EscalationPolicy:
    """
    Deterministic rule-based policy engine for customer support escalation.
    Evaluates signals from Phase 1 (Intent), Phase 2 (Retrieval), and Phase 3 (Generation)
    in strict priority order.
    """

    def __init__(self, config: Optional[EscalationPolicyConfig] = None):
        self.config = config or EscalationPolicyConfig()

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        top_retrieval_similarity: float,
        evidence_count: int,
        grounding_status: str,
        reply_text: Optional[str] = None
    ) -> EscalationDecision:
        """
        Evaluates support signals against deterministic safety rules in explicit priority order.

        Priority Order:
        1. E0: Empty / Invalid input
        2. E3: Generation failure
        3. E4: Insufficient grounding
        4. E2: Weak retrieval evidence (< min_retrieval_similarity or < min_evidence_count)
        5. E1: Low intent confidence (< min_intent_confidence)
        6. E5: Sensitive account/security credentials issue (when strict)
        7. E6: Billing / refund action requests requiring human agent (when strict)
        8. E7_AMBIGUOUS: Ambiguous or unclear request
        9. Auto-handle rules:
           - E7_GREETING: Polite greeting / acknowledgement
           - E8_FEATURE_REQUEST: Grounded feature suggestion
           - A1: Safe grounded auto-handle

        Returns:
            EscalationDecision: Fully populated, validated Pydantic decision object.
        """
        rules_triggered: List[str] = []
        normalized_msg = str(customer_message or "").strip()
        normalized_reply = str(reply_text or "").strip()

        # Clamp and sanitize numerical signals
        sim = float(top_retrieval_similarity)
        conf = float(intent_confidence)
        count = int(evidence_count)
        intent = str(predicted_intent)

        # ------------------------------------------------------------------
        # Rule Detection (Gather all matching rule candidates)
        # ------------------------------------------------------------------

        # Rule E0: Empty or invalid input
        if not normalized_msg or len(normalized_msg) < 2:
            rules_triggered.append("E0")

        # Rule E3: Generation failure
        if grounding_status == "generation_failed" or not normalized_reply:
            rules_triggered.append("E3")

        # Rule E4: Insufficient Grounding
        if grounding_status == "insufficient_evidence":
            rules_triggered.append("E4")

        # Rule E2: Insufficient Retrieval Evidence
        if sim < self.config.min_retrieval_similarity or count < self.config.min_evidence_count:
            rules_triggered.append("E2")

        # Rule E1: Low Intent Confidence
        if conf < self.config.min_intent_confidence:
            rules_triggered.append("E1")

        # Rule E5: Sensitive Account & Security Issues
        account_kw = _match_keywords(normalized_msg, SENSITIVE_ACCOUNT_KEYWORDS)
        if intent == "account_and_login" and account_kw and self.config.strict_account_security:
            rules_triggered.append("E5")

        # Rule E6: Transactional Billing Actions
        billing_kw = _match_keywords(normalized_msg, BILLING_ACTION_KEYWORDS)
        if intent == "billing_and_payment" and billing_kw and self.config.strict_billing_actions:
            rules_triggered.append("E6")

        # Rule E7: Other / Non-actionable Handling
        is_greeting = False
        if intent == "other_non_actionable":
            if self.config.auto_handle_greetings and _is_simple_greeting(normalized_msg):
                is_greeting = True
                rules_triggered.append("E7_GREETING")
            else:
                rules_triggered.append("E7_AMBIGUOUS")

        # Rule E8: Feature Requests
        if intent == "feature_requests" and self.config.auto_handle_feature_requests:
            rules_triggered.append("E8_FEATURE_REQUEST")

        # ------------------------------------------------------------------
        # Deterministic Priority Selection
        # ------------------------------------------------------------------

        # Priority 1: E0 (Empty / Invalid input)
        if "E0" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="high",
                reason="Escalated because the customer message is empty, whitespace, or invalid.",
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E0"
            )

        # Priority 2: E3 (Generation Failure)
        if "E3" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="high",
                reason="Escalated because response generation failed or could not produce a valid reply.",
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E3"
            )

        # Priority 3: E4 (Insufficient Grounding)
        if "E4" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="medium",
                reason="Escalated because the response lacks sufficient grounding in retrieved historical cases.",
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E4"
            )

        # Priority 4: E2 (Insufficient Retrieval Evidence)
        if "E2" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="medium",
                reason=(
                    f"Escalated because historical evidence is insufficient: top retrieval similarity is "
                    f"{sim:.2f}, below the configured threshold of {self.config.min_retrieval_similarity:.2f} "
                    f"(usable cases: {count})."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E2"
            )

        # Priority 5: E1 (Low Intent Confidence)
        if "E1" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="medium",
                reason=(
                    f"Escalated because intent confidence is below the configured threshold "
                    f"({conf:.2f} < {self.config.min_intent_confidence:.2f}), so the system cannot "
                    f"reliably determine the customer's issue."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E1"
            )

        # Priority 6: E5 (Sensitive Account & Security Issues)
        if "E5" in rules_triggered:
            matched_kw_str = f" ('{account_kw}')" if account_kw else ""
            return EscalationDecision(
                decision="escalate",
                risk_level="high",
                reason=(
                    f"Escalated because sensitive account or security credentials/access issue{matched_kw_str} "
                    f"requires secure human verification."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E5"
            )

        # Priority 7: E6 (Billing Actions / Refunds)
        if "E6" in rules_triggered:
            matched_kw_str = f" ('{billing_kw}')" if billing_kw else ""
            return EscalationDecision(
                decision="escalate",
                risk_level="high",
                reason=(
                    f"Escalated because financial transactions, refund requests, or disputed charges{matched_kw_str} "
                    f"require authorized human account investigation."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E6"
            )

        # Priority 8: E7_AMBIGUOUS (Ambiguous or non-actionable other message)
        if "E7_AMBIGUOUS" in rules_triggered:
            return EscalationDecision(
                decision="escalate",
                risk_level="medium",
                reason=(
                    "Escalated because the request is ambiguous, non-actionable, or lacks sufficient issue "
                    "details to resolve safely."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E7"
            )

        # Priority 9: Auto-Handle Rules
        if "E7_GREETING" in rules_triggered:
            return EscalationDecision(
                decision="auto_handle",
                risk_level="low",
                reason=(
                    "Auto-handled because the message is a polite greeting or acknowledgment that does not "
                    "require customer support intervention."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E7_GREETING"
            )

        if "E8_FEATURE_REQUEST" in rules_triggered:
            return EscalationDecision(
                decision="auto_handle",
                risk_level="low",
                reason=(
                    "Auto-handled because the customer is submitting a straightforward feature suggestion "
                    "and received a grounded informational response."
                ),
                intent=intent,
                intent_confidence=conf,
                top_retrieval_similarity=sim,
                evidence_count=count,
                grounding_status=grounding_status,
                policy_rules_triggered=rules_triggered,
                primary_rule="E8"
            )

        # Default Auto-Handle (Rule A1)
        rules_triggered.append("A1")
        return EscalationDecision(
            decision="auto_handle",
            risk_level="low",
            reason=(
                f"Auto-handled because the intent is confident ('{intent}' @ {conf:.2f}), "
                f"relevant historical cases were retrieved (top similarity {sim:.2f}), "
                f"and the generated reply is grounded in the retrieved evidence."
            ),
            intent=intent,
            intent_confidence=conf,
            top_retrieval_similarity=sim,
            evidence_count=count,
            grounding_status=grounding_status,
            policy_rules_triggered=rules_triggered,
            primary_rule="A1"
        )


def decide_escalation(
    customer_message: str,
    predicted_intent: str,
    intent_confidence: float,
    top_retrieval_similarity: float,
    evidence_count: int,
    grounding_status: str,
    reply_text: Optional[str] = None,
    config: Optional[EscalationPolicyConfig] = None
) -> EscalationDecision:
    """
    Functional convenience wrapper for EscalationPolicy.evaluate().
    """
    policy = EscalationPolicy(config=config)
    return policy.evaluate(
        customer_message=customer_message,
        predicted_intent=predicted_intent,
        intent_confidence=intent_confidence,
        top_retrieval_similarity=top_retrieval_similarity,
        evidence_count=evidence_count,
        grounding_status=grounding_status,
        reply_text=reply_text
    )
