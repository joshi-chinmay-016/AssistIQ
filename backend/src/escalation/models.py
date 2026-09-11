"""
AssistIQ: Escalation Policy Data Models & Configuration
Defines structured Pydantic schemas for deterministic auto-handle vs escalate decisions.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class EscalationPolicyConfig(BaseModel):
    """
    Configurable parameters and safety thresholds for the escalation policy.
    Avoids magic numbers and allows tuning based on empirical human reviews.
    """
    min_intent_confidence: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum softmax probability required to consider intent confident (11 classes, random baseline ~0.091)."
    )
    min_retrieval_similarity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity required for historical retrieval grounding (aligns with Phase 3 threshold)."
    )
    min_evidence_count: int = Field(
        default=1,
        ge=1,
        description="Minimum number of relevant historical cases required for auto-handling."
    )
    strict_account_security: bool = Field(
        default=True,
        description="Whether to strictly escalate sensitive account security/credentials/access queries."
    )
    strict_billing_actions: bool = Field(
        default=True,
        description="Whether to strictly escalate transactional billing actions (refunds, duplicate charges, disputes)."
    )
    auto_handle_greetings: bool = Field(
        default=True,
        description="Whether simple polite greetings and acknowledgments may be auto-handled if safe."
    )
    auto_handle_feature_requests: bool = Field(
        default=True,
        description="Whether straightforward feature suggestions may be auto-handled if grounded."
    )


class EscalationDecision(BaseModel):
    """
    Deterministic escalation decision returned for each customer-support interaction.
    Provides complete transparency, risk assessment, and machine-readable audit signals.
    """
    decision: Literal["auto_handle", "escalate"] = Field(
        ...,
        description="Final action: 'auto_handle' to send reply autonomously, or 'escalate' to route to human agent."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Estimated operational risk level for the interaction."
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of why this decision was reached."
    )
    intent: str = Field(
        ...,
        description="Predicted intent taxonomy label from Phase 1."
    )
    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized confidence signal (e.g. softmax probability over classes)."
    )
    top_retrieval_similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Highest cosine similarity score among retrieved historical cases."
    )
    evidence_count: int = Field(
        ...,
        ge=0,
        description="Number of usable historical cases retrieved."
    )
    grounding_status: Literal["grounded", "insufficient_evidence", "generation_failed"] = Field(
        ...,
        description="Phase 3 grounded generation verification status."
    )
    policy_rules_triggered: List[str] = Field(
        default_factory=list,
        description="All rule IDs that matched the interaction signals."
    )
    primary_rule: str = Field(
        ...,
        description="The winning priority rule ID that determined the final action."
    )

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Escalation reason cannot be empty.")
        return v.strip()
