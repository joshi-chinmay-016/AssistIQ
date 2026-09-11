"""
AssistIQ: FastAPI Public API Request & Response Schemas
Defines strongly-typed Pydantic contracts for frontend and external client interactions.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AssistRequest(BaseModel):
    """
    Incoming customer support query payload.
    """
    message: str = Field(
        ...,
        description="The customer's incoming support question or tweet.",
        example="I was charged twice for Spotify Premium this month. Can I get a refund?"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of historical support cases to retrieve as grounding evidence."
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Message must be a string.")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message cannot be empty or whitespace-only.")
        if len(trimmed) > 2000:
            raise ValueError("Message exceeds maximum length of 2000 characters.")
        return trimmed


class IntentInfo(BaseModel):
    """
    Classified customer intent details from Phase 1.
    """
    name: str = Field(..., description="Predicted intent taxonomy category label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence signal (softmax probability over classes).")


class ReplyInfo(BaseModel):
    """
    Grounded LLM drafted support reply from Phase 3.
    """
    text: str = Field(..., description="Drafted customer-facing support reply.")
    grounding_status: Literal["grounded", "insufficient_evidence", "generation_failed"] = Field(
        ...,
        description="Verification status indicating whether reply is grounded in historical evidence."
    )
    grounding_summary: str = Field(..., description="Internal grounding explanation or safety note.")
    evidence_case_ids: List[str] = Field(
        default_factory=list,
        description="Historical case IDs cited as supporting evidence."
    )


class DecisionInfo(BaseModel):
    """
    Deterministic escalation decision details from Phase 4.
    """
    decision: Literal["auto_handle", "escalate"] = Field(
        ...,
        description="Final action: 'auto_handle' to dispatch autonomously, or 'escalate' to route to human agent."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Operational risk assessment level."
    )
    reason: str = Field(..., description="Human-readable explanation of why this decision was reached.")
    primary_rule: str = Field(..., description="Winning policy rule ID that determined the action.")
    policy_rules_triggered: List[str] = Field(
        default_factory=list,
        description="All rule IDs that matched the interaction signals."
    )


class EvidenceCase(BaseModel):
    """
    A retrieved historical support case from Phase 2 vector search.
    """
    case_id: str = Field(..., description="Unique case identifier in the historical knowledge base.")
    similarity: float = Field(..., description="Cosine similarity score relative to customer message.")
    customer_text: str = Field(..., description="Historical customer support inquiry text.")
    support_text: str = Field(..., description="Historical official Spotify support resolution text.")
    rank: Optional[int] = Field(default=None, description="Retrieval ranking position (1-indexed).")
    conversation_id: Optional[int] = Field(default=None, description="Root Twitter conversation ID.")


class LatencyInfo(BaseModel):
    """
    Granular execution latencies across all pipeline stages in milliseconds.
    """
    intent_ms: float = Field(..., description="Phase 1 Intent classification latency (ms).")
    retrieval_ms: float = Field(..., description="Phase 2 FAISS vector search latency (ms).")
    generation_ms: float = Field(..., description="Phase 3 LLM generation latency (ms).")
    escalation_ms: float = Field(..., description="Phase 4 Escalation policy latency (ms).")
    total_ms: float = Field(..., description="Total end-to-end pipeline latency (ms).")


class AssistResponse(BaseModel):
    """
    Complete structured response returned by the AssistIQ API.
    """
    message: str = Field(..., description="Sanitized customer input message.")
    intent: IntentInfo = Field(..., description="Classified intent context.")
    reply: ReplyInfo = Field(..., description="Drafted reply and grounding verification.")
    decision: DecisionInfo = Field(..., description="Deterministic escalation decision.")
    evidence: List[EvidenceCase] = Field(..., description="Retrieved historical support evidence cases.")
    latency: LatencyInfo = Field(..., description="Execution timing breakdown in milliseconds.")


class HealthResponse(BaseModel):
    """
    Service health check response schema.
    """
    status: str = Field(default="ok", description="Overall health status.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the check.")
    version: str = Field(default="0.5.0", description="AssistIQ API version.")


class RootInfoResponse(BaseModel):
    """
    Root API service metadata response schema.
    """
    name: str = Field(default="AssistIQ API", description="Service name.")
    version: str = Field(default="0.5.0", description="Service version.")
    status: str = Field(default="ok", description="Operational status.")
    brand: str = Field(default="SpotifyCares", description="Target support brand.")
    docs_url: str = Field(default="/docs", description="Interactive API documentation URL.")
