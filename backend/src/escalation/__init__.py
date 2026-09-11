"""
AssistIQ: Escalation Policy Module (Phase 4)
Exports deterministic policy engine, configuration, models, and decision functions.
"""

from backend.src.escalation.models import EscalationDecision, EscalationPolicyConfig
from backend.src.escalation.policy import EscalationPolicy, decide_escalation

__all__ = [
    "EscalationDecision",
    "EscalationPolicyConfig",
    "EscalationPolicy",
    "decide_escalation"
]
