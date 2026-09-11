"""
AssistIQ: Generation Package
Phase 3 Grounded LLM Reply Generation.
"""

from backend.src.generation.config import GenerationConfig
from backend.src.generation.prompt import (
    SYSTEM_INSTRUCTION,
    format_evidence_block,
    build_generation_prompt
)
from backend.src.generation.llm import (
    SupportReply,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    get_llm_client
)
from backend.src.generation.generate import (
    generate_support_reply,
    assist_customer
)

__all__ = [
    "GenerationConfig",
    "SYSTEM_INSTRUCTION",
    "format_evidence_block",
    "build_generation_prompt",
    "SupportReply",
    "BaseLLMClient",
    "GeminiLLMClient",
    "MockLLMClient",
    "get_llm_client",
    "generate_support_reply",
    "assist_customer"
]
