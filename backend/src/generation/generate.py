"""
AssistIQ: Grounded Generation Pipeline & End-to-End Orchestrator
Provides generate_support_reply() for grounding responses on retrieved evidence
and assist_customer() for the complete Phase 1 -> Phase 2 -> Phase 3 pipeline.
"""

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from backend.src.generation.config import GenerationConfig
from backend.src.generation.prompt import SYSTEM_INSTRUCTION, build_generation_prompt
from backend.src.generation.llm import SupportReply, BaseLLMClient, get_llm_client
from backend.src.retrieval.retrieve import get_retriever
from backend.src.intent.data import load_and_split_data, find_dataset_path
from backend.src.intent.models import build_proposed_model

# Global singleton cache for models
_GLOBAL_CLASSIFIER = None
_GLOBAL_LLM_CLIENT = None


def get_cached_classifier():
    """
    Lazily trains and caches the Phase 1 LinearSVC intent classifier once.
    """
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        golden_path = find_dataset_path("golden_set.csv")
        X_train, _, y_train, _, _, _ = load_and_split_data(golden_path)
        clf = build_proposed_model(random_state=2026)
        clf.fit(X_train, y_train)
        _GLOBAL_CLASSIFIER = clf
    return _GLOBAL_CLASSIFIER


def generate_support_reply(
    customer_message: str,
    predicted_intent: str,
    retrieved_cases: List[Dict[str, Any]],
    config: Optional[GenerationConfig] = None,
    llm_client: Optional[BaseLLMClient] = None
) -> SupportReply:
    """
    Generates a concise, grounded customer support reply based on the customer message,
    predicted intent context, and retrieved historical Spotify support cases.

    Args:
        customer_message (str): The customer's incoming support question.
        predicted_intent (str): The intent classified by Phase 1.
        retrieved_cases (List[Dict[str, Any]]): Top-K historical cases from Phase 2 FAISS retrieval.
        config (Optional[GenerationConfig]): Configuration overrides.
        llm_client (Optional[BaseLLMClient]): Injected LLM client (e.g. for testing).

    Returns:
        SupportReply: Validated structured reply with grounding metadata.
    """
    # 1. Input Validation
    if not customer_message or not str(customer_message).strip():
        return SupportReply(
            reply="Hey there! Could you tell us a bit more about what you need help with?",
            grounding_summary="Customer message was empty or whitespace.",
            evidence_case_ids=[],
            grounding_status="insufficient_evidence"
        )

    # 2. Safety & Grounding Guard: Detect empty or extremely weak retrieval evidence
    valid_cases = [c for c in retrieved_cases if isinstance(c, dict) and "case_id" in c]
    highest_sim = max([float(c.get("similarity", 0.0)) for c in valid_cases], default=0.0)

    if not valid_cases:
        return SupportReply(
            reply=(
                "Hey there! We'd be glad to look into this for you. "
                "Could you send us a quick DM with your account details and what device you're on? "
                "We'll see what we can do to help!"
            ),
            grounding_summary="No historical support cases were retrieved; safely requested account details via DM.",
            evidence_case_ids=[],
            grounding_status="insufficient_evidence"
        )

    # 3. Build Prompt with Grounding Constraints and Injection Defense
    top_k = config.top_k if config else 5
    user_prompt = build_generation_prompt(
        customer_message=customer_message,
        predicted_intent=predicted_intent,
        retrieved_cases=valid_cases,
        max_cases=top_k
    )

    # 4. Resolve LLM Client and Invoke Model
    try:
        client = llm_client or get_llm_client(config)
        reply_obj = client.generate_reply(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt=user_prompt
        )
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        return SupportReply(
            reply="We are currently experiencing technical difficulties retrieving support guidance. Your request has been queued for human support review.",
            grounding_summary=f"Generation failed: {str(e)}",
            evidence_case_ids=[],
            grounding_status="generation_failed"
        )

    # 6. Post-Generation Grounding Verification
    retrieved_case_ids = {str(c["case_id"]) for c in valid_cases}
    # Scrub hallucinated case IDs that were not in the retrieved evidence
    verified_citations = [cid for cid in reply_obj.evidence_case_ids if cid in retrieved_case_ids]
    reply_obj.evidence_case_ids = verified_citations

    # Enforce grounding status integrity
    if not reply_obj.evidence_case_ids and reply_obj.grounding_status == "grounded":
        reply_obj.grounding_status = "insufficient_evidence"

    # Flag low-confidence retrievals
    if highest_sim < 0.45 and reply_obj.grounding_status == "grounded":
        reply_obj.grounding_status = "insufficient_evidence"
        reply_obj.grounding_summary += f" (Note: Highest retrieval similarity was low at {highest_sim:.4f})."

    return reply_obj


def assist_customer(
    customer_message: str,
    top_k: int = 5,
    config: Optional[GenerationConfig] = None,
    llm_client: Optional[BaseLLMClient] = None
) -> Dict[str, Any]:
    """
    End-to-End local inference pipeline executing:
    Customer message -> Phase 1 Intent -> Phase 2 Retrieval -> Phase 3 LLM Generation.
    """
    t_start = time.perf_counter()

    # Step 1: Phase 1 Intent Classification
    t_intent_start = time.perf_counter()
    classifier = get_cached_classifier()
    predicted_intent = str(classifier.predict([customer_message])[0])
    intent_ms = (time.perf_counter() - t_intent_start) * 1000.0

    # Step 2: Phase 2 Historical Support Retrieval
    t_retrieval_start = time.perf_counter()
    retriever = get_retriever()
    retrieved_cases = retriever.search(customer_message, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # Step 3: Phase 3 Grounded Reply Generation
    t_gen_start = time.perf_counter()
    support_reply = generate_support_reply(
        customer_message=customer_message,
        predicted_intent=predicted_intent,
        retrieved_cases=retrieved_cases,
        config=config,
        llm_client=llm_client
    )
    gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "customer_message": customer_message,
        "predicted_intent": predicted_intent,
        "retrieved_cases": retrieved_cases,
        "support_reply": support_reply,
        "reply": support_reply.reply,
        "grounding_summary": support_reply.grounding_summary,
        "evidence_case_ids": support_reply.evidence_case_ids,
        "grounding_status": support_reply.grounding_status,
        "latency": {
            "intent_ms": round(intent_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round(total_ms, 2)
        },
        "latency_ms": {
            "intent_ms": round(intent_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round(total_ms, 2)
        }
    }


def main():
    """
    Demonstration CLI entry point:
    Run via: python -m backend.src.generation.generate
    """
    print("==================================================================")
    print("ASSISTIQ END-TO-END ASSISTANT DEMO (PHASES 1 -> 2 -> 3)")
    print("==================================================================")

    config = GenerationConfig.from_env()
    # If no key is set yet, gracefully notify and use mock for demonstration
    if not config.has_valid_api_key():
        print("Notice: No live GEMINI_API_KEY detected in backend/.env.")
        print("Using MockLLMClient for demonstration.\n")
        config.provider = "mock"

    test_queries = [
        "I was charged twice for Spotify Premium this month. Can I get a refund?",
        "Music keeps pausing on my iPhone whenever the screen turns off",
        "How do I recover my hacked Spotify account and reset my password?"
    ]

    for q in test_queries:
        print(f"\n------------------------------------------------------------------")
        print(f"CUSTOMER: '{q}'")
        res = assist_customer(q, top_k=3, config=config)
        print(f"• PREDICTED INTENT:    {res['predicted_intent']}")
        print(f"• TOP HISTORICAL CASE: {res['retrieved_cases'][0]['case_id']} (Sim: {res['retrieved_cases'][0]['similarity']:.4f})")
        print(f"• GROUNDING STATUS:    {res['grounding_status']}")
        print(f"• CITED EVIDENCE:      {res['evidence_case_ids']}")
        print(f"• DRAFTED REPLY:\n\"{res['reply']}\"")
        print(f"• INTERNAL SUMMARY:    {res['grounding_summary']}")
        print(f"• LATENCY:             Total: {res['latency']['total_ms']:.1f}ms (Intent: {res['latency']['intent_ms']:.1f}ms, Retrieval: {res['latency']['retrieval_ms']:.1f}ms, LLM: {res['latency']['generation_ms']:.1f}ms)")

    print("==================================================================")


if __name__ == "__main__":
    main()
