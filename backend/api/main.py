"""
AssistIQ: FastAPI Application Server
Provides HTTP endpoints for health checks and customer support assistance (Phases 1-5).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Callable
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    AssistRequest,
    AssistResponse,
    IntentInfo,
    ReplyInfo,
    DecisionInfo,
    EvidenceCase,
    LatencyInfo,
    HealthResponse,
    RootInfoResponse
)
from backend.api.dependencies import get_api_config, get_pipeline_runner, ApiConfig

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="AssistIQ API",
    description="Deterministic AI Customer Support Agent for @SpotifyCares (Phases 1-5)",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS for local development and frontend integration
config = get_api_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Sanitizes validation errors into clean, readable client messages without internal trace leakage.
    """
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", [])])
        msg = err.get("msg", "Invalid value")
        errors.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
        content={
            "error": "Validation Error",
            "detail": "; ".join(errors) if errors else "Invalid request payload."
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Safe catch-all error handler preventing secret, path, or stack trace exposure.
    """
    logger.exception(f"Unhandled server error while processing {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred while processing your request."
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
    description="Ultra-fast server liveness check. Does not invoke ML models or LLMs."
)
async def health_check() -> HealthResponse:
    """
    Returns server operational status and UTC timestamp.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="0.5.0"
    )


@app.get(
    "/",
    response_model=RootInfoResponse,
    summary="API Root Information",
    tags=["System"],
    description="Returns metadata about the AssistIQ service, active brand, and docs link."
)
async def root_info() -> RootInfoResponse:
    """
    Returns service name, version, and documentation link.
    """
    return RootInfoResponse(
        name="AssistIQ API",
        version="0.5.0",
        status="ok",
        brand="SpotifyCares",
        docs_url="/docs"
    )


@app.post(
    "/api/v1/assist",
    response_model=AssistResponse,
    summary="Assist Customer",
    tags=["Customer Support"],
    description=(
        "Executes the full 4-phase AssistIQ pipeline for an incoming customer question:\n"
        "1. Phase 1: Intent Classification\n"
        "2. Phase 2: Historical Knowledge Retrieval\n"
        "3. Phase 3: Grounded Reply Generation\n"
        "4. Phase 4: Deterministic Auto-handle vs Escalate Policy"
    )
)
async def assist(
    request: AssistRequest,
    pipeline_runner: Callable[..., Dict[str, Any]] = Depends(get_pipeline_runner),
    api_config: ApiConfig = Depends(get_api_config)
) -> AssistResponse:
    """
    Processes a customer message through AssistIQ and returns the complete structured decision.
    """
    try:
        raw_result = pipeline_runner(
            customer_message=request.message,
            top_k=request.top_k or 5,
            config=api_config.generation_config,
            escalation_config=api_config.escalation_config
        )
    except Exception as e:
        logger.error(f"Pipeline execution failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "AssistIQ pipeline failed to process customer message."
            }
        )

    # Transform raw internal dict into strongly-typed API response contract
    evidence_cases = [
        EvidenceCase(
            case_id=str(c.get("case_id", "")),
            similarity=round(float(c.get("similarity", 0.0)), 4),
            customer_text=str(c.get("customer_text", "")),
            support_text=str(c.get("support_text", "")),
            rank=c.get("rank"),
            conversation_id=c.get("conversation_id")
        )
        for c in raw_result.get("retrieved_cases", [])
        if isinstance(c, dict)
    ]

    latencies = raw_result.get("latency", {})

    return AssistResponse(
        message=request.message,
        intent=IntentInfo(
            name=raw_result["predicted_intent"],
            confidence=raw_result["intent_confidence"]
        ),
        reply=ReplyInfo(
            text=raw_result["reply"],
            grounding_status=raw_result["grounding_status"],
            grounding_summary=raw_result["grounding_summary"],
            evidence_case_ids=raw_result["evidence_case_ids"]
        ),
        decision=DecisionInfo(
            decision=raw_result["decision"],
            risk_level=raw_result["risk_level"],
            reason=raw_result["escalation_reason"],
            primary_rule=raw_result["primary_rule"],
            policy_rules_triggered=raw_result["policy_rules_triggered"]
        ),
        evidence=evidence_cases,
        latency=LatencyInfo(
            intent_ms=latencies.get("intent_ms", 0.0),
            retrieval_ms=latencies.get("retrieval_ms", 0.0),
            generation_ms=latencies.get("generation_ms", 0.0),
            escalation_ms=latencies.get("escalation_ms", 0.0),
            total_ms=latencies.get("total_ms", 0.0)
        )
    )
