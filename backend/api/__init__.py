"""
AssistIQ: API Module (Phase 5)
Exposes FastAPI application and data contracts.
"""

from backend.api.main import app
from backend.api.schemas import AssistRequest, AssistResponse, HealthResponse, RootInfoResponse

__all__ = [
    "app",
    "AssistRequest",
    "AssistResponse",
    "HealthResponse",
    "RootInfoResponse"
]
