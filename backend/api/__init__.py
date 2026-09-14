import os
import sys

# Ensure repository root is in sys.path so 'from backend...' works
# even when uvicorn is executed from within the backend directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.api.main import app
from backend.api.schemas import AssistRequest, AssistResponse, HealthResponse, RootInfoResponse

__all__ = [
    "app",
    "AssistRequest",
    "AssistResponse",
    "HealthResponse",
    "RootInfoResponse"
]
