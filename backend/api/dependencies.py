"""
AssistIQ: API Dependencies & Configuration Management
Provides centralized settings, environment parsing, and dependency injection helpers.
"""

import os
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

from backend.src.generation.generate import assist_customer
from backend.src.generation.config import GenerationConfig
from backend.src.escalation.models import EscalationPolicyConfig
from backend.src.generation.llm import BaseLLMClient


class ApiConfig:
    """
    Central configuration for the FastAPI service.
    """
    def __init__(self):
        self.host: str = os.getenv("ASSISTIQ_API_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("ASSISTIQ_API_PORT", "8000"))
        
        # Parse comma-separated CORS origins
        raw_origins = os.getenv(
            "ASSISTIQ_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000"
        )
        self.cors_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]
        
        # Generation config from environment
        self.generation_config: GenerationConfig = GenerationConfig.from_env()
        self.escalation_config: EscalationPolicyConfig = EscalationPolicyConfig()


_GLOBAL_API_CONFIG = ApiConfig()


def get_api_config() -> ApiConfig:
    """
    Returns the singleton API configuration instance.
    """
    return _GLOBAL_API_CONFIG


def get_pipeline_runner() -> Callable[..., Dict[str, Any]]:
    """
    Dependency returning the AssistIQ customer assist execution callable.
    Allows clean dependency overriding in unit tests.
    """
    return assist_customer
