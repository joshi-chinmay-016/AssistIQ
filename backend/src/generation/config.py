"""
AssistIQ: Generation Configuration Module
Manages environment variables, API keys, and model parameters for Phase 3 LLM generation.
Strictly ensures no API keys are logged, exposed, or committed.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


def find_project_root() -> str:
    """
    Locates the AssistIQ repository root directory.
    """
    if os.path.exists("dataset") and os.path.isdir("dataset"):
        return os.path.abspath(".")
    if os.path.exists(os.path.join("..", "dataset")):
        return os.path.abspath("..")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_environment():
    """
    Searches and loads environment variables from backend/.env and .env.
    """
    root = find_project_root()
    env_paths = [
        os.path.join(root, "backend", ".env"),
        os.path.join(root, ".env")
    ]
    for p in env_paths:
        if os.path.exists(p):
            load_dotenv(p, override=True)


# Auto-load on module import
load_environment()


@dataclass
class GenerationConfig:
    """
    Configuration parameters for grounded response drafting.
    """
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    provider: str = "gemini"
    temperature: float = 0.1
    max_output_tokens: int = 512
    top_k: int = 5
    eval_limit: int = 30

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        """
        Loads configuration from environment variables with safe defaults.
        """
        load_environment()

        # Check key aliases in priority order
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("ASSISTIQ_LLM_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("API_KEY")
        )

        raw_model = os.getenv("ASSISTIQ_LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        # Gracefully translate superseded 1.5 model name to current 2.5 flash
        if raw_model.strip() in ["gemini-1.5-flash", "gemini-1.5-flash-latest"]:
            model_name = "gemini-2.5-flash"
        else:
            model_name = raw_model.strip()

        provider = (os.getenv("ASSISTIQ_LLM_PROVIDER") or "gemini").strip().lower()

        try:
            temperature = float(os.getenv("ASSISTIQ_LLM_TEMPERATURE", "0.1"))
        except ValueError:
            temperature = 0.1

        try:
            max_output_tokens = int(os.getenv("ASSISTIQ_LLM_MAX_TOKENS", "512"))
        except ValueError:
            max_output_tokens = 512

        try:
            top_k = int(os.getenv("ASSISTIQ_TOP_K", "5"))
        except ValueError:
            top_k = 5

        try:
            eval_limit = int(os.getenv("ASSISTIQ_EVAL_LIMIT", "30"))
        except ValueError:
            eval_limit = 30

        return cls(
            api_key=api_key if api_key and api_key.strip() else None,
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_k=top_k,
            eval_limit=eval_limit
        )

    def has_valid_api_key(self) -> bool:
        """
        Safely checks whether a non-empty API key is present without exposing its value.
        """
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def validate(self):
        """
        Validates configuration and fails with actionable guidance if credentials are missing.
        Never prints or exposes secret keys.
        """
        if self.provider == "gemini":
            if not self.has_valid_api_key():
                raise ValueError(
                    "Missing Gemini API Key. Please add GEMINI_API_KEY=your_key to backend/.env "
                    "or set the GEMINI_API_KEY environment variable. "
                    "Alternatively, set ASSISTIQ_LLM_PROVIDER=mock for offline testing."
                )
