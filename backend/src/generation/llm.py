"""
AssistIQ: LLM Provider Abstraction Module
Implements provider-agnostic LLM client interfaces, Google Gemini integration,
and a deterministic mock client for offline testing.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

from backend.src.generation.config import GenerationConfig


class SupportReply(BaseModel):
    """
    Validated structured output for drafted customer-support replies.
    """
    reply: str = Field(
        ...,
        description="The customer-facing drafted customer-support reply. Must be polite, concise, and grounded in evidence."
    )
    grounding_summary: str = Field(
        ...,
        description="Internal explanation describing which historical evidence supports this reply and why."
    )
    evidence_case_ids: List[str] = Field(
        default_factory=list,
        description="List of historical case_ids explicitly referenced or utilized to ground the draft."
    )
    grounding_status: Literal["grounded", "insufficient_evidence", "generation_failed"] = Field(
        ...,
        description="Grounding status flag: 'grounded', 'insufficient_evidence', or 'generation_failed'."
    )


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM drafting clients.
    """

    @abstractmethod
    def generate_reply(
        self,
        system_instruction: str,
        user_prompt: str
    ) -> SupportReply:
        """
        Sends system instructions and formatted user prompt to the model,
        returning a validated SupportReply instance.
        """
        pass


class GeminiLLMClient(BaseLLMClient):
    """
    Google Gemini LLM client implementing structured JSON generation
    conforming to SupportReply using the official google-genai SDK.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.config.validate()

        from google import genai
        self._client = genai.Client(api_key=self.config.api_key)

    def generate_reply(
        self,
        system_instruction: str,
        user_prompt: str
    ) -> SupportReply:
        """
        Invokes Google Gemini with JSON schema enforcement and validates response.
        """
        try:
            from google.genai import types

            generation_config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
                response_schema=SupportReply,
                system_instruction=system_instruction,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )

            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=user_prompt,
                config=generation_config
            )

            # In google-genai, if response_schema is provided, response.parsed contains the Pydantic instance
            if response.parsed is not None:
                if isinstance(response.parsed, SupportReply):
                    return response.parsed
                elif isinstance(response.parsed, dict):
                    return SupportReply.model_validate(response.parsed)
                elif isinstance(response.parsed, BaseModel):
                    return SupportReply.model_validate(response.parsed.model_dump())

            # Fallback: parse from response.text if parsed is None
            raw_text = (response.text or "").strip()
            if not raw_text:
                raise ValueError("Received empty response from Gemini API.")

            try:
                return SupportReply.model_validate_json(raw_text)
            except Exception:
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    return SupportReply.model_validate_json(json_match.group(0))
                raise ValueError(f"Could not parse valid JSON from Gemini response: {raw_text[:200]}")

        except Exception as exc:
            # Safe error handling that never leaks secret keys
            err_msg = str(exc)
            # Scrub any potential key leaks in error message
            if self.config.api_key and self.config.api_key in err_msg:
                err_msg = err_msg.replace(self.config.api_key, "[REDACTED_API_KEY]")

            return SupportReply(
                reply="I'm having trouble retrieving account information right now. Please reach out to our support team directly via DM for assistance.",
                grounding_summary=f"LLM generation failed: {err_msg[:120]}",
                evidence_case_ids=[],
                grounding_status="generation_failed"
            )


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock LLM client for offline unit testing and local verification
    without requiring API keys or incurring network latency/costs.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig(provider="mock")

    def generate_reply(
        self,
        system_instruction: str,
        user_prompt: str
    ) -> SupportReply:
        """
        Parses the prompt and generates realistic grounded outputs or handles
        insufficient evidence deterministically.
        """
        # Check for insufficient evidence triggers in prompt
        if "No historical support cases available" in user_prompt or "HISTORICAL EVIDENCE" not in user_prompt:
            return SupportReply(
                reply="Hey there! Could you give us a bit more detail on what's happening? Let us know your device and app version so we can help.",
                grounding_summary="No historical evidence was available; safely drafted a request for more context.",
                evidence_case_ids=[],
                grounding_status="insufficient_evidence"
            )

        # Extract available Case IDs from prompt
        case_ids = re.findall(r"Case ID:\s*(case_\d+)", user_prompt)
        cited_ids = case_ids[:2] if case_ids else []

        # Extract customer query from prompt
        msg_match = re.search(r'CUSTOMER MESSAGE:\s*\n"([^"]+)"', user_prompt)
        customer_msg = msg_match.group(1).lower() if msg_match else ""

        # Craft grounded replies based on query intent domain
        if any(w in customer_msg for w in ["charge", "paid", "refund", "bill", "money", "fee", "cost"]):
            reply = (
                "Hey there! We can certainly look into those charges for you. "
                "Could you send us a DM with your account's email address and username? "
                "We'll check things out backstage."
            )
            summary = "Grounded in historical Spotify billing resolutions recommending secure DM account verification for refund inquiries."
            status = "grounded"
        elif any(w in customer_msg for w in ["pause", "stop", "skip", "crash", "glitch", "play", "won't play"]):
            reply = (
                "Hi! That definitely doesn't sound right. Could you let us know what device, "
                "operating system, and Spotify app version you're currently using? "
                "We'll see what troubleshooting steps we can recommend."
            )
            summary = "Grounded in historical Spotify playback resolutions gathering client OS/version telemetry."
            status = "grounded"
        elif any(w in customer_msg for w in ["password", "login", "log in", "hack", "email", "account"]):
            reply = (
                "Hey! Help is here. If you're having trouble accessing your account, "
                "please send us a DM with the email address linked to your Spotify account "
                "so we can help you regain access securely."
            )
            summary = "Grounded in historical Spotify account recovery procedures via private message verification."
            status = "grounded"
        elif any(w in customer_msg for w in ["hello", "hey", "hi", "thanks", "thank you", "cool"]):
            reply = "Hey there! Thanks for reaching out. Let us know how we can help you out today!"
            summary = "Grounded in conversational greeting support responses."
            status = "grounded"
        else:
            if cited_ids:
                reply = (
                    "Hey there! We'd be happy to look into this for you. "
                    "Can you let us know what device and Spotify version you're using? "
                    "Feel free to send us a DM if you'd prefer to share account details privately."
                )
                summary = "Grounded in general technical and support inquiry procedures from retrieved historical cases."
                status = "grounded"
            else:
                reply = "Hey! Could you send us a quick DM with more details? We'll be happy to take a closer look."
                summary = "Insufficient specific evidence; safely requested further details."
                status = "insufficient_evidence"

        return SupportReply(
            reply=reply,
            grounding_summary=summary,
            evidence_case_ids=cited_ids,
            grounding_status=status
        )


def get_llm_client(config: Optional[GenerationConfig] = None) -> BaseLLMClient:
    """
    Factory function returning the appropriate LLM client based on configuration.
    """
    if config is None:
        config = GenerationConfig.from_env()

    if config.provider == "mock":
        return MockLLMClient(config)

    if config.provider == "gemini":
        if config.has_valid_api_key():
            return GeminiLLMClient(config)
        else:
            # When API key is not yet set, provide helpful error
            raise ValueError(
                "Missing Gemini API Key. Please add GEMINI_API_KEY=your_key to backend/.env "
                "or set the GEMINI_API_KEY environment variable. "
                "To run tests or local exploration without an API key, set ASSISTIQ_LLM_PROVIDER=mock."
            )

    raise ValueError(f"Unsupported LLM provider: {config.provider}")
