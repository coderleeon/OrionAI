"""
OrionAI — LLM Inference Client
================================
Abstracts all LLM calls behind a single async interface.
Currently wraps OpenAI's chat completions API.
To swap to vLLM or a local model, replace only this module.

Modes
-----
* **Real mode**: OPENAI_API_KEY is set → makes actual API calls.
* **Mock mode**: key is placeholder → returns deterministic mocked responses
  so the full pipeline runs without any API key during development/testing.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from config.settings import settings
from utils.logger import StructuredLogger

log = StructuredLogger(agent_name="LLMClient")

# ── Prompt prefix that instructs every agent to respond in JSON ────────────
_JSON_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: You MUST respond with a valid JSON object only. "
    "No preamble. No markdown. No explanation outside the JSON."
)

# ── Mock responses keyed by agent role ─────────────────────────────────────
_MOCK_RESPONSES: dict[str, Any] = {
    "planner": {
        "steps": [
            {"id": 1, "description": "Understand the core question", "type": "analysis"},
            {"id": 2, "description": "Gather background context", "type": "retrieval"},
            {"id": 3, "description": "Synthesize a comprehensive answer", "type": "synthesis"},
        ]
    },
    "retriever": {
        "context": (
            "Photosynthesis is the process by which green plants, algae, and some bacteria "
            "convert sunlight energy into chemical energy stored as glucose. "
            "The overall equation: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂."
        ),
        "sources": [
            {"title": "Biology Textbook Ch.7", "relevance": 0.93},
            {"title": "Encyclopedia Britannica — Photosynthesis", "relevance": 0.88},
        ],
    },
    "executor": {
        "results": [
            {"step_id": 1, "output": "The query asks for a clear explanation of photosynthesis."},
            {"step_id": 2, "output": "Context retrieved: light-dependent and light-independent reactions."},
            {"step_id": 3, "output": "Photosynthesis converts CO₂ and H₂O into glucose using solar energy."},
        ],
        "combined": (
            "Photosynthesis is how plants make food. Using sunlight, they convert carbon dioxide "
            "and water into glucose and oxygen. The process has two stages: light-dependent reactions "
            "(in the thylakoids) and the Calvin cycle (in the stroma)."
        ),
    },
    "critic": {
        "approved": True,
        "confidence": 0.91,
        "feedback": "The answer is factually accurate, well-structured, and covers the key stages.",
    },
}


class LLMClient:
    """
    Async LLM client with automatic JSON parsing and fallback mock mode.

    Parameters
    ----------
    role : str
        Agent role label used for mock lookups ("planner", "retriever", etc.)
    """

    def __init__(self, role: str = "executor") -> None:
        self._role = role
        self._mock_mode = settings.openai_api_key == "sk-placeholder"

        if not self._mock_mode:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None  # type: ignore[assignment]
            log.warning(
                "LLMClient running in MOCK mode — no real API calls will be made.",
                role=role,
            )

    async def chat_complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        """
        Send a chat completion request and return a parsed JSON dict.

        Parameters
        ----------
        system_prompt : str
            The system instruction for this agent.
        user_prompt : str
            The user-facing input (query + context).
        session_id : str
            Logged for traceability.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response from the LLM (or mock).
        """
        logger = log.bind(session_id=session_id, role=self._role)

        if self._mock_mode:
            return await self._mock_response(logger)

        # Append JSON enforcement to system prompt
        enforced_system = system_prompt + _JSON_SYSTEM_SUFFIX

        logger.info(
            "Calling LLM",
            model=settings.model_name,
            user_prompt_chars=len(user_prompt),
        )

        try:
            response = await self._client.chat.completions.create(
                model=settings.model_name,
                temperature=settings.model_temperature,
                max_tokens=settings.model_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": enforced_system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            logger.info(
                "LLM responded",
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
            return self._parse_json(raw, logger)

        except OpenAIError as exc:
            logger.error("OpenAI API error", error=str(exc))
            raise

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _mock_response(self, logger: StructuredLogger) -> dict[str, Any]:
        """Return a deterministic mock response after a brief artificial delay."""
        await asyncio.sleep(0.05)  # Simulate network latency
        payload = _MOCK_RESPONSES.get(self._role, {"result": "mock"})
        logger.debug("Returning mock LLM response", mock_role=self._role)
        return payload

    @staticmethod
    def _parse_json(raw: str, logger: StructuredLogger) -> dict[str, Any]:
        """Attempt to parse raw LLM output as JSON, with fallback extraction."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract a JSON block from markdown code fences
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            logger.warning("Could not parse LLM output as JSON; wrapping in raw field.", raw=raw[:200])
            return {"raw": raw}
