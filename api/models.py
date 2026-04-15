"""
OrionAI — Pydantic API Models
==============================
Defines all request/response models for the FastAPI layer.
Using strict Pydantic v2 validation throughout.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ──────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """
    Payload for POST /run.

    Fields
    ------
    query : str
        The user's task or question. Must be non-empty.
    session_id : str
        Caller-managed session identifier. Auto-generated UUID if omitted.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=4096,
        description="The user query or task to process.",
        examples=["Explain photosynthesis in simple terms."],
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=128,
        description="Session identifier for memory continuity.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must be a non-empty string after stripping whitespace.")
        return stripped


# ── Nested response models ────────────────────────────────────────────────────

class PlanStep(BaseModel):
    id: int
    description: str
    type: str


class AgentLatencies(BaseModel):
    """Wall-clock time (ms) spent in each agent call."""

    class Config:
        extra = "allow"  # Allow dynamic keys like executor_attempt_0


# ── Response ──────────────────────────────────────────────────────────────────

class InferenceResponse(BaseModel):
    """
    Structured response from POST /run.

    Fields
    ------
    request_id : str
        Unique ID for this inference request (for tracing/logging).
    session_id : str
        Echo of the request's session_id.
    query : str
        Echo of the original query.
    steps : list[PlanStep]
        The plan produced by the Planner Agent.
    final_answer : str
        The answer approved (or last-attempted) by the Critic Agent.
    retries : int
        Number of Executor→Critic retry cycles performed (0 = first attempt succeeded).
    latency : float
        Total pipeline wall-clock time in seconds.
    critic_confidence : float
        Final confidence score from the Critic Agent (0–1).
    critic_feedback : str
        Last Critic feedback message.
    agent_latencies_ms : dict
        Per-agent wall-clock times in milliseconds.
    from_cache : bool
        True if this result was served from the query cache.
    error : str | None
        Error message if the pipeline encountered an unrecoverable failure.
    """

    request_id: str
    session_id: str
    query: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    retries: int = Field(ge=0)
    latency: float = Field(ge=0.0, description="Total pipeline latency in seconds.")
    critic_confidence: float = Field(ge=0.0, le=1.0)
    critic_feedback: str = ""
    agent_latencies_ms: dict[str, Any] = Field(default_factory=dict)
    from_cache: bool = False
    error: Optional[str] = None


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    memory_backend: str
    cache_enabled: bool
