"""
OrionAI — Application Settings
================================
Config-driven system using Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the OrionAI system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="sk-placeholder",
        description="OpenAI API key. Set via OPENAI_API_KEY env var.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier (e.g. gpt-4o-mini, gpt-4o, gpt-3.5-turbo).",
    )
    model_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM calls.",
    )
    model_max_tokens: int = Field(
        default=2048,
        ge=64,
        description="Maximum tokens per LLM completion.",
    )

    # ── Orchestration ─────────────────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum Critic-triggered retry attempts per request.",
    )
    critic_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0–1) from Critic to pass without retry.",
    )

    # ── Memory / Redis ────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL. Falls back to in-memory if unavailable.",
    )
    session_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="TTL for session data stored in Redis.",
    )

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_enabled: bool = Field(
        default=True,
        description="Enable in-process TTL cache for identical queries.",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=10,
        description="TTL (seconds) for cached query results.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application log level.",
    )
    log_file: str = Field(
        default="logs/orion.log",
        description="Path to the structured log file.",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_title: str = "OrionAI Inference Engine"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Retriever simulation ──────────────────────────────────────────────────
    retriever_top_k: int = Field(
        default=3,
        ge=1,
        description="Number of context documents to return from retriever.",
    )

    @field_validator("openai_api_key")
    @classmethod
    def warn_placeholder_key(cls, v: str) -> str:
        if v == "sk-placeholder":
            import warnings
            warnings.warn(
                "OPENAI_API_KEY is not set. LLM calls will use mock mode.",
                stacklevel=2,
            )
        return v


# Module-level singleton — import this everywhere
settings = Settings()
